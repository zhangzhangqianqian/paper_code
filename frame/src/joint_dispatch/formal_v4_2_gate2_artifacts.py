"""Immutable Gate 2 rollout and row evidence for formal-v4.2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .formal_v4_2_artifacts import LineageError, canonical_sha256, sha256_file, write_once_json


ROW_SCHEMA = "formal-v4.2-gate2-row-v1"
ROLLOUT_SCHEMA = "formal-v4.2-gate2-rollout-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HASH_FIELDS = (
    "contract_sha256",
    "source_manifest_sha256",
    "train_manifest_sha256",
    "calibration_manifest_sha256",
    "evaluation_manifest_sha256",
    "normalization_sha256",
)
_STOCHASTIC_FILES = ("CHECKPOINT.pt", "TRAINING_RECEIPT.json")
_COMMON_FILES = ("ROLLOUT.npz", "METRICS.json")


def _require_hash(value: Any, name: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise LineageError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing.npz")
    if temporary.exists():
        raise FileExistsError(f"stale artifact write exists: {temporary}")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def write_gate2_rollout(
    path: str | Path,
    arrays: Mapping[str, Any],
    lineage: Mapping[str, str],
) -> str:
    """Persist chronological arrays with their complete upstream lineage."""

    target = Path(path)
    for field in _HASH_FIELDS:
        _require_hash(lineage.get(field, ""), field)
    if not arrays:
        raise LineageError("Gate 2 rollout must contain arrays")
    encoded: dict[str, Any] = {}
    lengths: set[int] = set()
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind in {"f", "c"} and not np.isfinite(array).all():
            raise LineageError(f"rollout array {name} contains non-finite values")
        encoded[str(name)] = array
        if array.ndim:
            lengths.add(int(array.shape[0]))
    if not lengths or len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise LineageError("rollout arrays must share one non-empty leading dimension")
    encoded["schema"] = np.asarray(ROLLOUT_SCHEMA)
    encoded["lineage_json"] = np.asarray(json.dumps(dict(lineage), sort_keys=True))
    _atomic_npz(target, encoded)
    return sha256_file(target)


def _expected_file_hash(row_dir: Path, name: str, payload: Mapping[str, Any]) -> str:
    field = {
        "CHECKPOINT.pt": "checkpoint_sha256",
        "TRAINING_RECEIPT.json": "training_receipt_sha256",
        "ROLLOUT.npz": "rollout_sha256",
        "METRICS.json": "metrics_sha256",
    }[name]
    path = row_dir / name
    if not path.is_file():
        raise LineageError(f"{name} is missing")
    actual = sha256_file(path)
    expected = _require_hash(payload.get(field, ""), field)
    if actual != expected:
        raise LineageError(f"{field} does not match {name}")
    return actual


def write_gate2_row_receipt(
    row_dir: str | Path,
    payload: Mapping[str, Any],
) -> str:
    """Validate material evidence before committing a complete row receipt."""

    directory = Path(row_dir)
    seed = payload.get("seed")
    stochastic = seed is not None
    for field in _HASH_FIELDS:
        _require_hash(payload.get(field, ""), field)
    required = (_STOCHASTIC_FILES if stochastic else ()) + _COMMON_FILES
    for name in required:
        _expected_file_hash(directory, name, payload)
    if stochastic:
        training = json.loads((directory / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
        if int(training.get("optimizer_steps", 0)) <= 0:
            raise LineageError("TRAINING_RECEIPT optimizer_steps must be positive")
    else:
        if payload.get("checkpoint_sha256") != "not_applicable":
            raise LineageError("deterministic checkpoint_sha256 must be not_applicable")
        if payload.get("training_receipt_sha256") != "not_applicable":
            raise LineageError("deterministic training_receipt_sha256 must be not_applicable")
    receipt = {
        **dict(payload),
        "schema": ROW_SCHEMA,
        "status": "complete",
        "stochastic": stochastic,
    }
    return write_once_json(directory / "ROW_RECEIPT.json", receipt)


def validate_gate2_row(
    row_dir: str | Path,
    expected: Mapping[str, str],
) -> Mapping[str, Any]:
    directory = Path(row_dir)
    receipt_path = directory / "ROW_RECEIPT.json"
    if not receipt_path.is_file():
        raise LineageError("ROW_RECEIPT.json is missing")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if payload.get("schema") != ROW_SCHEMA or payload.get("status") != "complete":
        raise LineageError("Gate 2 row receipt is not complete")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise LineageError(f"{field} lineage mismatch")
    for field in _HASH_FIELDS:
        _require_hash(payload.get(field, ""), field)
    required = (_STOCHASTIC_FILES if payload.get("seed") is not None else ()) + _COMMON_FILES
    for name in required:
        _expected_file_hash(directory, name, payload)
    if payload.get("seed") is not None:
        training = json.loads((directory / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
        if int(training.get("optimizer_steps", 0)) <= 0:
            raise LineageError("trained checkpoint evidence has no optimizer steps")
    return payload


__all__ = [
    "ROW_SCHEMA",
    "ROLLOUT_SCHEMA",
    "validate_gate2_row",
    "write_gate2_rollout",
    "write_gate2_row_receipt",
]
