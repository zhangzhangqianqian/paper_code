from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_2_artifacts import LineageError, sha256_file
from src.joint_dispatch.formal_v4_2_gate2_artifacts import (
    validate_gate2_row,
    write_gate2_rollout,
    write_gate2_row_receipt,
)


HASHES = {
    "contract_sha256": "1" * 64,
    "source_manifest_sha256": "2" * 64,
    "train_manifest_sha256": "3" * 64,
    "calibration_manifest_sha256": "4" * 64,
    "evaluation_manifest_sha256": "5" * 64,
    "normalization_sha256": "6" * 64,
}


def _json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return sha256_file(path)


def _stochastic_row(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    (root / "CHECKPOINT.pt").write_bytes(b"trained")
    training_hash = _json(root / "TRAINING_RECEIPT.json", {"optimizer_steps": 4})
    rollout_hash = write_gate2_rollout(
        root / "ROLLOUT.npz",
        {"origin_index": np.arange(3), "objective": np.ones(3)},
        HASHES,
    )
    metrics_hash = _json(root / "METRICS.json", {"penalized_objective": 1.0})
    return {
        **HASHES,
        "method_id": "RSC-PF",
        "seed": 2026,
        "checkpoint_sha256": sha256_file(root / "CHECKPOINT.pt"),
        "training_receipt_sha256": training_hash,
        "rollout_sha256": rollout_hash,
        "metrics_sha256": metrics_hash,
    }


def test_row_receipt_requires_persisted_checkpoint_and_rollout(tmp_path: Path) -> None:
    payload = {**HASHES, "method_id": "RSC-PF", "seed": 2026}
    with pytest.raises(LineageError, match="CHECKPOINT"):
        write_gate2_row_receipt(tmp_path, payload)


def test_complete_row_round_trip(tmp_path: Path) -> None:
    payload = _stochastic_row(tmp_path)
    write_gate2_row_receipt(tmp_path, payload)
    restored = validate_gate2_row(tmp_path, HASHES)
    assert restored["method_id"] == "RSC-PF"
    assert restored["seed"] == 2026


def test_resume_rejects_changed_rollout_hash(tmp_path: Path) -> None:
    payload = _stochastic_row(tmp_path)
    write_gate2_row_receipt(tmp_path, payload)
    with (tmp_path / "ROLLOUT.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(LineageError, match="rollout_sha256"):
        validate_gate2_row(tmp_path, HASHES)


def test_training_receipt_requires_optimizer_steps(tmp_path: Path) -> None:
    payload = _stochastic_row(tmp_path)
    payload["training_receipt_sha256"] = _json(
        tmp_path / "TRAINING_RECEIPT.json",
        {"optimizer_steps": 0},
    )
    with pytest.raises(LineageError, match="optimizer_steps"):
        write_gate2_row_receipt(tmp_path, payload)


def test_deterministic_row_uses_explicit_not_applicable_hashes(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    rollout_hash = write_gate2_rollout(
        tmp_path / "ROLLOUT.npz",
        {"origin_index": np.arange(2), "objective": np.ones(2)},
        HASHES,
    )
    metrics_hash = _json(tmp_path / "METRICS.json", {"penalized_objective": 1.0})
    payload = {
        **HASHES,
        "method_id": "Seasonal-Naive-PTO",
        "seed": None,
        "checkpoint_sha256": "not_applicable",
        "training_receipt_sha256": "not_applicable",
        "rollout_sha256": rollout_hash,
        "metrics_sha256": metrics_hash,
    }
    write_gate2_row_receipt(tmp_path, payload)
    assert validate_gate2_row(tmp_path, HASHES)["stochastic"] is False
