"""Lineage and write-once checks for formal-v4.3 artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target


def validate_v43_checkpoint(lineage: Mapping[str, Any]) -> None:
    if lineage.get("contract_version") != "formal-v4.3":
        raise ValueError("checkpoint lineage must belong to formal-v4.3")
    required = (
        "contract_sha256", "normalization_sha256", "thermal_magnitude_receipt_sha256",
        "parent_checkpoint_sha256", "teacher_sha256", "method_id", "seed",
    )
    missing = [name for name in required if name not in lineage]
    if missing:
        raise ValueError(f"checkpoint lineage is missing {missing[0]}")
    if not isinstance(lineage["seed"], int) or lineage["seed"] < 0:
        raise ValueError("checkpoint seed must be a non-negative integer")


def assert_shared_parent_v43(joint: Mapping[str, Any], decoupled: Mapping[str, Any]) -> None:
    for name in ("stage_p_parent_sha256", "stage_s_parent_sha256", "training_seed"):
        if joint.get(name) != decoupled.get(name):
            raise ValueError(f"joint and decoupled branches differ at {name}")


__all__ = ["assert_shared_parent_v43", "sha256_file", "validate_v43_checkpoint", "write_once_json"]
