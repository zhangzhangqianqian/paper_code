"""Seal an abruptly interrupted complete-v1 Gate 1 run for safe recovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .formal_v4_2_artifacts import sha256_file, write_once_json


FAILURE_SCHEMA = "rsc-pf-complete-formal-gate1-failure-v1"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"interrupted Gate 1 JSON artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"interrupted Gate 1 JSON artifact must be an object: {path}")
    return payload


def _assert_no_evaluation_access(run_root: Path) -> None:
    for path in run_root.rglob("*.json"):
        payload = _load_object(path)
        for field in (
            "evaluation_year_accessed",
            "test_set_accessed",
            "excluded_years_accessed",
            "excluded_year_accessed",
        ):
            if payload.get(field) is True:
                raise PermissionError(
                    f"cannot seal interrupted Gate 1 run after {field}=true: {path}"
                )


def seal_interrupted_gate1_run(
    run_root: str | Path,
    *,
    reason: str = "run interrupted before Gate 1 completion; no evaluation was performed",
) -> Path:
    """Write an immutable failure receipt for a run stopped without cleanup.

    This function does not train, evaluate, copy, or modify existing artifacts.
    It only adds the missing failure receipt after proving that the run has no
    completion transition and no artifact reporting evaluation/test-set access.
    """

    root = Path(run_root).resolve()
    gate1 = root / "gate1"
    protocol = root / "protocol"
    lineage_path = gate1 / "DATA_LINEAGE.json"
    failure_path = gate1 / "GATE1_FAILURE.json"
    if not gate1.is_dir() or not protocol.is_dir() or not lineage_path.is_file():
        raise FileNotFoundError(f"interrupted Gate 1 run is incomplete: {root}")
    if failure_path.exists():
        raise FileExistsError(f"Gate 1 failure receipt already exists: {failure_path}")
    for completion in (gate1 / "GATE1_EVIDENCE.json", protocol / "GATE1_TRANSITION.json"):
        if completion.exists():
            raise ValueError(f"refusing to seal a run with completion artifact: {completion}")

    lineage = _load_object(lineage_path)
    if lineage.get("evaluation_year_accessed") is not False:
        raise PermissionError("interrupted Gate 1 lineage does not prove evaluation_year_accessed=false")
    _assert_no_evaluation_access(root)

    payload = {
        "schema_version": FAILURE_SCHEMA,
        "run_id": root.name,
        "contract_sha256": lineage.get("contract_sha256"),
        "error_type": "InterruptedRun",
        "reason": str(reason),
        "evaluation_year_accessed": False,
        "resume_from": None,
        "recovery_manifest_sha256": None,
        "interrupted": True,
        "source_modified": False,
    }
    if not isinstance(payload["contract_sha256"], str) or len(payload["contract_sha256"]) != 64:
        raise ValueError("interrupted Gate 1 lineage has no valid contract_sha256")
    write_once_json(failure_path, payload)
    return failure_path


__all__ = ["FAILURE_SCHEMA", "seal_interrupted_gate1_run"]
