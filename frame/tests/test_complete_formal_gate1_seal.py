from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.complete_formal_gate1_seal import seal_interrupted_gate1_run


def _make_run(tmp_path: Path) -> Path:
    root = tmp_path / "interrupted"
    gate1 = root / "gate1"
    (root / "protocol").mkdir(parents=True)
    gate1.mkdir(parents=True)
    (gate1 / "DATA_LINEAGE.json").write_text(
        json.dumps({
            "contract_sha256": "a" * 64,
            "evaluation_year_accessed": False,
        }),
        encoding="utf-8",
    )
    return root


def test_seal_writes_interrupted_failure_receipt(tmp_path):
    root = _make_run(tmp_path)
    path = seal_interrupted_gate1_run(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "rsc-pf-complete-formal-gate1-failure-v1"
    assert payload["error_type"] == "InterruptedRun"
    assert payload["interrupted"] is True
    assert payload["evaluation_year_accessed"] is False


def test_seal_is_immutable_and_refuses_second_write(tmp_path):
    root = _make_run(tmp_path)
    seal_interrupted_gate1_run(root)
    with pytest.raises(FileExistsError):
        seal_interrupted_gate1_run(root)


def test_seal_refuses_completion_artifact(tmp_path):
    root = _make_run(tmp_path)
    (root / "protocol" / "GATE1_TRANSITION.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="completion"):
        seal_interrupted_gate1_run(root)


def test_seal_refuses_any_evaluation_access_marker(tmp_path):
    root = _make_run(tmp_path)
    (root / "gate1" / "ROWS.json").write_text(
        json.dumps({"evaluation_year_accessed": True}),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="evaluation_year_accessed"):
        seal_interrupted_gate1_run(root)
