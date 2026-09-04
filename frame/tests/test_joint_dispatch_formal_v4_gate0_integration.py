from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_gate0 import Gate0Context, MANDATORY_CHECK_IDS, execute_gate0


def _checkers(failing: str | None = None):
    return {
        check_id: (lambda check_id=check_id: {"passed": check_id != failing, "evidence": check_id})
        for check_id in MANDATORY_CHECK_IDS
    }


def test_transactional_gate0_authorizes_only_after_final_receipt(tmp_path: Path):
    result = execute_gate0(Gate0Context(tmp_path, _checkers(), metadata={"commit": "fixture"}), "clean-run")
    assert result.authorized_gate1
    assert result.receipt_path and result.receipt_path.exists()
    assert result.authorization_path and result.authorization_path.exists()
    assert not (result.run_root / "IN_PROGRESS.json").exists()
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["authorized_gate1"] is True
    assert receipt["mandatory_failures"] == []
    auth = json.loads(result.authorization_path.read_text(encoding="utf-8"))
    assert auth["receipt_sha256"]


@pytest.mark.parametrize("failure", MANDATORY_CHECK_IDS)
def test_every_injected_mandatory_failure_denies_and_writes_no_authorization(tmp_path: Path, failure: str):
    result = execute_gate0(Gate0Context(tmp_path, _checkers(failure)), f"failed-{failure}")
    assert result.authorized_gate1 is False
    assert result.authorization_path is None
    assert result.receipt_path and result.receipt_path.exists()
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["authorized_gate1"] is False
    assert failure in receipt["mandatory_failures"]
    assert not (result.run_root / "GATE0_AUTHORIZATION.json").exists()


def test_gate0_rejects_reuse_or_unsafe_run_id(tmp_path: Path):
    execute_gate0(Gate0Context(tmp_path, _checkers()), "one")
    with pytest.raises(FileExistsError):
        execute_gate0(Gate0Context(tmp_path, _checkers()), "one")
    with pytest.raises(ValueError):
        execute_gate0(Gate0Context(tmp_path, _checkers()), "../escape")


def test_gate0_rejects_missing_and_unknown_checkers(tmp_path: Path):
    with pytest.raises(ValueError, match="registry mismatch"):
        execute_gate0(Gate0Context(tmp_path, {"protocol_freeze": lambda: True}), "missing")
