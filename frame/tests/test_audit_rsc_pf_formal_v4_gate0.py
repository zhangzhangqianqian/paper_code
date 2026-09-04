from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_rsc_pf_formal_v4_gate0 import audit_gate0_run
from src.joint_dispatch.formal_v4_gate0 import Gate0Context, MANDATORY_CHECK_IDS, execute_gate0


def _checkers():
    return {check_id: (lambda check_id=check_id: {"passed": True}) for check_id in MANDATORY_CHECK_IDS}


def test_independent_auditor_verifies_clean_run(tmp_path: Path):
    result = execute_gate0(Gate0Context(tmp_path, _checkers()), "audit-clean")
    payload = audit_gate0_run(result.run_root)
    assert payload["verified"] is True
    persisted = json.loads((result.run_root.parent / "INDEPENDENT_GATE0_AUDIT.json").read_text(encoding="utf-8"))
    assert persisted["receipt_sha256"] == payload["receipt_sha256"]


def test_independent_auditor_rejects_tampered_receipt(tmp_path: Path):
    result = execute_gate0(Gate0Context(tmp_path, _checkers()), "audit-tampered")
    path = result.receipt_path
    assert path is not None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["authorized_gate1"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    audit = audit_gate0_run(result.run_root, write_receipt=False)
    assert audit["verified"] is False
    assert any("authorization marker" in error for error in audit["errors"])


def test_independent_auditor_rejects_evaluation_artifact(tmp_path: Path):
    result = execute_gate0(Gate0Context(tmp_path, _checkers()), "audit-eval")
    forbidden = result.run_root / "evaluation" / "2020.npz"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"blocked")
    audit = audit_gate0_run(result.run_root, write_receipt=False)
    assert audit["verified"] is False
    assert audit["forbidden_artifacts"]


def test_auditor_does_not_import_gate0_decision_function():
    source = Path(__file__).parents[1] / "scripts" / "audit_rsc_pf_formal_v4_gate0.py"
    text = source.read_text(encoding="utf-8")
    assert "execute_gate0" not in text
