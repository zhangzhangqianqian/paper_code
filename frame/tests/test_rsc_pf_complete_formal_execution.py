from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.complete_formal_execution import (
    decide_gate1,
    run_gate1,
    run_gate2,
    synthetic_rows,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def contract() -> CompleteFormalContract:
    return CompleteFormalContract.from_path(ROOT / "configs" / "rsc_pf_complete_formal_v1.json")


@pytest.fixture()
def fake_rows(contract):
    return synthetic_rows(contract, "gate1", origin_count=contract.selection_origin_count)


def test_gate1_rejects_one_missing_difflp_seed(contract, fake_rows):
    fake_rows.pop(("Differentiable-LP", 2030))
    decision = decide_gate1(contract, fake_rows)
    assert decision.authorized_gate2 is False
    assert "Differentiable-LP/2030" in decision.missing_rows


def test_gate1_writes_transition_only_after_independent_audit(contract, tmp_path):
    decision, audit, stage_dir = run_gate1(contract, tmp_path)
    assert decision.authorized_next_gate is False
    assert audit.status == "pass"
    transition = json.loads((stage_dir / "GATE1_TRANSITION.json").read_text(encoding="utf-8"))
    assert transition["authorized_gate2"] is False
    assert transition["synthetic"] is True
    assert transition["paper_result"] is False
    assert transition["evaluation_year_accessed"] is False
    assert transition["test_set_accessed"] is False
    assert (stage_dir / "EXECUTION_RECEIPT.json").is_file()
    assert (stage_dir / "AUDIT_RECEIPT.json").is_file()


def test_gate2_refuses_rejected_gate1(contract, tmp_path):
    rejected = tmp_path / "GATE1_TRANSITION.json"
    rejected.write_text(json.dumps({"contract_sha256": contract.contract_sha256, "authorized_gate2": False, "evaluation_year_accessed": False, "test_set_accessed": False}), encoding="utf-8")
    with pytest.raises(PermissionError):
        run_gate2(contract, tmp_path, rejected)


def test_gate2_rejects_synthetic_transition(contract, tmp_path):
    _, _, stage_dir = run_gate1(contract, tmp_path, run_id="gate1")
    with pytest.raises(PermissionError, match="authorized Gate 1|non-synthetic"):
        run_gate2(contract, tmp_path, stage_dir / "GATE1_TRANSITION.json", run_id="gate2")


def test_gate1_rejects_future_or_excluded_access(contract, fake_rows):
    fake_rows[("RSC-PF", 2026)]["evaluation_year_accessed"] = True
    fake_rows[("RSC-PF", 2026)]["excluded_year_accessed"] = True
    decision = decide_gate1(contract, fake_rows)
    assert decision.authorized_next_gate is False
    assert any("evaluation year" in value for value in decision.failures)
    assert any("excluded year" in value for value in decision.failures)


def test_row_reuse_rejects_lineage_mismatch(contract, tmp_path):
    run_gate1(contract, tmp_path, run_id="reuse")
    changed = synthetic_rows(contract, "gate1", origin_count=contract.selection_origin_count)
    changed[("RSC-PF", 2026)]["source_sha256"] = "f" * 64
    with pytest.raises(PermissionError, match="lineage/hash mismatch"):
        run_gate1(contract, tmp_path, changed, run_id="reuse")
