from __future__ import annotations

from pathlib import Path

import pytest

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.complete_formal_execution import audit_rows, synthetic_rows


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def contract() -> CompleteFormalContract:
    return CompleteFormalContract.from_path(ROOT / "configs" / "rsc_pf_complete_formal_v1.json")


@pytest.fixture()
def complete_rows(contract):
    return synthetic_rows(contract, "gate1", origin_count=8709)


def test_audit_recomputes_lp_calls(contract, complete_rows):
    audit = audit_rows(contract, complete_rows, origin_count=8709)
    assert audit.status == "pass"
    assert audit.rows["Differentiable-LP/2026"].inference_lp_calls == 8709
    assert audit.rows["RSC-PF/2026"].inference_lp_calls == 0


def test_audit_rejects_declared_lp_call_mismatch(contract, complete_rows):
    complete_rows[("Differentiable-LP", 2026)]["inference_lp_calls"] = 0
    audit = audit_rows(contract, complete_rows, origin_count=8709)
    assert audit.status == "fail"
    assert any("inference LP calls" in failure for failure in audit.failures)


def test_audit_rejects_access_to_excluded_year(contract, complete_rows):
    complete_rows[("Seasonal-Naive-PTO", None)]["excluded_year_accessed"] = True
    audit = audit_rows(contract, complete_rows, origin_count=8709)
    assert audit.status == "fail"
    assert audit.excluded_year_accessed is True
