from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_rsc_pf_complete_formal_gate0 import run_gate0
from src.joint_dispatch.complete_formal_contract import CompleteFormalContract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "rsc_pf_complete_formal_v1.json"


def _ops(*, official=True, difflp=True, hours=1.0):
    return {
        "rsc_training": lambda: {"loss_finite": True, "native_gradient_nonzero": True, "timing": {"p95_seconds": 0.001}},
        "exact_lp": lambda: {"success": True, "timing": {"p95_seconds": 0.001}},
        "difflp": lambda: {"native_gradient_nonzero": difflp, "timing": {"p95_seconds": 0.001}},
        "direct_policy": lambda: {"success": True},
        "seasonal_naive": lambda: {"success": True},
        "itransformer": lambda: {"official_source_hash_verified": official},
    }


def test_gate0_requires_native_difflp_and_official_itransformer(tmp_path):
    receipt = run_gate0(contract_path=CONTRACT, output_root=tmp_path, run_id="gate0_missing_native", operations=_ops(official=False, difflp=False))
    assert receipt["difflp"]["native_gradient_nonzero"] is False
    assert receipt["itransformer"]["official_source_hash_verified"] is False
    assert receipt["all_method_families_smoked"] is False
    assert receipt["authorized_gate1_training"] is False


def test_gate0_authorizes_with_native_family_coverage(tmp_path):
    receipt = run_gate0(contract_path=CONTRACT, output_root=tmp_path, run_id="gate0_ok", train_windows=16, operations=_ops())
    assert receipt["difflp"]["native_gradient_nonzero"] is True
    assert receipt["itransformer"]["official_source_hash_verified"] is True
    assert receipt["all_method_families_smoked"] is True
    assert receipt["authorized_gate1_training"] is True
    gate = tmp_path / "gate0_ok" / "gate0"
    assert (gate / "GATE0_RECEIPT.json").is_file()
    assert (gate / "GATE0_AUDIT.json").is_file()
    transition = json.loads((gate / "GATE0_TRANSITION.json").read_text(encoding="utf-8"))
    assert transition["authorized_gate1_training"] is True


def test_gate0_reports_long_projection_without_falsifying_pass(tmp_path):
    slow = {
        "rsc_training": lambda: {"loss_finite": True, "native_gradient_nonzero": True, "timing": {"p95_seconds": 1.0}},
        "exact_lp": lambda: {"success": True, "timing": {"p95_seconds": 1.0}},
        "difflp": lambda: {"native_gradient_nonzero": True, "timing": {"p95_seconds": 1.0}},
        "direct_policy": lambda: {"success": True},
        "seasonal_naive": lambda: {"success": True},
        "itransformer": lambda: {"official_source_hash_verified": True},
    }
    receipt = run_gate0(contract_path=CONTRACT, output_root=tmp_path, run_id="gate0_slow", train_windows=34959, operations=slow)
    assert receipt["resource_projection"]["projected_total_hours"] > 24.0
    assert receipt["authorized_gate1_training"] is False
    assert receipt["status"] == "requires_authorization"
    audit = json.loads((tmp_path / "gate0_slow" / "gate0" / "GATE0_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["requires_explicit_authorization"] is True
    assert "projected formal run exceeds 24 hours" in audit["failures"]


def test_gate0_transition_denies_evaluation_access(tmp_path):
    run_gate0(contract_path=CONTRACT, output_root=tmp_path, run_id="gate0_boundary", train_windows=16, operations=_ops())
    gate = tmp_path / "gate0_boundary" / "gate0"
    transition = json.loads((gate / "GATE0_TRANSITION.json").read_text(encoding="utf-8"))
    assert transition["evaluation_year_accessed"] is False
    assert transition["excluded_year_accessed"] is False
