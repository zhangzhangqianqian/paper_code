from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from scripts.audit_rsc_pf_architecture_freeze import (  # noqa: E402
    GATES,
    EXPECTED_SHAPES,
    audit_artifact_gate,
    audit_data_gate,
    audit_forward_gate,
    audit_gradient_gate,
    audit_training_gate,
    build_architecture_inventory,
    load_freeze_spec,
    run_architecture_audit,
    sha256_file,
)
from src.joint_dispatch.formal_protocol import load_formal_experiment_spec  # noqa: E402


FREEZE_SPEC = ROOT / "configs" / "rsc_pf_architecture_freeze_v1.json"
V2_CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_contract_v2.json"


def _spec():
    return load_formal_experiment_spec(V2_CONTRACT)


def test_freeze_spec_designates_v2_as_authority() -> None:
    spec = load_freeze_spec(FREEZE_SPEC)
    assert spec["schema_version"] == "rsc-pf-architecture-freeze-v1"
    assert spec["status"] == "candidate"
    assert spec["authority_contract"] == "configs/joint_forecast_dispatch_contract_v2.json"
    assert "configs/joint_forecast_dispatch_contract_v1.json" in spec["legacy_contracts"]
    assert spec["shape_contract"]["control_logits"] == ["B", 15]
    assert spec["shape_contract"]["dispatch"] == ["B", 4, 21]
    assert tuple(spec["required_gates"]) == GATES


def test_inventory_hashes_authoritative_sources() -> None:
    spec = load_freeze_spec(FREEZE_SPEC)
    inventory = build_architecture_inventory(ROOT, spec)
    assert inventory["authority"]["schema_version"] == "joint-forecast-dispatch-v2"
    assert inventory["authority"]["contract_status"] in {"calibration_candidate", "frozen"}
    assert len(inventory["authority"]["sha256"]) == 64
    assert inventory["legacy"][0]["role"] == "legacy_compatibility_only"
    assert set(inventory["source_files"]) == set(spec["source_files"])


def test_actual_v2_splits_have_complete_device_history() -> None:
    evidence = audit_data_gate(_spec(), ROOT)
    assert evidence["passed"] is True
    assert evidence["history_source"] == {"train": "causal_lp", "validation": "causal_lp", "test": "causal_lp"}
    for split in evidence["split_evidence"].values():
        assert split["fields"]["load_history"]["shape"][1:] == [24, 4]
        assert split["fields"]["exog_history"]["shape"][1:] == [24, 12]
        assert split["fields"]["device_history"]["shape"][1:] == [24, 21]
        assert split["fields"]["device_status"]["shape"][1:] == [24, 6]
        assert split["fields"]["teacher_dispatch"]["shape"][1:] == [4, 21]


def test_actual_status_is_binary_and_canonical() -> None:
    evidence = audit_data_gate(_spec(), ROOT)
    assert evidence["status_binary"] is True
    assert all(item["max_absolute_difference"] == 0.0 for item in evidence["status_consistency"].values())


def test_forward_contract_is_one_connected_graph() -> None:
    evidence = audit_forward_gate()
    assert evidence["passed"] is True
    assert evidence["shapes"]["control_logits"] == [3, 15]
    assert evidence["shapes"]["dispatch"] == [3, 4, 21]
    assert evidence["forecast_bottleneck"]["control_logits_max_delta"] > 0.0
    assert evidence["forecast_bottleneck"]["dispatch_max_delta"] > 0.0
    assert evidence["future_binary_decision_head"] is False


def test_dispatch_gradients_reach_forecaster_scheduler_and_all_groups() -> None:
    evidence = audit_gradient_gate(ROOT)
    assert evidence["passed"] is True
    assert evidence["dispatch_to_forecaster"]["loss_only_l2_norm"] > 0.0
    assert evidence["dispatch_to_scheduler"]["loss_only_l2_norm"] > 0.0
    assert all(item["finite"] and item["nonzero"] for item in evidence["decoder_group_gradients"].values())
    assert evidence["piecewise_differentiable"] is True
    assert evidence["static_scan"]["disallowed_hits"] == {}


def test_training_gate_keeps_warm_start_jointly_trainable() -> None:
    evidence = audit_training_gate(_spec(), ROOT)
    assert evidence["passed"] is True
    warm = evidence["methods"]["Warm-Start-Joint"]
    assert warm["initialization"] == "pretrained_weights"
    assert warm["forecaster_trainable"] is True
    assert warm["scheduler_trainable"] is True
    assert warm["one_optimizer_contains_both"] is True
    assert evidence["online_exact_lp_calls"] == 0
    assert evidence["future_binary_decisions"] is False


def test_existing_artifacts_are_compatible_and_receipts_are_present() -> None:
    evidence = audit_artifact_gate(_spec(), ROOT)
    assert evidence["passed"] is True
    assert evidence["checkpoint_count"] >= 10
    assert evidence["state_coverage"]["missing_keys"] == []
    assert evidence["state_coverage"]["unexpected_keys"] == []
    assert evidence["one_batch_forward"]["control_logits"] == [2, 15]
    assert evidence["one_batch_forward"]["dispatch"] == [2, 4, 21]


def test_full_architecture_receipt_is_frozen_without_contract_mutation() -> None:
    before = sha256_file(V2_CONTRACT)
    receipt = run_architecture_audit(ROOT, FREEZE_SPEC, write_reports=False, regression_verified=True)
    after = sha256_file(V2_CONTRACT)
    assert receipt["status"] == "frozen"
    assert receipt["failed_gates"] == []
    assert receipt["gates"] == {gate: True for gate in GATES}
    assert after == before
    assert receipt["authorized_claims"]
