from __future__ import annotations

import pytest

from src.joint_dispatch.formal_v4_6_contract import load_formal_v4_6_contract
from src.joint_dispatch.formal_v4_6_pilot_gate import authorize_pilot_v46


CONTRACT = load_formal_v4_6_contract("configs/joint_forecast_dispatch_formal_v4_6.json")


def _receipt():
    return {
        "four_task_score_ratio": 1.0,
        "electricity_wape_ratio": 1.0,
        "gas_wape_ratio": 1.0,
        "active_thermal_wape_ratio": 1.0,
        "normalized_inactive_leakage_ratio": 1.0,
        "joint_decision_objective": 0.9,
        "decoupled_decision_objective": 1.0,
        "joint_shortage": 0.1,
        "decoupled_shortage": 0.2,
        "decoupled_decision_gradient": 0.0,
        "cap_violation": 0.0,
        "gas_adjustment": 0.0,
        "physical_residual": 1e-7,
        "evaluation_year_accessed": False,
        "hashes_valid": True,
    }


def test_gate_accepts_nominal_noninferiority_and_decision_improvement():
    decision = authorize_pilot_v46(_receipt(), CONTRACT)
    assert decision.authorized_gate1
    assert decision.failure_names == ()


@pytest.mark.parametrize("field", ["gas_adjustment", "cap_violation", "evaluation_year_accessed"])
def test_gate_fails_closed_on_semantic_or_lineage_violation(field):
    receipt = _receipt()
    receipt[field] = True if field == "evaluation_year_accessed" else 1.0
    assert not authorize_pilot_v46(receipt, CONTRACT).authorized_gate1
