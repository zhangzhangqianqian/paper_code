from __future__ import annotations

import numpy as np

from src.joint_dispatch.formal_v4_3_gate1 import (
    authorize_gate1_v43,
    build_full_chronology_view,
    candidate_is_eligible_v43,
)


def _candidate() -> dict[str, object]:
    return {
        "candidate_id": "lr0.5_dec0.25",
        "views": {
            "full_chronology": {
                "forecast_guardrail_passed": True,
                "inactive_leakage_guardrail_passed": True,
                "regime_guardrail_passed": True,
            },
            "stress_sample": {"forecast_guardrail_passed": True},
        },
        "physical_feasibility_passed": True,
        "dispatch_improvement_passed": True,
        "gradient_boundary_passed": True,
        "full_chronology_penalized_objective": 10.0,
    }


def test_full_chronology_uses_every_selection_origin_once() -> None:
    selection = {"forecast_target": np.zeros((7, 4, 4), dtype=np.float32)}
    view = build_full_chronology_view(selection)
    np.testing.assert_array_equal(view.indices, np.arange(7))
    np.testing.assert_array_equal(view.weights, np.ones(7))


def test_stress_sample_cannot_authorize_candidate_by_itself() -> None:
    candidate = _candidate()
    candidate["views"]["full_chronology"]["forecast_guardrail_passed"] = False
    assert candidate_is_eligible_v43(candidate) is False


def test_gate1_selects_only_eligible_candidate() -> None:
    first = _candidate()
    second = _candidate()
    second["candidate_id"] = "lr2.0_dec1.0"
    second["full_chronology_penalized_objective"] = 5.0
    result = authorize_gate1_v43([first, second])
    assert result["authorized_gate2"] is True
    assert result["selected_candidate_id"] == "lr2.0_dec1.0"
    assert result["evaluation_year_accessed"] is False
