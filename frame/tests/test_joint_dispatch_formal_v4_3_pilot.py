from __future__ import annotations

from src.joint_dispatch.formal_v4_3_pilot import PILOT_METHODS, authorize_pilot_v43


def _row(method_id: str, leakage: float) -> dict[str, object]:
    return {
        "method_id": method_id,
        "inactive_leakage": {"cooling_mae": leakage, "heating_mae": leakage},
        "active_forecast_guardrail_passed": True,
        "physical_feasibility_passed": True,
        "decision_regime_gradient_norm": 1.0,
        "decision_magnitude_gradient_norm": 1.0,
        "decision_forecast_gradient_norm": 1.0,
        "state_carry_passed": True,
    }


def test_pilot_contains_exact_mechanism_rows_and_requires_improvement() -> None:
    rows = [_row(PILOT_METHODS[0], 1.0), _row(PILOT_METHODS[1], 0.8), _row(PILOT_METHODS[2], 0.8), _row(PILOT_METHODS[3], 2.0)]
    rows[2]["decision_forecast_gradient_norm"] = 0.0
    receipt = {"rows": rows, "evaluation_year_accessed": False}
    assert authorize_pilot_v43(receipt) is True


def test_pilot_rejects_missing_row_or_evaluation_access() -> None:
    rows = [_row(method_id, 1.0) for method_id in PILOT_METHODS[:-1]]
    assert authorize_pilot_v43({"rows": rows, "evaluation_year_accessed": False}) is False
    rows = [_row(PILOT_METHODS[0], 1.0), _row(PILOT_METHODS[1], 0.8), _row(PILOT_METHODS[2], 0.8), _row(PILOT_METHODS[3], 2.0)]
    rows[2]["decision_forecast_gradient_norm"] = 0.0
    assert authorize_pilot_v43({"rows": rows, "evaluation_year_accessed": True}) is False
