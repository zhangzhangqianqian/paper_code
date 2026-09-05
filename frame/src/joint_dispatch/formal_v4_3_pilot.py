"""Fail-closed pilot checks for the regime-aware formal-v4.3 model."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


PILOT_METHODS = ("stage_p_regime", "rsc_pf_joint", "decoupled_rsc_pf", "continuous_thermal_head")


def authorize_pilot_v43(receipt: Mapping[str, Any]) -> bool:
    rows = receipt.get("rows")
    if not isinstance(rows, list) or tuple(row.get("method_id") for row in rows) != PILOT_METHODS:
        return False
    if receipt.get("evaluation_year_accessed") is not False:
        return False
    regime = rows[0]
    continuous = rows[-1]
    regime_leak = regime.get("inactive_leakage", {})
    continuous_leak = continuous.get("inactive_leakage", {})
    for task in ("cooling_mae", "heating_mae"):
        if not np.isfinite(float(regime_leak.get(task, np.inf))) or not np.isfinite(float(continuous_leak.get(task, np.inf))):
            return False
        if float(regime_leak[task]) >= float(continuous_leak[task]):
            return False
    joint = rows[1]
    decoupled = rows[2]
    checks = (
        bool(regime.get("active_forecast_guardrail_passed", False)),
        bool(regime.get("physical_feasibility_passed", False)),
        bool(joint.get("physical_feasibility_passed", False)),
        float(joint.get("decision_regime_gradient_norm", 0.0)) > 0.0,
        float(joint.get("decision_magnitude_gradient_norm", 0.0)) > 0.0,
        float(decoupled.get("decision_forecast_gradient_norm", np.inf)) == 0.0,
        bool(joint.get("state_carry_passed", False)),
    )
    return bool(all(checks))


def pilot_receipt_template(*, run_id: str, contract_sha256: str) -> dict[str, Any]:
    return {
        "schema": "formal-v4.3-pilot-decision-v1",
        "run_id": str(run_id),
        "contract_sha256": str(contract_sha256),
        "rows": [],
        "evaluation_year_accessed": False,
        "paper_eligible": False,
        "authorized_gate1": False,
    }


__all__ = ["PILOT_METHODS", "authorize_pilot_v43", "pilot_receipt_template"]
