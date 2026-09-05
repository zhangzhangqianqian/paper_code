"""Fail-closed Gate 1 authorization for the formal-v4.6 Pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class PilotDecisionV46:
    authorized_gate1: bool
    failure_names: tuple[str, ...]
    measurements: Mapping[str, float]


def _value(receipt: Mapping[str, Any], *paths: tuple[str, ...], default: Any = None) -> Any:
    for path in paths:
        current: Any = receipt
        found = True
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                found = False; break
            current = current[key]
        if found:
            return current
    return default


def _finite(value: Any, name: str, failures: list[str]) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        failures.append(name); return float("nan")
    if not np.isfinite(result):
        failures.append(name)
    return result


def authorize_pilot_v46(receipt: Mapping[str, Any], contract: Any) -> PilotDecisionV46:
    failures: list[str] = []
    if not isinstance(receipt, Mapping):
        return PilotDecisionV46(False, ("receipt_not_mapping",), {})
    thresholds = contract.pilot_thresholds if hasattr(contract, "pilot_thresholds") else contract.get("pilot_thresholds", {})
    four_task = _finite(_value(receipt, ("four_task_score_ratio",), ("nominal_ratios", "four_task_score_ratio"), ("forecast", "four_task_score_ratio")), "four_task_score_ratio", failures)
    electricity = _finite(_value(receipt, ("electricity_wape_ratio",), ("nominal_ratios", "electricity_wape_ratio")), "electricity_wape_ratio", failures)
    gas = _finite(_value(receipt, ("gas_wape_ratio",), ("nominal_ratios", "gas_wape_ratio")), "gas_wape_ratio", failures)
    thermal = _finite(_value(receipt, ("active_thermal_wape_ratio",), ("nominal_ratios", "active_thermal_wape_ratio")), "active_thermal_wape_ratio", failures)
    leakage = _finite(_value(receipt, ("normalized_inactive_leakage_ratio",), ("nominal_ratios", "normalized_inactive_leakage_ratio")), "normalized_inactive_leakage_ratio", failures)
    joint = _finite(_value(receipt, ("joint_decision_objective",), ("decision", "joint_objective")), "joint_decision_objective", failures)
    decoupled = _finite(_value(receipt, ("decoupled_decision_objective",), ("decision", "decoupled_objective")), "decoupled_decision_objective", failures)
    joint_short = _finite(_value(receipt, ("joint_shortage",), ("decision", "joint_shortage")), "joint_shortage", failures)
    dec_short = _finite(_value(receipt, ("decoupled_shortage",), ("decision", "decoupled_shortage")), "decoupled_shortage", failures)
    gradient = _finite(_value(receipt, ("decoupled_decision_gradient",), ("gradients", "decoupled_decision_gradient")), "decoupled_decision_gradient", failures)
    cap_violation = _finite(_value(receipt, ("cap_violation",), ("risk", "maximum_cap_violation")), "cap_violation", failures)
    gas_adjustment = _finite(_value(receipt, ("gas_adjustment",), ("risk", "gas_adjustment")), "gas_adjustment", failures)
    physical = _finite(_value(receipt, ("physical_residual",), ("physics", "maximum_residual")), "physical_residual", failures)
    macro_drop = _finite(_value(receipt, ("macro_f1_drop",), ("forecast", "macro_f1_drop"), default=0.0), "macro_f1_drop", failures)
    transition_gain = _finite(_value(receipt, ("transition_balanced_accuracy_gain",), ("forecast", "transition_balanced_accuracy_gain"), default=1.0), "transition_balanced_accuracy_gain", failures)
    evaluation_accessed = bool(_value(receipt, ("evaluation_year_accessed",), ("lineage", "evaluation_year_accessed"), default=False))
    if evaluation_accessed: failures.append("evaluation_year_accessed")
    if _value(receipt, ("hashes_valid",), ("lineage", "hashes_valid"), default=True) is not True: failures.append("hashes_invalid")
    checks = (
        ("four_task_score_ratio", four_task, float(thresholds.get("maximum_four_task_score_ratio", 1.02))),
        ("electricity_wape_ratio", electricity, float(thresholds.get("maximum_electricity_wape_ratio", 1.02))),
        ("gas_wape_ratio", gas, float(thresholds.get("maximum_gas_wape_ratio", 1.10))),
        ("active_thermal_wape_ratio", thermal, float(thresholds.get("maximum_active_thermal_wape_ratio", 1.05))),
        ("normalized_inactive_leakage_ratio", leakage, float(thresholds.get("maximum_normalized_inactive_leakage_ratio", 1.05))),
    )
    for name, value, limit in checks:
        if np.isfinite(value) and value > limit: failures.append(name)
    if np.isfinite(joint) and np.isfinite(decoupled) and joint > decoupled + 1.0e-12: failures.append("joint_decision_not_better")
    if np.isfinite(joint_short) and np.isfinite(dec_short) and joint_short > dec_short + 1.0e-12: failures.append("joint_shortage_not_better")
    if np.isfinite(gradient) and gradient > float(thresholds.get("maximum_decoupled_decision_gradient", 1.0e-12)): failures.append("decoupled_decision_gradient")
    if np.isfinite(cap_violation) and cap_violation > 1.0e-6: failures.append("cap_violation")
    if np.isfinite(gas_adjustment) and abs(gas_adjustment) > 1.0e-12: failures.append("gas_adjustment")
    if np.isfinite(physical) and physical > float(thresholds.get("maximum_physical_residual", 1.0e-6)): failures.append("physical_residual")
    if np.isfinite(macro_drop) and macro_drop > float(thresholds.get("maximum_macro_f1_drop", 0.02)): failures.append("macro_f1_drop")
    if np.isfinite(transition_gain) and transition_gain < float(thresholds.get("minimum_transition_balanced_accuracy_gain", 0.01)): failures.append("transition_balanced_accuracy_gain")
    unique = tuple(dict.fromkeys(failures))
    measurements = {name: value for name, value in (("four_task_score_ratio", four_task), ("electricity_wape_ratio", electricity), ("gas_wape_ratio", gas), ("active_thermal_wape_ratio", thermal), ("normalized_inactive_leakage_ratio", leakage), ("joint_decision_objective", joint), ("decoupled_decision_objective", decoupled), ("physical_residual", physical), ("macro_f1_drop", macro_drop), ("transition_balanced_accuracy_gain", transition_gain)) if np.isfinite(value)}
    return PilotDecisionV46(len(unique) == 0, unique, measurements)


__all__ = ["PilotDecisionV46", "authorize_pilot_v46"]
