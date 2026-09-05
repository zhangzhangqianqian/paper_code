"""Fail-closed authorization criteria for the bounded formal-v4.4 Pilot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .formal_v4_4_contract import FormalV44Contract


@dataclass(frozen=True)
class PilotDecisionV44:
    authorized_gate1: bool
    criteria: Mapping[str, bool]
    failures: tuple[str, ...]
    measured: Mapping[str, Any]
    accessed_years: tuple[int, ...] = ()
    rows: tuple[str, ...] = ()
    audit_sha256: str = ""


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _get(mapping: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value: Any = mapping
        ok = True
        for key in path.split("."):
            if not isinstance(value, Mapping) or key not in value:
                ok = False; break
            value = value[key]
        if ok: return value
    return default


def authorize_pilot_v44(receipt: Mapping[str, Any], contract: FormalV44Contract) -> PilotDecisionV44:
    """Evaluate all Pilot criteria; missing or malformed evidence fails closed."""

    thresholds = contract.pilot_thresholds
    failures: list[str] = []; measured: dict[str, Any] = {}
    lineage = receipt.get("lineage") if isinstance(receipt, Mapping) else None
    hashes_ok = isinstance(lineage, Mapping) and all(isinstance(lineage.get(key), str) and len(lineage[key]) == 64 for key in ("source_manifest_sha256", "train_data_sha256", "selection_data_sha256", "contract_sha256"))
    accessed = receipt.get("accessed_years", [])
    integrity = hashes_ok and isinstance(accessed, list) and contract.evaluation_year not in [int(v) for v in accessed] and all(_finite(v) for v in receipt.get("finite_values", [0.0]))
    measured["accessed_years"] = accessed
    comparisons = receipt.get("comparisons", {}) if isinstance(receipt.get("comparisons", {}), Mapping) else {}
    leakage_ratios = comparisons.get("leakage_ratio", {}) if isinstance(comparisons.get("leakage_ratio", {}), Mapping) else {}
    active_ratios = comparisons.get("active_wape_ratio", {}) if isinstance(comparisons.get("active_wape_ratio", {}), Mapping) else {}
    unaffected_ratios = comparisons.get("electricity_gas_wape_ratio", {}) if isinstance(comparisons.get("electricity_gas_wape_ratio", {}), Mapping) else {}
    overall_ratio = comparisons.get("four_task_score_ratio")
    leakage = all(_finite(leakage_ratios.get(name)) and float(leakage_ratios[name]) <= 1.0 - float(thresholds["minimum_leakage_reduction"]) for name in ("cooling", "heating"))
    active = all(_finite(active_ratios.get(name)) and float(active_ratios[name]) <= 1.0 + float(thresholds["maximum_active_wape_relative_degradation"]) for name in ("cooling", "heating"))
    unaffected = all(_finite(unaffected_ratios.get(name)) and float(unaffected_ratios[name]) <= 1.0 + float(thresholds["maximum_electricity_gas_wape_relative_degradation"]) for name in ("electricity", "gas"))
    overall = _finite(overall_ratio) and float(overall_ratio) <= 1.0 + float(thresholds["maximum_four_task_score_relative_degradation"])
    transition_gain = _get(receipt, "regime.transition_balanced_accuracy_gain", "transition_balanced_accuracy_gain")
    macro_f1 = _get(receipt, "regime.macro_f1", "macro_f1"); prior_f1 = _get(receipt, "regime.prior_macro_f1", "prior_macro_f1")
    regime = _finite(transition_gain) and float(transition_gain) >= float(thresholds["minimum_transition_balanced_accuracy_gain"]) and _finite(macro_f1) and _finite(prior_f1) and float(macro_f1) >= float(prior_f1)
    joint_objective = _get(receipt, "joint.penalized_objective", "joint_objective"); dec_objective = _get(receipt, "decoupled.penalized_objective", "decoupled_objective")
    joint_shortage = _get(receipt, "joint.shortage", "joint_shortage"); dec_shortage = _get(receipt, "decoupled.shortage", "decoupled_shortage")
    decision = all(_finite(value) for value in (joint_objective, dec_objective, joint_shortage, dec_shortage)) and float(joint_objective) <= float(dec_objective) and float(joint_shortage) <= float(dec_shortage)
    joint_grad = _get(receipt, "joint.gradient_norms", default={}); dec_grad = _get(receipt, "decoupled.gradient_norms", default={})
    gradient = isinstance(joint_grad, Mapping) and isinstance(dec_grad, Mapping) and all(_finite(joint_grad.get(name)) and float(joint_grad[name]) > 0.0 for name in ("decision_to_gate", "decision_to_magnitude", "decision_to_scheduler")) and _finite(dec_grad.get("decision_to_base")) and float(dec_grad["decision_to_base"]) <= float(thresholds["maximum_decoupled_decision_gradient"])
    physical_residual = _get(receipt, "physics.max_residual", "max_physical_residual"); physics = _finite(physical_residual) and float(physical_residual) <= float(thresholds["maximum_physical_residual"])
    criteria = {"leakage": leakage, "active": active, "unaffected": unaffected, "overall": overall, "regime": regime, "decision": decision, "gradient": gradient, "physics": physics, "integrity": integrity}
    for name, passed in criteria.items():
        measured[name] = passed
        if not passed: failures.append(name)
    return PilotDecisionV44(not failures, criteria, tuple(failures), measured)


__all__ = ["PilotDecisionV44", "authorize_pilot_v44"]
