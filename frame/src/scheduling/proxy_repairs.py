"""Deterministic, non-LP repairs for raw scheduling-proxy outputs.

These candidates are intentionally conservative.  They preserve the raw
neural output, never call the exact LP, and return a separate repaired array
with explicit provenance.  The physical decoder is a deterministic feasible
projection for the registered continuous benchmark; it is not presented as a
learned scheduler until it passes the validation-only selection gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

# Imported only so tests can prove that a repair never invokes the exact LP.
# No repair path below calls this symbol.
from .dispatch_lp import solve_dispatch_lp  # noqa: F401
from .proxy_contract import FEATURE_ORDER, HORIZON, LABEL_ORDER
from .proxy_physics import _parameter


_INDEX = {name: idx for idx, name in enumerate(LABEL_ORDER)}
_FEATURE_INDEX = {name: idx for idx, name in enumerate(FEATURE_ORDER)}
_UNBOUNDED_LABELS = {"slack_e", "slack_c", "slack_h", "q_dump"}
_RENEWABLE_PAIRS = (("pv_use", "pv_curt", "pv_available"), ("wt_use", "wt_curt", "wt_available"))
_CONVERSION_LABELS = (
    "g_chp", "g_gb", "p_chp", "q_chp", "q_gb", "p_ec", "q_ec", "q_ac_in", "q_ac",
)
_SOC_LABELS = ("p_charge", "p_discharge", "soc")


@dataclass(frozen=True)
class ProxyRepairResult:
    """Raw and repaired predictions with auditable repair metadata."""

    raw_prediction: np.ndarray
    repaired_prediction: np.ndarray
    repair_mask: np.ndarray
    family_magnitudes: Mapping[str, np.ndarray]
    method: str
    label_order: tuple[str, ...] = LABEL_ORDER
    feature_order: tuple[str, ...] = FEATURE_ORDER
    provenance: Mapping[str, Any] | None = None


def _validate_inputs(raw_prediction: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw_prediction, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (HORIZON, len(LABEL_ORDER)):
        raise ValueError(f"raw_prediction must have shape [N,{HORIZON},{len(LABEL_ORDER)}]")
    if x.shape != (raw.shape[0], HORIZON, len(FEATURE_ORDER)):
        raise ValueError(f"features must have shape [N,{HORIZON},{len(FEATURE_ORDER)}]")
    if not np.isfinite(raw).all() or not np.isfinite(x).all():
        raise ValueError("raw_prediction and features must be finite")
    if (x < 0.0).any():
        raise ValueError("proxy features must be non-negative")
    return raw, x


def _upper_bounds(parameters: Mapping[str, Any]) -> dict[str, float]:
    return {
        "grid": _parameter(parameters, "grid_import_capacity"),
        "g_chp": _parameter(parameters, "chp_electric_capacity") / max(_parameter(parameters, "chp_electric_efficiency"), 1e-12),
        "g_gb": _parameter(parameters, "gas_boiler_capacity") / max(_parameter(parameters, "gas_boiler_efficiency"), 1e-12),
        "p_chp": _parameter(parameters, "chp_electric_capacity"),
        "q_chp": _parameter(parameters, "chp_heat_capacity"),
        "q_gb": _parameter(parameters, "gas_boiler_capacity"),
        "p_ec": _parameter(parameters, "electric_chiller_capacity") / max(_parameter(parameters, "electric_chiller_cop"), 1e-12),
        "q_ec": _parameter(parameters, "electric_chiller_capacity"),
        "q_ac_in": _parameter(parameters, "absorption_chiller_capacity") / max(_parameter(parameters, "absorption_chiller_cop"), 1e-12),
        "q_ac": _parameter(parameters, "absorption_chiller_capacity"),
        "p_charge": _parameter(parameters, "bess_power_capacity"),
        "p_discharge": _parameter(parameters, "bess_power_capacity"),
        "soc": _parameter(parameters, "bess_energy_capacity"),
    }


def _bound_and_split(raw: np.ndarray, features: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    repaired = np.maximum(raw.copy(), 0.0)
    for name, upper in _upper_bounds(parameters).items():
        repaired[:, :, _INDEX[name]] = np.minimum(repaired[:, :, _INDEX[name]], float(upper))
    for use_name, curt_name, feature_name in _RENEWABLE_PAIRS:
        use = np.minimum(repaired[:, :, _INDEX[use_name]], features[:, :, _FEATURE_INDEX[feature_name]])
        repaired[:, :, _INDEX[use_name]] = use
        repaired[:, :, _INDEX[curt_name]] = np.maximum(features[:, :, _FEATURE_INDEX[feature_name]] - use, 0.0)
    return repaired


def _physical_decoder(raw: np.ndarray, features: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    """Project to a feasible dispatch using deterministic physical equations.

    The decoder keeps bounded cooling and boiler decisions when possible, but
    uses grid import and explicit unserved-load slack to guarantee balances.
    CHP and battery outputs are set to a ramp- and SOC-safe zero trajectory;
    this deliberately avoids inventing an unsupported operating policy.
    """

    n_samples = raw.shape[0]
    repaired = np.zeros_like(raw)
    demand_e = np.maximum(features[:, :, _FEATURE_INDEX["electricity"]], 0.0)
    demand_c = np.maximum(features[:, :, _FEATURE_INDEX["cooling"]], 0.0)
    demand_h = np.maximum(features[:, :, _FEATURE_INDEX["heating"]], 0.0)
    pv_available = np.maximum(features[:, :, _FEATURE_INDEX["pv_available"]], 0.0)
    wt_available = np.maximum(features[:, :, _FEATURE_INDEX["wt_available"]], 0.0)
    parameters = {str(k): float(v) for k, v in parameters.items()}

    # Cooling: retain raw requested outputs, then scale them to the demand.
    q_ec = np.minimum(np.maximum(raw[:, :, _INDEX["q_ec"]], 0.0), parameters["electric_chiller_capacity"])
    q_ac = np.minimum(np.maximum(raw[:, :, _INDEX["q_ac"]], 0.0), parameters["absorption_chiller_capacity"])
    cooling_sum = q_ec + q_ac
    cooling_scale = np.minimum(1.0, demand_c / np.maximum(cooling_sum, 1e-12))
    q_ec *= cooling_scale
    q_ac *= cooling_scale
    repaired[:, :, _INDEX["q_ec"]] = q_ec
    repaired[:, :, _INDEX["p_ec"]] = q_ec / max(parameters["electric_chiller_cop"], 1e-12)
    repaired[:, :, _INDEX["q_ac"]] = q_ac
    repaired[:, :, _INDEX["q_ac_in"]] = q_ac / max(parameters["absorption_chiller_cop"], 1e-12)
    repaired[:, :, _INDEX["slack_c"]] = np.maximum(demand_c - q_ec - q_ac, 0.0)

    # Heat: retain a bounded boiler trajectory, with slack covering the rest.
    q_gb = np.minimum(
        np.minimum(np.maximum(raw[:, :, _INDEX["q_gb"]], 0.0), parameters["gas_boiler_capacity"]),
        demand_h,
    )
    repaired[:, :, _INDEX["q_gb"]] = q_gb
    repaired[:, :, _INDEX["g_gb"]] = q_gb / max(parameters["gas_boiler_efficiency"], 1e-12)
    heat_surplus = q_gb - demand_h - repaired[:, :, _INDEX["q_ac_in"]]
    repaired[:, :, _INDEX["q_dump"]] = np.maximum(heat_surplus, 0.0)
    repaired[:, :, _INDEX["slack_h"]] = np.maximum(-heat_surplus, 0.0)

    # CHP is intentionally zero: this is the only trajectory that guarantees
    # both the zero-prior ramp convention and electricity balance without an
    # unsupported electric-dump variable.
    repaired[:, :, _INDEX["slack_e"]] = 0.0

    # Use renewables up to the residual electricity demand, PV first then WT;
    # the remainder is supplied by grid import and, if necessary, slack.
    electric_target = demand_e + repaired[:, :, _INDEX["p_ec"]]
    renewable_target = np.minimum(electric_target, pv_available + wt_available)
    pv_use = np.minimum(pv_available, renewable_target)
    wt_use = np.minimum(wt_available, np.maximum(renewable_target - pv_use, 0.0))
    repaired[:, :, _INDEX["pv_use"]] = pv_use
    repaired[:, :, _INDEX["pv_curt"]] = pv_available - pv_use
    repaired[:, :, _INDEX["wt_use"]] = wt_use
    repaired[:, :, _INDEX["wt_curt"]] = wt_available - wt_use
    grid_needed = np.maximum(electric_target - pv_use - wt_use, 0.0)
    grid_capacity = parameters["grid_import_capacity"]
    repaired[:, :, _INDEX["grid"]] = np.minimum(grid_needed, grid_capacity)
    repaired[:, :, _INDEX["slack_e"]] = np.maximum(grid_needed - grid_capacity, 0.0)

    # Fixed SOC trajectory satisfies state recursion and terminal equality.
    initial_soc = features[:, 0, _FEATURE_INDEX["initial_soc"]]
    repaired[:, :, _INDEX["soc"]] = initial_soc[:, None] * parameters["bess_energy_capacity"]
    return repaired


def _family_magnitudes(raw: np.ndarray, repaired: np.ndarray) -> dict[str, np.ndarray]:
    delta = np.abs(repaired - raw)
    all_labels = np.max(delta, axis=(1, 2))
    conversion = np.max(delta[:, :, [_INDEX[name] for name in _CONVERSION_LABELS]], axis=(1, 2))
    soc = np.max(delta[:, :, [_INDEX[name] for name in _SOC_LABELS]], axis=(1, 2))
    renewable = np.max(delta[:, :, [_INDEX[name] for pair in _RENEWABLE_PAIRS for name in pair[:2]]], axis=(1, 2))
    ramp = np.max(delta[:, :, [_INDEX["p_chp"]]], axis=(1, 2))
    return {
        "balance": all_labels,
        "conversion": conversion,
        "soc_state": soc,
        "soc_terminal": delta[:, -1, _INDEX["soc"]],
        "bounds": all_labels,
        "renewable_split": renewable,
        "ramp": ramp,
    }


def apply_repair(
    raw_prediction: np.ndarray,
    features: np.ndarray,
    parameters: Mapping[str, Any],
    method: str,
    tolerance: float = 1.0e-3,
) -> ProxyRepairResult:
    """Apply one bounded deterministic repair without invoking an exact LP."""

    raw, x = _validate_inputs(raw_prediction, features)
    aliases = {
        "candidate_a": "bound_and_split",
        "candidate_b": "physical_decoder",
    }
    canonical = aliases.get(str(method), str(method))
    if canonical == "bound_and_split":
        repaired = _bound_and_split(raw, x, parameters)
    elif canonical == "physical_decoder":
        repaired = _physical_decoder(raw, x, parameters)
    else:
        raise ValueError("unknown repair method; choose bound_and_split or physical_decoder")
    if repaired.shape != raw.shape or not np.isfinite(repaired).all():
        raise ValueError("repair must preserve a finite [N,4,21] output")
    magnitudes = _family_magnitudes(raw, repaired)
    mask = np.max(np.abs(repaired - raw), axis=(1, 2)) > float(tolerance)
    provenance = {
        "repair_version": "scheduling-proxy-repair-v1",
        "method": canonical,
        "exact_lp_used": False,
        "raw_prediction_preserved": True,
        "value_space": "physical",
        "tolerance": float(tolerance),
        "label_order": list(LABEL_ORDER),
        "feature_order": list(FEATURE_ORDER),
    }
    return ProxyRepairResult(
        raw_prediction=raw.copy(),
        repaired_prediction=repaired,
        repair_mask=mask,
        family_magnitudes=magnitudes,
        method=canonical,
        provenance=provenance,
    )


__all__ = ["ProxyRepairResult", "apply_repair"]
