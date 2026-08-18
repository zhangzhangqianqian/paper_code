"""Constraint-family diagnostics for raw scheduling-proxy predictions.

The diagnostics in this module are deliberately descriptive.  They evaluate
the neural proxy output before any exact-LP fallback or deterministic repair
is applied, and therefore cannot silently turn an infeasible raw output into
an apparently feasible result.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dispatch_schema import VARIABLES
from .proxy_contract import FEATURE_ORDER, HORIZON, LABEL_ORDER
from .proxy_physics import _parameter, balance_residuals, conversion_residuals, soc_residuals


_LABEL_INDEX = {name: idx for idx, name in enumerate(LABEL_ORDER)}
_FAMILY_ORDER = (
    "balance",
    "conversion",
    "soc_state",
    "soc_terminal",
    "bounds",
    "renewable_split",
    "ramp",
)


def _as_dispatch(prediction: np.ndarray) -> np.ndarray:
    value = np.asarray(prediction, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (HORIZON, len(LABEL_ORDER)):
        raise ValueError(f"prediction must have shape [N,{HORIZON},{len(LABEL_ORDER)}]")
    return value


def _as_features(features: np.ndarray, n_samples: int) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.shape != (n_samples, HORIZON, len(FEATURE_ORDER)):
        raise ValueError(f"features must have shape [N,{HORIZON},{len(FEATURE_ORDER)}]")
    if not np.isfinite(value).all():
        raise ValueError("features must be finite")
    return value


def _scenario_ids(scenario_ids: Sequence[Any] | None, n_samples: int) -> np.ndarray:
    if scenario_ids is None:
        return np.asarray([str(index) for index in range(n_samples)], dtype=str)
    value = np.asarray(scenario_ids).astype(str)
    if value.shape != (n_samples,):
        raise ValueError("scenario_ids must have shape [N]")
    return value


def _finite_violation(values: np.ndarray) -> np.ndarray:
    """Convert non-finite raw predictions to explicit infinite violations."""

    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values), np.maximum(values, 0.0), np.inf)


def _summary(values: np.ndarray, scenario_ids: np.ndarray, tolerance: float) -> dict[str, Any]:
    values = _finite_violation(values)
    mask = values > float(tolerance)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        p50 = p95 = float("inf")
    else:
        p50 = float(np.percentile(finite_values, 50.0))
        p95 = float(np.percentile(finite_values, 95.0))
    worst_index = int(np.argmax(values)) if values.size else -1
    return {
        "violation_count": int(mask.sum()),
        "violation_rate": float(np.mean(mask)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
        "p50": p50,
        "p95": p95,
        "worst_scenario_id": str(scenario_ids[worst_index]) if worst_index >= 0 else None,
    }


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


def diagnose_dispatch(
    prediction: np.ndarray,
    features: np.ndarray,
    parameters: Mapping[str, Any],
    tolerance: float,
    scenario_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return raw per-sample and aggregate violations for each constraint family.

    ``prediction`` and ``features`` are physical-unit arrays.  No normalization
    or safety fallback is performed here.  The returned ``per_sample`` arrays
    contain one maximum violation and one boolean mask per family; the
    ``aggregate`` section contains counts, rates, quantiles, and worst IDs.
    """

    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    y = _as_dispatch(prediction)
    n_samples = y.shape[0]
    x = _as_features(features, n_samples)
    ids = _scenario_ids(scenario_ids, n_samples)
    finite = np.isfinite(y).all(axis=(1, 2))
    safe_y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    yt = torch.as_tensor(safe_y, dtype=torch.float64)
    xt = torch.as_tensor(x, dtype=torch.float64)

    balance_tensor = balance_residuals(yt, xt).detach().numpy()
    conversion_tensor = conversion_residuals(yt, parameters).detach().numpy()
    soc_tensor = soc_residuals(yt, xt, parameters)
    balance = np.max(np.abs(balance_tensor), axis=(1, 2))
    conversion = np.max(np.abs(conversion_tensor), axis=(1, 2))
    soc_state = np.max(np.abs(soc_tensor["state"].detach().numpy()), axis=1)
    soc_terminal = np.abs(soc_tensor["terminal"].detach().numpy())

    lower_violation = np.max(np.maximum(-safe_y, 0.0), axis=(1, 2))
    upper_limits = _upper_bounds(parameters)
    upper_violation = np.zeros(n_samples, dtype=np.float64)
    for name, limit in upper_limits.items():
        upper_violation = np.maximum(
            upper_violation,
            np.max(np.maximum(safe_y[:, :, _LABEL_INDEX[name]] - float(limit), 0.0), axis=1),
        )
    bounds = np.maximum(lower_violation, upper_violation)

    pv_split = np.abs(
        safe_y[:, :, _LABEL_INDEX["pv_use"]]
        + safe_y[:, :, _LABEL_INDEX["pv_curt"]]
        - x[:, :, 4]
    )
    wt_split = np.abs(
        safe_y[:, :, _LABEL_INDEX["wt_use"]]
        + safe_y[:, :, _LABEL_INDEX["wt_curt"]]
        - x[:, :, 5]
    )
    pv_split_max = np.max(pv_split, axis=1)
    wt_split_max = np.max(wt_split, axis=1)
    renewable_split = np.maximum(pv_split_max, wt_split_max)

    ramp_limit = _parameter(parameters, "chp_ramp_fraction") * _parameter(parameters, "chp_electric_capacity")
    p_chp = safe_y[:, :, _LABEL_INDEX["p_chp"]]
    p_change = np.concatenate([p_chp[:, :1], np.diff(p_chp, axis=1)], axis=1)
    ramp = np.max(np.maximum(np.abs(p_change) - ramp_limit, 0.0), axis=1)

    violations = {
        "balance": balance,
        "conversion": conversion,
        "soc_state": soc_state,
        "soc_terminal": soc_terminal,
        "bounds": bounds,
        "renewable_split": renewable_split,
        "ramp": ramp,
    }
    family_masks = {
        name: _finite_violation(values) > tolerance
        for name, values in violations.items()
    }
    raw_feasible = finite.copy()
    for mask in family_masks.values():
        raw_feasible &= ~mask
    per_sample: dict[str, np.ndarray] = {
        "scenario_ids": ids,
        "finite_mask": finite,
        "raw_feasible_mask": raw_feasible,
        "pv_split_max": _finite_violation(pv_split_max),
        "wt_split_max": _finite_violation(wt_split_max),
        "lower_bound_max": _finite_violation(lower_violation),
        "upper_bound_max": _finite_violation(upper_violation),
    }
    for name, values in violations.items():
        per_sample[f"{name}_max"] = _finite_violation(values)
        per_sample[f"{name}_mask"] = family_masks[name]

    aggregate = {
        name: _summary(values, ids, tolerance)
        for name, values in violations.items()
    }
    aggregate["raw_feasible_rate"] = float(np.mean(raw_feasible)) if n_samples else 0.0
    aggregate["raw_feasible_count"] = int(raw_feasible.sum())
    aggregate["nonfinite_count"] = int((~finite).sum())
    aggregate["tolerance"] = tolerance
    return {
        "schema_version": "scheduling-proxy-constraint-diagnostics-v1",
        "value_space": "physical",
        "sample_count": int(n_samples),
        "horizon": HORIZON,
        "label_order": list(LABEL_ORDER),
        "feature_order": list(FEATURE_ORDER),
        "family_order": list(_FAMILY_ORDER),
        "per_sample": per_sample,
        "aggregate": aggregate,
    }


def diagnostics_npz_payload(report: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Convert a diagnostic report into a stable, array-only NPZ payload."""

    per_sample = report.get("per_sample")
    if not isinstance(per_sample, Mapping):
        raise ValueError("diagnostic report is missing per_sample arrays")
    payload: dict[str, np.ndarray] = {}
    for name, value in per_sample.items():
        array = np.asarray(value)
        if array.dtype.kind in "OUS":
            array = array.astype(str)
        payload[str(name)] = array
    return payload


__all__ = ["diagnose_dispatch", "diagnostics_npz_payload"]
