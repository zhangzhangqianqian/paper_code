"""Metrics for the matched chronological closed-loop evaluation.

The module deliberately keeps forecast quality, physical feasibility, and
shortage adequacy as separate quantities.  It does not import a model or a
solver, so it is safe to use from the independent audit process.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..scheduling.dispatch_schema import VARIABLES


PHYSICAL_TOLERANCE = 1.0e-6
SHORTAGE_TOLERANCE = 1.0e-8
THERMAL_ACTIVE_EPSILON = 1.0e-9
RECOURSE_CHANNELS = (
    "grid", "pv_use", "wt_use", "p_chp", "q_gb", "q_ec", "q_ac",
    "p_charge", "p_discharge",
)
_TASKS = ("electricity", "cooling", "heating", "gas")


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _finite_scalar(value: Any, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _max_abs(value: Any, *, name: str) -> float:
    array = _finite_array(value, name=name)
    return float(np.max(np.abs(array))) if array.size else 0.0


def residual_vector(residuals: Any) -> np.ndarray:
    """Return the eight canonical maximum residual magnitudes."""

    names = (
        "balance", "capacity", "conversion", "soc", "ramp",
        "exclusivity", "renewable_accounting", "finite",
    )
    return np.asarray([_max_abs(getattr(residuals, name), name=name) for name in names], dtype=np.float64)


def physical_feasible(vector: Any, tolerance: float = PHYSICAL_TOLERANCE) -> bool:
    value = _finite_array(vector, name="physical residual vector")
    tol = _finite_scalar(tolerance, name="tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be non-negative")
    return bool(np.max(np.abs(value)) <= tol) if value.size else True


def shortage_free(shortage: Any, tolerance: float = SHORTAGE_TOLERANCE) -> bool:
    value = _finite_array(shortage, name="shortage")
    tol = _finite_scalar(tolerance, name="tolerance")
    if tol < 0.0:
        raise ValueError("tolerance must be non-negative")
    return bool(np.max(value) <= tol) if value.size else True


def _summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    denominator = float(np.abs(target).sum())
    return {
        "mae": float(np.abs(error).mean()) if error.size else float("nan"),
        "rmse": float(np.sqrt(np.square(error).mean())) if error.size else float("nan"),
        "wape": float(np.abs(error).sum() / denominator) if denominator > 0.0 else float("nan"),
        "count": float(error.size),
    }


def regime_aware_forecast_metrics(
    prediction: Any,
    target: Any,
    thermal_active_scales: Mapping[str, float],
    epsilon: float = THERMAL_ACTIVE_EPSILON,
) -> dict[str, Any]:
    """Compute ordinary and seasonal on/off-aware forecast diagnostics.

    ``thermal_active_scales`` must be fitted once on the training split.  No
    selection or evaluation target is used to estimate those denominators.
    """

    pred = _finite_array(prediction, name="prediction", ndim=3)
    truth = _finite_array(target, name="target", ndim=3)
    if pred.shape != truth.shape or pred.shape[-1] != 4:
        raise ValueError("prediction and target must have shape [N,H,4]")
    active_epsilon = _finite_scalar(epsilon, name="epsilon")
    if active_epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    ordinary: dict[str, Any] = {}
    for task_index, task in enumerate(_TASKS):
        per_horizon = [_summary(pred[:, h, task_index], truth[:, h, task_index]) for h in range(pred.shape[1])]
        ordinary[task] = {
            "by_horizon": per_horizon,
            "overall": _summary(pred[..., task_index], truth[..., task_index]),
        }

    thermal: dict[str, Any] = {}
    for task_index, task in ((1, "cooling"), (2, "heating")):
        if task not in thermal_active_scales:
            raise ValueError(f"missing frozen active scale for {task}")
        scale = _finite_scalar(thermal_active_scales[task], name=f"{task} active scale")
        if scale <= 0.0:
            raise ValueError(f"{task} active scale must be positive")
        active = truth[..., task_index] > active_epsilon
        inactive = ~active
        active_by_horizon: list[dict[str, float]] = []
        leakage_by_horizon: list[dict[str, float]] = []
        for horizon in range(pred.shape[1]):
            active_pred = pred[:, horizon, task_index][active[:, horizon]]
            active_truth = truth[:, horizon, task_index][active[:, horizon]]
            inactive_pred = np.abs(pred[:, horizon, task_index][inactive[:, horizon]])
            active_by_horizon.append(_summary(active_pred, active_truth) if active_pred.size else {
                "mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "count": 0.0,
            })
            leakage_by_horizon.append({
                "mean": float(inactive_pred.mean() / scale) if inactive_pred.size else float("nan"),
                "p95": float(np.percentile(inactive_pred, 95.0) / scale) if inactive_pred.size else float("nan"),
                "count": float(inactive_pred.size),
            })
        inactive_pred_all = np.abs(pred[..., task_index][inactive])
        thermal[task] = {
            "active": {"by_horizon": active_by_horizon, "overall": _summary(pred[..., task_index][active], truth[..., task_index][active]) if active.any() else {
                "mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "count": 0.0,
            }, "count": int(active.sum()), "epsilon": active_epsilon},
            "inactive_leakage": {
                "by_horizon": leakage_by_horizon,
                "mean": float(inactive_pred_all.mean() / scale) if inactive_pred_all.size else float("nan"),
                "p95": float(np.percentile(inactive_pred_all, 95.0) / scale) if inactive_pred_all.size else float("nan"),
                "count": int(inactive.sum()),
                "normalization_scale": scale,
            },
        }
    return {"ordinary": ordinary, "thermal": thermal, "epsilon": active_epsilon}


def recourse_distance(
    planned: Any,
    settled: Any,
    realized_demand: Any,
) -> tuple[float, float, np.ndarray]:
    planned_array = _finite_array(planned, name="planned dispatch")
    settled_array = _finite_array(settled, name="settled dispatch")
    if planned_array.shape[-1] != len(VARIABLES) or settled_array.shape[-1] != len(VARIABLES):
        raise ValueError("planned and settled dispatch must have final dimension 21")
    if planned_array.ndim > 2 or settled_array.ndim > 2:
        raise ValueError("dispatch must be [21] or [H,21]")
    planned_row = planned_array[0] if planned_array.ndim == 2 else planned_array
    settled_row = settled_array[0] if settled_array.ndim == 2 else settled_array
    if planned_row.ndim != 1:
        raise ValueError("dispatch must be [21] or [H,21]")
    indices = np.asarray([VARIABLES.index(name) for name in RECOURSE_CHANNELS], dtype=np.int64)
    by_channel = np.abs(settled_row[indices] - planned_row[indices])
    demand = _finite_array(realized_demand, name="realized demand")
    denominator = max(float(np.abs(demand).sum()), 1.0e-12)
    absolute = float(by_channel.sum())
    return absolute, absolute / denominator, by_channel


def terminal_stock_adjustment(
    initial_soc: float,
    final_soc: float,
    energy_capacity: float,
    roundtrip_efficiency: float,
    final_grid_price: float,
    final_carbon_price: float,
    grid_emission_factor: float,
) -> float:
    """Value terminal battery-stock difference without claiming restoration."""

    initial = _finite_scalar(initial_soc, name="initial_soc")
    final = _finite_scalar(final_soc, name="final_soc")
    capacity = _finite_scalar(energy_capacity, name="energy_capacity")
    roundtrip = _finite_scalar(roundtrip_efficiency, name="roundtrip_efficiency")
    grid_price = _finite_scalar(final_grid_price, name="final_grid_price")
    carbon_price = _finite_scalar(final_carbon_price, name="final_carbon_price")
    emission = _finite_scalar(grid_emission_factor, name="grid_emission_factor")
    if capacity < 0.0:
        raise ValueError("energy_capacity must be non-negative")
    if not 0.0 < roundtrip <= 1.0:
        raise ValueError("roundtrip_efficiency must be in (0, 1]")
    eta = float(np.sqrt(roundtrip))
    initial_energy = initial * capacity
    final_energy = final * capacity
    marginal_value = grid_price + carbon_price * emission
    if final_energy < initial_energy:
        return float((initial_energy - final_energy) / eta * marginal_value)
    if final_energy > initial_energy:
        return float(-(final_energy - initial_energy) * eta * marginal_value)
    return 0.0


def _percentile(values: Any, q: float) -> float:
    array = _finite_array(values, name="metric values")
    return float(np.percentile(array, q)) if array.size else float("nan")


def summarize_closed_loop(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize raw runner arrays while preserving metric boundaries."""

    if not isinstance(arrays, Mapping):
        raise ValueError("arrays must be a mapping")
    result: dict[str, Any] = {}
    for required in ("planned_residuals", "settled_residuals", "shortage"):
        if required not in arrays:
            raise ValueError(f"missing {required}")
        _finite_array(arrays[required], name=required)
    planned_residuals = _finite_array(arrays["planned_residuals"], name="planned_residuals")
    settled_residuals = _finite_array(arrays["settled_residuals"], name="settled_residuals")
    shortage = _finite_array(arrays["shortage"], name="shortage")
    if planned_residuals.ndim != 2 or settled_residuals.ndim != 2:
        raise ValueError("residual arrays must have shape [N,K]")
    result["rates"] = {
        "planned_physical_feasibility": float(np.mean(np.max(np.abs(planned_residuals), axis=1) <= PHYSICAL_TOLERANCE)),
        "settled_physical_feasibility": float(np.mean(np.max(np.abs(settled_residuals), axis=1) <= PHYSICAL_TOLERANCE)),
        "no_shortage": float(np.mean(np.max(shortage, axis=1) <= SHORTAGE_TOLERANCE)),
    }
    result["shortage"] = {
        "mean": float(np.mean(shortage)),
        "total": float(np.sum(shortage)),
        "p95": _percentile(shortage.reshape(-1), 95.0),
    }
    for source, key in (("operating_cost", "operating_cost"), ("physical_carbon", "physical_carbon"), ("penalized_objective", "objective")):
        if source in arrays:
            values = _finite_array(arrays[source], name=source)
            result[key] = {"mean": float(np.mean(values)), "total": float(np.sum(values)), "p95": _percentile(values.reshape(-1), 95.0)}
    if "penalized_objective" in arrays:
        result["cumulative_raw_objective"] = float(np.sum(_finite_array(arrays["penalized_objective"], name="penalized_objective")))
    if "initial_soc" in arrays:
        result["initial_soc"] = float(_finite_array(arrays["initial_soc"], name="initial_soc").reshape(-1)[0])
    if "final_soc" in arrays:
        result["final_soc"] = float(_finite_array(arrays["final_soc"], name="final_soc").reshape(-1)[-1])
    if "settled_dispatch" in arrays:
        settled_dispatch = _finite_array(arrays["settled_dispatch"], name="settled_dispatch")
        charge = settled_dispatch[..., VARIABLES.index("p_charge")]
        discharge = settled_dispatch[..., VARIABLES.index("p_discharge")]
        delta = _finite_scalar(arrays.get("delta_hours", 1.0), name="delta_hours")
        if delta <= 0.0:
            raise ValueError("delta_hours must be positive")
        result["battery_throughput"] = float(np.sum((charge + discharge) * delta))
    if "planned_dispatch" in arrays and "settled_dispatch" in arrays and "realized_demand" in arrays:
        planned = _finite_array(arrays["planned_dispatch"], name="planned_dispatch")
        settled = _finite_array(arrays["settled_dispatch"], name="settled_dispatch")
        demand = _finite_array(arrays["realized_demand"], name="realized_demand")
        distances = [recourse_distance(p, s, d) for p, s, d in zip(planned, settled, demand)]
        absolute = np.asarray([item[0] for item in distances], dtype=np.float64)
        normalized = np.asarray([item[1] for item in distances], dtype=np.float64)
        by_channel = np.stack([item[2] for item in distances], axis=0) if distances else np.empty((0, len(RECOURSE_CHANNELS)))
        result["recourse_adjustment"] = {
            "absolute_mean": float(np.mean(absolute)) if absolute.size else float("nan"),
            "absolute_median": _percentile(absolute, 50.0),
            "absolute_p95": _percentile(absolute, 95.0),
            "normalized_mean": float(np.mean(normalized)) if normalized.size else float("nan"),
            "normalized_median": _percentile(normalized, 50.0),
            "normalized_p95": _percentile(normalized, 95.0),
            "by_channel_mean": by_channel.mean(axis=0).tolist() if by_channel.size else [float("nan")] * len(RECOURSE_CHANNELS),
            "channels": list(RECOURSE_CHANNELS),
        }
    if "latency_ms" in arrays:
        latency = _finite_array(arrays["latency_ms"], name="latency_ms").reshape(-1)
        result["latency"] = {"median_ms": _percentile(latency, 50.0), "p95_ms": _percentile(latency, 95.0), "count": int(latency.size)}
    if "terminal_stock_adjustment" in arrays:
        adjustment = _finite_array(arrays["terminal_stock_adjustment"], name="terminal_stock_adjustment")
        result["terminal_stock_valuation_sensitivity"] = float(np.sum(adjustment))
    if "cumulative_raw_objective" in result and "terminal_stock_valuation_sensitivity" in result:
        result["cumulative_adjusted_objective"] = result["cumulative_raw_objective"] + result["terminal_stock_valuation_sensitivity"]
    return result


__all__ = [
    "PHYSICAL_TOLERANCE", "SHORTAGE_TOLERANCE", "THERMAL_ACTIVE_EPSILON",
    "RECOURSE_CHANNELS", "residual_vector", "physical_feasible",
    "shortage_free", "regime_aware_forecast_metrics", "recourse_distance",
    "terminal_stock_adjustment", "summarize_closed_loop",
]
