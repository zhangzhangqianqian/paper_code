"""Recomputable metrics and paired uncertainty for joint experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contract import DISPATCH_ORDER, TASK_ORDER


def _array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _check_forecast(prediction: object, target: object) -> tuple[np.ndarray, np.ndarray]:
    pred, truth = _array(prediction, "prediction"), _array(target, "target")
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != len(TASK_ORDER):
        raise ValueError("forecast arrays must have shape [N,H,4] and match")
    return pred, truth


def mae_by_task_horizon(prediction: object, target: object) -> np.ndarray:
    pred, truth = _check_forecast(prediction, target)
    return np.mean(np.abs(pred - truth), axis=0)


def rmse_by_task_horizon(prediction: object, target: object) -> np.ndarray:
    pred, truth = _check_forecast(prediction, target)
    return np.sqrt(np.mean((pred - truth) ** 2, axis=0))


def wape_by_task_horizon(prediction: object, target: object) -> np.ndarray:
    pred, truth = _check_forecast(prediction, target)
    numerator = np.sum(np.abs(pred - truth), axis=0)
    denominator = np.sum(np.abs(truth), axis=0)
    result = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 1e-12)
    zero = denominator <= 1e-12
    result[zero & (numerator <= 1e-12)] = 0.0
    result[zero & (numerator > 1e-12)] = np.inf
    return result


def dispatch_mae_by_variable(prediction: object, target: object) -> np.ndarray:
    pred, truth = _array(prediction, "prediction"), _array(target, "target")
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != len(DISPATCH_ORDER):
        raise ValueError("dispatch arrays must have shape [N,H,21] and match")
    return np.mean(np.abs(pred - truth), axis=(0, 1))


def physical_carbon(dispatch: object, grid_emission_factor: object = 0.5, gas_emission_factor: object = 0.25) -> np.ndarray:
    values = _array(dispatch, "dispatch")
    if values.ndim != 3 or values.shape[-1] != len(DISPATCH_ORDER):
        raise ValueError("dispatch must have shape [N,H,21]")
    grid = np.asarray(grid_emission_factor, dtype=np.float64)
    gas = np.asarray(gas_emission_factor, dtype=np.float64)
    return values[..., DISPATCH_ORDER.index("grid")] * grid + (
        values[..., DISPATCH_ORDER.index("g_chp")] + values[..., DISPATCH_ORDER.index("g_gb")]
    ) * gas


def paired_moving_block_bootstrap(
    method_a: object,
    method_b: object,
    *,
    block_hours: int = 168,
    replicates: int = 2000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict[str, float | int | str]:
    """Bootstrap paired method differences using contiguous hourly blocks.

    Arrays may be ``[T]`` or ``[seeds,T]``.  The same sampled block indices are
    used for both methods, preserving pairing and serial dependence.
    """

    a, b = _array(method_a, "method_a"), _array(method_b, "method_b")
    if a.shape != b.shape or a.ndim not in {1, 2}:
        raise ValueError("paired metrics must be [T] or [seeds,T] and match")
    if a.ndim == 1:
        a, b = a[None, :], b[None, :]
    seeds, hours = a.shape
    if block_hours <= 0 or block_hours > hours or replicates <= 0:
        raise ValueError("invalid moving-block bootstrap dimensions")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0,1)")
    rng = np.random.default_rng(seed)
    starts = np.arange(hours - block_hours + 1)
    block_count = int(np.ceil(hours / block_hours))
    estimates = np.empty(replicates, dtype=np.float64)
    difference = a - b
    for replicate in range(replicates):
        selected = starts[rng.integers(0, len(starts), size=block_count)]
        indices = np.concatenate([np.arange(start, start + block_hours) for start in selected])[:hours]
        estimates[replicate] = float(difference[:, indices].mean())
    observed = float(difference.mean())
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "observed_difference": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "alpha": float(alpha),
        "replicates": int(replicates),
        "block_hours": int(block_hours),
        "dependence_model": "paired_contiguous_moving_block",
    }


def benjamini_hochberg(p_values: object, alpha: float = 0.05) -> dict[str, np.ndarray | float]:
    p = _array(p_values, "p_values").reshape(-1)
    if ((p < 0.0) | (p > 1.0)).any() or not 0.0 < alpha < 1.0:
        raise ValueError("p-values must be in [0,1] and alpha in (0,1)")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return {"raw_p": p, "fdr_adjusted_p": adjusted, "alpha": float(alpha), "reject": adjusted <= alpha}


@dataclass(frozen=True)
class ForecastMetricTable:
    mae: np.ndarray
    rmse: np.ndarray
    wape: np.ndarray


def evaluate_forecast(prediction: object, target: object) -> ForecastMetricTable:
    return ForecastMetricTable(mae_by_task_horizon(prediction, target), rmse_by_task_horizon(prediction, target), wape_by_task_horizon(prediction, target))


def realized_dispatch_summary(
    dispatch: object,
    demand: object,
    *,
    grid_price: object = 1.0,
    gas_price: object = 0.6,
    carbon_price: object = 0.0,
    grid_emission_factor: float = 0.5,
    gas_emission_factor: float = 0.25,
    unserved_penalty: float = 100.0,
) -> Mapping[str, float]:
    """Summarize physical outcomes from realized dispatch arrays."""

    values = _array(dispatch, "dispatch")
    actual = _array(demand, "demand")
    if values.ndim != 3 or values.shape[-1] != len(DISPATCH_ORDER) or actual.shape != values.shape[:2] + (3,):
        raise ValueError("dispatch/demand shapes are inconsistent")
    index = {name: DISPATCH_ORDER.index(name) for name in DISPATCH_ORDER}
    served_e = values[..., index["grid"]] + values[..., index["pv_use"]] + values[..., index["wt_use"]] + values[..., index["p_chp"]] + values[..., index["p_discharge"]] - values[..., index["p_ec"]] - values[..., index["p_charge"]]
    served_c = values[..., index["q_ec"]] + values[..., index["q_ac"]]
    served_h = values[..., index["q_chp"]] + values[..., index["q_gb"]] - values[..., index["q_ac_in"]] - values[..., index["q_dump"]]
    shortage = np.maximum(actual - np.stack((served_e, served_c, served_h), axis=-1), 0.0)
    gas = values[..., index["g_chp"]] + values[..., index["g_gb"]]
    op_cost = values[..., index["grid"]] * np.asarray(grid_price) + gas * np.asarray(gas_price)
    carbon = physical_carbon(values, grid_emission_factor, gas_emission_factor)
    objective = op_cost + np.asarray(carbon_price) * carbon + unserved_penalty * shortage.sum(axis=-1)
    return {
        "operating_cost": float(op_cost.sum()),
        "physical_carbon": float(carbon.sum()),
        "shortage": float(shortage.sum()),
        "penalized_objective": float(objective.sum()),
        "feasibility_rate": float(np.mean(np.max(shortage, axis=-1) <= 1.0e-8)),
    }


__all__ = [
    "ForecastMetricTable", "benjamini_hochberg", "dispatch_mae_by_variable",
    "evaluate_forecast", "mae_by_task_horizon", "paired_moving_block_bootstrap",
    "physical_carbon", "realized_dispatch_summary", "rmse_by_task_horizon", "wape_by_task_horizon",
]
