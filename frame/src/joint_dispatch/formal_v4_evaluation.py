"""Common chronological evaluation and statistical summaries for formal-v4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .evaluation import (
    mae_by_task_horizon,
    paired_moving_block_bootstrap,
    physical_carbon,
    realized_dispatch_summary,
    rmse_by_task_horizon,
    wape_by_task_horizon,
)


def _finite(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class ForecastMetricTable:
    """Task-by-horizon forecast metrics with gas reported separately."""

    mae: np.ndarray
    rmse: np.ndarray
    wape: np.ndarray

    def __post_init__(self) -> None:
        for name in ("mae", "rmse", "wape"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 2 or value.shape[0] != 4:
                raise ValueError("forecast metric arrays must have shape [4,H]")
            if name != "wape" and not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")

    @property
    def task_wape(self) -> np.ndarray:
        values = np.asarray(self.wape, dtype=np.float64)
        finite = np.where(np.isfinite(values), values, np.nan)
        return np.nanmean(finite, axis=1)

    @property
    def rigid_macro_wape(self) -> float:
        return float(np.mean(self.task_wape[:3]))

    @property
    def gas_wape(self) -> float:
        return float(self.task_wape[3])


@dataclass(frozen=True)
class DispatchMetricTable:
    """Closed-loop dispatch metrics; fields are aggregates over executed hours."""

    operating_cost: float
    physical_carbon: float
    penalized_objective: float
    shortage_energy: float
    shortage_rate: float
    balance_residual: float
    capacity_violation: float = 0.0
    conversion_violation: float = 0.0
    soc_violation: float = 0.0
    ramp_violation: float = 0.0
    p_dump: float = 0.0
    q_dump: float = 0.0
    pv_curtailment: float = 0.0
    wt_curtailment: float = 0.0
    optimizer_calls: int = 0
    latency_ms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("operating_cost", "physical_carbon", "penalized_objective", "shortage_energy", "shortage_rate", "balance_residual", "capacity_violation", "conversion_violation", "soc_violation", "ramp_violation", "p_dump", "q_dump", "pv_curtailment", "wt_curtailment"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if int(self.optimizer_calls) < 0:
            raise ValueError("optimizer_calls must be non-negative")


@dataclass(frozen=True)
class FormalV4EvaluationResult:
    method_id: str
    metric_names: tuple[str, ...]
    forecast: ForecastMetricTable | None
    dispatch: DispatchMetricTable
    per_window: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        forbidden = {"per_window_regret_vs_oracle", "per_window_regret_vs_pi"}
        if forbidden.intersection(self.metric_names):
            raise ValueError("oracle comparisons must be aggregated closed-loop differences, never per-window regret")
        if "oracle" in self.method_id.lower() and self.metadata.get("deployable", True):
            raise ValueError("perfect-information references must be marked non-deployable")


def evaluate_forecast_table(prediction: object, target: object) -> ForecastMetricTable:
    return ForecastMetricTable(
        mae=mae_by_task_horizon(prediction, target).T,
        rmse=rmse_by_task_horizon(prediction, target).T,
        wape=wape_by_task_horizon(prediction, target).T,
    )


def _dispatch_hourly_metrics(dispatch: np.ndarray, demand: np.ndarray, *, grid_price: object = 1.0, gas_price: object = 0.6, carbon_price: object = 0.0, grid_emission_factor: float = 0.5, gas_emission_factor: float = 0.25, unserved_penalty: float = 100.0) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if dispatch.ndim != 3 or dispatch.shape[-1] != 21 or demand.shape != dispatch.shape[:2] + (3,):
        raise ValueError("dispatch must be [N,H,21] and demand [N,H,3]")
    idx = {"grid": 0, "pv_use": 1, "pv_curt": 2, "wt_use": 3, "wt_curt": 4, "g_chp": 5, "g_gb": 6, "p_chp": 7, "q_chp": 8, "q_gb": 9, "p_ec": 10, "q_ec": 11, "q_ac_in": 12, "q_ac": 13, "p_charge": 14, "p_discharge": 15, "soc": 16, "slack_e": 17, "slack_c": 18, "slack_h": 19, "q_dump": 20}
    served = np.stack((dispatch[..., idx["grid"]] + dispatch[..., idx["pv_use"]] + dispatch[..., idx["wt_use"]] + dispatch[..., idx["p_chp"]] + dispatch[..., idx["p_discharge"]] - dispatch[..., idx["p_ec"]] - dispatch[..., idx["p_charge"]], dispatch[..., idx["q_ec"]] + dispatch[..., idx["q_ac"]], dispatch[..., idx["q_chp"]] + dispatch[..., idx["q_gb"]] - dispatch[..., idx["q_ac_in"]] - dispatch[..., idx["q_dump"]]), axis=-1)
    shortage = np.maximum(demand - served, 0.0)
    grid = dispatch[..., idx["grid"]]
    gas = dispatch[..., idx["g_chp"]] + dispatch[..., idx["g_gb"]]
    op = grid * np.asarray(grid_price) + gas * np.asarray(gas_price)
    carbon = physical_carbon(dispatch, grid_emission_factor, gas_emission_factor)
    objective = op + np.asarray(carbon_price) * carbon + unserved_penalty * shortage.sum(axis=-1)
    residual = np.max(np.abs(demand - served), axis=-1)
    per_window = {
        "operating_cost": op.sum(axis=1), "physical_carbon": carbon.sum(axis=1),
        "penalized_objective": objective.sum(axis=1), "shortage_energy": shortage.sum(axis=(1, 2)),
        "shortage_rate": np.mean(np.max(shortage, axis=-1) > 1.0e-8, axis=1),
        "balance_residual": residual.max(axis=1), "p_dump": dispatch[..., idx["pv_curt"]].sum(axis=1),
        "q_dump": dispatch[..., idx["q_dump"]].sum(axis=1), "pv_curtailment": dispatch[..., idx["pv_curt"]].sum(axis=1),
        "wt_curtailment": dispatch[..., idx["wt_curt"]].sum(axis=1),
    }
    aggregate = {name: float(np.mean(value)) for name, value in per_window.items()}
    return aggregate, per_window


def evaluate_dispatch_table(dispatch: object, demand: object, **kwargs: object) -> tuple[DispatchMetricTable, dict[str, np.ndarray]]:
    optimizer_calls = int(kwargs.pop("optimizer_calls", 0))
    values, per_window = _dispatch_hourly_metrics(_finite(dispatch, "dispatch"), _finite(demand, "demand"), **kwargs)
    table = DispatchMetricTable(**values, optimizer_calls=optimizer_calls)
    return table, per_window


def evaluate_formal_v4_arrays(method_id: str, prediction: object | None, target: object | None, dispatch: object, demand: object, *, perfect_information_objective: float | None = None, optimizer_calls: int = 0, deployable: bool = True, **kwargs: object) -> FormalV4EvaluationResult:
    forecast = None if prediction is None or target is None else evaluate_forecast_table(prediction, target)
    table, per_window = evaluate_dispatch_table(dispatch, demand, **kwargs)
    table = DispatchMetricTable(**{**table.__dict__, "optimizer_calls": int(optimizer_calls)})
    names = tuple(per_window.keys())
    metadata: dict[str, Any] = {"deployable": bool(deployable)}
    if perfect_information_objective is not None:
        metadata["closed_loop_objective_difference_vs_pi"] = float(table.penalized_objective - perfect_information_objective)
        names += ("closed_loop_objective_difference_vs_pi",)
    return FormalV4EvaluationResult(method_id, names, forecast, table, per_window, metadata)


def hierarchical_paired_interval(method_a: object, method_b: object, *, block_hours: int = 168, replicates: int = 2000, alpha: float = 0.05, seed: int = 2026) -> dict[str, float | int | str]:
    """Seed-paired, contiguous-block uncertainty; five seeds are not pseudo-replicates."""
    result = paired_moving_block_bootstrap(method_a, method_b, block_hours=block_hours, replicates=replicates, alpha=alpha, seed=seed)
    a, b = _finite(method_a, "method_a"), _finite(method_b, "method_b")
    difference = a - b
    if difference.ndim == 1:
        result["direction_count"] = int(np.sum(difference.mean() < 0.0))
    else:
        result["direction_count"] = int(np.sum(difference.mean(axis=1) < 0.0))
    result["seed_count"] = int(a.shape[0]) if a.ndim == 2 else 1
    result["standardized_paired_effect"] = float(difference.mean() / max(float(difference.std(ddof=1)), 1.0e-12))
    result["uncertainty_unit"] = "seed_matched_168h_block"
    return result


def run_formal_v4_rollout(method: Any, windows: Sequence[Any], *, initial_state: Any, execute_first_step: Callable[..., Any] | None = None) -> FormalV4EvaluationResult:
    """Execute chronological windows without resetting the carried state.

    A method exposes ``predict_and_dispatch(window, state)`` and returns a
    mapping containing ``forecast``, ``dispatch`` and optional ``target`` and
    ``demand`` arrays.  The callback can perform one-pass realized recourse;
    state is replaced only after that callback returns.
    """
    state = initial_state
    forecasts, targets, dispatches, demands = [], [], [], []
    previous_time = None
    for window in windows:
        if isinstance(window, Mapping) and window.get("target_time") is not None:
            current_time = np.datetime64(window["target_time"])
            if previous_time is not None and current_time <= previous_time:
                raise ValueError("rollout windows must be strictly chronological")
            previous_time = current_time
        result = method.predict_and_dispatch(window, state)
        if not isinstance(result, Mapping):
            raise TypeError("rollout method must return a mapping")
        if execute_first_step is not None:
            result = execute_first_step(result, state, window)
        forecasts.append(np.asarray(result["forecast"], dtype=np.float64) if result.get("forecast") is not None else None)
        targets.append(np.asarray(result["target"], dtype=np.float64) if result.get("target") is not None else None)
        dispatches.append(np.asarray(result["dispatch"], dtype=np.float64))
        demands.append(np.asarray(result["demand"], dtype=np.float64))
        next_state = result.get("next_state", state)
        if not isinstance(next_state, Mapping):
            raise ValueError("rollout next_state must be a mapping")
        for name in ("soc", "previous_chp"):
            if name in next_state and not np.isfinite(float(next_state[name])):
                raise ValueError("rollout next_state must be finite")
        state = next_state
    prediction = None if any(value is None for value in forecasts) else np.stack(forecasts, axis=0)
    target = None if any(value is None for value in targets) else np.stack(targets, axis=0)
    canonical_calls = getattr(method, "online_optimizer_calls_per_window", None)
    legacy_calls = getattr(method, "online_lp_calls_per_window", None)
    if canonical_calls is None:
        canonical_calls = legacy_calls if legacy_calls is not None else 0
    elif legacy_calls is not None and int(canonical_calls) != int(legacy_calls):
        raise ValueError("canonical and legacy optimizer-call fields disagree")
    return evaluate_formal_v4_arrays(getattr(method, "method_id", method.__class__.__name__), prediction, target, np.stack(dispatches, axis=0), np.stack(demands, axis=0), optimizer_calls=int(canonical_calls) * len(windows), deployable=bool(getattr(method, "deployable", True)))


__all__ = [
    "DispatchMetricTable", "FormalV4EvaluationResult", "ForecastMetricTable",
    "evaluate_dispatch_table", "evaluate_formal_v4_arrays", "evaluate_forecast_table",
    "hierarchical_paired_interval", "run_formal_v4_rollout",
]
