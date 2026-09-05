"""Matched chronological closed-loop evaluation engine.

The runner exposes a strict causal boundary: deployable providers receive a
``CausalOriginInput`` and never the future labels kept in
``RealizedOriginLabels``.  Only the runner uses labels for settlement and
auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol
import time

import numpy as np
import torch

from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from ..scheduling.dispatch_schema import VARIABLES
from .closed_loop_metrics import residual_vector, summarize_closed_loop
from .external_v46_data import ExternalV46Split
from .formal_v4_2_rollout import (
    advance_with_executed_first_hour,
    calculate_all_residual_families,
    settle_and_advance_v42,
)
from .formal_v4_recourse import settle_first_step_v4
from .formal_v4_state import FormalV4ClosedLoopState


HORIZON = 4
TASK_COUNT = 4


@dataclass(frozen=True)
class CausalOriginInput:
    """Only deployment-available data for one chronological origin."""

    load_history: np.ndarray
    exog_history: np.ndarray
    device_history: np.ndarray
    activity_history: np.ndarray
    scheduler_context: np.ndarray
    previous_chp: np.ndarray
    renewable_forecast: np.ndarray
    origin_time: np.datetime64
    trajectory_id: str

    def __post_init__(self) -> None:
        for name, shape in (
            ("load_history", (1, 24, 4)),
            ("exog_history", (1, 24, 12)),
            ("device_history", (1, 24, 17)),
            ("activity_history", (1, 24, 6)),
            ("scheduler_context", (1, 4, 6)),
            ("previous_chp", (1, 1)),
            ("renewable_forecast", (4, 2)),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must have finite shape {shape}")
        activity = np.asarray(self.activity_history)
        if not np.isin(activity, (0.0, 1.0)).all():
            raise ValueError("activity_history must be binary")
        if not np.allclose(self.scheduler_context[..., 5], self.scheduler_context[:, :1, 5], atol=1.0e-6):
            raise ValueError("scheduler_context SOC must be constant over the horizon")
        timestamp = np.asarray(self.origin_time, dtype="datetime64[ns]")
        if timestamp.ndim != 0 or not self.trajectory_id:
            raise ValueError("origin_time and trajectory_id are invalid")
        object.__setattr__(self, "origin_time", timestamp)


@dataclass(frozen=True)
class RealizedOriginLabels:
    """Future observations retained for settlement/audit, never providers."""

    forecast_target: np.ndarray
    renewable_realized: np.ndarray
    next_observed_exog: np.ndarray

    def __post_init__(self) -> None:
        for name, shape in (
            ("forecast_target", (4, 4)),
            ("renewable_realized", (4, 2)),
            ("next_observed_exog", (12,)),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must have finite shape {shape}")
            if name != "next_observed_exog" and (value < 0.0).any():
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PlannedStep:
    forecast: np.ndarray
    scheduler_demand: np.ndarray
    renewable_forecast: np.ndarray
    dispatch: np.ndarray
    inference_lp_calls: int = 0

    def __post_init__(self) -> None:
        for name, shape in (
            ("forecast", (4, 4)),
            ("scheduler_demand", (4, 4)),
            ("renewable_forecast", (4, 2)),
            ("dispatch", (4, len(VARIABLES))),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must have finite shape {shape}")
            object.__setattr__(self, name, value)
        if int(self.inference_lp_calls) < 0:
            raise ValueError("inference_lp_calls must be non-negative")


class ActionProvider(Protocol):
    method_id: str
    optimizer_role: str

    def plan(self, origin: CausalOriginInput) -> PlannedStep: ...


@dataclass(frozen=True)
class MatchedClosedLoopResult:
    method_id: str
    seed: int
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping[str, Any]
    inference_lp_calls: int
    reference_lp_calls: int = 0


def _require_metadata(selection: ExternalV46Split) -> tuple[np.ndarray, np.ndarray]:
    if selection.trajectory_ids is None or selection.source_state_hashes is None:
        raise ValueError("matched evaluation requires trajectory_ids and source_state_hashes")
    trajectory_ids = np.asarray(selection.trajectory_ids, dtype=str)
    source_hashes = np.asarray(selection.source_state_hashes, dtype=str)
    if trajectory_ids.shape != (len(selection),) or source_hashes.shape != (len(selection),):
        raise ValueError("trajectory metadata must align with selection")
    if np.unique(trajectory_ids).tolist() != ["capacity_bound_causal"]:
        raise ValueError(f"unexpected selection trajectory IDs: {np.unique(trajectory_ids).tolist()}")
    if any(not value for value in source_hashes.tolist()):
        raise ValueError("selection contains an empty source state hash")
    return trajectory_ids, source_hashes


def initial_state_from_selection(selection: ExternalV46Split) -> FormalV4ClosedLoopState:
    trajectory_ids, _ = _require_metadata(selection)
    return FormalV4ClosedLoopState(
        torch.as_tensor(selection.load_history[:1]),
        torch.as_tensor(selection.exog_history[:1]),
        torch.as_tensor(selection.device_history[:1]),
        torch.as_tensor(selection.device_status[:1]),
        torch.as_tensor(selection.scheduler_context[:1, 0, 5:6]),
        torch.as_tensor(selection.previous_chp[:1]),
        selection.target_times[0],
        str(trajectory_ids[0]),
    )


def split_origin(
    selection: ExternalV46Split,
    index: int,
    state: FormalV4ClosedLoopState,
) -> tuple[CausalOriginInput, RealizedOriginLabels]:
    if index < 0 or index >= len(selection):
        raise IndexError("origin index is out of range")
    trajectory_ids, _ = _require_metadata(selection)
    row = selection.take(np.asarray([index], dtype=np.int64))
    context = np.asarray(row.scheduler_context, dtype=np.float64).copy()
    context[:, :, 5] = float(state.initial_soc[0, 0].detach().cpu().item())
    causal = CausalOriginInput(
        load_history=state.load_history.detach().cpu().numpy().astype(np.float32, copy=True),
        exog_history=state.exog_history.detach().cpu().numpy().astype(np.float32, copy=True),
        device_history=state.device_history.detach().cpu().numpy().astype(np.float32, copy=True),
        activity_history=state.activity_history.detach().cpu().numpy().astype(np.float32, copy=True),
        scheduler_context=context,
        previous_chp=state.previous_chp.detach().cpu().numpy().astype(np.float32, copy=True),
        renewable_forecast=context[0, :, :2].copy(),
        origin_time=row.target_times[0],
        trajectory_id=str(trajectory_ids[index]),
    )
    labels = RealizedOriginLabels(
        forecast_target=row.forecast_target[0].astype(np.float32, copy=True),
        renewable_realized=row.renewable_realized[0].astype(np.float32, copy=True),
        next_observed_exog=selection.exog_history[min(index + 1, len(selection) - 1), -1].astype(np.float32, copy=True),
    )
    return causal, labels


def _planned_residuals_for_row(
    row: np.ndarray,
    state: FormalV4ClosedLoopState,
    demand: np.ndarray,
    renewable: np.ndarray,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    residuals = calculate_all_residual_families(row, state, {"demand": demand[:3], "renewable": renewable}, parameters)
    return residual_vector(residuals)


def planned_horizon_residuals(
    plan: PlannedStep,
    state: FormalV4ClosedLoopState,
    parameters: Mapping[str, Any],
) -> np.ndarray:
    """Check each raw planned row without invoking recourse."""

    shadow = state
    values: list[np.ndarray] = []
    for horizon in range(4):
        row = plan.dispatch[horizon]
        demand = plan.scheduler_demand[horizon]
        renewable = plan.renewable_forecast[horizon]
        # A forecast head may emit a small signed residual around zero.  The
        # physical decoder and LP both interpret demand/renewables as
        # non-negative quantities, so the audit uses the same boundary clamp.
        demand = np.maximum(demand, 0.0)
        renewable = np.maximum(renewable, 0.0)
        values.append(_planned_residuals_for_row(row, shadow, demand, renewable, parameters))
        shadow = advance_with_executed_first_hour(
            shadow,
            row,
            {"demand": demand[:3], "renewable": renewable, "realized_load": demand, "realized_exog": shadow.exog_history[0, -1].detach().cpu().numpy()},
            parameters,
        )
    return np.stack(values, axis=0)


def _validate_selection_chronology(selection: ExternalV46Split) -> None:
    if len(selection) == 0:
        raise ValueError("selection chronology is empty")
    years = set(int(value) for value in selection.target_times.astype("datetime64[Y]").astype(int) + 1970)
    if years != {2019}:
        raise ValueError("selection must contain exactly 2019")
    if not np.all(np.diff(selection.target_times) == np.timedelta64(1, "h")):
        raise ValueError("selection timestamps must be consecutive hourly origins")


def _settled_arrays(
    result: Any,
    plan: PlannedStep,
    labels: RealizedOriginLabels,
    state: FormalV4ClosedLoopState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    settled = result.settled
    residuals = residual_vector(result.residuals)
    return (
        settled,
        residuals,
        result.shortage.copy(),
        float(result.operating_cost),
        float(result.physical_carbon),
        float(result.penalized_objective),
    )


def run_matched_closed_loop(
    selection: ExternalV46Split,
    provider: ActionProvider,
    parameters: Mapping[str, Any],
    *,
    method_id: str,
    seed: int,
    warmup_origins: int = 100,
    thermal_active_scales: Mapping[str, float] | None = None,
) -> MatchedClosedLoopResult:
    """Run one provider chronologically, settling only each first action."""

    _validate_selection_chronology(selection)
    _require_metadata(selection)
    if warmup_origins < 0 or warmup_origins >= len(selection):
        raise ValueError("warmup_origins must be in [0, len(selection))")
    state = initial_state_from_selection(selection)
    forecasts: list[np.ndarray] = []
    forecast_targets: list[np.ndarray] = []
    scheduler_demands: list[np.ndarray] = []
    renewable_forecasts: list[np.ndarray] = []
    planned_dispatch: list[np.ndarray] = []
    settled_dispatch: list[np.ndarray] = []
    planned_residuals: list[np.ndarray] = []
    settled_residuals: list[np.ndarray] = []
    shortages: list[np.ndarray] = []
    operating: list[float] = []
    carbon: list[float] = []
    objective: list[float] = []
    realized_demand: list[np.ndarray] = []
    times: list[np.datetime64] = []
    state_hashes: list[str] = []
    initial_socs: list[float] = []
    final_socs: list[float] = []
    latency_ms: list[float] = []
    lp_calls = 0

    for index in range(len(selection)):
        origin, labels = split_origin(selection, index, state)
        started = time.perf_counter_ns()
        plan = provider.plan(origin)
        if not isinstance(plan, PlannedStep):
            raise TypeError("provider must return PlannedStep")
        # Settlement is part of the timed deployment step; planned checks and
        # artifact writes are deliberately outside the latency measurement.
        planned_residual = planned_horizon_residuals(plan, state, parameters)
        settled_result = settle_and_advance_v42(
            state,
            plan.dispatch,
            {"demand": labels.forecast_target[0, :3], "renewable": labels.renewable_realized[0], "realized_load": labels.forecast_target[0], "realized_exog": labels.next_observed_exog},
            parameters,
        )
        elapsed = (time.perf_counter_ns() - started) / 1.0e6
        if index >= warmup_origins:
            latency_ms.append(float(elapsed))
        settled, settled_residual, shortage, operating_cost, physical_carbon, penalized = _settled_arrays(settled_result, plan, labels, state)
        forecasts.append(plan.forecast.copy())
        forecast_targets.append(labels.forecast_target.copy())
        scheduler_demands.append(plan.scheduler_demand.copy())
        renewable_forecasts.append(plan.renewable_forecast.copy())
        planned_dispatch.append(plan.dispatch.copy())
        settled_dispatch.append(settled.copy())
        planned_residuals.append(np.max(planned_residual, axis=0))
        settled_residuals.append(settled_residual)
        shortages.append(shortage)
        operating.append(operating_cost)
        carbon.append(physical_carbon)
        objective.append(penalized)
        realized_demand.append(labels.forecast_target[0, :3].copy())
        times.append(origin.origin_time)
        state_hashes.append(state.state_hash)
        initial_socs.append(float(state.initial_soc[0, 0].detach().cpu().item()))
        state = settled_result.next_state
        final_socs.append(float(state.initial_soc[0, 0].detach().cpu().item()))
        lp_calls += int(plan.inference_lp_calls)

    arrays: dict[str, np.ndarray] = {
        "forecast": np.stack(forecasts),
        "forecast_target": np.stack(forecast_targets),
        "scheduler_demand": np.stack(scheduler_demands),
        "renewable_forecast": np.stack(renewable_forecasts),
        "planned_dispatch": np.stack(planned_dispatch),
        "settled_dispatch": np.stack(settled_dispatch),
        "planned_residuals": np.stack(planned_residuals),
        "settled_residuals": np.stack(settled_residuals),
        "shortage": np.stack(shortages),
        "operating_cost": np.asarray(operating),
        "physical_carbon": np.asarray(carbon),
        "penalized_objective": np.asarray(objective),
        "realized_demand": np.stack(realized_demand),
        "times": np.asarray(times, dtype="datetime64[ns]"),
        "state_hashes": np.asarray(state_hashes, dtype=str),
        "initial_soc": np.asarray(initial_socs),
        "final_soc": np.asarray(final_socs),
        "latency_ms": np.asarray(latency_ms),
    }
    metrics = summarize_closed_loop(arrays, thermal_active_scales=thermal_active_scales)
    return MatchedClosedLoopResult(str(method_id), int(seed), arrays, metrics, lp_calls, 0)


def _reference_parameters(origin: CausalOriginInput, parameters: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(parameters)
    context = origin.scheduler_context[0]
    result["grid_energy_price"] = context[:, 2].copy()
    result["gas_energy_price"] = context[:, 3].copy()
    result["carbon_price"] = context[:, 4].copy()
    return result


def _dispatch_from_lp(values: Mapping[str, np.ndarray], horizon: int) -> np.ndarray:
    dispatch = np.zeros((horizon, len(VARIABLES)), dtype=np.float64)
    for index, name in enumerate(VARIABLES):
        if name in values:
            dispatch[:, index] = np.asarray(values[name], dtype=np.float64)
    return dispatch


def run_perfect_information_mpc_reference(
    selection: ExternalV46Split,
    parameters: Mapping[str, Any],
) -> MatchedClosedLoopResult:
    """Run one non-deployable realized-information rolling LP reference."""

    _validate_selection_chronology(selection)
    _require_metadata(selection)
    state = initial_state_from_selection(selection)
    plans: list[np.ndarray] = []
    settled: list[np.ndarray] = []
    planned_residuals: list[np.ndarray] = []
    settled_residuals: list[np.ndarray] = []
    shortages: list[np.ndarray] = []
    operating: list[float] = []
    carbon: list[float] = []
    objective: list[float] = []
    demands: list[np.ndarray] = []
    initial_socs: list[float] = []
    final_socs: list[float] = []
    times: list[np.datetime64] = []
    hashes: list[str] = []
    for index in range(len(selection)):
        origin, labels = split_origin(selection, index, state)
        lp_parameters = _reference_parameters(origin, parameters)
        dispatch_result = solve_dispatch_lp(DispatchInputs(
            demand=labels.forecast_target[:, :3].astype(np.float64),
            pv_available=labels.renewable_realized[:, 0].astype(np.float64),
            wt_available=labels.renewable_realized[:, 1].astype(np.float64),
            parameters=lp_parameters,
            initial_soc=float(state.initial_soc[0, 0].item()),
            # The settled recourse path is float64, while the carried state is
            # stored in float32.  A one-ULP round-up at the capacity boundary
            # must not make the reference LP reject an otherwise feasible
            # state.
            previous_chp=min(float(state.previous_chp[0, 0].item()), float(parameters["chp_electric_capacity"])),
        ))
        if not dispatch_result.success:
            raise RuntimeError(f"Perfect-Information-MPC LP failed at origin {index}: {dispatch_result.message}")
        plan = _dispatch_from_lp(dispatch_result.values, 4)
        planned = PlannedStep(labels.forecast_target.copy(), labels.forecast_target.copy(), labels.renewable_realized.copy(), plan, 0)
        planned_residuals.append(np.max(planned_horizon_residuals(planned, state, lp_parameters), axis=0))
        settlement_parameters = dict(lp_parameters)
        settlement_parameters["grid_energy_price"] = float(lp_parameters["grid_energy_price"][0])
        settlement_parameters["gas_energy_price"] = float(lp_parameters["gas_energy_price"][0])
        settlement_parameters["carbon_price"] = float(lp_parameters["carbon_price"][0])
        settled_result = settle_and_advance_v42(
            state,
            plan,
            {"demand": labels.forecast_target[0, :3], "renewable": labels.renewable_realized[0], "realized_load": labels.forecast_target[0], "realized_exog": labels.next_observed_exog},
            settlement_parameters,
        )
        plans.append(plan)
        settled.append(settled_result.settled)
        settled_residuals.append(residual_vector(settled_result.residuals))
        shortages.append(settled_result.shortage)
        operating.append(settled_result.operating_cost)
        carbon.append(settled_result.physical_carbon)
        objective.append(settled_result.penalized_objective)
        demands.append(labels.forecast_target[0, :3].copy())
        initial_socs.append(float(state.initial_soc[0, 0].item()))
        state = settled_result.next_state
        final_socs.append(float(state.initial_soc[0, 0].item()))
        times.append(origin.origin_time)
        hashes.append(state.state_hash)
    arrays = {
        "planned_dispatch": np.stack(plans),
        "settled_dispatch": np.stack(settled),
        "planned_residuals": np.stack(planned_residuals),
        "settled_residuals": np.stack(settled_residuals),
        "shortage": np.stack(shortages),
        "operating_cost": np.asarray(operating),
        "physical_carbon": np.asarray(carbon),
        "penalized_objective": np.asarray(objective),
        "realized_demand": np.stack(demands),
        "initial_soc": np.asarray(initial_socs),
        "final_soc": np.asarray(final_socs),
        "times": np.asarray(times, dtype="datetime64[ns]"),
        "state_hashes": np.asarray(hashes, dtype=str),
        "latency_ms": np.empty(0, dtype=np.float64),
    }
    metrics = summarize_closed_loop(arrays)
    return MatchedClosedLoopResult("Perfect-Information-MPC", 0, arrays, metrics, 0, len(selection))


__all__ = [
    "CausalOriginInput", "RealizedOriginLabels", "PlannedStep", "ActionProvider",
    "MatchedClosedLoopResult", "split_origin", "initial_state_from_selection",
    "planned_horizon_residuals", "run_matched_closed_loop",
    "run_perfect_information_mpc_reference",
]
