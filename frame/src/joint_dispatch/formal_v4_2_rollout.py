"""Canonical first-step settlement and chronological rollout for formal-v4.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from ..scheduling.dispatch_schema import VARIABLES
from .formal_v4_recourse import FormalV4RealizedOutcome, settle_first_step_v4
from .formal_v4_state import FormalV4ClosedLoopState, advance_formal_v4_state


_I = {name: index for index, name in enumerate(VARIABLES)}
_STATUS_FROM = ("p_chp", "q_gb", "q_ec", "q_ac", "p_charge", "p_discharge")


@dataclass(frozen=True)
class PhysicalResidualsV42:
    balance: np.ndarray
    capacity: np.ndarray
    conversion: np.ndarray
    soc: np.ndarray
    ramp: np.ndarray
    exclusivity: np.ndarray
    renewable_accounting: np.ndarray
    finite: np.ndarray

    @property
    def balance_residual_max(self) -> float:
        return float(np.max(np.abs(self.balance))) if self.balance.size else 0.0

    @property
    def capacity_violation_max(self) -> float:
        return float(np.max(np.abs(self.capacity))) if self.capacity.size else 0.0


@dataclass(frozen=True)
class SettledStepV42:
    planned: np.ndarray
    settled: np.ndarray
    shortage: np.ndarray
    p_dump: float
    q_dump: float
    residuals: PhysicalResidualsV42
    next_state: FormalV4ClosedLoopState
    executed_plan_index: int
    operating_cost: float = 0.0
    physical_carbon: float = 0.0
    penalized_objective: float = 0.0


@dataclass(frozen=True)
class ChronologicalRolloutV42:
    method_id: str
    forecast_target: np.ndarray
    forecast_prediction: np.ndarray
    planned_dispatch: np.ndarray
    settled_dispatch: np.ndarray
    shortage_energy: np.ndarray
    p_dump: np.ndarray
    q_dump: np.ndarray
    residuals: tuple[PhysicalResidualsV42, ...]
    operating_cost: np.ndarray
    physical_carbon: np.ndarray
    penalized_objective: np.ndarray
    target_times: np.ndarray
    next_states: tuple[FormalV4ClosedLoopState, ...]


def _realized_fields(realized: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if isinstance(realized, Mapping):
        demand = realized.get("demand", realized.get("realized_demand", realized.get("load")))
        renewable = realized.get("renewable", realized.get("realized_renewable"))
        load = realized.get("realized_load", realized.get("load"))
        exog = realized.get("realized_exog", realized.get("exog"))
    elif isinstance(realized, (tuple, list)) and len(realized) >= 2:
        demand, renewable = realized[0], realized[1]
        load = realized[2] if len(realized) > 2 else None
        exog = realized[3] if len(realized) > 3 else None
    else:
        array = np.asarray(realized, dtype=np.float64)
        if array.shape == (5,):
            demand, renewable = array[:3], array[3:]
        else:
            raise ValueError("realized_first_hour must provide demand [3] and renewable [2]")
        load = exog = None
    demand_array = np.asarray(demand, dtype=np.float64)
    renewable_array = np.asarray(renewable, dtype=np.float64)
    if demand_array.shape != (3,) or renewable_array.shape != (2,):
        raise ValueError("realized demand and renewable must have shapes [3] and [2]")
    if not np.isfinite(demand_array).all() or not np.isfinite(renewable_array).all() or (demand_array < 0).any() or (renewable_array < 0).any():
        raise ValueError("realized demand and renewable must be finite and non-negative")
    return demand_array, renewable_array, None if load is None else np.asarray(load, dtype=np.float64), None if exog is None else np.asarray(exog, dtype=np.float64)


def _settle_numpy(planned: np.ndarray, demand: np.ndarray, renewable: np.ndarray, parameters: Mapping[str, Any], state: FormalV4ClosedLoopState | None = None) -> FormalV4RealizedOutcome:
    action = torch.as_tensor(planned, dtype=torch.float64).reshape(1, len(VARIABLES))
    demand_tensor = torch.as_tensor(demand, dtype=torch.float64).reshape(1, 3)
    renew_tensor = torch.as_tensor(renewable, dtype=torch.float64).reshape(1, 2)
    if state is None:
        soc = torch.tensor([[0.5]], dtype=torch.float64)
        previous = torch.tensor([[0.0]], dtype=torch.float64)
    else:
        if state.load_history.shape[0] != 1:
            raise ValueError("formal-v4.2 NumPy settlement currently requires a batch of one")
        soc = state.initial_soc.detach().to(dtype=torch.float64, device="cpu")
        previous = state.previous_chp.detach().to(dtype=torch.float64, device="cpu")
    with torch.no_grad():
        return settle_first_step_v4(action, demand_tensor, renew_tensor, parameters, initial_soc=soc, previous_chp=previous)


def canonical_recourse_once(planned: np.ndarray, realized_first_hour: Any, parameters: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Settle exactly the first plan row against the realized first hour.

    The function intentionally returns only the settled row and diagnostics;
    callers that need state advancement should use :func:`settle_and_advance_v42`
    so the underlying recourse projection is performed once.
    """

    plan = np.asarray(planned, dtype=np.float64)
    if plan.ndim == 2:
        if plan.shape != (4, len(VARIABLES)):
            raise ValueError("planned four-hour action must have shape [4,21]")
        row = plan[0]
    elif plan.shape == (len(VARIABLES),):
        row = plan
    else:
        raise ValueError("planned action must have shape [4,21] or [21]")
    demand, renewable, _, _ = _realized_fields(realized_first_hour)
    outcome = _settle_numpy(row, demand, renewable, parameters)
    return (
        outcome.realized_dispatch[0].detach().cpu().numpy(),
        outcome.shortage[0].detach().cpu().numpy(),
        float(outcome.p_dump[0].item()),
        float(outcome.recomputed_heat_surplus[0].item()),
    )


def _capacity_residuals(settled: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    limits = np.full(len(VARIABLES), np.inf, dtype=np.float64)
    values = {
        "grid": parameters.get("grid_import_capacity"),
        "g_chp": None if parameters.get("chp_electric_efficiency") is None else float(parameters["chp_electric_capacity"]) / float(parameters["chp_electric_efficiency"]),
        "g_gb": None if parameters.get("gas_boiler_efficiency") is None else float(parameters["gas_boiler_capacity"]) / float(parameters["gas_boiler_efficiency"]),
        "p_chp": parameters.get("chp_electric_capacity"), "q_chp": parameters.get("chp_heat_capacity"),
        "q_gb": parameters.get("gas_boiler_capacity"),
        "p_ec": None if parameters.get("electric_chiller_cop") is None else float(parameters["electric_chiller_capacity"]) / float(parameters["electric_chiller_cop"]),
        "q_ec": parameters.get("electric_chiller_capacity"),
        "q_ac_in": None if parameters.get("absorption_chiller_cop") is None else float(parameters["absorption_chiller_capacity"]) / float(parameters["absorption_chiller_cop"]),
        "q_ac": parameters.get("absorption_chiller_capacity"),
        "p_charge": parameters.get("bess_power_capacity"), "p_discharge": parameters.get("bess_power_capacity"),
        "soc": parameters.get("bess_energy_capacity"),
    }
    for name, limit in values.items():
        if limit is not None:
            limits[_I[name]] = float(limit)
    return np.maximum(np.maximum(-settled, 0.0), np.maximum(settled - limits, 0.0))


def calculate_all_residual_families(
    settled: np.ndarray,
    state: FormalV4ClosedLoopState,
    realized_first_hour: Any,
    parameters: Mapping[str, Any],
) -> PhysicalResidualsV42:
    """Compute independent physical residual families for one settled row."""

    values = np.asarray(settled, dtype=np.float64)
    if values.shape != (len(VARIABLES),):
        raise ValueError("settled action must have shape [21]")
    demand, renewable, _, _ = _realized_fields(realized_first_hour)
    def g(name: str) -> float:
        return float(values[_I[name]])
    derived_p_dump = max(
        g("grid") + g("pv_use") + g("wt_use") + g("p_chp") + g("p_discharge")
        + g("slack_e") - g("p_ec") - g("p_charge") - demand[0],
        0.0,
    )
    balance = np.asarray([
        g("grid") + g("pv_use") + g("wt_use") + g("p_chp") + g("p_discharge") + g("slack_e") - g("p_ec") - g("p_charge") - demand[0] - derived_p_dump,
        g("q_ec") + g("q_ac") + g("slack_c") - demand[1],
        g("q_chp") + g("q_gb") + g("slack_h") - g("q_ac_in") - g("q_dump") - demand[2],
    ], dtype=np.float64)
    eta_e = float(parameters["chp_electric_efficiency"]); eta_h = float(parameters["chp_heat_efficiency"])
    conversion = np.asarray([
        g("p_chp") - eta_e * g("g_chp"), g("q_chp") - eta_h * g("g_chp"),
        g("q_gb") - float(parameters["gas_boiler_efficiency"]) * g("g_gb"),
        g("q_ec") - float(parameters["electric_chiller_cop"]) * g("p_ec"),
        g("q_ac") - float(parameters["absorption_chiller_cop"]) * g("q_ac_in"),
    ], dtype=np.float64)
    capacity = _capacity_residuals(values, parameters)
    energy_capacity = float(parameters["bess_energy_capacity"])
    eta_b = float(parameters["bess_roundtrip_efficiency"]) ** 0.5
    prior_soc = float(state.initial_soc[0, 0].item()) * energy_capacity
    soc = np.asarray([values[_I["soc"]] - prior_soc - eta_b * values[_I["p_charge"]] + values[_I["p_discharge"]] / eta_b, values[_I["soc"]] - values[_I["soc"]]], dtype=np.float64)
    ramp = float(parameters["chp_ramp_fraction"]) * float(parameters["chp_electric_capacity"])
    ramp_residual = np.asarray([max(abs(g("p_chp") - float(state.previous_chp[0, 0].item())) - ramp, 0.0)], dtype=np.float64)
    exclusivity = np.asarray([min(max(g("p_charge"), 0.0), max(g("p_discharge"), 0.0))], dtype=np.float64)
    renewable_accounting = np.asarray([g("pv_use") + g("pv_curt") - renewable[0], g("wt_use") + g("wt_curt") - renewable[1]], dtype=np.float64)
    finite = np.asarray([0.0 if np.isfinite(values).all() else 1.0], dtype=np.float64)
    return PhysicalResidualsV42(balance, capacity, conversion, soc, ramp_residual, exclusivity, renewable_accounting, finite)


def advance_with_executed_first_hour(
    state: FormalV4ClosedLoopState,
    settled: np.ndarray,
    realized_first_hour: Any,
    parameters: Mapping[str, Any],
) -> FormalV4ClosedLoopState:
    """Append the settled row (never a later planned row) to the state."""

    values = np.asarray(settled, dtype=np.float64)
    if values.shape != (len(VARIABLES),):
        raise ValueError("settled action must have shape [21]")
    _, _, realized_load, realized_exog = _realized_fields(realized_first_hour)
    capacity = float(parameters["bess_energy_capacity"])
    if capacity <= 0.0:
        raise ValueError("bess_energy_capacity must be positive")
    action = torch.as_tensor(values, dtype=state.device_history.dtype, device=state.device_history.device).unsqueeze(0)
    activity = torch.stack([(action[:, _I[name]] > 1.0e-6).to(action.dtype) for name in _STATUS_FROM], dim=-1)
    next_soc = (action[:, _I["soc"]] / capacity).clamp(0.0, 1.0).unsqueeze(-1)
    next_chp = action[:, _I["p_chp"]].clamp_min(0.0).unsqueeze(-1)
    zeros = torch.zeros_like(next_soc)
    outcome = FormalV4RealizedOutcome(
        realized_dispatch=action, observable_dispatch=action[:, :17], activity_indicators=activity,
        p_dump=zeros[:, 0], recomputed_heat_surplus=zeros[:, 0], balance_residuals=torch.zeros((1, 3), dtype=action.dtype, device=action.device),
        conversion_residuals=torch.zeros((1, 5), dtype=action.dtype, device=action.device), next_soc=next_soc,
        next_previous_chp=next_chp, shortage=torch.zeros((1, 3), dtype=action.dtype, device=action.device),
        operating_cost=zeros[:, 0], physical_carbon=zeros[:, 0], penalized_objective=zeros[:, 0],
    )
    load = None if realized_load is None else torch.as_tensor(realized_load, dtype=state.load_history.dtype, device=state.load_history.device).reshape(1, 4)
    exog = None if realized_exog is None else torch.as_tensor(realized_exog, dtype=state.exog_history.dtype, device=state.exog_history.device).reshape(1, 12)
    return advance_formal_v4_state(state, outcome, realized_load=load, realized_exog=exog)


def settle_and_advance_v42(
    state: FormalV4ClosedLoopState,
    four_hour_plan: np.ndarray,
    realized_first_hour: Any,
    parameters: Mapping[str, Any],
) -> SettledStepV42:
    """Perform one canonical recourse projection and then advance the state."""

    plan = np.asarray(four_hour_plan, dtype=np.float64)
    if plan.shape != (4, len(VARIABLES)):
        raise ValueError("four_hour_plan must have shape [4,21]")
    demand, renewable, _, _ = _realized_fields(realized_first_hour)
    outcome = _settle_numpy(plan[0], demand, renewable, parameters, state)
    settled = outcome.realized_dispatch[0].detach().cpu().numpy()
    shortage = outcome.shortage[0].detach().cpu().numpy()
    residuals = calculate_all_residual_families(settled, state, realized_first_hour, parameters)
    next_state = advance_with_executed_first_hour(state, settled, realized_first_hour, parameters)
    return SettledStepV42(
        planned=plan[0].copy(), settled=settled, shortage=shortage,
        p_dump=float(outcome.p_dump[0].item()), q_dump=float(outcome.recomputed_heat_surplus[0].item()),
        residuals=residuals, next_state=next_state, executed_plan_index=0,
        operating_cost=float(outcome.operating_cost[0].item()), physical_carbon=float(outcome.physical_carbon[0].item()),
        penalized_objective=float(outcome.penalized_objective[0].item()),
    )


def _window_attr(window: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if isinstance(window, Mapping) and name in window:
            return window[name]
        if hasattr(window, name):
            return getattr(window, name)
    return default


def _method_plan(method: Any, window: Any, state: FormalV4ClosedLoopState) -> tuple[np.ndarray, np.ndarray | None]:
    result = method(window, state) if callable(method) else (method.predict(window, state) if hasattr(method, "predict") else method)
    forecast = None
    if isinstance(result, Mapping):
        forecast = result.get("forecast", result.get("prediction", result.get("predicted_forecast")))
        result = result.get("dispatch", result.get("plan", result.get("planned_dispatch")))
    elif hasattr(result, "dispatch"):
        forecast = getattr(result, "forecast", None)
        result = result.dispatch
    plan = np.asarray(result, dtype=np.float64)
    if plan.ndim == 3 and plan.shape[0] == 1:
        plan = plan[0]
    if plan.shape != (4, len(VARIABLES)):
        raise ValueError("chronological method must return a [4,21] dispatch plan")
    return plan, None if forecast is None else np.asarray(forecast, dtype=np.float64)


def evaluate_chronological_v42(
    method: Any,
    windows: Iterable[Any],
    *,
    initial_state: FormalV4ClosedLoopState,
    parameters: Mapping[str, Any] | None = None,
) -> ChronologicalRolloutV42:
    """Evaluate windows in chronological order with one first-step settlement."""

    state = initial_state
    forecast_targets: list[np.ndarray] = []
    forecasts: list[np.ndarray] = []
    plans: list[np.ndarray] = []
    settled: list[np.ndarray] = []
    shortages: list[np.ndarray] = []
    p_dumps: list[float] = []
    q_dumps: list[float] = []
    residuals: list[PhysicalResidualsV42] = []
    operating: list[float] = []
    carbon: list[float] = []
    objective: list[float] = []
    times: list[Any] = []
    states: list[FormalV4ClosedLoopState] = []
    method_id = str(getattr(method, "method_id", getattr(method, "__name__", "method")))
    for window in windows:
        plan, forecast = _method_plan(method, window, state)
        target = _window_attr(window, ("realized_target", "forecast_target", "target"))
        if target is None:
            raise ValueError("each rollout window must expose realized_target/forecast_target")
        target = np.asarray(target, dtype=np.float64)
        if target.shape == (1, 4, 4):
            target = target[0]
        if target.shape != (4, 4):
            raise ValueError("rollout target must have shape [4,4]")
        renewable = _window_attr(window, ("renewable_realized", "realized_renewable"))
        if renewable is None:
            raise ValueError("each rollout window must expose realized renewable output")
        renewable = np.asarray(renewable, dtype=np.float64)
        if renewable.shape == (1, 4, 2):
            renewable = renewable[0]
        if renewable.shape != (4, 2):
            raise ValueError("rollout renewable target must have shape [4,2]")
        context = parameters if parameters is not None else _window_attr(window, ("parameters",), None)
        if context is None:
            raise ValueError("chronological evaluation requires benchmark parameters")
        realized = {
            "demand": target[0, :3], "renewable": renewable[0],
            "realized_load": target[0], "realized_exog": _window_attr(window, ("realized_exog", "exog"), None),
        }
        result = settle_and_advance_v42(state, plan, realized, context)
        forecast_targets.append(target)
        forecasts.append(target if forecast is None else forecast.reshape(4, 4))
        plans.append(plan); settled.append(result.settled); shortages.append(result.shortage)
        p_dumps.append(result.p_dump); q_dumps.append(result.q_dump); residuals.append(result.residuals)
        operating.append(result.operating_cost); carbon.append(result.physical_carbon); objective.append(result.penalized_objective)
        times.append(_window_attr(window, ("target_time", "target_times"), state.origin))
        state = result.next_state; states.append(state)
    return ChronologicalRolloutV42(
        method_id=method_id, forecast_target=np.stack(forecast_targets), forecast_prediction=np.stack(forecasts),
        planned_dispatch=np.stack(plans), settled_dispatch=np.stack(settled), shortage_energy=np.stack(shortages),
        p_dump=np.asarray(p_dumps), q_dump=np.asarray(q_dumps), residuals=tuple(residuals),
        operating_cost=np.asarray(operating), physical_carbon=np.asarray(carbon), penalized_objective=np.asarray(objective),
        target_times=np.asarray(times, dtype="datetime64[ns]"), next_states=tuple(states),
    )


__all__ = [
    "ChronologicalRolloutV42", "PhysicalResidualsV42", "SettledStepV42",
    "advance_with_executed_first_hour", "calculate_all_residual_families",
    "canonical_recourse_once", "evaluate_chronological_v42", "settle_and_advance_v42",
]
