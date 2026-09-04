"""Differentiable first-hour recourse and closed-loop state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from ..scheduling.dispatch_schema import VARIABLES
from .data import STATUS_ORDER


_I = {name: index for index, name in enumerate(VARIABLES)}


def _param(parameters: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = float(parameters.get(name, default))
    if not torch.isfinite(torch.tensor(value)):
        raise ValueError(f"parameter {name} must be finite")
    return value


def _vector(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
    if not isinstance(value, Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True)
class RealizedFirstStepOutcome:
    realized_dispatch: Tensor
    shortage: Tensor
    surplus: Tensor
    operating_cost: Tensor
    carbon: Tensor
    penalized_objective: Tensor


@dataclass(frozen=True)
class ClosedLoopState:
    soc: Tensor
    previous_chp: Tensor
    device_history: Tensor
    device_status: Tensor
    load_history: Tensor | None = None
    exog_history: Tensor | None = None


@dataclass(frozen=True)
class ClosedLoopRollout:
    outcomes: tuple[RealizedFirstStepOutcome, ...]
    states: tuple[ClosedLoopState, ...]
    forecasts: tuple[Tensor, ...] = ()


def _status_from_dispatch(dispatch: Tensor, epsilon: float) -> Tensor:
    return torch.stack(
        [
            (dispatch[..., _I["p_chp"]] > epsilon),
            (dispatch[..., _I["q_gb"]] > epsilon),
            (dispatch[..., _I["q_ec"]] > epsilon),
            (dispatch[..., _I["q_ac"]] > epsilon),
            (dispatch[..., _I["p_charge"]] > epsilon),
            (dispatch[..., _I["p_discharge"]] > epsilon),
        ],
        dim=-1,
    ).to(dtype=dispatch.dtype)


def apply_first_step_recourse(
    planned_dispatch: Tensor,
    actual_demand: Tensor,
    actual_pv: Tensor,
    actual_wt: Tensor,
    parameters: Mapping[str, Any],
    *,
    grid_price: Tensor | None = None,
    gas_price: Tensor | None = None,
    carbon_price: Tensor | None = None,
) -> RealizedFirstStepOutcome:
    """Settle one hour without an LP, preserving planned slow controls.

    Electricity recourse first changes grid import, then PV use, then WT use.
    Remaining mismatch is reported as shortage/surplus; cooling and heat have
    no hidden optimization step.  The operations stay in the autograd graph.
    """

    if planned_dispatch.ndim != 2 or planned_dispatch.shape[-1] != len(VARIABLES):
        raise ValueError("planned_dispatch must have shape [B,21]")
    batch = planned_dispatch.shape[0]
    demand = _vector(actual_demand, (batch, 3), "actual_demand")
    pv = _vector(actual_pv, (batch,), "actual_pv")
    wt = _vector(actual_wt, (batch,), "actual_wt")
    if bool((demand < 0).any()) or bool((pv < 0).any()) or bool((wt < 0).any()):
        raise ValueError("actual demand and renewables must be non-negative")
    if not bool(torch.isfinite(planned_dispatch).all()):
        raise ValueError("planned_dispatch must be finite")
    # Compute all recourse quantities as new tensors, then assemble the
    # realized dispatch in one ``stack``.  Avoiding in-place writes here is
    # essential: the realized plan is part of the joint forecast-to-dispatch
    # autograd graph and every source tensor may be used by the loss.
    p_ec_plan = torch.clamp(planned_dispatch[:, _I["p_ec"]], min=0.0)
    p_charge_plan = torch.clamp(planned_dispatch[:, _I["p_charge"]], min=0.0)
    p_chp_plan = torch.clamp(planned_dispatch[:, _I["p_chp"]], min=0.0)
    p_discharge_plan = torch.clamp(planned_dispatch[:, _I["p_discharge"]], min=0.0)
    pv_use_plan = torch.minimum(torch.clamp(planned_dispatch[:, _I["pv_use"]], min=0.0), pv)
    wt_use_plan = torch.minimum(torch.clamp(planned_dispatch[:, _I["wt_use"]], min=0.0), wt)
    grid_plan = torch.clamp(planned_dispatch[:, _I["grid"]], min=0.0, max=_param(parameters, "grid_import_capacity"))
    electric_balance = demand[:, 0] + p_ec_plan + p_charge_plan - p_chp_plan - p_discharge_plan - grid_plan - pv_use_plan - wt_use_plan
    deficit = torch.clamp(electric_balance, min=0.0)
    grid_capacity = planned_dispatch.new_tensor(_param(parameters, "grid_import_capacity"))
    extra_grid = torch.minimum(deficit, torch.clamp(grid_capacity - grid_plan, min=0.0))
    grid_after_deficit = grid_plan + extra_grid
    electric_shortage = torch.clamp(deficit - extra_grid, min=0.0)
    surplus = torch.clamp(-electric_balance, min=0.0)
    reduce_grid = torch.minimum(surplus, grid_after_deficit)
    grid_after_surplus = grid_after_deficit - reduce_grid
    remaining_surplus = surplus - reduce_grid
    reduce_pv = torch.minimum(remaining_surplus, pv_use_plan)
    pv_after_first = pv_use_plan - reduce_pv
    remaining_surplus = remaining_surplus - reduce_pv
    reduce_wt = torch.minimum(remaining_surplus, wt_use_plan)
    wt_after_first = wt_use_plan - reduce_wt
    electric_surplus = torch.clamp(remaining_surplus - reduce_wt, min=0.0)

    q_ec_plan = torch.clamp(planned_dispatch[:, _I["q_ec"]], min=0.0)
    q_ac_plan = torch.clamp(planned_dispatch[:, _I["q_ac"]], min=0.0)
    q_cooling_plan = q_ec_plan + q_ac_plan
    # Cooling production can be curtailed when realized cooling demand is
    # lower than the forecast plan.  Keep conversion identities intact by
    # curtailing the corresponding electric/thermal inputs as well.
    cooling_scale = torch.minimum(torch.ones_like(q_cooling_plan), demand[:, 1] / q_cooling_plan.clamp_min(1.0e-8))
    q_ec = q_ec_plan * cooling_scale
    q_ac = q_ac_plan * cooling_scale
    # The lightweight rollout tests intentionally omit COP parameters.  In
    # that compatibility mode, retain the legacy zero input convention;
    # benchmark parameter bundles always provide positive COP values.
    if "electric_chiller_cop" in parameters:
        p_ec = q_ec / max(_param(parameters, "electric_chiller_cop"), 1.0e-8)
    else:
        p_ec = torch.zeros_like(q_ec)
    if "absorption_chiller_cop" in parameters:
        q_ac_in = q_ac / max(_param(parameters, "absorption_chiller_cop"), 1.0e-8)
    else:
        q_ac_in = torch.zeros_like(q_ac)
    q_cooling = q_ec + q_ac
    cooling_error = demand[:, 1] - q_cooling
    cooling_shortage = torch.clamp(cooling_error, min=0.0)
    cooling_surplus = torch.clamp(-cooling_error, min=0.0)
    q_chp = torch.clamp(planned_dispatch[:, _I["q_chp"]], min=0.0)
    q_gb = torch.clamp(planned_dispatch[:, _I["q_gb"]], min=0.0)
    q_heat = q_chp + q_gb - torch.clamp(q_ac_in, min=0.0)
    heat_error = demand[:, 2] - q_heat
    heat_shortage = torch.clamp(heat_error, min=0.0)
    heat_surplus = torch.clamp(-heat_error, min=0.0)
    q_dump = torch.clamp(planned_dispatch[:, _I["q_dump"]], min=0.0) + heat_surplus

    # Reconcile electricity once more after cooling curtailment.  The first
    # pass may have used electric-chiller consumption to absorb a forecast
    # surplus; after curtailment, reduce sources in the same order.
    electric_balance = demand[:, 0] + p_ec + p_charge_plan - p_chp_plan - p_discharge_plan - grid_after_surplus - pv_after_first - wt_after_first
    final_deficit = torch.clamp(electric_balance, min=0.0)
    final_extra_grid = torch.minimum(final_deficit, torch.clamp(grid_capacity - grid_after_surplus, min=0.0))
    grid_after_deficit_final = grid_after_surplus + final_extra_grid
    electric_shortage = torch.clamp(final_deficit - final_extra_grid, min=0.0)
    final_surplus = torch.clamp(-electric_balance, min=0.0)
    reduce_grid = torch.minimum(final_surplus, grid_after_deficit_final)
    grid_after = grid_after_deficit_final - reduce_grid
    remaining = final_surplus - reduce_grid
    reduce_pv = torch.minimum(remaining, pv_after_first)
    pv_after = pv_after_first - reduce_pv
    remaining = remaining - reduce_pv
    reduce_wt = torch.minimum(remaining, wt_after_first)
    wt_after = wt_after_first - reduce_wt
    electric_surplus = torch.clamp(remaining - reduce_wt, min=0.0)
    realized_columns = [planned_dispatch[:, index] for index in range(len(VARIABLES))]
    replacements = {
        "grid": grid_after,
        "pv_use": pv_after,
        "pv_curt": torch.clamp(pv - pv_after, min=0.0),
        "wt_use": wt_after,
        "wt_curt": torch.clamp(wt - wt_after, min=0.0),
        "p_ec": p_ec,
        "q_ec": q_ec,
        "q_ac_in": q_ac_in,
        "q_ac": q_ac,
        "slack_e": electric_shortage,
        "slack_c": cooling_shortage,
        "slack_h": heat_shortage,
        "q_dump": q_dump,
    }
    for name, value in replacements.items():
        realized_columns[_I[name]] = value
    realized = torch.stack(realized_columns, dim=-1)
    shortage = torch.stack((electric_shortage, cooling_shortage, heat_shortage), dim=-1)
    surplus_vector = torch.stack((electric_surplus, cooling_surplus, heat_surplus), dim=-1)
    if grid_price is None:
        grid_price = planned_dispatch.new_tensor(_param(parameters, "grid_energy_price"))
    if gas_price is None:
        gas_price = planned_dispatch.new_tensor(_param(parameters, "gas_energy_price"))
    if carbon_price is None:
        raw_carbon_price = parameters.get("carbon_price", parameters.get("carbon_price_default", 0.0))
        carbon_price = planned_dispatch.new_tensor(float(raw_carbon_price))
    grid_price = grid_price.to(dtype=planned_dispatch.dtype)
    gas_price = gas_price.to(dtype=planned_dispatch.dtype)
    carbon_price = carbon_price.to(dtype=planned_dispatch.dtype)
    gas = torch.clamp(realized[:, _I["g_chp"]], min=0.0) + torch.clamp(realized[:, _I["g_gb"]], min=0.0)
    operating = (
        grid_after * grid_price
        + gas * gas_price
        + _param(parameters, "bess_throughput_cost") * (
            torch.clamp(realized[:, _I["p_charge"]], min=0.0) + torch.clamp(realized[:, _I["p_discharge"]], min=0.0)
        )
    )
    carbon = grid_after * _param(parameters, "grid_emission_factor") + gas * _param(parameters, "gas_emission_factor")
    surplus_penalty = _param(parameters, "surplus_penalty", 0.0) * surplus_vector.sum(dim=-1)
    penalized = operating + carbon_price * carbon + _param(parameters, "unserved_penalty") * shortage.sum(dim=-1) + surplus_penalty
    return RealizedFirstStepOutcome(realized, shortage, surplus_vector, operating, carbon, penalized)


def advance_closed_loop_state(
    previous: ClosedLoopState,
    outcome: RealizedFirstStepOutcome,
    *,
    status_epsilon: float = 1e-6,
    bess_energy_capacity: float = 1.0,
) -> ClosedLoopState:
    if previous.device_history.ndim != 3 or previous.device_history.shape[-1] != len(VARIABLES) or previous.device_history.shape[1] != 24:
        raise ValueError("device_history must have shape [B,24,21]")
    if previous.device_status.shape != (previous.device_history.shape[0], 24, len(STATUS_ORDER)):
        raise ValueError("device_status must have shape [B,24,6]")
    dispatch = outcome.realized_dispatch
    if dispatch.shape != (previous.device_history.shape[0], len(VARIABLES)):
        raise ValueError("outcome dispatch shape does not match state")
    if bess_energy_capacity <= 0.0:
        raise ValueError("bess_energy_capacity must be positive")
    history_dispatch = dispatch.to(dtype=previous.device_history.dtype)
    status = _status_from_dispatch(dispatch, status_epsilon).to(dtype=previous.device_status.dtype)
    next_history = torch.cat((previous.device_history[:, 1:, :], history_dispatch.unsqueeze(1)), dim=1)
    next_status = torch.cat((previous.device_status[:, 1:, :], status.unsqueeze(1)), dim=1)
    next_soc = (dispatch[:, _I["soc"]] / float(bess_energy_capacity)).clamp(0.0, 1.0).to(dtype=previous.soc.dtype).unsqueeze(-1)
    next_chp = dispatch[:, _I["p_chp"]].to(dtype=previous.previous_chp.dtype).unsqueeze(-1)
    return ClosedLoopState(next_soc, next_chp, next_history, next_status, previous.load_history, previous.exog_history)


def rollout_joint_policy(
    model: Any,
    hourly_inputs: Mapping[str, Tensor],
    initial_state: ClosedLoopState,
    parameters: Mapping[str, Any],
    *,
    reveal_realized_load_after_action: bool = True,
    retain_states: bool = True,
    state_callback: Callable[[ClosedLoopState], None] | None = None,
) -> ClosedLoopRollout:
    """Run a model-generated closed loop with zero LP calls."""

    loads = hourly_inputs["loads"]
    # ``loads`` are the realized physical demands used by recourse.  During
    # training/evaluation the model's history buffer may be normalized, so a
    # separate causal representation can be supplied for the post-action
    # history update.  Keeping these two streams explicit prevents mixing raw
    # and normalized values after the first rollout step.
    loads_for_history = hourly_inputs.get("loads_for_history", loads)
    device_history_mean = hourly_inputs.get("device_history_mean")
    device_history_scale = hourly_inputs.get("device_history_scale")
    exog = hourly_inputs["exog"]
    pv = hourly_inputs["pv_available"]
    wt = hourly_inputs["wt_available"]
    for name, value in (("loads", loads), ("exog", exog)):
        if not isinstance(value, Tensor) or value.ndim < 2:
            raise ValueError(f"hourly_inputs[{name}] must be a tensor with a feature dimension")
    if not isinstance(loads_for_history, Tensor) or loads_for_history.shape != loads.shape:
        raise ValueError("hourly_inputs[loads_for_history] must match loads shape")
    if (device_history_mean is None) != (device_history_scale is None):
        raise ValueError("device_history_mean and device_history_scale must be supplied together")
    if device_history_mean is not None:
        state_dtype = initial_state.device_history.dtype
        device_history_mean = torch.as_tensor(device_history_mean, dtype=state_dtype, device=initial_state.device_history.device)
        device_history_scale = torch.as_tensor(device_history_scale, dtype=state_dtype, device=initial_state.device_history.device)
        if device_history_mean.shape != (len(VARIABLES),) or device_history_scale.shape != (len(VARIABLES),):
            raise ValueError("device history normalization vectors must have shape [21]")
        if bool((device_history_scale <= 0).any()) or not bool(torch.isfinite(device_history_mean).all() and torch.isfinite(device_history_scale).all()):
            raise ValueError("device history normalization vectors must be finite and positive")
    for name, value in (("pv_available", pv), ("wt_available", wt)):
        if not isinstance(value, Tensor) or value.ndim != 1:
            raise ValueError(f"hourly_inputs[{name}] must be a one-dimensional tensor")
    total = loads.shape[0]
    if loads.shape[-1] != 4 or exog.shape[0] != total or pv.shape[0] != total or wt.shape[0] != total:
        raise ValueError("hourly input lengths/tasks are inconsistent")
    if initial_state.device_history.shape[1] != 24:
        raise ValueError("initial state must contain a 24-hour history")
    state = initial_state
    outcomes: list[RealizedFirstStepOutcome] = []
    states: list[ClosedLoopState] = [state] if retain_states else []
    forecasts: list[Tensor] = []
    for current in range(total):
        if current + 4 > total:
            break
        if state_callback is not None:
            state_callback(state)
        soc_column = state.soc.reshape(-1)[0].expand(4)
        context = torch.stack(
            (
                pv[current : current + 4], wt[current : current + 4],
                hourly_inputs.get("grid_price", torch.zeros_like(pv))[current : current + 4],
                hourly_inputs.get("gas_price", torch.zeros_like(pv))[current : current + 4],
                hourly_inputs.get("carbon_price", torch.zeros_like(pv))[current : current + 4],
                soc_column,
            ), dim=-1,
        ).unsqueeze(0)
        # The rollout is intentionally batch-one; callers can wrap separate
        # timelines in a DataLoader without leaking a future realized load.
        load_history = state.load_history if state.load_history is not None else hourly_inputs["load_history"]
        exog_history = state.exog_history if state.exog_history is not None else hourly_inputs["exog_history"]
        model_output = model(
            load_history=load_history,
            exog_history=exog_history,
            device_history=state.device_history,
            device_status=state.device_status,
            scheduler_context=context,
            previous_chp=state.previous_chp,
        )
        forecasts.append(model_output.forecast_physical.detach())
        actual = loads[current].reshape(1, 4)
        outcome = apply_first_step_recourse(
            model_output.dispatch[:, 0, :], actual[:, :3], pv[current].reshape(1), wt[current].reshape(1), parameters,
            grid_price=context[:, 0, 2], gas_price=context[:, 0, 3], carbon_price=context[:, 0, 4],
        )
        outcomes.append(outcome)
        state = advance_closed_loop_state(
            state, outcome, bess_energy_capacity=_param(parameters, "bess_energy_capacity", 1.0)
        )
        if device_history_mean is not None:
            # The model consumes normalized device histories.  The shared
            # state transition is intentionally raw, so normalize only the
            # newly revealed row before the next model call.
            normalized_dispatch = ((outcome.realized_dispatch - device_history_mean) / device_history_scale).to(dtype=state.device_history.dtype)
            next_device_history = torch.cat((state.device_history[:, 1:, :], normalized_dispatch.unsqueeze(1)), dim=1)
            state = ClosedLoopState(
                state.soc, state.previous_chp, next_device_history,
                state.device_status, state.load_history, state.exog_history,
            )
        # The realized current load is revealed only after action.  Keep the
        # 24-hour buffers explicit so the next model call cannot see future
        # realized demand.
        if state.load_history is not None:
            # The forecaster consumes the four-task history (including the
            # station-side gas task); recourse itself uses only the first three
            # physical demands above.
            observed_load = loads_for_history[current].reshape(1, 4).to(dtype=state.load_history.dtype)
            next_load = torch.cat((state.load_history[:, 1:, :], observed_load.unsqueeze(1)), dim=1)
            state = ClosedLoopState(state.soc, state.previous_chp, state.device_history, state.device_status, next_load, state.exog_history)
        if state.exog_history is not None and "exog" in hourly_inputs:
            exog_now = hourly_inputs["exog"][current].reshape(1, -1)
            next_exog = torch.cat((state.exog_history[:, 1:, :], exog_now.unsqueeze(1)), dim=1)
            state = ClosedLoopState(state.soc, state.previous_chp, state.device_history, state.device_status, state.load_history, next_exog)
        if retain_states:
            states.append(state)
        if not reveal_realized_load_after_action:
            raise ValueError("rollout requires reveal_realized_load_after_action=True")
    return ClosedLoopRollout(tuple(outcomes), tuple(states), tuple(forecasts))


__all__ = [
    "ClosedLoopRollout", "ClosedLoopState", "RealizedFirstStepOutcome",
    "advance_closed_loop_state", "apply_first_step_recourse", "rollout_joint_policy",
]
