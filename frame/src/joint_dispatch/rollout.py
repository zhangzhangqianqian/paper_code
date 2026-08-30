"""Differentiable first-hour recourse and closed-loop state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

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
    realized = planned_dispatch.clone()
    # Bound planned renewable use by what is available in the realized hour.
    pv_use = torch.minimum(torch.clamp(planned_dispatch[:, _I["pv_use"]], min=0.0), pv)
    wt_use = torch.minimum(torch.clamp(planned_dispatch[:, _I["wt_use"]], min=0.0), wt)
    grid = torch.clamp(planned_dispatch[:, _I["grid"]], min=0.0, max=_param(parameters, "grid_import_capacity"))
    electric_balance = (
        demand[:, 0]
        + torch.clamp(planned_dispatch[:, _I["p_ec"]], min=0.0)
        + torch.clamp(planned_dispatch[:, _I["p_charge"]], min=0.0)
        - torch.clamp(planned_dispatch[:, _I["p_chp"]], min=0.0)
        - torch.clamp(planned_dispatch[:, _I["p_discharge"]], min=0.0)
        - grid
        - pv_use
        - wt_use
    )
    deficit = torch.clamp(electric_balance, min=0.0)
    extra_grid = torch.minimum(deficit, planned_dispatch.new_tensor(_param(parameters, "grid_import_capacity")) - grid)
    grid = grid + torch.clamp(extra_grid, min=0.0)
    electric_shortage = torch.clamp(deficit - torch.clamp(extra_grid, min=0.0), min=0.0)
    surplus = torch.clamp(-electric_balance, min=0.0)
    reduce_grid = torch.minimum(surplus, grid)
    grid = grid - reduce_grid
    remaining_surplus = surplus - reduce_grid
    reduce_pv = torch.minimum(remaining_surplus, pv_use)
    pv_use = pv_use - reduce_pv
    remaining_surplus = remaining_surplus - reduce_pv
    reduce_wt = torch.minimum(remaining_surplus, wt_use)
    wt_use = wt_use - reduce_wt
    electric_surplus = torch.clamp(remaining_surplus - reduce_wt, min=0.0)
    realized[:, _I["grid"]] = grid
    realized[:, _I["pv_use"]] = pv_use
    realized[:, _I["wt_use"]] = wt_use
    realized[:, _I["pv_curt"]] = torch.clamp(pv - pv_use, min=0.0)
    realized[:, _I["wt_curt"]] = torch.clamp(wt - wt_use, min=0.0)
    realized[:, _I["slack_e"]] = electric_shortage
    q_cooling = torch.clamp(planned_dispatch[:, _I["q_ec"]], min=0.0) + torch.clamp(planned_dispatch[:, _I["q_ac"]], min=0.0)
    cooling_error = demand[:, 1] - q_cooling
    cooling_shortage = torch.clamp(cooling_error, min=0.0)
    cooling_surplus = torch.clamp(-cooling_error, min=0.0)
    realized[:, _I["slack_c"]] = cooling_shortage
    q_heat = torch.clamp(planned_dispatch[:, _I["q_chp"]], min=0.0) + torch.clamp(planned_dispatch[:, _I["q_gb"]], min=0.0) - torch.clamp(planned_dispatch[:, _I["q_ac_in"]], min=0.0)
    heat_error = demand[:, 2] - q_heat
    heat_shortage = torch.clamp(heat_error, min=0.0)
    heat_surplus = torch.clamp(-heat_error, min=0.0)
    realized[:, _I["slack_h"]] = heat_shortage
    realized[:, _I["q_dump"]] = torch.clamp(planned_dispatch[:, _I["q_dump"]], min=0.0) + heat_surplus
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
        grid * grid_price
        + gas * gas_price
        + _param(parameters, "bess_throughput_cost") * (
            torch.clamp(realized[:, _I["p_charge"]], min=0.0) + torch.clamp(realized[:, _I["p_discharge"]], min=0.0)
        )
    )
    carbon = grid * _param(parameters, "grid_emission_factor") + gas * _param(parameters, "gas_emission_factor")
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
    next_history = torch.cat((previous.device_history[:, 1:, :], dispatch.unsqueeze(1)), dim=1)
    next_status = torch.cat((previous.device_status[:, 1:, :], _status_from_dispatch(dispatch, status_epsilon).unsqueeze(1)), dim=1)
    next_soc = (dispatch[:, _I["soc"]] / float(bess_energy_capacity)).clamp(0.0, 1.0).unsqueeze(-1)
    next_chp = dispatch[:, _I["p_chp"]].unsqueeze(-1)
    return ClosedLoopState(next_soc, next_chp, next_history, next_status, previous.load_history, previous.exog_history)


def rollout_joint_policy(
    model: Any,
    hourly_inputs: Mapping[str, Tensor],
    initial_state: ClosedLoopState,
    parameters: Mapping[str, Any],
    *,
    reveal_realized_load_after_action: bool = True,
) -> ClosedLoopRollout:
    """Run a model-generated closed loop with zero LP calls."""

    loads = hourly_inputs["loads"]
    exog = hourly_inputs["exog"]
    pv = hourly_inputs["pv_available"]
    wt = hourly_inputs["wt_available"]
    for name, value in (("loads", loads), ("exog", exog), ("pv_available", pv), ("wt_available", wt)):
        if not isinstance(value, Tensor) or value.ndim < 2:
            raise ValueError(f"hourly_inputs[{name}] must be a tensor")
    total = loads.shape[0]
    if loads.shape[-1] != 4 or exog.shape[0] != total or pv.shape[0] != total or wt.shape[0] != total:
        raise ValueError("hourly input lengths/tasks are inconsistent")
    if initial_state.device_history.shape[1] != 24:
        raise ValueError("initial state must contain a 24-hour history")
    state = initial_state
    outcomes: list[RealizedFirstStepOutcome] = []
    states: list[ClosedLoopState] = [state]
    for current in range(total):
        if current + 4 > total:
            break
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
        actual = loads[current].reshape(1, 4)
        outcome = apply_first_step_recourse(
            model_output.dispatch[:, 0, :], actual[:, :3], pv[current].reshape(1), wt[current].reshape(1), parameters,
            grid_price=context[:, 0, 2], gas_price=context[:, 0, 3], carbon_price=context[:, 0, 4],
        )
        outcomes.append(outcome)
        state = advance_closed_loop_state(
            state, outcome, bess_energy_capacity=_param(parameters, "bess_energy_capacity", 1.0)
        )
        # The realized current load is revealed only after action.  Keep the
        # 24-hour buffers explicit so the next model call cannot see future
        # realized demand.
        if state.load_history is not None:
            next_load = torch.cat((state.load_history[:, 1:, :], actual[:, :3].unsqueeze(1)), dim=1)
            state = ClosedLoopState(state.soc, state.previous_chp, state.device_history, state.device_status, next_load, state.exog_history)
        if state.exog_history is not None and "exog" in hourly_inputs:
            exog_now = hourly_inputs["exog"][current].reshape(1, -1)
            next_exog = torch.cat((state.exog_history[:, 1:, :], exog_now.unsqueeze(1)), dim=1)
            state = ClosedLoopState(state.soc, state.previous_chp, state.device_history, state.device_status, state.load_history, next_exog)
        states.append(state)
        if not reveal_realized_load_after_action:
            raise ValueError("rollout requires reveal_realized_load_after_action=True")
    return ClosedLoopRollout(tuple(outcomes), tuple(states))


__all__ = [
    "ClosedLoopRollout", "ClosedLoopState", "RealizedFirstStepOutcome",
    "advance_closed_loop_state", "apply_first_step_recourse", "rollout_joint_policy",
]
