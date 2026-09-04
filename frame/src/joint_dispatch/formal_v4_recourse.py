"""One-pass differentiable realized settlement for formal-v4.

The learned contract remains 21 dispatch columns.  Electric curtailment is an
evaluation/settlement quantity (``p_dump``), not a new learned output, while
heat dump is recomputed exactly once from realized heat balance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from ..scheduling.dispatch_schema import VARIABLES


_I = {name: index for index, name in enumerate(VARIABLES)}
_STATUS_FROM = ("p_chp", "q_gb", "q_ec", "q_ac", "p_charge", "p_discharge")


def _param(parameters: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = float(parameters.get(name, default))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _column(value: Tensor | None, batch: int, name: str, default: float) -> Tensor:
    if value is None:
        return torch.full((batch, 1), default, dtype=torch.float64)
    if value.ndim == 1 and value.shape[0] == batch:
        value = value.unsqueeze(-1)
    if value.shape != (batch, 1) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must have shape [B] or [B,1] and be finite")
    return value


def _price(parameters: Mapping[str, Any], name: str, reference: Tensor, override: Tensor | None, default: float = 0.0) -> Tensor:
    value = override if override is not None else parameters.get(name, default)
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim == 0:
        return result.expand(reference.shape[0])
    if result.shape == reference.shape:
        return result
    raise ValueError(f"{name} must be scalar or [B]")


@dataclass(frozen=True)
class FormalV4RealizedOutcome:
    realized_dispatch: Tensor
    observable_dispatch: Tensor
    activity_indicators: Tensor
    p_dump: Tensor
    recomputed_heat_surplus: Tensor
    balance_residuals: Tensor
    conversion_residuals: Tensor
    next_soc: Tensor
    next_previous_chp: Tensor
    shortage: Tensor
    operating_cost: Tensor
    physical_carbon: Tensor
    penalized_objective: Tensor


def settle_first_step_v4(
    planned_action: Tensor,
    demand: Tensor,
    renewable: Tensor,
    parameters: Mapping[str, Any],
    *,
    initial_soc: Tensor | None = None,
    previous_chp: Tensor | None = None,
    grid_price: Tensor | None = None,
    gas_price: Tensor | None = None,
    carbon_price: Tensor | None = None,
) -> FormalV4RealizedOutcome:
    """Project one planned action onto the realized, balanced physical state."""

    if planned_action.ndim != 2 or planned_action.shape[-1] != len(VARIABLES):
        raise ValueError(f"planned_action must have shape [B,{len(VARIABLES)}]")
    batch = planned_action.shape[0]
    if demand.shape != (batch, 3) or renewable.shape != (batch, 2):
        raise ValueError("demand and renewable must have shapes [B,3] and [B,2]")
    if not bool(torch.isfinite(planned_action).all() and torch.isfinite(demand).all() and torch.isfinite(renewable).all()):
        raise ValueError("planned action, demand and renewable must be finite")
    if bool((demand < 0.0).any()) or bool((renewable < 0.0).any()):
        raise ValueError("demand and renewable must be non-negative")

    soc0 = _column(initial_soc, batch, "initial_soc", 0.5).to(dtype=planned_action.dtype, device=planned_action.device)
    prev_chp = _column(previous_chp, batch, "previous_chp", 0.0).to(dtype=planned_action.dtype, device=planned_action.device)
    pmax = _param(parameters, "chp_electric_capacity")
    ramp = _param(parameters, "chp_ramp_fraction") * pmax
    eta_e = _param(parameters, "chp_electric_efficiency")
    eta_h = _param(parameters, "chp_heat_efficiency")
    eta_gb = _param(parameters, "gas_boiler_efficiency")
    cop_ec = _param(parameters, "electric_chiller_cop")
    cop_ac = _param(parameters, "absorption_chiller_cop")
    grid_cap = _param(parameters, "grid_import_capacity")
    boiler_cap = _param(parameters, "gas_boiler_capacity")
    ec_cap = _param(parameters, "electric_chiller_capacity")
    ac_cap = _param(parameters, "absorption_chiller_capacity")
    bess_power = _param(parameters, "bess_power_capacity")
    bess_energy = _param(parameters, "bess_energy_capacity")
    eta_b = _param(parameters, "bess_roundtrip_efficiency") ** 0.5

    # Physics projection: CHP output is ramp-limited, and gas/heat columns are
    # derived from the projected electric output to enforce conversion exactly.
    p_chp_plan = planned_action[:, _I["p_chp"]].clamp(0.0, pmax)
    lower = (prev_chp[:, 0] - ramp).clamp_min(0.0)
    upper = torch.minimum(prev_chp[:, 0] + ramp, planned_action.new_tensor(pmax))
    p_chp = p_chp_plan.clamp_min(0.0).minimum(upper).maximum(lower)
    g_chp = p_chp / eta_e
    q_chp = p_chp * eta_h / eta_e

    q_gb = planned_action[:, _I["q_gb"]].clamp(0.0, boiler_cap)
    g_gb = q_gb / eta_gb

    # Curtail cooling once, preserving COP identities.
    q_ec_plan = planned_action[:, _I["q_ec"]].clamp(0.0, ec_cap)
    q_ac_plan = planned_action[:, _I["q_ac"]].clamp(0.0, ac_cap)
    cooling_total = q_ec_plan + q_ac_plan
    cooling_scale = torch.minimum(torch.ones_like(cooling_total), demand[:, 1] / cooling_total.clamp_min(1.0e-12))
    q_ec = q_ec_plan * cooling_scale
    q_ac = q_ac_plan * cooling_scale
    p_ec = q_ec / cop_ec
    q_ac_in = q_ac / cop_ac
    slack_c = (demand[:, 1] - q_ec - q_ac).clamp_min(0.0)

    # Project battery controls to one direction and then to the reachable SOC.
    raw_charge = planned_action[:, _I["p_charge"]].clamp(0.0, bess_power)
    raw_discharge = planned_action[:, _I["p_discharge"]].clamp(0.0, bess_power)
    net = raw_charge - raw_discharge
    prev_energy = (soc0[:, 0] * bess_energy).clamp(0.0, bess_energy)
    charge_room = ((bess_energy - prev_energy) / max(eta_b, 1.0e-12)).clamp_min(0.0)
    discharge_room = (prev_energy * eta_b).clamp_min(0.0)
    p_charge = torch.minimum(net.clamp_min(0.0), charge_room)
    p_discharge = torch.minimum((-net).clamp_min(0.0), discharge_room)
    next_energy = (prev_energy + eta_b * p_charge - p_discharge / max(eta_b, 1.0e-12)).clamp(0.0, bess_energy)

    # Renewable and grid recourse.  Remaining electric surplus is explicit
    # ``p_dump``; it is not hidden in a second slack or duplicate curtailment.
    pv_avail, wt_avail = renewable[:, 0], renewable[:, 1]
    pv_use_plan = planned_action[:, _I["pv_use"]].clamp_min(0.0).minimum(pv_avail)
    wt_use_plan = planned_action[:, _I["wt_use"]].clamp_min(0.0).minimum(wt_avail)
    grid_plan = planned_action[:, _I["grid"]].clamp(0.0, grid_cap)
    electric_balance = demand[:, 0] + p_ec + p_charge - p_chp - p_discharge - grid_plan - pv_use_plan - wt_use_plan
    deficit = electric_balance.clamp_min(0.0)
    extra_grid = torch.minimum(deficit, (grid_cap - grid_plan).clamp_min(0.0))
    grid_after_deficit = grid_plan + extra_grid
    slack_e = (deficit - extra_grid).clamp_min(0.0)
    surplus = (-electric_balance).clamp_min(0.0)
    reduce_grid = torch.minimum(surplus, grid_after_deficit)
    grid_after = grid_after_deficit - reduce_grid
    remaining = surplus - reduce_grid
    reduce_pv = torch.minimum(remaining, pv_use_plan)
    pv_after = pv_use_plan - reduce_pv
    remaining = remaining - reduce_pv
    reduce_wt = torch.minimum(remaining, wt_use_plan)
    wt_after = wt_use_plan - reduce_wt
    p_dump = (remaining - reduce_wt).clamp_min(0.0)

    # Heat balance is recomputed exactly once from realized outputs.
    heat_output = q_chp + q_gb - q_ac_in
    heat_surplus = (heat_output - demand[:, 2]).clamp_min(0.0)
    slack_h = (demand[:, 2] - heat_output).clamp_min(0.0)
    q_dump = heat_surplus

    values = [planned_action[:, index] for index in range(len(VARIABLES))]
    replacements = {
        "grid": grid_after, "pv_use": pv_after, "pv_curt": pv_avail - pv_after,
        "wt_use": wt_after, "wt_curt": wt_avail - wt_after,
        "g_chp": g_chp, "g_gb": g_gb, "p_chp": p_chp, "q_chp": q_chp,
        "q_gb": q_gb, "p_ec": p_ec, "q_ec": q_ec, "q_ac_in": q_ac_in,
        "q_ac": q_ac, "p_charge": p_charge, "p_discharge": p_discharge,
        "soc": next_energy, "slack_e": slack_e, "slack_c": slack_c,
        "slack_h": slack_h, "q_dump": q_dump,
    }
    for name, value in replacements.items():
        values[_I[name]] = value
    realized = torch.stack(values, dim=-1)
    activity = torch.stack([(realized[:, _I[name]] > 1.0e-6).to(realized.dtype) for name in _STATUS_FROM], dim=-1)

    balance = torch.stack((
        grid_after + pv_after + wt_after + p_chp + p_discharge + slack_e - p_ec - p_charge - demand[:, 0] - p_dump,
        q_ec + q_ac + slack_c - demand[:, 1],
        q_chp + q_gb + slack_h - q_ac_in - q_dump - demand[:, 2],
    ), dim=-1)
    conversion = torch.stack((
        p_chp - eta_e * g_chp, q_chp - eta_h * g_chp, q_gb - eta_gb * g_gb,
        q_ec - cop_ec * p_ec, q_ac - cop_ac * q_ac_in,
    ), dim=-1)
    shortage = torch.stack((slack_e, slack_c, slack_h), dim=-1)
    grid_p = _price(parameters, "grid_energy_price", grid_after, grid_price)
    gas_p = _price(parameters, "gas_energy_price", grid_after, gas_price)
    carbon_p = _price(parameters, "carbon_price", grid_after, carbon_price if carbon_price is not None else torch.full_like(grid_after, _param(parameters, "carbon_price_default", 0.0)))
    gas = g_chp + g_gb
    operating = grid_after * grid_p + gas * gas_p + _param(parameters, "bess_throughput_cost") * (p_charge + p_discharge)
    physical_carbon = grid_after * _param(parameters, "grid_emission_factor") + gas * _param(parameters, "gas_emission_factor")
    surplus_penalty = _param(parameters, "surplus_penalty", 0.0) * (p_dump + q_dump)
    penalized = operating + carbon_p * physical_carbon + _param(parameters, "unserved_penalty") * shortage.sum(dim=-1) + surplus_penalty
    return FormalV4RealizedOutcome(
        realized_dispatch=realized, observable_dispatch=realized[..., :17].clone(), activity_indicators=activity,
        p_dump=p_dump, recomputed_heat_surplus=heat_surplus, balance_residuals=balance,
        conversion_residuals=conversion, next_soc=(next_energy / bess_energy).unsqueeze(-1),
        next_previous_chp=p_chp.unsqueeze(-1), shortage=shortage, operating_cost=operating,
        physical_carbon=physical_carbon, penalized_objective=penalized,
    )


def settle_first_step_v4_numpy(
    planned_action: np.ndarray,
    demand: np.ndarray,
    renewable: np.ndarray,
    parameters: Mapping[str, Any],
    *,
    initial_soc: float = 0.5,
    previous_chp: float = 0.0,
) -> dict[str, np.ndarray]:
    """NumPy adapter for the canonical one-step settlement.

    This helper is intentionally a thin boundary adapter: all projection,
    balance and conversion logic remains in :func:`settle_first_step_v4`.
    """

    action = torch.as_tensor(planned_action, dtype=torch.float64)
    if action.ndim == 1:
        action = action.unsqueeze(0)
    demand_tensor = torch.as_tensor(demand, dtype=torch.float64)
    if demand_tensor.ndim == 1:
        demand_tensor = demand_tensor.unsqueeze(0)
    renewable_tensor = torch.as_tensor(renewable, dtype=torch.float64)
    if renewable_tensor.ndim == 1:
        renewable_tensor = renewable_tensor.unsqueeze(0)
    with torch.no_grad():
        outcome = settle_first_step_v4(
            action, demand_tensor, renewable_tensor, parameters,
            initial_soc=torch.as_tensor([[initial_soc]], dtype=torch.float64),
            previous_chp=torch.as_tensor([[previous_chp]], dtype=torch.float64),
        )
    return {
        "realized_dispatch": outcome.realized_dispatch.detach().cpu().numpy(),
        "observable_dispatch": outcome.observable_dispatch.detach().cpu().numpy(),
        "activity_indicators": outcome.activity_indicators.detach().cpu().numpy(),
        "balance_residuals": outcome.balance_residuals.detach().cpu().numpy(),
        "conversion_residuals": outcome.conversion_residuals.detach().cpu().numpy(),
        "next_soc": outcome.next_soc.detach().cpu().numpy(),
        "next_previous_chp": outcome.next_previous_chp.detach().cpu().numpy(),
        "shortage": outcome.shortage.detach().cpu().numpy(),
        "operating_cost": outcome.operating_cost.detach().cpu().numpy(),
        "physical_carbon": outcome.physical_carbon.detach().cpu().numpy(),
        "penalized_objective": outcome.penalized_objective.detach().cpu().numpy(),
    }


__all__ = ["FormalV4RealizedOutcome", "settle_first_step_v4", "settle_first_step_v4_numpy"]
