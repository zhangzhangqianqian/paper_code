"""Canonical four-hour settlement and deployment metrics for RSC-PF v3.

The module is deliberately independent of the LP solver.  It mirrors the LP
objective in ordinary NumPy code and provides a differentiable Torch
settlement used by the joint training loss.  All price and penalty terms come
from the frozen dispatch parameter mapping; no evaluator-local scientific
defaults are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from ..scheduling.dispatch_schema import VARIABLES


_I = {name: index for index, name in enumerate(VARIABLES)}


def _scalar(parameters: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = parameters.get(name, default)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 0 or not np.isfinite(float(array)):
        raise ValueError(f"{name} must be a finite scalar")
    return float(array)


def _horizon(parameters: Mapping[str, Any], name: str, horizon: int, default: float = 0.0) -> np.ndarray:
    value = np.asarray(parameters.get(name, default), dtype=np.float64)
    if value.ndim == 0:
        value = np.full(horizon, float(value), dtype=np.float64)
    if value.shape != (horizon,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite scalar or [{horizon}] vector")
    return value


def _check_numpy_inputs(dispatch: object, demand: object, renewables: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.asarray(dispatch, dtype=np.float64)
    y = np.asarray(demand, dtype=np.float64)
    r = np.asarray(renewables, dtype=np.float64)
    if d.ndim != 2 or d.shape[1] != len(VARIABLES):
        raise ValueError(f"dispatch must have shape [H,{len(VARIABLES)}]")
    h = d.shape[0]
    if y.shape != (h, 3) or r.shape != (h, 2):
        raise ValueError("demand and renewables shapes must match the four-hour dispatch")
    if not np.isfinite(d).all() or not np.isfinite(y).all() or not np.isfinite(r).all():
        raise ValueError("dispatch, demand and renewables must be finite")
    if (y < 0).any() or (r < 0).any():
        raise ValueError("demand and renewables must be non-negative")
    return d, y, r


@dataclass(frozen=True)
class FourHourEvaluation:
    operating_cost: float
    carbon: float
    shortage: float
    surplus: float
    penalized_objective: float
    balance_residual: np.ndarray
    conversion_residual: np.ndarray
    soc_residual: np.ndarray
    capacity_violation: float
    ramp_violation: float
    renewable_residual: np.ndarray
    simultaneous_charge_discharge: float
    constraint_valid: bool
    hourly_objective: np.ndarray = field(repr=False)


def evaluate_four_hour_plan(
    *,
    dispatch: np.ndarray,
    demand: np.ndarray,
    renewables: np.ndarray,
    initial_soc: float,
    parameters: Mapping[str, Any],
    previous_chp: float = 0.0,
    tolerance: float = 1.0e-5,
) -> FourHourEvaluation:
    """Evaluate one plan using the same objective as ``solve_dispatch_lp``."""

    values, actual, available = _check_numpy_inputs(dispatch, demand, renewables)
    h = values.shape[0]
    initial_soc = float(initial_soc)
    previous_chp = float(previous_chp)
    if not 0.0 <= initial_soc <= 1.0 or not np.isfinite(previous_chp) or previous_chp < 0.0:
        raise ValueError("initial_soc must be in [0,1] and previous_chp non-negative")
    capacity = _scalar(parameters, "bess_energy_capacity")
    eta = _scalar(parameters, "bess_roundtrip_efficiency") ** 0.5
    grid_price = _horizon(parameters, "grid_energy_price", h)
    gas_price = _horizon(parameters, "gas_energy_price", h)
    grid_ef = _horizon(parameters, "grid_emission_factor", h)
    gas_ef = _horizon(parameters, "gas_emission_factor", h)
    carbon_price = _horizon(parameters, "carbon_price", h, _scalar(parameters, "carbon_price_default", 0.0))
    throughput_cost = _scalar(parameters, "bess_throughput_cost")
    unserved_penalty = _scalar(parameters, "unserved_penalty")
    surplus_penalty = _scalar(parameters, "surplus_penalty", 0.0)
    g = lambda name: values[:, _I[name]]
    served = np.stack(
        (
            g("grid") + g("pv_use") + g("wt_use") + g("p_chp") + g("p_discharge") + g("slack_e") - g("p_ec") - g("p_charge"),
            g("q_ec") + g("q_ac") + g("slack_c"),
            g("q_chp") + g("q_gb") + g("slack_h") - g("q_ac_in") - g("q_dump"),
        ), axis=1,
    )
    balance = served - actual
    shortage_vector = np.maximum(actual - (served - np.stack((g("slack_e"), g("slack_c"), g("slack_h")), axis=1)), 0.0)
    # For an exact LP solution this is exactly the explicit slack vector.  For
    # a candidate plan it remains a useful physical shortage diagnostic.
    explicit_slack = np.stack((g("slack_e"), g("slack_c"), g("slack_h")), axis=1)
    shortage = np.maximum(explicit_slack, 0.0).sum(axis=1)
    surplus = np.maximum(-balance, 0.0).sum(axis=1)
    gas = np.maximum(g("g_chp"), 0.0) + np.maximum(g("g_gb"), 0.0)
    operating_hourly = g("grid") * grid_price + gas * gas_price + throughput_cost * (np.maximum(g("p_charge"), 0.0) + np.maximum(g("p_discharge"), 0.0)) + unserved_penalty * np.maximum(explicit_slack, 0.0).sum(axis=1)
    carbon_hourly = g("grid") * grid_ef + gas * gas_ef
    hourly = operating_hourly + carbon_price * carbon_hourly + surplus_penalty * surplus
    conversion = np.stack(
        (
            g("p_chp") - _scalar(parameters, "chp_electric_efficiency") * g("g_chp"),
            g("q_chp") - _scalar(parameters, "chp_heat_efficiency") * g("g_chp"),
            g("q_gb") - _scalar(parameters, "gas_boiler_efficiency") * g("g_gb"),
            g("q_ec") - _scalar(parameters, "electric_chiller_cop") * g("p_ec"),
            g("q_ac") - _scalar(parameters, "absorption_chiller_cop") * g("q_ac_in"),
        ), axis=1,
    )
    soc_state = np.empty(h, dtype=np.float64)
    previous_energy = initial_soc * capacity
    for t in range(h):
        soc_state[t] = g("soc")[t] - previous_energy - eta * g("p_charge")[t] + g("p_discharge")[t] / eta
        previous_energy = g("soc")[t]
    soc_residual = np.concatenate((soc_state, np.asarray([g("soc")[-1] - initial_soc * capacity])))
    renewable = np.column_stack((g("pv_use") + g("pv_curt") - available[:, 0], g("wt_use") + g("wt_curt") - available[:, 1]))
    upper = {
        "grid": _scalar(parameters, "grid_import_capacity"),
        "g_chp": _scalar(parameters, "chp_electric_capacity") / _scalar(parameters, "chp_electric_efficiency"),
        "g_gb": _scalar(parameters, "gas_boiler_capacity") / _scalar(parameters, "gas_boiler_efficiency"),
        "p_chp": _scalar(parameters, "chp_electric_capacity"),
        "q_chp": _scalar(parameters, "chp_heat_capacity"),
        "q_gb": _scalar(parameters, "gas_boiler_capacity"),
        "p_ec": _scalar(parameters, "electric_chiller_capacity") / _scalar(parameters, "electric_chiller_cop"),
        "q_ec": _scalar(parameters, "electric_chiller_capacity"),
        "q_ac_in": _scalar(parameters, "absorption_chiller_capacity") / _scalar(parameters, "absorption_chiller_cop"),
        "q_ac": _scalar(parameters, "absorption_chiller_capacity"),
        "p_charge": _scalar(parameters, "bess_power_capacity"),
        "p_discharge": _scalar(parameters, "bess_power_capacity"),
        "soc": capacity,
    }
    violation = []
    for name, limit in upper.items():
        violation.append(np.maximum(values[:, _I[name]] - limit, 0.0))
    capacity_violation = float(max(np.max(np.maximum(-values, 0.0)), *(np.max(item) for item in violation)))
    ramp = _scalar(parameters, "chp_ramp_fraction") * _scalar(parameters, "chp_electric_capacity")
    p_chp = g("p_chp")
    prior = np.concatenate(([previous_chp], p_chp[:-1]))
    ramp_violation = float(np.max(np.maximum(np.abs(p_chp - prior) - ramp, 0.0)))
    simultaneous = float(np.max(np.minimum(np.maximum(g("p_charge"), 0.0), np.maximum(g("p_discharge"), 0.0))))
    max_violation = max(float(np.max(np.abs(balance))), float(np.max(np.abs(conversion))), float(np.max(np.abs(soc_residual))), float(np.max(np.abs(renewable))), capacity_violation, ramp_violation, simultaneous)
    return FourHourEvaluation(
        operating_cost=float(np.sum(operating_hourly)), carbon=float(np.sum(carbon_hourly)),
        shortage=float(np.sum(shortage)), surplus=float(np.sum(surplus)),
        penalized_objective=float(np.sum(hourly)), balance_residual=balance,
        conversion_residual=conversion, soc_residual=soc_residual,
        capacity_violation=capacity_violation, ramp_violation=ramp_violation,
        renewable_residual=renewable, simultaneous_charge_discharge=simultaneous,
        constraint_valid=bool(max_violation <= tolerance), hourly_objective=hourly,
    )


def four_hour_optimality_gap(evaluation: FourHourEvaluation, oracle_four_step_objective: float, *, tolerance: float = 1.0e-5) -> float:
    if not np.isfinite(float(oracle_four_step_objective)):
        raise ValueError("oracle objective must be finite")
    if not evaluation.constraint_valid:
        return float("nan")
    return float(evaluation.penalized_objective - float(oracle_four_step_objective))


def _torch_parameter(parameters: Mapping[str, Any], name: str, reference: Tensor, default: float = 0.0) -> Tensor:
    value = parameters.get(name, default)
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


@dataclass(frozen=True)
class FourHourTensorEvaluation:
    operating_cost: Tensor
    carbon: Tensor
    shortage: Tensor
    surplus: Tensor
    penalized_objective: Tensor
    normalized_shortage: Tensor
    constraint_penalty: Tensor
    settled_dispatch: Tensor


def settle_four_hour_plan_torch(
    planned_dispatch: Tensor,
    actual_demand: Tensor,
    actual_renewables: Tensor,
    initial_soc: Tensor,
    previous_chp: Tensor,
    parameters: Mapping[str, Any],
) -> FourHourTensorEvaluation:
    """Differentiably settle all four actions against realized labels."""

    if planned_dispatch.ndim != 3 or planned_dispatch.shape[-1] != len(VARIABLES):
        raise ValueError("planned_dispatch must have shape [B,4,21]")
    b, h, _ = planned_dispatch.shape
    if h != 4 or actual_demand.shape != (b, h, 3) or actual_renewables.shape != (b, h, 2):
        raise ValueError("four-hour demand and renewable shapes are invalid")
    if initial_soc.shape != (b, 1) or previous_chp.shape != (b, 1):
        raise ValueError("initial_soc and previous_chp must have shape [B,1]")
    if not bool(torch.isfinite(planned_dispatch).all() and torch.isfinite(actual_demand).all() and torch.isfinite(actual_renewables).all()):
        raise ValueError("torch settlement inputs must be finite")
    from .rollout import apply_first_step_recourse

    outcomes = []
    # Carry the realized state through the four one-hour settlements.  The
    # first action is constrained by the supplied previous CHP output; later
    # actions must be constrained by the preceding realized action, exactly as
    # in the rolling deployment loop.
    current_soc = initial_soc
    current_previous_chp = previous_chp
    bess_capacity = _torch_parameter(parameters, "bess_energy_capacity", planned_dispatch[:, 0, 0]).clamp_min(1.0e-8)
    for t in range(h):
        outcome = apply_first_step_recourse(
            planned_dispatch[:, t, :], actual_demand[:, t, :], actual_renewables[:, t, 0], actual_renewables[:, t, 1], parameters,
            grid_price=_torch_parameter(parameters, "grid_energy_price", planned_dispatch[:, t, 0]),
            gas_price=_torch_parameter(parameters, "gas_energy_price", planned_dispatch[:, t, 0]),
            carbon_price=_torch_parameter(parameters, "carbon_price", planned_dispatch[:, t, 0], float(parameters.get("carbon_price_default", 0.0))),
        )
        outcomes.append(outcome)
        current_soc = (outcome.realized_dispatch[:, _I["soc"]] / bess_capacity).clamp(0.0, 1.0).unsqueeze(-1)
        current_previous_chp = outcome.realized_dispatch[:, _I["p_chp"]].unsqueeze(-1)
    settled = torch.stack([item.realized_dispatch for item in outcomes], dim=1)
    operating = torch.stack([item.operating_cost for item in outcomes], dim=1).sum(dim=1)
    carbon = torch.stack([item.carbon for item in outcomes], dim=1).sum(dim=1)
    shortage_vectors = torch.stack([item.shortage for item in outcomes], dim=1)
    surplus_vectors = torch.stack([item.surplus for item in outcomes], dim=1)
    shortage = shortage_vectors.sum(dim=(1, 2))
    surplus = surplus_vectors.sum(dim=(1, 2))
    penalized = torch.stack([item.penalized_objective for item in outcomes], dim=1).sum(dim=1)
    demand_scale = actual_demand.abs().sum(dim=(1, 2)).clamp_min(1.0)
    normalized_shortage = shortage / demand_scale
    balance = torch.stack((
        settled[..., _I["grid"]] + settled[..., _I["pv_use"]] + settled[..., _I["wt_use"]] + settled[..., _I["p_chp"]] + settled[..., _I["p_discharge"]] + settled[..., _I["slack_e"]] - settled[..., _I["p_ec"]] - settled[..., _I["p_charge"]] - actual_demand[..., 0],
        settled[..., _I["q_ec"]] + settled[..., _I["q_ac"]] + settled[..., _I["slack_c"]] - actual_demand[..., 1],
        settled[..., _I["q_chp"]] + settled[..., _I["q_gb"]] + settled[..., _I["slack_h"]] - settled[..., _I["q_ac_in"]] - settled[..., _I["q_dump"]] - actual_demand[..., 2],
    ), dim=-1)
    conversion = torch.stack((
        settled[..., _I["p_chp"]] - _torch_parameter(parameters, "chp_electric_efficiency", settled) * settled[..., _I["g_chp"]],
        settled[..., _I["q_chp"]] - _torch_parameter(parameters, "chp_heat_efficiency", settled) * settled[..., _I["g_chp"]],
        settled[..., _I["q_gb"]] - _torch_parameter(parameters, "gas_boiler_efficiency", settled) * settled[..., _I["g_gb"]],
        settled[..., _I["q_ec"]] - _torch_parameter(parameters, "electric_chiller_cop", settled) * settled[..., _I["p_ec"]],
        settled[..., _I["q_ac"]] - _torch_parameter(parameters, "absorption_chiller_cop", settled) * settled[..., _I["q_ac_in"]],
    ), dim=-1)
    eta = _torch_parameter(parameters, "bess_roundtrip_efficiency", settled).clamp_min(1.0e-8).sqrt()
    capacity = _torch_parameter(parameters, "bess_energy_capacity", settled).clamp_min(1.0e-8)
    prev_energy = initial_soc[:, 0] * capacity
    soc_residual = []
    for t in range(h):
        soc = settled[:, t, _I["soc"]]
        soc_residual.append(soc - prev_energy - eta * settled[:, t, _I["p_charge"]] + settled[:, t, _I["p_discharge"]] / eta)
        prev_energy = soc
    soc_residual_tensor = torch.stack(soc_residual, dim=1)
    violation_penalty = balance.abs().mean(dim=(1, 2)) + conversion.abs().mean(dim=(1, 2)) + soc_residual_tensor.abs().mean(dim=1)
    return FourHourTensorEvaluation(operating, carbon, shortage, surplus, penalized, normalized_shortage, violation_penalty, settled)


FORMAL_DISPATCH_METRICS = (
    "operating_cost", "physical_carbon", "shortage", "four_hour_optimality_gap",
    "rolling_objective_difference_vs_pi_mpc", "constraint_violation_rate",
)


class RollingMetricAccumulator:
    """Accumulate realized one-step outcomes without reusing four-hour gaps."""

    def __init__(self) -> None:
        self._operating: list[float] = []
        self._carbon: list[float] = []
        self._shortage: list[float] = []
        self._objective: list[float] = []

    def update(self, *, operating_cost: float, carbon: float, shortage: float, penalized_objective: float) -> None:
        values = (operating_cost, carbon, shortage, penalized_objective)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("rolling metrics must be finite")
        self._operating.append(float(operating_cost)); self._carbon.append(float(carbon))
        self._shortage.append(float(shortage)); self._objective.append(float(penalized_objective))

    def summary(self) -> dict[str, float | int]:
        if not self._objective:
            return {"windows": 0, "operating_cost": 0.0, "physical_carbon": 0.0, "shortage": 0.0, "penalized_objective": 0.0}
        return {"windows": len(self._objective), "operating_cost": float(np.sum(self._operating)), "physical_carbon": float(np.sum(self._carbon)), "shortage": float(np.sum(self._shortage)), "penalized_objective": float(np.sum(self._objective))}


__all__ = [
    "FORMAL_DISPATCH_METRICS", "FourHourEvaluation", "FourHourTensorEvaluation",
    "RollingMetricAccumulator", "evaluate_four_hour_plan", "four_hour_optimality_gap",
    "settle_four_hour_plan_torch",
]
