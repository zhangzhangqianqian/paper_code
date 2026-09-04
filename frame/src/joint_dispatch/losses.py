"""Curriculum and joint forecast--dispatch losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from ..scheduling.proxy_physics import balance_residuals
from .contract import DISPATCH_ORDER, FORECAST_TASK_WEIGHTS
from .common_evaluation import settle_four_hour_plan_torch
from .model import JointForwardOutput
from .rollout import apply_first_step_recourse


@dataclass(frozen=True)
class CurriculumWeights:
    forecast: float
    imitation: float
    decision: float


def weights_for_epoch(
    *,
    epoch: int,
    ramp_epochs: int,
    decision_start: float = 0.05,
    decision_final: float = 1.0,
    imitation_start: float = 1.0,
    imitation_final: float = 0.25,
    forecast: float = 1.0,
) -> CurriculumWeights:
    if epoch < 0 or ramp_epochs <= 0:
        raise ValueError("epoch must be non-negative and ramp_epochs must be positive")
    if not (0.0 < decision_start <= decision_final) or not (0.0 < imitation_final <= imitation_start):
        raise ValueError("invalid curriculum endpoints")
    alpha = min(max(float(epoch) / float(ramp_epochs), 0.0), 1.0)
    return CurriculumWeights(
        forecast=float(forecast),
        imitation=float(imitation_start + alpha * (imitation_final - imitation_start)),
        decision=float(decision_start + alpha * (decision_final - decision_start)),
    )


@dataclass(frozen=True)
class JointLossBreakdown:
    total: Tensor
    forecast: Tensor
    imitation: Tensor
    regret: Tensor
    shortage: Tensor
    carbon_metric: Tensor
    planned_feasibility: Tensor
    normalized_shortage: Tensor | None = None


@dataclass(frozen=True)
class JointV3LossBreakdown:
    """Four-hour decision-aware loss components used by the v3 pilot."""

    total: Tensor
    forecast: Tensor
    imitation: Tensor
    four_hour_optimality_gap: Tensor
    shortage: Tensor
    constraint_penalty: Tensor
    normalized_shortage: Tensor

    @property
    def regret(self) -> Tensor:
        """Compatibility name used by legacy training receipts."""

        return self.four_hour_optimality_gap


def _idx(name: str) -> int:
    return DISPATCH_ORDER.index(name)


def _parameter(parameters: Mapping[str, Any], name: str, default: float = 1.0) -> float:
    value = float(parameters.get(name, default))
    if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
        raise ValueError(f"parameter {name} must be finite and positive")
    return value


def joint_forecast_dispatch_loss(
    output: JointForwardOutput,
    target_normalized: Tensor,
    target_physical: Tensor,
    teacher_dispatch: Tensor,
    oracle_first_step_objective: Tensor,
    parameters: Mapping[str, Any],
    weights: CurriculumWeights,
    *,
    normalize_decision: bool = False,
) -> JointLossBreakdown:
    """Compute one scalar loss while retaining all gradient paths."""

    if target_normalized.shape != output.forecast_normalized.shape or target_physical.shape != output.forecast_physical.shape:
        raise ValueError("forecast targets must match model output shape [B,4,4]")
    if teacher_dispatch.ndim != 3 or teacher_dispatch.shape != output.dispatch.shape:
        raise ValueError("teacher_dispatch must have shape [B,4,21]")
    if oracle_first_step_objective.shape != (output.dispatch.shape[0],):
        raise ValueError("oracle_first_step_objective must have shape [B]")
    for value in (target_normalized, target_physical, teacher_dispatch, oracle_first_step_objective):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("loss inputs must be finite")
    task_weights = target_normalized.new_tensor(FORECAST_TASK_WEIGHTS)
    per_task = F.smooth_l1_loss(output.forecast_normalized, target_normalized, reduction="none").mean(dim=1)
    forecast = (per_task * task_weights).sum(dim=-1).mean() / task_weights.sum()
    # Independent control coordinates avoid overweighting decoder-dependent
    # variables that are algebraically determined by the four controls.
    independent = ("q_ec", "p_chp", "soc", "pv_use")
    denominators = (
        _parameter(parameters, "electric_chiller_capacity"),
        _parameter(parameters, "chp_electric_capacity"),
        _parameter(parameters, "bess_energy_capacity"),
        _parameter(parameters, "pv_capacity", 1.0),
    )
    imitation_terms = []
    for name, denominator in zip(independent, denominators):
        prediction = output.dispatch[..., _idx(name)]
        target = teacher_dispatch[..., _idx(name)].to(dtype=prediction.dtype)
        imitation_terms.append(F.smooth_l1_loss(prediction / denominator, target / denominator))
    imitation = torch.stack(imitation_terms).mean()
    actual_demand = target_physical[:, 0, :3]
    context = output.physical_features[:, 0, :]
    outcome = apply_first_step_recourse(
        output.dispatch[:, 0, :],
        actual_demand,
        context[:, 4],
        context[:, 5],
        parameters,
        grid_price=context[:, 6],
        gas_price=context[:, 7],
        carbon_price=context[:, 8],
    )
    oracle = oracle_first_step_objective.to(dtype=outcome.penalized_objective.dtype)
    regret = (torch.relu(outcome.penalized_objective - oracle) / oracle.abs().clamp_min(1.0)).mean()
    shortage = outcome.shortage.mean()
    # ``shortage`` is retained in physical units for reporting.  The formal
    # v2 objective also uses a dimensionless version so that the decision term
    # is not dominated merely by the magnitude of the demand series.
    demand_scale = target_physical[:, 0, :3].detach().abs().mean().clamp_min(1.0)
    normalized_shortage = shortage / demand_scale
    carbon_metric = outcome.carbon.mean()
    planned_feasibility = balance_residuals(output.dispatch, output.physical_features).abs().mean()
    decision_shortage = normalized_shortage if normalize_decision else shortage
    total = weights.forecast * forecast + weights.imitation * imitation + weights.decision * (regret + decision_shortage)
    return JointLossBreakdown(total, forecast, imitation, regret, shortage, carbon_metric, planned_feasibility, normalized_shortage)


def joint_forecast_dispatch_loss_v3(
    output: JointForwardOutput,
    target_normalized: Tensor,
    target_physical: Tensor,
    realized_renewables: Tensor,
    initial_soc: Tensor,
    previous_chp: Tensor,
    teacher_dispatch: Tensor,
    oracle_four_step_objective: Tensor,
    parameters: Mapping[str, Any],
    weights: CurriculumWeights,
    *,
    forecast_task_weights: Sequence[float] | None = None,
) -> JointV3LossBreakdown:
    """Compute the connected v3 objective against realized four-hour labels."""

    if target_normalized.shape != output.forecast_normalized.shape or target_physical.shape != output.forecast_physical.shape:
        raise ValueError("forecast targets must match model output shape [B,4,4]")
    if teacher_dispatch.shape != output.dispatch.shape:
        raise ValueError("teacher_dispatch must match output dispatch shape [B,4,21]")
    batch = output.dispatch.shape[0]
    if realized_renewables.shape != (batch, 4, 2):
        raise ValueError("realized_renewables must have shape [B,4,2]")
    if initial_soc.shape != (batch, 1) or previous_chp.shape != (batch, 1):
        raise ValueError("initial_soc and previous_chp must have shape [B,1]")
    if oracle_four_step_objective.shape != (batch,):
        raise ValueError("oracle_four_step_objective must have shape [B]")
    for value in (target_normalized, target_physical, realized_renewables, initial_soc, previous_chp, teacher_dispatch, oracle_four_step_objective):
        if not bool(torch.isfinite(value).all()):
            raise ValueError("v3 loss inputs must be finite")
    task_weights = target_normalized.new_tensor(
        FORECAST_TASK_WEIGHTS if forecast_task_weights is None else tuple(float(value) for value in forecast_task_weights)
    )
    if task_weights.shape != (4,) or bool((task_weights < 0.0).any()) or float(task_weights.sum()) <= 0.0:
        raise ValueError("forecast_task_weights must contain four non-negative values with a positive sum")
    per_task = F.smooth_l1_loss(output.forecast_normalized, target_normalized, reduction="none").mean(dim=1)
    forecast = (per_task * task_weights).sum(dim=-1).mean() / task_weights.sum()
    independent = ("q_ec", "p_chp", "soc", "pv_use")
    denominators = (
        _parameter(parameters, "electric_chiller_capacity"),
        _parameter(parameters, "chp_electric_capacity"),
        _parameter(parameters, "bess_energy_capacity"),
        _parameter(parameters, "pv_capacity", 1.0),
    )
    imitation_terms = []
    for name, denominator in zip(independent, denominators):
        prediction = output.dispatch[..., _idx(name)]
        target = teacher_dispatch[..., _idx(name)].to(dtype=prediction.dtype, device=prediction.device)
        imitation_terms.append(F.smooth_l1_loss(prediction / denominator, target / denominator))
    imitation = torch.stack(imitation_terms).mean()
    settled = settle_four_hour_plan_torch(
        output.dispatch,
        target_physical[:, :, :3],
        realized_renewables,
        initial_soc,
        previous_chp,
        parameters,
    )
    oracle = oracle_four_step_objective.to(dtype=settled.penalized_objective.dtype, device=settled.penalized_objective.device)
    gap_per_sample = (settled.penalized_objective - oracle) / oracle.abs().clamp_min(1.0)
    positive_gap = torch.relu(gap_per_sample).mean()
    shortage = settled.shortage.mean()
    normalized_shortage = settled.normalized_shortage.mean()
    constraint_penalty = settled.constraint_penalty.mean()
    decision = positive_gap + normalized_shortage + constraint_penalty
    total = weights.forecast * forecast + weights.imitation * imitation + weights.decision * decision
    return JointV3LossBreakdown(
        total=total, forecast=forecast, imitation=imitation,
        four_hour_optimality_gap=positive_gap, shortage=shortage,
        constraint_penalty=constraint_penalty, normalized_shortage=normalized_shortage,
    )


__all__ = [
    "CurriculumWeights", "JointLossBreakdown", "JointV3LossBreakdown",
    "joint_forecast_dispatch_loss", "joint_forecast_dispatch_loss_v3", "weights_for_epoch",
]
