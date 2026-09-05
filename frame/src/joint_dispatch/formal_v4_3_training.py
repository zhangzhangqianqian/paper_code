"""Regime-aware forecast losses and v4.3 joint-loss helpers."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .formal_v4_3_data import ThermalMagnitudeReceiptV43
from .formal_v4_objective import formal_v4_joint_loss
from .formal_v4_2_training import (
    StageBudgetV42,
    StageResultV42,
    _batches,
    _decoder_parameters,
    _dispatch_targets,
    _grad_norm,
    _model_groups,
    _optimizer_step,
    _optimizer_steps,
    _run_model,
    _target_normalized,
    _target_physical,
    _to_tensor,
    _value_grad_norm,
    seed_everything,
)


@dataclass(frozen=True)
class ForecastLossBreakdownV43:
    total: Tensor
    continuous: Tensor
    regime: Tensor
    active_magnitude: Tensor
    point: Tensor
    inactive_leakage: Tensor


def _target(batch: Mapping[str, Any], key: str, reference: Tensor) -> Tensor:
    value = batch.get(key)
    if value is None:
        raise ValueError(f"training batch requires {key}")
    result = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=reference.dtype)
    result = result.to(device=reference.device, dtype=reference.dtype)
    if result.shape != reference.shape:
        raise ValueError(f"{key} must have shape {tuple(reference.shape)}")
    return result


def _thermal_labels(target_physical: Tensor, epsilon: float = 1.0e-9) -> Tensor:
    cooling = target_physical[..., 1] > float(epsilon)
    heating = target_physical[..., 2] > float(epsilon)
    if bool((cooling & heating).any()):
        raise ValueError("simultaneous cooling and heating targets are not representable")
    return torch.where(cooling, 1, torch.where(heating, 2, 0)).long()


def _sample_mean(values: Tensor, weights: Tensor | None = None) -> Tensor:
    if values.ndim != 1:
        values = values.reshape(values.shape[0], -1).mean(dim=1)
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-12)


def _class_weights(receipt: ThermalMagnitudeReceiptV43, reference: Tensor) -> Tensor:
    counts = torch.as_tensor(receipt.class_count, dtype=reference.dtype, device=reference.device)
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1.0e-12)
    lower, upper = 0.5, 2.0
    return weights.clamp(lower, upper)


def forecast_loss_v43(
    output: Any,
    batch: Mapping[str, Any],
    receipt: ThermalMagnitudeReceiptV43,
    weights: Mapping[str, float],
) -> ForecastLossBreakdownV43:
    """Calculate the auditable five-component v4.3 forecast loss."""

    forecast = getattr(output, "forecast_normalized", None)
    physical = getattr(output, "forecast_physical", None)
    logits = getattr(output, "regime_logits", None)
    magnitudes = getattr(output, "thermal_magnitudes", None)
    if not all(isinstance(value, Tensor) for value in (forecast, physical, logits, magnitudes)):
        raise ValueError("regime-aware output is missing forecast tensors")
    if forecast.ndim != 3 or tuple(forecast.shape[1:]) != (4, 4):
        raise ValueError("forecast_normalized must have shape [B,4,4]")
    if physical.shape != forecast.shape or logits.shape != (forecast.shape[0], 4, 3) or magnitudes.shape != (forecast.shape[0], 4, 2):
        raise ValueError("regime-aware output shapes are inconsistent")
    target_normalized = _target(batch, "target_normalized", forecast)
    target_physical = _target(batch, "target_physical", physical)
    labels = _thermal_labels(target_physical)
    transition = batch.get("thermal_transition_mask")
    if transition is None:
        sample_weights = forecast.new_ones((forecast.shape[0],))
    else:
        sample_weights = torch.as_tensor(transition, dtype=forecast.dtype, device=forecast.device).reshape(-1)
        if sample_weights.shape != (forecast.shape[0],):
            raise ValueError("thermal_transition_mask must have shape [B]")
        sample_weights = torch.where(sample_weights > 0.5, forecast.new_tensor(float(weights.get("transition_window_weight", 1.0))), forecast.new_ones(()))

    point_per_sample = F.smooth_l1_loss(forecast, target_normalized, reduction="none").mean(dim=(1, 2))
    continuous_per_sample = (
        F.smooth_l1_loss(forecast[..., 0], target_normalized[..., 0], reduction="none").mean(dim=1)
        + float(weights.get("gas_task_weight", 0.25))
        * F.smooth_l1_loss(forecast[..., 3], target_normalized[..., 3], reduction="none").mean(dim=1)
    )
    class_weight = _class_weights(receipt, forecast)
    regime_per_step = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), weight=class_weight, reduction="none")
    regime_per_sample = regime_per_step.reshape(forecast.shape[0], 4).mean(dim=1)

    magnitude_mean = torch.as_tensor(receipt.mean, dtype=physical.dtype, device=physical.device)
    magnitude_scale = torch.as_tensor(receipt.scale, dtype=physical.dtype, device=physical.device).clamp_min(1.0e-12)
    true_magnitude = torch.stack((target_physical[..., 1], target_physical[..., 2]), dim=-1)
    magnitude_error = F.smooth_l1_loss(
        (magnitudes - magnitude_mean) / magnitude_scale,
        (true_magnitude - magnitude_mean) / magnitude_scale,
        reduction="none",
    )
    active_mask = torch.stack((labels == 1, labels == 2), dim=-1)
    active_magnitude_per_sample = (magnitude_error * active_mask).sum(dim=(1, 2)) / active_mask.sum(dim=(1, 2)).clamp_min(1.0)
    inactive_mask = ~active_mask
    leakage = (physical[..., 1:3] / magnitude_scale).abs()
    inactive_leakage_per_sample = (leakage * inactive_mask).sum(dim=(1, 2)) / inactive_mask.sum(dim=(1, 2)).clamp_min(1.0)

    continuous = _sample_mean(continuous_per_sample, sample_weights)
    regime = _sample_mean(regime_per_sample, sample_weights)
    active_magnitude = _sample_mean(active_magnitude_per_sample, sample_weights)
    point = _sample_mean(point_per_sample, sample_weights)
    inactive_leakage = _sample_mean(inactive_leakage_per_sample, sample_weights)
    total = (
        float(weights.get("continuous_weight", 1.0)) * continuous
        + float(weights.get("regime_weight", 1.0)) * regime
        + float(weights.get("active_magnitude_weight", 1.0)) * active_magnitude
        + float(weights.get("point_weight", 0.5)) * point
        + float(weights.get("inactive_leakage_weight", 0.25)) * inactive_leakage
    )
    return ForecastLossBreakdownV43(total, continuous, regime, active_magnitude, point, inactive_leakage)


def run_stage_p_v43(
    model: torch.nn.Module,
    loaders: Any,
    receipt: ThermalMagnitudeReceiptV43,
    loss_weights: Mapping[str, float],
    budget: StageBudgetV42 | None = None,
    seed: int = 2026,
) -> StageResultV42:
    budget = budget or StageBudgetV42()
    seed_everything(seed)
    train = _batches(loaders.get("train") if isinstance(loaders, Mapping) else loaders)
    if not train:
        raise ValueError("v4.3 Stage P requires non-empty train batches")
    forecaster, scheduler = _model_groups(model)
    for parameter in forecaster:
        parameter.requires_grad_(True)
    for parameter in scheduler:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(forecaster), lr=budget.forecaster_lr, weight_decay=budget.weight_decay)
    history: list[float] = []
    last_loss = 0.0
    for epoch in range(budget.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train:
            output = _run_model(model, batch)
            breakdown = forecast_loss_v43(output, batch, receipt, loss_weights)
            last_loss = float(breakdown.total.detach())
            epoch_losses.append(last_loss)
            _optimizer_step(optimizer, model, forecaster, breakdown.total, budget.max_grad_norm)
        history.append(float(np.mean(epoch_losses)))
    return StageResultV42("P", "forecast_pretrain_v43", model, optimizer, budget.max_epochs, _optimizer_steps(optimizer), last_loss, loss_history=tuple(history))


def run_stage_s_v43(
    model: torch.nn.Module,
    loaders: Any,
    budget: StageBudgetV42 | None = None,
    seed: int = 2026,
) -> StageResultV42:
    budget = budget or StageBudgetV42()
    seed_everything(seed)
    train = _batches(loaders.get("train") if isinstance(loaders, Mapping) else loaders)
    if not train:
        raise ValueError("v4.3 Stage S requires non-empty train batches")
    forecaster, scheduler = _model_groups(model)
    for parameter in forecaster:
        parameter.requires_grad_(False)
    for parameter in scheduler:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(list(scheduler), lr=budget.scheduler_lr, weight_decay=budget.weight_decay)
    history: list[float] = []
    last_loss = 0.0
    for epoch in range(budget.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train:
            output = _run_model(model, batch, decouple_decision=True)
            teacher = _dispatch_targets(batch, output.dispatch)
            if teacher is None:
                raise ValueError("v4.3 Stage S requires teacher_dispatch")
            scale = teacher.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
            loss = F.smooth_l1_loss(output.dispatch / scale, teacher.to(output.dispatch) / scale)
            last_loss = float(loss.detach())
            epoch_losses.append(last_loss)
            _optimizer_step(optimizer, model, scheduler, loss, budget.max_grad_norm)
        history.append(float(np.mean(epoch_losses)))
    return StageResultV42("S", "scheduler_pretrain_v43", model, optimizer, budget.max_epochs, _optimizer_steps(optimizer), last_loss, loss_history=tuple(history))


def run_stage_j_v43(
    model: torch.nn.Module,
    loaders: Any,
    *,
    mode: Literal["joint", "decoupled"],
    receipt: ThermalMagnitudeReceiptV43,
    loss_weights: Mapping[str, float],
    budget: StageBudgetV42 | None = None,
    c_ref: float = 1.0,
    parameters: Mapping[str, Any] | None = None,
    seed: int = 2026,
) -> StageResultV42:
    if mode not in {"joint", "decoupled"}:
        raise ValueError("v4.3 Stage J mode must be joint or decoupled")
    budget = budget or StageBudgetV42()
    seed_everything(seed)
    train = _batches(loaders.get("train") if isinstance(loaders, Mapping) else loaders)
    if not train:
        raise ValueError("v4.3 Stage J requires non-empty train batches")
    forecast_params, scheduler_params = _model_groups(model)
    for parameter in forecast_params:
        parameter.requires_grad_(mode == "joint")
    for parameter in scheduler_params:
        parameter.requires_grad_(True)
    active_forecast = [p for p in forecast_params if p.requires_grad]
    active_scheduler = [p for p in scheduler_params if p.requires_grad]
    groups = ([{"params": active_forecast, "lr": budget.forecaster_lr}] if active_forecast else [])
    groups.append({"params": active_scheduler, "lr": budget.scheduler_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=budget.weight_decay)
    decoder_parameters = _decoder_parameters(model, parameters)
    history: list[float] = []
    last_loss = forecast_gradient_norm = scheduler_gradient_norm = 0.0
    for epoch in range(budget.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train:
            output = _run_model(model, batch, decouple_decision=mode == "decoupled")
            forecast_breakdown = forecast_loss_v43(output, batch, receipt, loss_weights)
            teacher = _dispatch_targets(batch, output.dispatch)
            realized = batch.get("realized_renewables", batch.get("renewable_realized"))
            initial_soc = batch.get("initial_soc")
            previous_chp = batch.get("previous_chp")
            if realized is None or initial_soc is None or previous_chp is None:
                if teacher is None:
                    total = budget.weights(epoch).forecast * forecast_breakdown.total
                else:
                    scale = teacher.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
                    imitation = F.smooth_l1_loss(output.dispatch / scale, teacher.to(output.dispatch) / scale)
                    total = budget.weights(epoch).forecast * forecast_breakdown.total + budget.weights(epoch).imitation * imitation
                decision = total.new_zeros(())
            else:
                realized_t = _to_tensor(realized, dtype=output.dispatch.dtype).to(output.dispatch)
                initial_t = _to_tensor(initial_soc, dtype=output.dispatch.dtype).to(output.dispatch)
                previous_t = _to_tensor(previous_chp, dtype=output.dispatch.dtype).to(output.dispatch)
                weights = budget.weights(epoch)
                target_normalized = _target_normalized(batch, output.forecast_normalized)
                target_physical = _target_physical(batch, output.forecast_physical)
                breakdown = formal_v4_joint_loss(
                    output, target_normalized, target_physical, realized_t, initial_t, previous_t, teacher,
                    decoder_parameters, c_ref=c_ref, forecast_weight=weights.forecast,
                    imitation_weight=weights.imitation, decision_weight=weights.decision,
                    supervised_forecast_loss=forecast_breakdown.total,
                )
                total = breakdown.total
                decision = breakdown.normalized_realized_objective + breakdown.constraint_penalty
            if active_forecast and decision.requires_grad:
                decision_grads = torch.autograd.grad(decision, active_forecast, allow_unused=True, retain_graph=True)
                forecast_gradient_norm = _value_grad_norm(decision_grads)
            if active_scheduler and decision.requires_grad:
                scheduler_grads = torch.autograd.grad(decision, active_scheduler, allow_unused=True, retain_graph=True)
                scheduler_gradient_norm = _value_grad_norm(scheduler_grads)
            last_loss = float(total.detach())
            epoch_losses.append(last_loss)
            _optimizer_step(optimizer, model, tuple((*active_forecast, *active_scheduler)), total, budget.max_grad_norm)
        history.append(float(np.mean(epoch_losses)))
    return StageResultV42(
        "J", mode + "_v43", model, optimizer, budget.max_epochs, _optimizer_steps(optimizer),
        last_loss, forecast_gradient_norm, scheduler_gradient_norm, tuple(history),
    )


__all__ = [
    "ForecastLossBreakdownV43", "forecast_loss_v43", "formal_v4_joint_loss",
    "run_stage_p_v43", "run_stage_s_v43", "run_stage_j_v43",
]
