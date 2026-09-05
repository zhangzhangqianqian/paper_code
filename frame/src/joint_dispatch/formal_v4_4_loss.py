"""Auditable forecast losses and fixed curriculum for formal-v4.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .formal_v4_4_regime import ThermalPriorReceiptV44


@dataclass(frozen=True)
class CurriculumWeightsV44:
    electricity_gas: float
    regime: float
    active_magnitude: float
    point: float
    inactive_leakage: float


@dataclass(frozen=True)
class ForecastLossV44:
    total: Tensor
    electricity_gas: Tensor
    regime: Tensor
    active_magnitude: Tensor
    point: Tensor
    inactive_leakage: Tensor
    inactive_leakage_physical: Tensor


def curriculum_weights(epoch: int, ramp_epochs: int, leakage_target: float) -> CurriculumWeightsV44:
    if int(epoch) < 0 or int(ramp_epochs) < 1:
        raise ValueError("epoch must be non-negative and ramp_epochs must be positive")
    target = float(leakage_target)
    if not 0.10 <= target <= 1.0 or not torch.isfinite(torch.as_tensor(target)):
        raise ValueError("leakage_target must be finite in [0.10,1.0]")
    progress = min(max(float(epoch) / float(ramp_epochs), 0.0), 1.0)
    return CurriculumWeightsV44(
        electricity_gas=1.0,
        regime=1.0,
        active_magnitude=1.0,
        point=0.25 + 0.75 * progress,
        inactive_leakage=0.10 + (target - 0.10) * progress,
    )


def _target(batch: Mapping[str, Any], key: str, reference: Tensor) -> Tensor:
    value = batch.get(key)
    if value is None:
        raise ValueError(f"batch requires {key}")
    result = value if isinstance(value, Tensor) else torch.as_tensor(value)
    result = result.to(device=reference.device, dtype=reference.dtype)
    if result.shape != reference.shape:
        raise ValueError(f"{key} must have shape {tuple(reference.shape)}")
    return result


def _weighted(values: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("sample loss and sample weights must be one-dimensional and aligned")
    return (values * weights).sum() / weights.sum().clamp_min(1.0e-12)


def _labels(target: Tensor) -> Tensor:
    cooling = target[..., 1] > 1.0e-9
    heating = target[..., 2] > 1.0e-9
    if bool((cooling & heating).any()):
        raise ValueError("simultaneous cooling and heating targets are not representable")
    return torch.where(cooling, torch.ones_like(cooling, dtype=torch.long), torch.where(heating, torch.full_like(cooling, 2, dtype=torch.long), torch.zeros_like(cooling, dtype=torch.long)))


def forecast_loss_v44(
    output: Any,
    batch: Mapping[str, Any],
    receipt: ThermalPriorReceiptV44,
    weights: CurriculumWeightsV44 | Mapping[str, float],
) -> ForecastLossV44:
    forecast = getattr(output, "forecast_normalized", None)
    physical = getattr(output, "forecast_physical", None)
    logits = getattr(output, "regime_logits", None)
    magnitudes = getattr(output, "thermal_magnitudes", None)
    if not all(isinstance(value, Tensor) for value in (forecast, physical, logits, magnitudes)):
        raise ValueError("output is missing v4.4 forecast tensors")
    if forecast.ndim != 3 or tuple(forecast.shape[1:]) != (4, 4):
        raise ValueError("forecast must have shape [B,4,4]")
    if physical.shape != forecast.shape or logits.shape != (forecast.shape[0], 4, 3) or magnitudes.shape != (forecast.shape[0], 4, 2):
        raise ValueError("v4.4 forecast shapes are inconsistent")
    target_normalized = _target(batch, "target_normalized", forecast)
    target_physical = _target(batch, "target_physical", physical)
    labels = _labels(target_physical)
    transition = batch.get("thermal_transition_mask")
    if transition is None:
        sample_weights = forecast.new_ones((forecast.shape[0],))
    else:
        transition_tensor = torch.as_tensor(transition, dtype=forecast.dtype, device=forecast.device).reshape(-1)
        if transition_tensor.shape != (forecast.shape[0],):
            raise ValueError("thermal_transition_mask must have shape [B]")
        transition_weight = float(weights.get("transition_window_weight", 1.5) if isinstance(weights, Mapping) else 1.5)
        sample_weights = torch.where(transition_tensor > 0.5, forecast.new_tensor(transition_weight), forecast.new_ones(()))

    gas_weight = float(weights.get("gas_weight", 0.25) if isinstance(weights, Mapping) else 0.25)
    regime_weight = float(weights.get("regime_weight", 1.0) if isinstance(weights, Mapping) else 1.0)
    magnitude_weight = float(weights.get("active_magnitude_weight", 1.0) if isinstance(weights, Mapping) else 1.0)
    point_weight = float(weights.get("point", 1.0) if isinstance(weights, Mapping) else weights.point)
    leakage_weight = float(weights.get("inactive_leakage", 0.25) if isinstance(weights, Mapping) else weights.inactive_leakage)
    class_count = torch.as_tensor(receipt.class_count, dtype=forecast.dtype, device=forecast.device)
    class_weight = (class_count.sum() / class_count.clamp_min(1.0))
    class_weight = (class_weight / class_weight.mean().clamp_min(1.0e-12)).clamp(0.5, 2.0)

    electricity = F.smooth_l1_loss(forecast[..., 0], target_normalized[..., 0], reduction="none").mean(dim=1)
    gas = F.smooth_l1_loss(forecast[..., 3], target_normalized[..., 3], reduction="none").mean(dim=1)
    point_sample = F.smooth_l1_loss(forecast, target_normalized, reduction="none").mean(dim=(1, 2))
    regime_sample = F.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), weight=class_weight, reduction="none").reshape(forecast.shape[0], 4).mean(dim=1)

    means = torch.as_tensor(receipt.active_mean, dtype=physical.dtype, device=physical.device)
    scales = torch.as_tensor(receipt.active_scale, dtype=physical.dtype, device=physical.device).clamp_min(1.0e-12)
    true_magnitude = torch.stack((target_physical[..., 1], target_physical[..., 2]), dim=-1)
    magnitude_error = F.smooth_l1_loss((magnitudes - means) / scales, (true_magnitude - means) / scales, reduction="none")
    active_mask = torch.stack((labels == 1, labels == 2), dim=-1)
    active_sample = (magnitude_error * active_mask).sum(dim=(1, 2)) / active_mask.sum(dim=(1, 2)).clamp_min(1.0)
    inactive_mask = ~active_mask
    leakage_physical = physical[..., 1:3].abs()
    leakage_sample = (leakage_physical * inactive_mask).sum(dim=(1, 2)) / inactive_mask.sum(dim=(1, 2)).clamp_min(1.0)
    eg = _weighted(electricity + gas_weight * gas, sample_weights)
    regime = _weighted(regime_sample, sample_weights)
    active_magnitude = _weighted(active_sample, sample_weights)
    point = _weighted(point_sample, sample_weights)
    leakage = _weighted(leakage_sample / scales.mean(), sample_weights)
    total = eg + regime_weight * regime + magnitude_weight * active_magnitude + point_weight * point + leakage_weight * leakage
    return ForecastLossV44(total, eg, regime, active_magnitude, point, leakage, _weighted(leakage_sample, sample_weights))


__all__ = ["CurriculumWeightsV44", "ForecastLossV44", "curriculum_weights", "forecast_loss_v44"]
