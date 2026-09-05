"""Residual-gated formal-v4.4 RSC-PF model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .formal_v4_models import _FormalV4Base, _pad_device_history


@dataclass(frozen=True)
class FormalV44ForwardOutput:
    base_forecast_normalized: Tensor
    base_forecast_physical: Tensor
    forecast_normalized: Tensor
    forecast_physical: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    thermal_magnitudes: Tensor
    thermal_residuals: Tensor
    controls: Tensor
    dispatch: Tensor


class ResidualThermalHead(nn.Module):
    """Independent zero-initialized gate and magnitude residual pathways."""

    def __init__(self, transition_probability: Tensor, temperature: float) -> None:
        super().__init__()
        probability = torch.as_tensor(transition_probability, dtype=torch.float32)
        if probability.shape != (4, 3, 3) or not bool(torch.isfinite(probability).all()):
            raise ValueError("transition probability must have shape [4,3,3]")
        if not torch.allclose(probability.sum(-1), torch.ones(4, 3), atol=1.0e-6, rtol=0.0):
            raise ValueError("transition probability rows must sum to one")
        if float(temperature) <= 0.0 or not torch.isfinite(torch.as_tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)
        self.register_buffer("transition_log_prior", probability.clamp_min(1.0e-12).log())
        self.gate_hidden = nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.LayerNorm(64))
        self.magnitude_hidden = nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.LayerNorm(64))
        self.gate_output = nn.Linear(64, 3)
        self.magnitude_output = nn.Linear(64, 2)
        for layer in (self.gate_output, self.magnitude_output):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, base_forecast: Tensor, state: Tensor, last_regime: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if base_forecast.ndim != 3 or tuple(base_forecast.shape[1:]) != (4, 4):
            raise ValueError("base_forecast must have shape [B,4,4]")
        if state.ndim != 2 or state.shape[0] != base_forecast.shape[0] or state.shape[1] != 32:
            raise ValueError("state must have shape [B,32]")
        if last_regime.shape != (base_forecast.shape[0],) or not torch.is_floating_point(last_regime) and last_regime.dtype not in (torch.int64, torch.int32):
            raise ValueError("last_regime must have shape [B] and integer dtype")
        if bool((last_regime < 0).any()) or bool((last_regime > 2).any()):
            raise ValueError("last_regime values must be 0, 1, or 2")
        last_regime = last_regime.to(dtype=torch.long)
        prior = torch.stack([
            self.transition_log_prior[h].index_select(0, last_regime)
            for h in range(4)
        ], dim=1)
        state_horizon = state.unsqueeze(1).expand(-1, 4, -1)
        features = torch.cat((base_forecast, state_horizon), dim=-1)
        gate_residual = self.gate_output(self.gate_hidden(features))
        magnitude_residual = self.magnitude_output(self.magnitude_hidden(features))
        logits = prior + gate_residual
        probabilities = F.softmax(logits / self.temperature, dim=-1)
        return logits, probabilities, magnitude_residual, prior


class ResidualGatedRSCPFModel(_FormalV4Base):
    """RSC-PF with a causal thermal transition prior and residual gate."""

    def __init__(
        self,
        *,
        transition_probability: Tensor,
        regime_temperature: float = 1.0,
        decoder_parameters: Mapping[str, Any] | None = None,
        task_mean: Tensor | None = None,
        task_scale: Tensor | None = None,
        physical_feature_mean: Tensor | None = None,
        physical_feature_scale: Tensor | None = None,
        previous_chp_mean: Tensor | float | None = None,
        previous_chp_scale: Tensor | float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            decoder_parameters=decoder_parameters,
            task_mean=task_mean,
            task_scale=task_scale,
            physical_feature_mean=physical_feature_mean,
            physical_feature_scale=physical_feature_scale,
            previous_chp_mean=previous_chp_mean,
            previous_chp_scale=previous_chp_scale,
            dropout=dropout,
        )
        self.thermal_head = ResidualThermalHead(transition_probability, regime_temperature)

    def base_forecaster_parameters(self):
        return tuple(super().forecaster_parameters())

    def gate_parameters(self):
        return tuple(self.thermal_head.gate_hidden.parameters()) + tuple(self.thermal_head.gate_output.parameters())

    def magnitude_residual_parameters(self):
        return tuple(self.thermal_head.magnitude_hidden.parameters()) + tuple(self.thermal_head.magnitude_output.parameters())

    def scheduler_parameters(self):
        return tuple(super().scheduler_parameters())

    def v44_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        return {
            "base": self.base_forecaster_parameters(),
            "gate": self.gate_parameters(),
            "magnitude": self.magnitude_residual_parameters(),
            "scheduler": self.scheduler_parameters(),
        }

    def forward(self, *, last_thermal_regime: Tensor, detach_forecast_for_dispatch: bool = False, **inputs: Tensor) -> FormalV44ForwardOutput:
        self._validate_common(inputs)
        state = self.state_encoder(inputs["device_history"], inputs["activity_history"])
        padded = _pad_device_history(inputs["device_history"])
        base_normalized, _ = self.core.forecaster.forward_with_details(
            inputs["load_history"], inputs["exog_history"], padded, inputs["activity_history"],
        )
        base_normalized = base_normalized + self.forecast_state_fusion(state).unsqueeze(1)
        base_physical = self.core.forecast_to_physical(base_normalized)
        logits, probabilities, thermal_residuals, _prior = self.thermal_head(base_normalized, state, last_thermal_regime)
        thermal_normalized = base_normalized[..., 1:3] + thermal_residuals
        task_mean = self.core.task_mean.to(dtype=base_normalized.dtype)
        task_scale = self.core.task_scale.to(dtype=base_normalized.dtype)
        thermal_magnitudes = F.softplus(task_mean[1:3] + task_scale[1:3] * thermal_normalized)
        physical = torch.stack((
            base_physical[..., 0],
            probabilities[..., 1] * thermal_magnitudes[..., 0],
            probabilities[..., 2] * thermal_magnitudes[..., 1],
            base_physical[..., 3],
        ), dim=-1)
        normalized = (physical - task_mean) / task_scale
        forecast_for_dispatch = physical.detach() if detach_forecast_for_dispatch else physical
        physical_features = self.core._raw_physical_features(forecast_for_dispatch, inputs["scheduler_context"])
        controls, dispatch = self._schedule(physical_features, state, inputs["previous_chp"])
        return FormalV44ForwardOutput(
            base_forecast_normalized=base_normalized,
            base_forecast_physical=base_physical,
            forecast_normalized=normalized,
            forecast_physical=physical,
            regime_logits=logits,
            regime_probabilities=probabilities,
            thermal_magnitudes=thermal_magnitudes,
            thermal_residuals=thermal_residuals,
            controls=controls,
            dispatch=dispatch,
        )


__all__ = ["FormalV44ForwardOutput", "ResidualGatedRSCPFModel", "ResidualThermalHead"]
