"""Formal-v4.6 RSC-PF model with a bounded, risk-adjusted dispatch bottleneck.

The nominal forecast remains a four-task supervised output.  A small bounded
head may add non-negative, task-specific reserve to the three dispatchable
demand channels (electricity, cooling, and heating); the gas prior is kept as
an auxiliary context signal and is never risk-adjusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .formal_v4_4_model import FormalV44ForwardOutput, ResidualGatedRSCPFModel, _pad_device_history
from .model import JointForecastDispatchModel


@dataclass(frozen=True)
class FormalV46ForwardOutput:
    """All nominal, risk, scheduler, and physical outputs of RSC-PF v4.6."""

    base_forecast_normalized: Tensor
    base_forecast_physical: Tensor
    forecast_nominal_normalized: Tensor
    forecast_nominal_physical: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    thermal_magnitudes: Tensor
    thermal_residuals: Tensor
    risk_adjustment: Tensor
    risk_cap: Tensor
    scheduler_demand: Tensor
    controls: Tensor
    dispatch: Tensor

    @property
    def forecast_normalized(self) -> Tensor:
        """Compatibility alias: the reported forecast is the nominal forecast."""

        return self.forecast_nominal_normalized

    @property
    def forecast_physical(self) -> Tensor:
        """Compatibility alias: the reported forecast is the nominal forecast."""

        return self.forecast_nominal_physical

    def as_v44_forecast_output(self) -> FormalV44ForwardOutput:
        """Expose the nominal path through the v4.4 loss/training contract."""

        return FormalV44ForwardOutput(
            base_forecast_normalized=self.base_forecast_normalized,
            base_forecast_physical=self.base_forecast_physical,
            forecast_normalized=self.forecast_nominal_normalized,
            forecast_physical=self.forecast_nominal_physical,
            regime_logits=self.regime_logits,
            regime_probabilities=self.regime_probabilities,
            thermal_magnitudes=self.thermal_magnitudes,
            thermal_residuals=self.thermal_residuals,
            controls=self.controls,
            dispatch=self.dispatch,
        )


class RiskAdjustmentHeadV46(nn.Module):
    """Bounded non-negative reserve head for the three dispatchable tasks.

    The input is exactly the normalized ten-feature scheduler context, the
    32-dimensional causal state, and the normalized previous CHP output.  A
    regime mask prevents thermal reserve from appearing in an off regime.
    """

    def __init__(
        self,
        risk_cap: Tensor,
        *,
        hidden_width: int = 64,
        initial_bias: float = -6.0,
    ) -> None:
        super().__init__()
        cap = torch.as_tensor(risk_cap, dtype=torch.float32)
        if cap.shape != (4, 3) or not bool(torch.isfinite(cap).all()) or bool((cap <= 0.0).any()):
            raise ValueError("risk_cap must have shape [4,3] and be finite and positive")
        if int(hidden_width) <= 0 or not torch.isfinite(torch.as_tensor(initial_bias)):
            raise ValueError("risk head width and bias must be finite and valid")
        self.register_buffer("risk_cap", cap)
        self.hidden = nn.Sequential(
            nn.Linear(10 + 32 + 1, int(hidden_width)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_width)),
        )
        self.output = nn.Linear(int(hidden_width), 3)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, float(initial_bias))

    def forward(
        self,
        normalized_features: Tensor,
        state: Tensor,
        normalized_previous_chp: Tensor,
        regime_probabilities: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if normalized_features.ndim != 3 or tuple(normalized_features.shape[1:]) != (4, 10):
            raise ValueError("normalized_features must have shape [B,4,10]")
        batch = normalized_features.shape[0]
        if state.ndim != 2 or state.shape != (batch, 32):
            raise ValueError("state must have shape [B,32]")
        if normalized_previous_chp.shape != (batch, 1):
            raise ValueError("normalized_previous_chp must have shape [B,1]")
        if regime_probabilities.shape != (batch, 4, 3):
            raise ValueError("regime_probabilities must have shape [B,4,3]")
        values = (normalized_features, state, normalized_previous_chp, regime_probabilities)
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("risk head inputs must be finite")
        state_horizon = state.unsqueeze(1).expand(-1, 4, -1)
        previous_horizon = normalized_previous_chp.unsqueeze(1).expand(-1, 4, -1)
        x = torch.cat((normalized_features, state_horizon, previous_horizon), dim=-1)
        raw = self.output(self.hidden(x))
        cap = self.risk_cap.to(dtype=raw.dtype).unsqueeze(0).expand(batch, -1, -1)
        mask = torch.stack(
            (
                torch.ones_like(regime_probabilities[..., 0]),
                regime_probabilities[..., 1],
                regime_probabilities[..., 2],
            ),
            dim=-1,
        )
        adjustment = torch.sigmoid(raw) * cap * mask
        return adjustment, cap


class RiskAdjustedRSCPFModelV46(ResidualGatedRSCPFModel):
    """RSC-PF v4.6 with an explicit nominal-to-risk-adjusted bottleneck."""

    def __init__(
        self,
        *,
        transition_probability: Tensor,
        risk_cap: Tensor,
        regime_temperature: float = 1.0,
        risk_hidden_width: int = 64,
        risk_initial_bias: float = -6.0,
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
            transition_probability=transition_probability,
            regime_temperature=regime_temperature,
            decoder_parameters=decoder_parameters,
            task_mean=task_mean,
            task_scale=task_scale,
            physical_feature_mean=physical_feature_mean,
            physical_feature_scale=physical_feature_scale,
            previous_chp_mean=previous_chp_mean,
            previous_chp_scale=previous_chp_scale,
            dropout=dropout,
        )
        self.risk_head = RiskAdjustmentHeadV46(
            risk_cap,
            hidden_width=risk_hidden_width,
            initial_bias=risk_initial_bias,
        )

    @classmethod
    def for_test(cls, risk_cap: Tensor | None = None) -> "RiskAdjustedRSCPFModelV46":
        transition = torch.full((4, 3, 3), 1.0 / 3.0)
        cap = torch.ones(4, 3) if risk_cap is None else torch.as_tensor(risk_cap, dtype=torch.float32)
        return cls(
            transition_probability=transition,
            risk_cap=cap,
            decoder_parameters=JointForecastDispatchModel._test_parameters(),
            dropout=0.0,
            task_mean=torch.zeros(4),
            task_scale=torch.ones(4),
            physical_feature_mean=torch.zeros(10),
            physical_feature_scale=torch.ones(10),
            previous_chp_mean=torch.zeros(1),
            previous_chp_scale=torch.ones(1),
        )

    def risk_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.risk_head.parameters())

    def v46_parameter_groups(self) -> dict[str, tuple[nn.Parameter, ...]]:
        groups = self.v44_parameter_groups()
        groups["risk"] = self.risk_parameters()
        return groups

    def _nominal_forecast(self, inputs: Mapping[str, Tensor]) -> tuple[Tensor, ...]:
        state = self.state_encoder(inputs["device_history"], inputs["activity_history"])
        padded = _pad_device_history(inputs["device_history"])
        base_normalized, _ = self.core.forecaster.forward_with_details(
            inputs["load_history"], inputs["exog_history"], padded, inputs["activity_history"],
        )
        base_normalized = base_normalized + self.forecast_state_fusion(state).unsqueeze(1)
        base_physical = self.core.forecast_to_physical(base_normalized)
        logits, probabilities, thermal_residuals, _prior = self.thermal_head(
            base_normalized, state, inputs["last_thermal_regime"],
        )
        thermal_normalized = base_normalized[..., 1:3] + thermal_residuals
        task_mean = self.core.task_mean.to(dtype=base_normalized.dtype)
        task_scale = self.core.task_scale.to(dtype=base_normalized.dtype)
        thermal_magnitudes = F.softplus(task_mean[1:3] + task_scale[1:3] * thermal_normalized)
        nominal_physical = torch.stack(
            (
                base_physical[..., 0],
                probabilities[..., 1] * thermal_magnitudes[..., 0],
                probabilities[..., 2] * thermal_magnitudes[..., 1],
                base_physical[..., 3],
            ),
            dim=-1,
        )
        nominal_normalized = (nominal_physical - task_mean) / task_scale
        return (
            state,
            base_normalized,
            base_physical,
            nominal_normalized,
            nominal_physical,
            logits,
            probabilities,
            thermal_magnitudes,
            thermal_residuals,
        )

    def forward(
        self,
        *,
        last_thermal_regime: Tensor,
        detach_forecast_for_dispatch: bool = False,
        **inputs: Tensor,
    ) -> FormalV46ForwardOutput:
        self._validate_common(inputs)
        inputs = dict(inputs)
        inputs["last_thermal_regime"] = last_thermal_regime
        (
            state,
            base_normalized,
            base_physical,
            nominal_normalized,
            nominal_physical,
            logits,
            probabilities,
            thermal_magnitudes,
            thermal_residuals,
        ) = self._nominal_forecast(inputs)
        nominal_for_scheduler = nominal_physical.detach() if detach_forecast_for_dispatch else nominal_physical
        probabilities_for_scheduler = probabilities.detach() if detach_forecast_for_dispatch else probabilities
        state_for_scheduler = state.detach() if detach_forecast_for_dispatch else state
        raw_features = self.core._raw_physical_features(nominal_for_scheduler, inputs["scheduler_context"])
        normalized_features = (raw_features - self.core.physical_feature_mean) / self.core.physical_feature_scale
        normalized_features = normalized_features.to(dtype=state_for_scheduler.dtype)
        chp_capacity = float(self.core.decoder_parameters.get("chp_electric_capacity", self.core.decoder_parameters.get("Pbar_chp", float("inf"))))
        bounded_previous = inputs["previous_chp"].to(dtype=torch.float64).clamp_min(0.0).clamp_max(chp_capacity)
        normalized_previous = (
            bounded_previous.to(dtype=state_for_scheduler.dtype)
            - self.previous_chp_mean.to(dtype=state_for_scheduler.dtype)
        ) / self.previous_chp_scale.to(dtype=state_for_scheduler.dtype)
        risk_adjustment, risk_cap = self.risk_head(
            normalized_features,
            state_for_scheduler,
            normalized_previous,
            probabilities_for_scheduler,
        )
        scheduler_demand = nominal_for_scheduler.clone()
        scheduler_demand = torch.cat(
            (scheduler_demand[..., :3] + risk_adjustment, scheduler_demand[..., 3:]),
            dim=-1,
        )
        scheduler_features = self.core._raw_physical_features(scheduler_demand, inputs["scheduler_context"])
        controls, dispatch = self._schedule(
            scheduler_features,
            state_for_scheduler,
            inputs["previous_chp"],
        )
        return FormalV46ForwardOutput(
            base_forecast_normalized=base_normalized,
            base_forecast_physical=base_physical,
            forecast_nominal_normalized=nominal_normalized,
            forecast_nominal_physical=nominal_physical,
            regime_logits=logits,
            regime_probabilities=probabilities,
            thermal_magnitudes=thermal_magnitudes,
            thermal_residuals=thermal_residuals,
            risk_adjustment=risk_adjustment,
            risk_cap=risk_cap,
            scheduler_demand=scheduler_demand,
            controls=controls,
            dispatch=dispatch,
        )


__all__ = [
    "FormalV46ForwardOutput",
    "RiskAdjustedRSCPFModelV46",
    "RiskAdjustmentHeadV46",
]
