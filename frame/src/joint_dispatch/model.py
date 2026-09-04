"""Connected forecasting--dispatch neural network.

The forecast tensor is an explicit differentiable bottleneck: the scheduler
only sees the four predicted tasks together with renewable, price, carbon, and
state context.  The physical decoder is reused unchanged, so the network
cannot bypass the canonical 21-variable dispatch representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..models import Scheme2RModel
from ..scheduling.proxy_decoder import CONTROL_TEMPERATURE, decode_feasible_dispatch
from ..scheduling.proxy_model import ResidualMLPBlock
from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER
from .data import SCHEDULER_CONTEXT_ORDER


def _check_finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite torch tensor")


class DeviceHistoryEncoder(nn.Module):
    """Encode continuous dispatch and derived status histories into state."""

    def __init__(self, input_dim: int = 27, hidden_dim: int = 32, state_dim: int = 16) -> None:
        super().__init__()
        if input_dim != len(DISPATCH_ORDER) + len(STATUS_ORDER):
            raise ValueError("device history input_dim must equal 21+6")
        if hidden_dim <= 0 or state_dim <= 0:
            raise ValueError("hidden_dim and state_dim must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_dim = int(state_dim)
        self.projection = nn.Sequential(
            nn.Linear(4 * input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, state_dim),
            nn.GELU(),
        )

    def forward(self, device_history: Tensor, device_status: Tensor) -> Tensor:
        if device_history.ndim != 3 or device_history.shape[-1] != len(DISPATCH_ORDER):
            raise ValueError("device_history must have shape [B,T,21]")
        if device_status.ndim != 3 or device_status.shape[:2] != device_history.shape[:2] or device_status.shape[-1] != len(STATUS_ORDER):
            raise ValueError("device_status must have shape [B,T,6]")
        _check_finite(device_history, "device_history")
        _check_finite(device_status, "device_status")
        if bool(((device_status != 0.0) & (device_status != 1.0)).any()):
            raise ValueError("device_status must be binary")
        sequence = torch.cat((device_history, device_status), dim=-1)
        first = sequence[:, :1, :]
        last = sequence[:, -1:, :]
        mean = sequence.mean(dim=1, keepdim=True)
        std = sequence.std(dim=1, keepdim=True, unbiased=False)
        trend = last - first
        summary = torch.cat((last, mean, std, trend), dim=-1).squeeze(1)
        return self.projection(summary)


class StateConditionedScheme2R(nn.Module):
    """Scheme2R whose router state also receives causal device history."""

    def __init__(self, exog_dim: int = len(EXOG_ORDER), state_dim: int = 16, **kwargs: Any) -> None:
        super().__init__()
        self.base = Scheme2RModel(exog_dim=exog_dim, task_count=len(TASK_ORDER), state_dim=state_dim, **kwargs)
        self.device_encoder = DeviceHistoryEncoder(state_dim=state_dim)
        base_state_dim = int(getattr(self.base.state_encoder, "state_dim", state_dim))
        self.state_fusion = nn.Sequential(nn.Linear(base_state_dim + state_dim, base_state_dim), nn.GELU())

    def forward_with_details(
        self,
        load_history: Tensor,
        exog_history: Tensor,
        device_history: Tensor,
        device_status: Tensor,
    ) -> tuple[Tensor, Mapping[str, Tensor]]:
        representations = self.base.encode_tasks(load_history, exog_history)
        base_state = self.base.encode_state(exog_history, representations)
        device_state = self.device_encoder(device_history, device_status)
        joint_state = self.state_fusion(torch.cat((base_state, device_state), dim=-1))
        step_embeddings = self.base.step_embeddings()
        target_embeddings, source_embeddings = self.base.role_embeddings()
        rho, pi, gates = self.base.router(
            joint_state, representations, step_embeddings, target_embeddings, source_embeddings
        )
        pair_messages = self.base.message_projector(representations)
        fused = self.base.fusion(representations, rho, pi, pair_messages)
        predictions = self.base.head(fused)
        return predictions, MappingProxyType(
            {
                "representations": representations,
                "base_state": base_state,
                "device_state": device_state,
                "joint_state": joint_state,
                "rho": rho,
                "pi": pi,
                "gates": gates,
                "pair_messages": pair_messages,
                "fused_representations": fused,
            }
        )

    def forward(self, load_history: Tensor, exog_history: Tensor, device_history: Tensor, device_status: Tensor) -> Tensor:
        return self.forward_with_details(load_history, exog_history, device_history, device_status)[0]


class JointSchedulingProxy(nn.Module):
    """15-logit scheduler conditioned on forecast and device state."""

    def __init__(self, hidden_width: int = 128, residual_blocks: int = 2, state_dim: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        if hidden_width <= 0 or residual_blocks <= 0 or state_dim <= 0:
            raise ValueError("scheduler dimensions must be positive")
        self.state_dim = int(state_dim)
        self.input_dim = 4 * 10 + state_dim + 1
        self.input_block = nn.Sequential(
            nn.Linear(self.input_dim, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Dropout(dropout),
        )
        self.residual = nn.Sequential(*[ResidualMLPBlock(hidden_width, dropout) for _ in range(residual_blocks)])
        self.output_projection = nn.Linear(hidden_width, 15)

    def forward(self, normalized_physical_features: Tensor, device_state: Tensor, previous_chp: Tensor) -> Tensor:
        if normalized_physical_features.ndim != 3 or tuple(normalized_physical_features.shape[1:]) != (4, 10):
            raise ValueError("normalized_physical_features must have shape [B,4,10]")
        if device_state.ndim != 2 or device_state.shape[-1] != self.state_dim or device_state.shape[0] != normalized_physical_features.shape[0]:
            raise ValueError("device_state must have shape [B,state_dim]")
        if previous_chp.ndim != 2 or previous_chp.shape != (normalized_physical_features.shape[0], 1):
            raise ValueError("previous_chp must have shape [B,1]")
        _check_finite(normalized_physical_features, "normalized_physical_features")
        _check_finite(device_state, "device_state")
        _check_finite(previous_chp, "previous_chp")
        x = torch.cat((normalized_physical_features.reshape(normalized_physical_features.shape[0], -1), device_state, previous_chp), dim=-1)
        return self.output_projection(self.residual(self.input_block(x)))


@dataclass(frozen=True)
class JointForwardOutput:
    forecast_normalized: Tensor
    forecast_physical: Tensor
    physical_features: Tensor
    control_logits: Tensor
    dispatch: Tensor
    details: Mapping[str, Tensor]


class JointForecastDispatchModel(nn.Module):
    """One connected graph with an explicit forecast-to-dispatch bottleneck."""

    def __init__(
        self,
        *,
        exog_dim: int = len(EXOG_ORDER),
        task_mean: Tensor | None = None,
        task_scale: Tensor | None = None,
        physical_feature_mean: Tensor | None = None,
        physical_feature_scale: Tensor | None = None,
        decoder_parameters: Mapping[str, Any] | None = None,
        scheduler_context_mean: Tensor | None = None,
        scheduler_context_scale: Tensor | None = None,
        dropout: float = 0.1,
        use_gas_prior: bool = True,
    ) -> None:
        super().__init__()
        self.forecaster = StateConditionedScheme2R(exog_dim=exog_dim, dropout=dropout)
        self.scheduler = JointSchedulingProxy(dropout=dropout)
        self.use_gas_prior = bool(use_gas_prior)
        self.register_buffer("task_mean", self._vector(task_mean, 4, 0.0))
        self.register_buffer("task_scale", self._positive_vector(task_scale, 4, 1.0))
        self.register_buffer("physical_feature_mean", self._vector(physical_feature_mean, 10, 0.0))
        self.register_buffer("physical_feature_scale", self._positive_vector(physical_feature_scale, 10, 1.0))
        self.register_buffer("scheduler_context_mean", self._vector(scheduler_context_mean, 6, 0.0))
        self.register_buffer("scheduler_context_scale", self._positive_vector(scheduler_context_scale, 6, 1.0))
        self.decoder_parameters = MappingProxyType(dict(decoder_parameters or self._test_parameters()))

    @staticmethod
    def _vector(value: Tensor | None, size: int, fill: float) -> Tensor:
        if value is None:
            return torch.full((size,), fill, dtype=torch.float32)
        result = torch.as_tensor(value, dtype=torch.float32)
        if result.shape != (size,):
            raise ValueError(f"vector must have shape [{size}]")
        return result

    @classmethod
    def _positive_vector(cls, value: Tensor | None, size: int, fill: float) -> Tensor:
        result = cls._vector(value, size, fill)
        if bool((result <= 0).any()) or not bool(torch.isfinite(result).all()):
            raise ValueError("scale vectors must be finite and positive")
        return result

    @staticmethod
    def _test_parameters() -> dict[str, float]:
        return {
            "grid_import_capacity": 100.0,
            "chp_electric_capacity": 20.0,
            "chp_heat_capacity": 30.0,
            "gas_boiler_capacity": 40.0,
            "electric_chiller_capacity": 30.0,
            "absorption_chiller_capacity": 30.0,
            "bess_power_capacity": 10.0,
            "bess_energy_capacity": 40.0,
            "chp_electric_efficiency": 0.35,
            "chp_heat_efficiency": 0.45,
            "gas_boiler_efficiency": 0.9,
            "electric_chiller_cop": 3.5,
            "absorption_chiller_cop": 0.75,
            "bess_roundtrip_efficiency": 0.9,
            "chp_ramp_fraction": 0.5,
        }

    @classmethod
    def for_test(cls, exog_dim: int = 12) -> "JointForecastDispatchModel":
        return cls(exog_dim=exog_dim, dropout=0.0)

    def forecast_to_physical(self, forecast_normalized: Tensor) -> Tensor:
        if forecast_normalized.ndim < 1 or forecast_normalized.shape[-1] != len(TASK_ORDER):
            raise ValueError("forecast_normalized must end with four tasks")
        _check_finite(forecast_normalized, "forecast_normalized")
        unconstrained = self.task_mean + self.task_scale * forecast_normalized
        return F.softplus(unconstrained, beta=1.0, threshold=20.0)

    def _raw_physical_features(self, forecast_physical: Tensor, scheduler_context: Tensor) -> Tensor:
        if scheduler_context.ndim != 3 or tuple(scheduler_context.shape[1:]) != (4, len(SCHEDULER_CONTEXT_ORDER)):
            raise ValueError("scheduler_context must have shape [B,4,6]")
        if scheduler_context.shape[0] != forecast_physical.shape[0]:
            raise ValueError("scheduler_context batch does not match forecast")
        _check_finite(scheduler_context, "scheduler_context")
        # Keep the physical decoder inputs in float64.  Forecast heads may be
        # trained in float32, but silently down-casting the carried SOC and
        # price/context state here creates avoidable 1e-5--1e-4 feasibility
        # residuals in the strict post-training audit.
        context = scheduler_context.to(dtype=torch.float64)
        forecast_physical = forecast_physical.to(dtype=torch.float64)
        soc = context[..., -1]
        if bool((soc < 0.0).any()) or bool((soc > 1.0).any()):
            raise ValueError("initial_soc must be in [0,1]")
        if not bool(torch.allclose(soc, soc[:, :1], atol=1e-6, rtol=0.0)):
            raise ValueError("initial_soc must be identical across horizon rows")
        return torch.cat((forecast_physical, context), dim=-1)

    def forward(
        self,
        *,
        load_history: Tensor,
        exog_history: Tensor,
        device_history: Tensor,
        device_status: Tensor,
        scheduler_context: Tensor,
        previous_chp: Tensor,
    ) -> JointForwardOutput:
        forecast_normalized, forecast_details = self.forecaster.forward_with_details(
            load_history, exog_history, device_history, device_status
        )
        forecast_physical = self.forecast_to_physical(forecast_normalized)
        if not self.use_gas_prior:
            # The gas forecast remains a reported auxiliary task, but it is
            # removed from the scheduler context in the named No-Gas-Prior
            # ablation.  This keeps the graph shape and decoder contract fixed.
            forecast_physical = forecast_physical.clone()
            forecast_physical[..., 3] = 0.0
        physical_features = self._raw_physical_features(forecast_physical, scheduler_context)
        normalized_features = (physical_features - self.physical_feature_mean) / self.physical_feature_scale
        normalized_features = normalized_features.to(dtype=forecast_normalized.dtype)
        device_state = forecast_details["device_state"]
        # The scheduler MLP follows the forecaster's training dtype, while the
        # decoder receives the original high-precision carried state below.
        scheduler_previous_chp = previous_chp.to(dtype=normalized_features.dtype)
        control_logits = self.scheduler(normalized_features, device_state, scheduler_previous_chp)
        dispatch = decode_feasible_dispatch(
            control_logits,
            physical_features,
            self.decoder_parameters,
            temperature=CONTROL_TEMPERATURE,
            previous_chp=previous_chp,
            allow_heat_dump=True,
        )
        details = dict(forecast_details)
        details.update({"normalized_physical_features": normalized_features})
        return JointForwardOutput(
            forecast_normalized=forecast_normalized,
            forecast_physical=forecast_physical,
            physical_features=physical_features,
            control_logits=control_logits,
            dispatch=dispatch,
            details=MappingProxyType(details),
        )


__all__ = [
    "DeviceHistoryEncoder",
    "JointForecastDispatchModel",
    "JointForwardOutput",
    "JointSchedulingProxy",
    "StateConditionedScheme2R",
]
