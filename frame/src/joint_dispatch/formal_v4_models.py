"""Formal-v4 model paths: RSC-PF and a transparent Direct-Policy control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..scheduling.proxy_decoder import CONTROL_TEMPERATURE, decode_feasible_controls, decode_feasible_dispatch
from .contract import DISPATCH_ORDER
from .model import JointForecastDispatchModel


class _CausalDSTCNBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        value = F.pad(value, (self.left_padding, 0))
        value = self.depthwise(value)
        value = self.pointwise(value)
        value = self.norm(value)
        value = self.dropout(F.gelu(value))
        return residual + value


class CausalStateDSTCNEncoder(nn.Module):
    """Three left-padded depthwise-separable blocks over 17+6 state fields."""

    def __init__(self, input_dim: int = 23, hidden_dim: int = 32, *, kernel_size: int = 5, dilations: tuple[int, ...] = (1, 2, 4), dropout: float = 0.0) -> None:
        super().__init__()
        if input_dim != 23 or hidden_dim <= 0:
            raise ValueError("formal-v4 state encoder fixes input_dim=23 and requires positive hidden_dim")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            _CausalDSTCNBlock(hidden_dim, kernel_size=kernel_size, dilation=dilation, dropout=dropout)
            for dilation in dilations
        ])
        self.output_projection = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))

    def forward(self, device_history: Tensor, activity_history: Tensor | None = None) -> Tensor:
        if activity_history is None:
            state = device_history
        else:
            if device_history.ndim != 3 or tuple(device_history.shape[1:]) != (24, 17):
                raise ValueError("device_history must have shape [B,24,17]")
            if activity_history.ndim != 3 or tuple(activity_history.shape[1:]) != (24, 6):
                raise ValueError("activity_history must have shape [B,24,6]")
            if device_history.shape[0] != activity_history.shape[0]:
                raise ValueError("device and activity histories must share batch size")
            state = torch.cat((device_history, activity_history), dim=-1)
        if state.ndim != 3 or state.shape[1] != 24 or state.shape[-1] != self.input_dim:
            raise ValueError("state sequence must have shape [B,24,23]")
        if not bool(torch.isfinite(state).all()):
            raise ValueError("state sequence must be finite")
        value = self.input_projection(state.transpose(1, 2))
        for block in self.blocks:
            value = block(value)
        value = value.transpose(1, 2)
        summary = torch.cat((value[:, -1, :], value.mean(dim=1)), dim=-1)
        return self.output_projection(summary)


@dataclass(frozen=True)
class FormalV4ForwardOutput:
    forecast_normalized: Tensor | None
    forecast_physical: Tensor | None
    latent_planning_demand: Tensor | None
    controls: Tensor
    dispatch: Tensor


def _pad_device_history(device_history: Tensor) -> Tensor:
    if device_history.ndim != 3 or tuple(device_history.shape[1:]) != (24, 17):
        raise ValueError("device_history must have shape [B,24,17]")
    return torch.cat((device_history, device_history.new_zeros((*device_history.shape[:2], 4))), dim=-1)


class _FormalV4Base(nn.Module):
    def __init__(self, *, decoder_parameters: Mapping[str, Any] | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.core = JointForecastDispatchModel(decoder_parameters=decoder_parameters, dropout=dropout)
        self.state_encoder = CausalStateDSTCNEncoder(dropout=dropout)
        self.forecast_state_fusion = nn.Linear(32, 4)
        self.state_to_scheduler = nn.Sequential(nn.Linear(32, 16), nn.GELU())

    def forecaster_parameters(self):
        """Parameters considered part of the forecast/state pathway."""

        return tuple(self.core.forecaster.parameters()) + tuple(self.state_encoder.parameters()) + tuple(self.forecast_state_fusion.parameters())

    def scheduler_parameters(self):
        return tuple(self.core.scheduler.parameters()) + tuple(self.state_to_scheduler.parameters())

    @staticmethod
    def _validate_common(inputs: Mapping[str, Tensor]) -> None:
        required = ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")
        missing = [name for name in required if name not in inputs]
        if missing:
            raise ValueError(f"missing formal-v4 model inputs: {missing}")
        if inputs["load_history"].ndim != 3 or tuple(inputs["load_history"].shape[1:]) != (24, 4):
            raise ValueError("load_history must have shape [B,24,4]")
        if inputs["exog_history"].ndim != 3 or tuple(inputs["exog_history"].shape[1:]) != (24, 12):
            raise ValueError("exog_history must have shape [B,24,12]")
        if inputs["activity_history"].shape[:2] != inputs["device_history"].shape[:2]:
            raise ValueError("activity/device history dimensions must align")
        if inputs["scheduler_context"].ndim != 3 or tuple(inputs["scheduler_context"].shape[1:]) != (4, 6):
            raise ValueError("scheduler_context must have shape [B,4,6]")
        if inputs["previous_chp"].ndim != 2 or inputs["previous_chp"].shape != (inputs["load_history"].shape[0], 1):
            raise ValueError("previous_chp must have shape [B,1]")

    def _schedule(self, physical_features: Tensor, state: Tensor, previous_chp: Tensor) -> tuple[Tensor, Tensor]:
        normalized = (physical_features - self.core.physical_feature_mean) / self.core.physical_feature_scale
        normalized_for_scheduler = normalized.to(dtype=state.dtype)
        state16 = self.state_to_scheduler(state)
        logits = self.core.scheduler(normalized_for_scheduler, state16, previous_chp.to(dtype=state.dtype))
        controls_flat = torch.sigmoid(logits / CONTROL_TEMPERATURE)
        dispatch = decode_feasible_controls(
            controls_flat, physical_features, self.core.decoder_parameters,
            previous_chp=previous_chp, allow_heat_dump=True,
        )
        controls = controls_flat.unsqueeze(1).expand(-1, 4, -1)
        return controls, dispatch


class RSCPFModel(_FormalV4Base):
    """RSC-PF: explicit four-task forecast bottleneck followed by scheduling."""

    def forward(self, *, detach_forecast_for_dispatch: bool = False, **inputs: Tensor) -> FormalV4ForwardOutput:
        self._validate_common(inputs)
        state = self.state_encoder(inputs["device_history"], inputs["activity_history"])
        padded = _pad_device_history(inputs["device_history"])
        forecast_normalized, _details = self.core.forecaster.forward_with_details(
            inputs["load_history"], inputs["exog_history"], padded, inputs["activity_history"],
        )
        # The causal 23-channel state is part of the forecast graph, not only
        # a post-hoc scheduler feature.  This gives Stage P a meaningful update
        # for the state encoder and preserves the explicit forecast bottleneck.
        forecast_normalized = forecast_normalized + self.forecast_state_fusion(state).unsqueeze(1)
        forecast_physical = self.core.forecast_to_physical(forecast_normalized)
        # Decoupled-RSC-PF uses the identical scheduler and initialization but
        # cuts only the decision-loss edge at this bottleneck.  The reported
        # forecast remains attached so its supervised loss still trains the
        # forecaster.
        forecast_for_dispatch = forecast_physical.detach() if detach_forecast_for_dispatch else forecast_physical
        physical_features = self.core._raw_physical_features(forecast_for_dispatch, inputs["scheduler_context"])
        controls, dispatch = self._schedule(physical_features, state, inputs["previous_chp"])
        return FormalV4ForwardOutput(forecast_normalized, forecast_physical, None, controls, dispatch)


class DirectPolicyModel(_FormalV4Base):
    """Decision-only comparator with no supervised forecast output or loss."""

    def __init__(self, *, decoder_parameters: Mapping[str, Any] | None = None, dropout: float = 0.0) -> None:
        super().__init__(decoder_parameters=decoder_parameters, dropout=dropout)
        self.planning_head = nn.Sequential(nn.Linear(32 + 6, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, **inputs: Tensor) -> FormalV4ForwardOutput:
        self._validate_common(inputs)
        state = self.state_encoder(inputs["device_history"], inputs["activity_history"])
        state_horizon = state.unsqueeze(1).expand(-1, 4, -1)
        planning = self.planning_head(torch.cat((state_horizon, inputs["scheduler_context"]), dim=-1))
        planning = F.softplus(planning)
        gas_zeros = planning.new_zeros((*planning.shape[:2], 1))
        physical_features = self.core._raw_physical_features(torch.cat((planning, gas_zeros), dim=-1), inputs["scheduler_context"])
        controls, dispatch = self._schedule(physical_features, state, inputs["previous_chp"])
        return FormalV4ForwardOutput(None, None, planning, controls, dispatch)


__all__ = ["CausalStateDSTCNEncoder", "DirectPolicyModel", "FormalV4ForwardOutput", "RSCPFModel"]
