"""Formal-v4 model paths: RSC-PF and a transparent Direct-Policy control."""

from __future__ import annotations

from dataclasses import dataclass
import math
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


@dataclass(frozen=True)
class RegimeForecastV43:
    electricity_normalized: Tensor
    gas_normalized: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    thermal_magnitude_normalized: Tensor
    thermal_magnitudes: Tensor


@dataclass(frozen=True)
class FormalV43ForwardOutput:
    forecast_normalized: Tensor
    forecast_physical: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    thermal_magnitudes: Tensor
    controls: Tensor
    dispatch: Tensor


class RegimeAwareForecastHead(nn.Module):
    """Differentiable occurrence/state and conditional-size forecast head."""

    def __init__(self, state_dim: int = 32, hidden_dim: int = 64, temperature: float = 1.0) -> None:
        super().__init__()
        if state_dim <= 0 or hidden_dim <= 0 or not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("invalid regime-aware forecast-head dimensions or temperature")
        self.temperature = float(temperature)
        self.hidden = nn.Sequential(
            nn.Linear(4 + int(state_dim), int(hidden_dim)),
            nn.GELU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        # electricity, gas, three regime logits, cooling magnitude, heating magnitude
        self.output = nn.Linear(int(hidden_dim), 7)

    def forward(self, base_forecast: Tensor, state: Tensor) -> RegimeForecastV43:
        if base_forecast.ndim != 3 or tuple(base_forecast.shape[1:]) != (4, 4):
            raise ValueError("base_forecast must have shape [B,4,4]")
        if state.ndim != 2 or state.shape[0] != base_forecast.shape[0]:
            raise ValueError("state must have shape [B,state_dim]")
        state_horizon = state.unsqueeze(1).expand(-1, 4, -1)
        raw = self.output(self.hidden(torch.cat((base_forecast, state_horizon), dim=-1)))
        probabilities = F.softmax(raw[..., 2:5] / self.temperature, dim=-1)
        return RegimeForecastV43(
            electricity_normalized=raw[..., 0],
            gas_normalized=raw[..., 1],
            regime_logits=raw[..., 2:5],
            regime_probabilities=probabilities,
            thermal_magnitude_normalized=raw[..., 5:7],
            thermal_magnitudes=raw[..., 5:7],
        )


def _pad_device_history(device_history: Tensor) -> Tensor:
    if device_history.ndim != 3 or tuple(device_history.shape[1:]) != (24, 17):
        raise ValueError("device_history must have shape [B,24,17]")
    return torch.cat((device_history, device_history.new_zeros((*device_history.shape[:2], 4))), dim=-1)


class _FormalV4Base(nn.Module):
    def __init__(
        self,
        *,
        decoder_parameters: Mapping[str, Any] | None = None,
        task_mean: Tensor | None = None,
        task_scale: Tensor | None = None,
        physical_feature_mean: Tensor | None = None,
        physical_feature_scale: Tensor | None = None,
        previous_chp_mean: Tensor | float | None = None,
        previous_chp_scale: Tensor | float | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.core = JointForecastDispatchModel(
            decoder_parameters=decoder_parameters,
            task_mean=task_mean,
            task_scale=task_scale,
            physical_feature_mean=physical_feature_mean,
            physical_feature_scale=physical_feature_scale,
            dropout=dropout,
        )
        previous_mean = torch.as_tensor(0.0 if previous_chp_mean is None else previous_chp_mean, dtype=torch.float32).reshape(1)
        previous_scale = torch.as_tensor(1.0 if previous_chp_scale is None else previous_chp_scale, dtype=torch.float32).reshape(1)
        if not bool(torch.isfinite(previous_mean).all()) or not bool(torch.isfinite(previous_scale).all()) or bool((previous_scale <= 0.0).any()):
            raise ValueError("previous CHP normalization must be finite with positive scale")
        self.register_buffer("previous_chp_mean", previous_mean)
        self.register_buffer("previous_chp_scale", previous_scale)
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
        # The closed-loop state is carried through float32 tensors while the
        # decoder checks physical capacities in float64.  Permit only the
        # sub-micro-unit round-off that can arise at this boundary; a genuine
        # capacity violation remains fail-closed.
        chp_capacity = float(self.core.decoder_parameters.get("chp_electric_capacity", self.core.decoder_parameters.get("Pbar_chp", float("inf"))))
        if bool((previous_chp > chp_capacity + 1.0e-4).any()):
            raise ValueError("previous_chp cannot exceed chp_electric_capacity")
        bounded_previous_chp = previous_chp.to(dtype=torch.float64).clamp_min(0.0).clamp_max(chp_capacity)
        normalized = (physical_features - self.core.physical_feature_mean) / self.core.physical_feature_scale
        normalized_for_scheduler = normalized.to(dtype=state.dtype)
        state16 = self.state_to_scheduler(state)
        normalized_previous = (bounded_previous_chp.to(dtype=state.dtype) - self.previous_chp_mean.to(dtype=state.dtype)) / self.previous_chp_scale.to(dtype=state.dtype)
        logits = self.core.scheduler(normalized_for_scheduler, state16, normalized_previous)
        controls_flat = torch.sigmoid(logits / CONTROL_TEMPERATURE)
        dispatch = decode_feasible_controls(
            controls_flat, physical_features, self.core.decoder_parameters,
            previous_chp=bounded_previous_chp, allow_heat_dump=True,
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


class RegimeAwareRSCPFModel(_FormalV4Base):
    """RSC-PF with a differentiable off/cooling/heating thermal head."""

    def __init__(
        self,
        *,
        decoder_parameters: Mapping[str, Any] | None = None,
        task_mean: Tensor | None = None,
        task_scale: Tensor | None = None,
        physical_feature_mean: Tensor | None = None,
        physical_feature_scale: Tensor | None = None,
        previous_chp_mean: Tensor | float | None = None,
        previous_chp_scale: Tensor | float | None = None,
        thermal_magnitude_mean: Tensor | None = None,
        thermal_magnitude_scale: Tensor | None = None,
        regime_temperature: float = 1.0,
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
        magnitude_mean = torch.as_tensor(
            [1.0, 1.0] if thermal_magnitude_mean is None else thermal_magnitude_mean,
            dtype=torch.float32,
        )
        magnitude_scale = torch.as_tensor(
            [1.0, 1.0] if thermal_magnitude_scale is None else thermal_magnitude_scale,
            dtype=torch.float32,
        )
        if magnitude_mean.shape != (2,) or magnitude_scale.shape != (2,):
            raise ValueError("thermal magnitude statistics must have shape [2]")
        if not bool(torch.isfinite(magnitude_mean).all() and torch.isfinite(magnitude_scale).all()) or bool((magnitude_scale <= 0.0).any()):
            raise ValueError("thermal magnitude statistics must be finite with positive scale")
        self.register_buffer("thermal_magnitude_mean", magnitude_mean)
        self.register_buffer("thermal_magnitude_scale", magnitude_scale)
        self.regime_head = RegimeAwareForecastHead(
            state_dim=32,
            hidden_dim=64,
            temperature=regime_temperature,
        )

    def forecaster_parameters(self):
        return tuple(super().forecaster_parameters()) + tuple(self.regime_head.parameters())

    @classmethod
    def for_test(cls) -> "RegimeAwareRSCPFModel":
        return cls(decoder_parameters=JointForecastDispatchModel._test_parameters(), dropout=0.0)

    def forward(self, *, detach_forecast_for_dispatch: bool = False, **inputs: Tensor) -> FormalV43ForwardOutput:
        self._validate_common(inputs)
        state = self.state_encoder(inputs["device_history"], inputs["activity_history"])
        padded = _pad_device_history(inputs["device_history"])
        base_forecast, _details = self.core.forecaster.forward_with_details(
            inputs["load_history"], inputs["exog_history"], padded, inputs["activity_history"],
        )
        base_forecast = base_forecast + self.forecast_state_fusion(state).unsqueeze(1)
        raw = self.regime_head(base_forecast, state)
        electricity = F.softplus(self.core.task_mean[0] + self.core.task_scale[0] * raw.electricity_normalized)
        gas = F.softplus(self.core.task_mean[3] + self.core.task_scale[3] * raw.gas_normalized)
        thermal_raw = self.thermal_magnitude_mean + self.thermal_magnitude_scale * raw.thermal_magnitude_normalized
        thermal_magnitudes = F.softplus(thermal_raw)
        cooling = raw.regime_probabilities[..., 1] * thermal_magnitudes[..., 0]
        heating = raw.regime_probabilities[..., 2] * thermal_magnitudes[..., 1]
        forecast_physical = torch.stack((electricity, cooling, heating, gas), dim=-1)
        forecast_normalized = (forecast_physical - self.core.task_mean) / self.core.task_scale
        forecast_for_dispatch = forecast_physical.detach() if detach_forecast_for_dispatch else forecast_physical
        physical_features = self.core._raw_physical_features(forecast_for_dispatch, inputs["scheduler_context"])
        controls, dispatch = self._schedule(physical_features, state, inputs["previous_chp"])
        return FormalV43ForwardOutput(
            forecast_normalized=forecast_normalized,
            forecast_physical=forecast_physical,
            regime_logits=raw.regime_logits,
            regime_probabilities=raw.regime_probabilities,
            thermal_magnitudes=thermal_magnitudes,
            controls=controls,
            dispatch=dispatch,
        )


class DirectPolicyModel(_FormalV4Base):
    """Decision-only comparator with no supervised forecast output or loss."""

    FEASIBILITY_ADAPTER_ID = "state_conditioned_chp_ramp_projection_v1"

    def __init__(self, *, decoder_parameters: Mapping[str, Any] | None = None, dropout: float = 0.0, **normalization: Any) -> None:
        super().__init__(decoder_parameters=decoder_parameters, dropout=dropout, **normalization)
        self.planning_head = nn.Sequential(nn.Linear(32 + 6, 64), nn.GELU(), nn.Linear(64, 3))

    def _project_planning_demand(self, planning: Tensor, previous_chp: Tensor) -> Tensor:
        """Project latent demand onto the decoder's CHP-ramp-feasible set.

        Direct-Policy has no forecast bottleneck, so its randomly initialized
        demand head can output values close to zero while the carried CHP state
        is still near its previous operating point.  With no export variable in
        the frozen 21-variable IES topology, that combination would make the
        physical decoder's ramp interval empty.  This projection is a
        differentiable, state-conditioned feasibility step: it raises only the
        electric-demand component needed to make every future ramp interval
        reachable, using the same cooling allocation bounds as the decoder.
        It does not use future realized load or an optimizer.
        """

        if planning.ndim != 3 or tuple(planning.shape[1:]) != (4, 3):
            raise ValueError("planning must have shape [B,4,3]")
        if previous_chp.ndim != 2 or previous_chp.shape != (planning.shape[0], 1):
            raise ValueError("previous_chp must have shape [B,1]")
        p = self.core.decoder_parameters
        ac_capacity = planning.new_tensor(float(p["absorption_chiller_capacity"]))
        cop_ac = planning.new_tensor(float(p["absorption_chiller_cop"]))
        boiler_capacity = planning.new_tensor(float(p["gas_boiler_capacity"]))
        ec_capacity = planning.new_tensor(float(p["electric_chiller_capacity"]))
        cop_ec = planning.new_tensor(float(p["electric_chiller_cop"]))
        chp_capacity = planning.new_tensor(float(p["chp_electric_capacity"]))
        ramp = planning.new_tensor(float(p["chp_ramp_fraction"]) * float(p["chp_electric_capacity"]))
        # The decoder performs its interval audit in float64 while the policy
        # head is float32.  Keep a tiny physical-unit margin so a boundary
        # projection cannot be overturned by the float32-to-float64 cast.
        projection_margin = planning.new_tensor(1.0e-3)

        demand_c = planning[..., 1]
        demand_h = planning[..., 2]
        qac_safe = torch.minimum(ac_capacity, cop_ac * (boiler_capacity - demand_h).clamp_min(0.0))
        cooling_served = torch.minimum(demand_c, ec_capacity + qac_safe)
        # The decoder's electric-chiller allocation is never below this value.
        lower_ec = (cooling_served - qac_safe).clamp_min(0.0)
        # This is a safe upper bound for the same allocation when bounding the
        # maximum CHP output that can be reached at the next step.
        upper_ec = torch.minimum(cooling_served, ec_capacity)

        projected_electric: list[Tensor] = []
        previous_bound = previous_chp.to(dtype=planning.dtype)[:, 0].clamp_min(0.0).clamp_max(chp_capacity)
        for index in range(4):
            lower_chp = (previous_bound - ramp).clamp_min(0.0)
            guaranteed_ec_power = lower_ec[:, index] / cop_ec
            electric_floor = (lower_chp - guaranteed_ec_power + projection_margin).clamp_min(0.0)
            current_electric = torch.maximum(planning[:, index, 0], electric_floor)
            projected_electric.append(current_electric)

            # Bound the largest CHP output the decoder could choose at this
            # step; the next step must be reachable from that bound as well.
            maximum_net_before_chp = current_electric + upper_ec[:, index] / cop_ec
            previous_bound = torch.minimum(
                torch.minimum(chp_capacity, previous_bound + ramp),
                maximum_net_before_chp,
            )
        projected = planning.clone()
        projected = torch.cat(
            (torch.stack(projected_electric, dim=1).unsqueeze(-1), projected[..., 1:]),
            dim=-1,
        )
        return projected

    def forward(self, **inputs: Tensor) -> FormalV4ForwardOutput:
        self._validate_common(inputs)
        state = self.state_encoder(inputs["device_history"], inputs["activity_history"])
        state_horizon = state.unsqueeze(1).expand(-1, 4, -1)
        planning = self.planning_head(torch.cat((state_horizon, inputs["scheduler_context"]), dim=-1))
        planning = F.softplus(planning)
        planning = self._project_planning_demand(planning, inputs["previous_chp"])
        gas_zeros = planning.new_zeros((*planning.shape[:2], 1))
        physical_features = self.core._raw_physical_features(torch.cat((planning, gas_zeros), dim=-1), inputs["scheduler_context"])
        controls, dispatch = self._schedule(physical_features, state, inputs["previous_chp"])
        return FormalV4ForwardOutput(None, None, planning, controls, dispatch)


__all__ = [
    "CausalStateDSTCNEncoder", "DirectPolicyModel", "FormalV4ForwardOutput", "FormalV43ForwardOutput",
    "RegimeAwareForecastHead", "RegimeAwareRSCPFModel", "RegimeForecastV43", "RSCPFModel",
]
