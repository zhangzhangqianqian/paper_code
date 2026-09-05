"""Isolated neural wrappers for the three frozen external baselines.

These wrappers share tensor contracts only.  They deliberately do not import
``StateConditionedScheme2R`` or the exact LP solver.  PTO and decision-focused
optimizer calls are performed by later evaluation code; direct-policy forward
is fully neural plus the canonical differentiable physical decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..scheduling.proxy_decoder import CONTROL_DIM, decode_feasible_dispatch
from .contract import DISPATCH_ORDER, EXOG_ORDER, TASK_ORDER
from .external_baseline_data import ExternalBaselineBatch


LOOKBACK = 24
HORIZON = 4
TASK_COUNT = len(TASK_ORDER)
EXOG_COUNT = len(EXOG_ORDER)
DISPATCH_COUNT = len(DISPATCH_ORDER)


def _finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be a finite torch tensor")


def _history_inputs(load_history: Tensor, exog_history: Tensor) -> Tensor:
    if load_history.ndim != 3 or tuple(load_history.shape[1:]) != (LOOKBACK, TASK_COUNT):
        raise ValueError(f"load_history must have shape [B,{LOOKBACK},{TASK_COUNT}]")
    if exog_history.ndim != 3 or tuple(exog_history.shape[1:]) != (LOOKBACK, EXOG_COUNT):
        raise ValueError(f"exog_history must have shape [B,{LOOKBACK},{EXOG_COUNT}]")
    if load_history.shape[0] != exog_history.shape[0]:
        raise ValueError("load_history and exog_history batch sizes must match")
    _finite(load_history, "load_history")
    _finite(exog_history, "exog_history")
    return torch.cat((load_history, exog_history), dim=-1)


class InvertedTokenForecaster(nn.Module):
    """Small faithful iTransformer core: variates are attention tokens."""

    def __init__(self, *, input_dim: int = TASK_COUNT + EXOG_COUNT, d_model: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        if input_dim <= 0 or d_model <= 0 or heads <= 0 or layers <= 0:
            raise ValueError("iTransformer dimensions must be positive")
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.token_projection = nn.Linear(LOOKBACK, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.forecast_head = nn.Linear(d_model, HORIZON)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3 or tuple(values.shape[1:]) != (LOOKBACK, self.input_dim):
            raise ValueError(f"values must have shape [B,{LOOKBACK},{self.input_dim}]")
        _finite(values, "values")
        # Inverted tokenization: [B,T,C] -> [B,C,T].  The first four
        # variate tokens are the forecast targets; exogenous tokens provide
        # cross-variate context but are not copied into the output.
        tokens = self.token_projection(values.transpose(1, 2))
        encoded = self.encoder(tokens)
        forecast = self.forecast_head(encoded[:, :TASK_COUNT, :])
        return forecast.transpose(1, 2).contiguous()


@dataclass(frozen=True)
class ExternalForecastOutput:
    forecast: Tensor
    exact_lp_calls: int = 0
    optimizer_role: str = "none at inference"
    details: Mapping[str, Tensor] = MappingProxyType({})

    def __post_init__(self) -> None:
        if self.forecast.ndim != 3 or tuple(self.forecast.shape[1:]) != (HORIZON, TASK_COUNT):
            raise ValueError(f"forecast must have shape [B,{HORIZON},{TASK_COUNT}]")
        _finite(self.forecast, "forecast")
        if int(self.exact_lp_calls) < 0:
            raise ValueError("exact_lp_calls must be non-negative")


@dataclass(frozen=True)
class ExternalPolicyOutput:
    forecast_physical: Tensor
    control_logits: Tensor
    dispatch: Tensor
    exact_lp_calls: int = 0
    optimizer_role: str = "none at inference"

    def __post_init__(self) -> None:
        if self.forecast_physical.ndim != 3 or tuple(self.forecast_physical.shape[1:]) != (HORIZON, TASK_COUNT):
            raise ValueError(f"forecast_physical must have shape [B,{HORIZON},{TASK_COUNT}]")
        if self.control_logits.ndim != 2 or self.control_logits.shape[-1] != CONTROL_DIM:
            raise ValueError(f"control_logits must have shape [B,{CONTROL_DIM}]")
        if self.dispatch.ndim != 3 or tuple(self.dispatch.shape[1:]) != (HORIZON, DISPATCH_COUNT):
            raise ValueError(f"dispatch must have shape [B,{HORIZON},{DISPATCH_COUNT}]")
        if self.control_logits.shape[0] != self.dispatch.shape[0] or self.forecast_physical.shape[0] != self.dispatch.shape[0]:
            raise ValueError("policy output batch sizes must match")
        for name, value in (("forecast_physical", self.forecast_physical), ("control_logits", self.control_logits), ("dispatch", self.dispatch)):
            _finite(value, name)
        if int(self.exact_lp_calls) != 0:
            raise ValueError("direct policy inference must make zero exact LP calls")


class ExternalForecastPTO(nn.Module):
    """Official-iTransformer-shaped forecast adapter for the PTO slot."""

    method_id = "iTransformer-PTO"
    # The complete forecast-then-optimize method invokes one exact LP per
    # chronological origin; the neural forward itself remains optimizer-free.
    optimizer_role = "exact optimizer at inference"

    def __init__(self, *, d_model: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.backbone = InvertedTokenForecaster(d_model=d_model, heads=heads, layers=layers)

    def forward(self, load_history: Tensor, exog_history: Tensor) -> Tensor:
        return self.backbone(_history_inputs(load_history, exog_history))


class DecisionFocusedOnline(nn.Module):
    """Decision-focused forecast adapter with an explicit optimizer role.

    The downstream exact optimizer is intentionally not called in this
    differentiable forward pass.  Evaluation code invokes it according to the
    source paper's declared online path and records the calls separately.
    """

    method_id = "DecisionFocused-Online"
    optimizer_role = "exact optimizer at inference"

    def __init__(self, *, d_model: int = 64, heads: int = 4, layers: int = 2) -> None:
        super().__init__()
        self.backbone = InvertedTokenForecaster(d_model=d_model, heads=heads, layers=layers)

    def forward(self, batch_or_load_history: ExternalBaselineBatch | Tensor, exog_history: Tensor | None = None) -> ExternalForecastOutput:
        if isinstance(batch_or_load_history, ExternalBaselineBatch):
            forecast = self.backbone(_history_inputs(batch_or_load_history.load_history, batch_or_load_history.exog_history))
        else:
            if exog_history is None:
                raise ValueError("exog_history is required when passing tensors")
            forecast = self.backbone(_history_inputs(batch_or_load_history, exog_history))
        return ExternalForecastOutput(
            forecast=forecast,
            exact_lp_calls=0,
            optimizer_role=self.optimizer_role,
            details=MappingProxyType({"forecast_gradient": forecast}),
        )


def _default_decoder_parameters() -> dict[str, float]:
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


class DigitalTwinsPolicy(nn.Module):
    """Direct continuous dispatch policy followed by the physical decoder."""

    method_id = "DigitalTwins-Policy"
    optimizer_role = "none at inference"

    def __init__(self, *, hidden_dim: int = 128, decoder_parameters: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.decoder_parameters = MappingProxyType(dict(decoder_parameters or _default_decoder_parameters()))
        input_dim = LOOKBACK * (TASK_COUNT + EXOG_COUNT) + HORIZON * 6 + 1 + LOOKBACK * TASK_COUNT
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.forecast_head = nn.Linear(hidden_dim, HORIZON * TASK_COUNT)
        self.policy_head = nn.Linear(hidden_dim, CONTROL_DIM)

    @staticmethod
    def _batch_inputs(batch: ExternalBaselineBatch) -> tuple[Tensor, Tensor, Tensor]:
        if batch.split not in {"train", "validation", "pilot"}:
            raise ValueError("direct policy cannot run on the test split")
        inputs = _history_inputs(batch.load_history, batch.exog_history)
        # The causal innovation proxy is built from observed history only;
        # retaining all 24 steps makes its use auditable in the input ledger.
        innovation = torch.zeros_like(batch.load_history)
        innovation[:, 1:, :] = batch.load_history[:, 1:, :] - batch.load_history[:, :-1, :]
        context = batch.scheduler_context
        if not bool(torch.isfinite(context).all().item()):
            raise ValueError("scheduler_context must be finite")
        if not bool(torch.allclose(context[..., 5], context[:, :1, 5], atol=1.0e-6, rtol=0.0)):
            raise ValueError("initial_soc must be identical across horizon rows")
        return inputs, innovation, context

    def forward(self, batch: ExternalBaselineBatch) -> ExternalPolicyOutput:
        inputs, innovation, context = self._batch_inputs(batch)
        flat = torch.cat((inputs.reshape(inputs.shape[0], -1), context.reshape(context.shape[0], -1), batch.previous_chp, innovation.reshape(innovation.shape[0], -1)), dim=-1)
        hidden = self.input_projection(flat)
        forecast_physical = F.softplus(self.forecast_head(hidden).reshape(-1, HORIZON, TASK_COUNT))
        physical_features = torch.cat((forecast_physical, context), dim=-1)
        control_logits = self.policy_head(hidden)
        dispatch = decode_feasible_dispatch(control_logits, physical_features, self.decoder_parameters)
        return ExternalPolicyOutput(
            forecast_physical=forecast_physical,
            control_logits=control_logits,
            dispatch=dispatch,
            exact_lp_calls=0,
            optimizer_role=self.optimizer_role,
        )


def build_external_baseline(method_id: str, config: Mapping[str, Any] | None = None) -> nn.Module:
    """Build one frozen external method; unknown IDs fail closed."""

    options = dict(config or {})
    if method_id == "iTransformer-PTO":
        return ExternalForecastPTO(
            d_model=int(options.get("d_model", 64)), heads=int(options.get("heads", 4)), layers=int(options.get("layers", 2))
        )
    if method_id == "DecisionFocused-Online":
        return DecisionFocusedOnline(
            d_model=int(options.get("d_model", 64)), heads=int(options.get("heads", 4)), layers=int(options.get("layers", 2))
        )
    if method_id == "DigitalTwins-Policy":
        return DigitalTwinsPolicy(hidden_dim=int(options.get("hidden_dim", 128)), decoder_parameters=options.get("decoder_parameters"))
    raise ValueError(f"unknown external baseline: {method_id}")


__all__ = [
    "DecisionFocusedOnline", "DigitalTwinsPolicy", "ExternalForecastOutput", "ExternalForecastPTO",
    "ExternalPolicyOutput", "InvertedTokenForecaster", "build_external_baseline",
]
