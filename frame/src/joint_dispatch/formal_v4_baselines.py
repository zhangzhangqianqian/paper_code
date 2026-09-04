"""Formal-v4 baseline identities and thin execution adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

from ..models import Scheme2RModel
from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from .formal_v4_models import DirectPolicyModel, RSCPFModel
from .reference_data import seasonal_naive_24h


@dataclass(frozen=True)
class Scheme2RPTOMetadata:
    """Original Scheme2R forecaster followed by one exact online LP call."""

    forecaster_class: str = "Scheme2RModel"
    uses_device_history: bool = False
    online_lp_calls_per_window: int = 1
    online_optimizer_calls_per_window: int = 1
    method_id: str = "Scheme2R-PTO"
    forecast_metrics_applicable: bool = True

    def __post_init__(self) -> None:
        if self.online_lp_calls_per_window != 1:
            raise ValueError("formal PTO uses exactly one LP call per window")

    def build_forecaster(self) -> Scheme2RModel:
        return Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StateConditionedPTO:
    stage_p_checkpoint_sha256: str
    forecaster_class: str = "StateConditionedScheme2R"
    uses_device_history: bool = True
    online_lp_calls_per_window: int = 1
    online_optimizer_calls_per_window: int = 1
    method_id: str = "State-Conditioned-PTO"
    forecast_metrics_applicable: bool = True

    def __post_init__(self) -> None:
        if not self.stage_p_checkpoint_sha256:
            raise ValueError("State-Conditioned-PTO requires a Stage P checkpoint hash")
        if self.online_lp_calls_per_window != 1:
            raise ValueError("formal PTO uses exactly one LP call per window")

    @property
    def forecaster_sha256(self) -> str:
        return self.stage_p_checkpoint_sha256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DirectPolicyAdapter:
    method_id = "Direct-Policy"
    forecast_metrics_applicable = False
    uses_realized_future = False
    online_lp_calls_per_window = 0
    online_optimizer_calls_per_window = 0

    def __init__(self, *, decoder_parameters: Mapping[str, Any] | None = None, dropout: float = 0.0) -> None:
        self.model = DirectPolicyModel(decoder_parameters=decoder_parameters, dropout=dropout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id, "forecast_metrics_applicable": self.forecast_metrics_applicable,
            "uses_realized_future": self.uses_realized_future, "online_lp_calls_per_window": self.online_lp_calls_per_window,
        }


class SeasonalNaivePTO:
    method_id = "Seasonal-Naive-PTO"
    forecast_metrics_applicable = True
    uses_realized_future = False
    online_lp_calls_per_window = 1
    online_optimizer_calls_per_window = 1

    @staticmethod
    def forecast(task_values: np.ndarray, *, origin_index: int, horizon: int = 4) -> np.ndarray:
        return seasonal_naive_24h(task_values, origin_index, horizon=horizon, task_count=3)

    def to_dict(self) -> dict[str, Any]:
        return {"method_id": self.method_id, "forecast_metrics_applicable": True, "online_lp_calls_per_window": 1}


class PerfectInformationMPC:
    """Non-deployable realized-future reference, not a competitive baseline."""

    method_id = "Perfect-Information-MPC"
    uses_realized_future = True
    deployable = False
    reference_only = True
    online_lp_calls_per_window = 1
    online_optimizer_calls_per_window = 1
    forecast_metrics_applicable = False

    @staticmethod
    def solve(realized_demand: np.ndarray, realized_renewable: np.ndarray, *, initial_soc: float, previous_chp: float, parameters: Mapping[str, Any]):
        result = solve_dispatch_lp(DispatchInputs(
            demand=np.asarray(realized_demand, dtype=np.float64),
            pv_available=np.asarray(realized_renewable, dtype=np.float64)[:, 0],
            wt_available=np.asarray(realized_renewable, dtype=np.float64)[:, 1],
            parameters=parameters, initial_soc=float(initial_soc), previous_chp=float(previous_chp),
        ))
        if not result.success:
            raise RuntimeError(f"perfect-information reference LP failed: {result.message}")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id, "uses_realized_future": True, "deployable": False,
            "reference_only": True, "online_lp_calls_per_window": 1,
        }


@dataclass(frozen=True)
class OfficialITransformerPTOMetadata:
    method_id: str = "Official iTransformer-PTO"
    forecaster_class: str = "model.iTransformer.Model"
    online_lp_calls_per_window: int = 1
    online_optimizer_calls_per_window: int = 1
    reproduction_level: str = "official_backbone_adaptation"
    forecast_metrics_applicable: bool = True

    def __post_init__(self) -> None:
        if self.online_lp_calls_per_window != 1:
            raise ValueError("formal PTO uses exactly one LP call per window")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DifferentiableLPMetadata:
    method_id: str = "Differentiable-LP"
    forecaster_class: str = "forecast-to-CVXPYlayers"
    online_lp_calls_per_window: int = 1
    online_optimizer_calls_per_window: int = 1
    optimizer_at_inference: bool = True
    reproduction_level: str = "cvxpylayers_method_adaptation"
    forecast_metrics_applicable: bool = True

    def __post_init__(self) -> None:
        if self.online_lp_calls_per_window != 1 or not self.optimizer_at_inference:
            raise ValueError("Differentiable-LP must disclose one optimizer call at inference")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_formal_v4_baseline(method_id: str, *, stage_p_checkpoint_sha256: str = "", decoder_parameters: Mapping[str, Any] | None = None, dropout: float = 0.0) -> Any:
    if method_id == "Scheme2R-PTO":
        return Scheme2RPTOMetadata()
    if method_id == "State-Conditioned-PTO":
        return StateConditionedPTO(stage_p_checkpoint_sha256=stage_p_checkpoint_sha256)
    if method_id == "Direct-Policy":
        return DirectPolicyAdapter(decoder_parameters=decoder_parameters, dropout=dropout)
    if method_id == "Seasonal-Naive-PTO":
        return SeasonalNaivePTO()
    if method_id == "Perfect-Information-MPC":
        return PerfectInformationMPC()
    if method_id == "Official iTransformer-PTO":
        return OfficialITransformerPTOMetadata()
    if method_id == "Differentiable-LP":
        return DifferentiableLPMetadata()
    raise ValueError(f"unknown formal-v4 baseline: {method_id}")


__all__ = [
    "DirectPolicyAdapter", "PerfectInformationMPC", "Scheme2RPTOMetadata", "SeasonalNaivePTO",
    "StateConditionedPTO", "OfficialITransformerPTOMetadata", "DifferentiableLPMetadata", "build_formal_v4_baseline",
]
