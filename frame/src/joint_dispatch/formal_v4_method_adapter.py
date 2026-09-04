"""Executable baseline adapters under the formal-v4 common interface."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np
import torch

from ..models import Scheme2RModel
from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from .contract import DISPATCH_ORDER
from .formal_v4_models import DirectPolicyModel, RSCPFModel
from .formal_v4_diffopt import DifferentiableIESLayer
from .formal_v4_itransformer import OfficialITransformerAdapter


def _array(window: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in window:
        raise ValueError(f"baseline window is missing {name}")
    value = np.asarray(window[name], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must have shape {shape} and be finite")
    return value


@dataclass(frozen=True)
class MethodAdapterResult:
    forecast: np.ndarray | None
    dispatch: np.ndarray
    demand: np.ndarray
    target: np.ndarray | None
    next_state: Mapping[str, float]
    optimizer_calls: int
    latency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast": self.forecast,
            "dispatch": self.dispatch,
            "demand": self.demand,
            "target": self.target,
            "next_state": dict(self.next_state),
            "optimizer_calls": self.optimizer_calls,
            "latency": self.latency,
        }


# Protocol name used by the formal-v4 evaluation contract.
MethodStepResult = MethodAdapterResult


class _BaseAdapter:
    method_id = ""
    online_exact_lp = False
    online_optimizer_calls_per_window = 0
    forecast_metrics_applicable = True

    def __init__(self, parameters: Mapping[str, Any], *, task_mean: np.ndarray | None = None, task_scale: np.ndarray | None = None) -> None:
        self.parameters = dict(parameters)
        self.task_mean = np.zeros(4, dtype=np.float64) if task_mean is None else np.asarray(task_mean, dtype=np.float64)
        self.task_scale = np.ones(4, dtype=np.float64) if task_scale is None else np.asarray(task_scale, dtype=np.float64)
        if self.task_mean.shape != (4,) or self.task_scale.shape != (4,) or (self.task_scale <= 0).any():
            raise ValueError("task normalization vectors must have shape [4] and positive scale")

    def _window(self, window: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        load_history = _array(window, "load_history", (24, 4))
        exog_history = _array(window, "exog_history", (24, 12))
        device_history = _array(window, "device_history", (24, 17))
        activity_history = _array(window, "activity_history", (24, 6))
        if not np.isin(activity_history, (0.0, 1.0)).all():
            raise ValueError("activity_history must be binary")
        initial_soc = float(np.asarray(window.get("initial_soc", [0.5]), dtype=np.float64).reshape(-1)[0])
        previous_chp = float(np.asarray(window.get("previous_chp", [0.0]), dtype=np.float64).reshape(-1)[0])
        if not 0.0 <= initial_soc <= 1.0 or previous_chp < 0.0:
            raise ValueError("rolling state is invalid")
        return load_history, exog_history, device_history, activity_history, initial_soc, previous_chp

    def _context(self, window: Mapping[str, Any], initial_soc: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        renewable = _array(window, "renewable_forecast", (4, 2))
        prices = _array(window, "prices_and_weights", (4, 3))
        return renewable, prices, np.concatenate((renewable, prices, np.full((4, 1), initial_soc)), axis=-1)

    def _result(self, started: float, forecast: np.ndarray | None, dispatch: np.ndarray, demand: np.ndarray, target: np.ndarray | None, next_state: Mapping[str, float], calls: int) -> MethodAdapterResult:
        return MethodAdapterResult(forecast, dispatch, demand, target, next_state, calls, time.perf_counter() - started)

    def _next_state(self, dispatch: np.ndarray) -> dict[str, float]:
        soc_index = DISPATCH_ORDER.index("soc")
        chp_index = DISPATCH_ORDER.index("p_chp")
        return {"soc": float(dispatch[-1, soc_index] / self.parameters["bess_energy_capacity"]), "previous_chp": float(dispatch[-1, chp_index])}


class Scheme2RPTOAdapter(_BaseAdapter):
    method_id = "Scheme2R-PTO"
    online_exact_lp = True
    online_optimizer_calls_per_window = 1

    def __init__(self, parameters: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(parameters, **kwargs)
        self.model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0)

    def predict_and_dispatch(self, window: Mapping[str, Any], rolling_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        loads, exog, _, _, soc, previous = self._window(window)
        renew, prices, _ = self._context(window, soc)
        with torch.no_grad():
            raw = self.model(torch.as_tensor(loads[None], dtype=torch.float32), torch.as_tensor(exog[None], dtype=torch.float32))[0].cpu().numpy()
        forecast = np.maximum(self.task_mean + self.task_scale * raw, 0.0)
        context = dict(self.parameters); context["grid_energy_price"] = prices[:, 0]; context["gas_energy_price"] = prices[:, 1]; context["carbon_price"] = prices[:, 2]
        solved = solve_dispatch_lp(DispatchInputs(forecast[:, :3], renew[:, 0], renew[:, 1], context, soc, previous))
        if not solved.success:
            raise RuntimeError(f"{self.method_id} LP failed: {solved.message}")
        dispatch = np.stack([solved.values[name] for name in DISPATCH_ORDER], axis=-1)
        return self._result(started, forecast, dispatch, forecast[:, :3], window.get("forecast_target"), self._next_state(dispatch), 1).to_dict()


class StateConditionedPTOAdapter(Scheme2RPTOAdapter):
    method_id = "State-Conditioned-PTO"

    def __init__(self, parameters: Mapping[str, Any], **kwargs: Any) -> None:
        _BaseAdapter.__init__(self, parameters, **kwargs)
        self.model = RSCPFModel(decoder_parameters=parameters, dropout=0.0)

    def predict_and_dispatch(self, window: Mapping[str, Any], rolling_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        loads, exog, device, activity, soc, previous = self._window(window)
        renew, prices, context_features = self._context(window, soc)
        inputs = {
            "load_history": torch.as_tensor(loads[None], dtype=torch.float32), "exog_history": torch.as_tensor(exog[None], dtype=torch.float32),
            "device_history": torch.as_tensor(device[None], dtype=torch.float32), "activity_history": torch.as_tensor(activity[None], dtype=torch.float32),
            "scheduler_context": torch.as_tensor(context_features[None], dtype=torch.float32), "previous_chp": torch.tensor([[previous]], dtype=torch.float32),
        }
        with torch.no_grad():
            output = self.model(**inputs)
            forecast = output.forecast_physical[0].cpu().numpy()
        context = dict(self.parameters); context["grid_energy_price"] = prices[:, 0]; context["gas_energy_price"] = prices[:, 1]; context["carbon_price"] = prices[:, 2]
        solved = solve_dispatch_lp(DispatchInputs(forecast[:, :3], renew[:, 0], renew[:, 1], context, soc, previous))
        if not solved.success:
            raise RuntimeError(f"{self.method_id} LP failed: {solved.message}")
        dispatch = np.stack([solved.values[name] for name in DISPATCH_ORDER], axis=-1)
        return self._result(started, forecast, dispatch, forecast[:, :3], window.get("forecast_target"), self._next_state(dispatch), 1).to_dict()


class SeasonalNaivePTOAdapter(_BaseAdapter):
    method_id = "Seasonal-Naive-PTO"
    online_exact_lp = True
    online_optimizer_calls_per_window = 1

    def predict_and_dispatch(self, window: Mapping[str, Any], rolling_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        loads, _, _, _, soc, previous = self._window(window)
        renew, prices, _ = self._context(window, soc)
        forecast = np.repeat(loads[:4, :], 1, axis=0)
        context = dict(self.parameters); context["grid_energy_price"] = prices[:, 0]; context["gas_energy_price"] = prices[:, 1]; context["carbon_price"] = prices[:, 2]
        solved = solve_dispatch_lp(DispatchInputs(forecast[:, :3], renew[:, 0], renew[:, 1], context, soc, previous))
        if not solved.success:
            raise RuntimeError(f"{self.method_id} LP failed: {solved.message}")
        dispatch = np.stack([solved.values[name] for name in DISPATCH_ORDER], axis=-1)
        return self._result(started, forecast, dispatch, forecast[:, :3], window.get("forecast_target"), self._next_state(dispatch), 1).to_dict()


class DirectPolicyAdapter(_BaseAdapter):
    method_id = "Direct-Policy"
    online_exact_lp = False
    online_optimizer_calls_per_window = 0
    forecast_metrics_applicable = False

    def __init__(self, parameters: Mapping[str, Any], **kwargs: Any) -> None:
        super().__init__(parameters, **kwargs)
        self.model = DirectPolicyModel(decoder_parameters=parameters, dropout=0.0)

    def predict_and_dispatch(self, window: Mapping[str, Any], rolling_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        loads, exog, device, activity, soc, previous = self._window(window)
        renew, prices, context_features = self._context(window, soc)
        inputs = {
            "load_history": torch.as_tensor(loads[None], dtype=torch.float32), "exog_history": torch.as_tensor(exog[None], dtype=torch.float32),
            "device_history": torch.as_tensor(device[None], dtype=torch.float32), "activity_history": torch.as_tensor(activity[None], dtype=torch.float32),
            "scheduler_context": torch.as_tensor(context_features[None], dtype=torch.float32), "previous_chp": torch.tensor([[previous]], dtype=torch.float32),
        }
        with torch.no_grad():
            output = self.model(**inputs)
        dispatch = output.dispatch[0].cpu().numpy()
        demand = output.latent_planning_demand[0].cpu().numpy() if output.latent_planning_demand is not None else np.zeros((4, 3), dtype=np.float64)
        return self._result(started, None, dispatch, demand, window.get("forecast_target"), self._next_state(dispatch), 0).to_dict()


class OfficialITransformerPTOAdapter(Scheme2RPTOAdapter):
    """PTO adapter around the verified upstream THUML backbone."""

    method_id = "Official iTransformer-PTO"

    def __init__(self, parameters: Mapping[str, Any], *, forecaster: torch.nn.Module | None = None, **kwargs: Any) -> None:
        _BaseAdapter.__init__(self, parameters, **kwargs)
        if forecaster is None:
            raise ValueError("Official iTransformer-PTO requires a verified OfficialITransformerAdapter instance")
        if not isinstance(forecaster, OfficialITransformerAdapter) and not isinstance(forecaster, torch.nn.Module):
            raise TypeError("forecaster must be a verified torch module")
        self.model = forecaster

    def predict_and_dispatch(self, window: Mapping[str, Any], rolling_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        loads, _, _, _, soc, previous = self._window(window)
        renew, prices, _ = self._context(window, soc)
        with torch.no_grad():
            raw = self.model(torch.as_tensor(loads[None], dtype=torch.float32))[0].cpu().numpy()
        forecast = np.maximum(self.task_mean + self.task_scale * raw, 0.0)
        context = dict(self.parameters); context["grid_energy_price"] = prices[:, 0]; context["gas_energy_price"] = prices[:, 1]; context["carbon_price"] = prices[:, 2]
        solved = solve_dispatch_lp(DispatchInputs(forecast[:, :3], renew[:, 0], renew[:, 1], context, soc, previous))
        if not solved.success:
            raise RuntimeError(f"{self.method_id} LP failed: {solved.message}")
        dispatch = np.stack([solved.values[name] for name in DISPATCH_ORDER], axis=-1)
        return self._result(started, forecast, dispatch, forecast[:, :3], window.get("forecast_target"), self._next_state(dispatch), 1).to_dict()


class DifferentiableLPAdapter(_BaseAdapter):
    """Forecast-to-differentiable-LP adapter; optimizer call is explicit."""

    method_id = "Differentiable-LP"
    online_exact_lp = True
    online_optimizer_calls_per_window = 1

    def __init__(self, parameters: Mapping[str, Any], *, forecaster: torch.nn.Module | None = None, layer: DifferentiableIESLayer | None = None, **kwargs: Any) -> None:
        super().__init__(parameters, **kwargs)
        if forecaster is None or layer is None:
            raise ValueError("Differentiable-LP requires both a forecaster and a verified DifferentiableIESLayer")
        self.model, self.layer = forecaster, layer

    def predict_and_dispatch(self, window: Mapping[str, Any], rolling_state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        loads, exog, _, _, soc, previous = self._window(window)
        renew, prices, _ = self._context(window, soc)
        with torch.no_grad():
            raw = self.model(torch.as_tensor(loads[None], dtype=torch.float32), torch.as_tensor(exog[None], dtype=torch.float32))[0].cpu().numpy()
        forecast = np.maximum(self.task_mean + self.task_scale * raw, 0.0)
        p4 = np.column_stack((prices[:, 0], prices[:, 1], np.zeros(4), np.zeros(4)))
        with torch.no_grad():
            values = self.layer(torch.as_tensor(forecast[None, :, :3], dtype=torch.float64), torch.as_tensor(renew[None], dtype=torch.float64), torch.as_tensor(p4[None], dtype=torch.float64), torch.tensor([[soc]], dtype=torch.float64), torch.tensor([[previous]], dtype=torch.float64))[0].cpu().numpy()
        return self._result(started, forecast, values, forecast[:, :3], window.get("forecast_target"), self._next_state(values), 1).to_dict()


def build_formal_v4_method_adapter(method_id: str, parameters: Mapping[str, Any], **kwargs: Any) -> _BaseAdapter:
    builders = {
        "Scheme2R-PTO": Scheme2RPTOAdapter,
        "State-Conditioned-PTO": StateConditionedPTOAdapter,
        "Seasonal-Naive-PTO": SeasonalNaivePTOAdapter,
        "Direct-Policy": DirectPolicyAdapter,
        "Official iTransformer-PTO": OfficialITransformerPTOAdapter,
        "Differentiable-LP": DifferentiableLPAdapter,
    }
    try:
        return builders[method_id](parameters, **kwargs)
    except KeyError as exc:
        raise ValueError(f"no executable formal-v4 adapter is registered for {method_id}") from exc


__all__ = [
    "DirectPolicyAdapter", "MethodAdapterResult", "MethodStepResult", "Scheme2RPTOAdapter", "SeasonalNaivePTOAdapter",
    "StateConditionedPTOAdapter", "build_formal_v4_method_adapter",
]
