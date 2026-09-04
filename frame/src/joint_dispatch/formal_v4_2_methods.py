"""Executable method registry for the formal-v4.2 comparison matrix."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np
import torch

from ..scheduling.dispatch_schema import VARIABLES
from .formal_v4_2_contract import METHODS, FormalV42Contract
from .formal_v4_method_adapter import build_formal_v4_method_adapter
from .formal_v4_state import FormalV4ClosedLoopState


DEPLOYABLE_METHODS = (
    "RSC-PF", "Decoupled-RSC-PF", "Direct-Policy", "Scheme2R-PTO",
    "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP",
)
STOCHASTIC_METHODS = DEPLOYABLE_METHODS
DETERMINISTIC_METHODS = ("Perfect-Information-MPC", "Seasonal-Naive-PTO")


@dataclass(frozen=True)
class MethodStepV42:
    forecast: np.ndarray | None
    four_hour_plan: np.ndarray
    state_sha256: str
    optimizer_calls: int
    latency_seconds: float
    metadata: Mapping[str, Any]

    @property
    def dispatch(self) -> np.ndarray:
        return self.four_hour_plan


class FormalV42Method(Protocol):
    method_id: str
    deployable: bool

    def plan(self, window: Any, rolling_state: FormalV4ClosedLoopState) -> MethodStepV42: ...


@dataclass(frozen=True)
class MethodRowV42:
    method_id: str
    seed: int | None
    role: str
    deployable: bool


def _state_hash(state: FormalV4ClosedLoopState) -> str:
    digest = hashlib.sha256()
    digest.update(str(state.state_hash).encode("utf-8"))
    digest.update(np.ascontiguousarray(state.initial_soc.detach().cpu().numpy()).tobytes())
    digest.update(np.ascontiguousarray(state.previous_chp.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _window_mapping(window: Any, state: FormalV4ClosedLoopState) -> dict[str, Any]:
    if isinstance(window, Mapping):
        result = dict(window)
    else:
        result = {}
        for name in (
            "load_history", "exog_history", "device_history", "activity_history",
            "renewable_forecast", "renewable_realized", "prices_and_weights",
            "forecast_target", "target", "target_times", "target_time", "parameters",
        ):
            if hasattr(window, name):
                value = getattr(window, name)
                if isinstance(value, np.ndarray) and value.shape[0] == 1:
                    value = value[0]
                result[name] = value
        if hasattr(window, "realized_target"):
            result["realized_target"] = getattr(window, "realized_target")
    result["initial_soc"] = np.asarray([float(state.initial_soc[0, 0].detach().cpu())], dtype=np.float64)
    result["previous_chp"] = np.asarray([float(state.previous_chp[0, 0].detach().cpu())], dtype=np.float64)
    result["load_history"] = state.load_history[0].detach().cpu().numpy()
    result["exog_history"] = state.exog_history[0].detach().cpu().numpy()
    result["device_history"] = state.device_history[0].detach().cpu().numpy()
    result["activity_history"] = state.activity_history[0].detach().cpu().numpy()
    return result


def _safe_zero_plan() -> np.ndarray:
    return np.zeros((4, len(VARIABLES)), dtype=np.float64)


def _as_plan(result: Any) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    forecast = None; metadata: dict[str, Any] = {}
    if isinstance(result, Mapping):
        forecast = result.get("forecast", result.get("prediction"))
        metadata = {str(key): value for key, value in result.items() if key not in {"forecast", "prediction", "dispatch", "plan", "planned_dispatch"}}
        result = result.get("dispatch", result.get("plan", result.get("planned_dispatch")))
    elif hasattr(result, "dispatch"):
        forecast = getattr(result, "forecast", None)
        result = result.dispatch
    plan = np.asarray(result, dtype=np.float64)
    if plan.ndim == 3 and plan.shape[0] == 1:
        plan = plan[0]
    if plan.shape != (4, len(VARIABLES)) or not np.isfinite(plan).all():
        raise ValueError("method planner must return finite [4,21] dispatch")
    return plan, None if forecast is None else np.asarray(forecast, dtype=np.float64), metadata


class _RegisteredMethod:
    def __init__(self, method_id: str, *, seed: int | None = None, parameters: Mapping[str, Any] | None = None, planner: Callable[..., Any] | None = None, adapter: Any = None, checkpoint: Any = None, normalization: Any = None, deployable: bool = True) -> None:
        self.method_id = method_id
        self.seed = seed
        self.parameters = dict(parameters or {})
        self.planner = planner
        self.adapter = adapter
        self.checkpoint = checkpoint
        self.normalization = normalization
        self.deployable = bool(deployable)
        self.optimizer_calls = 1 if method_id in {"Scheme2R-PTO", "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP", "Perfect-Information-MPC", "Seasonal-Naive-PTO"} else 0
        self.forecast_metrics_applicable = method_id != "Direct-Policy" and method_id != "Perfect-Information-MPC"

    @property
    def forecaster_sha256(self) -> str:
        if isinstance(self.checkpoint, Mapping):
            return str(self.checkpoint.get("model_sha256", self.checkpoint.get("stage_p_checkpoint_sha256", "")))
        return str(getattr(self.checkpoint, "model_sha256", getattr(self.checkpoint, "stage_p_checkpoint_sha256", "")))

    def plan(self, window: Any, rolling_state: FormalV4ClosedLoopState) -> MethodStepV42:
        started = time.perf_counter()
        state_hash = _state_hash(rolling_state)
        mapped = _window_mapping(window, rolling_state)
        if self.planner is not None:
            try:
                result = self.planner(mapped, rolling_state)
            except TypeError:
                result = self.planner(mapped)
        elif self.adapter is not None:
            result = self.adapter.predict_and_dispatch(mapped, rolling_state)
        else:
            result = {"dispatch": _safe_zero_plan(), "forecast": None}
        plan, forecast, metadata = _as_plan(result)
        metadata.update({"method_id": self.method_id, "deployable": self.deployable, "seed": self.seed, "state_sha256": state_hash})
        return MethodStepV42(forecast, plan, state_hash, self.optimizer_calls, time.perf_counter() - started, metadata)


def build_v42_method(
    method_id: str,
    *,
    seed: int | None = None,
    checkpoint: Any = None,
    normalization: Any = None,
    parameters: Mapping[str, Any] | None = None,
    planner: Callable[..., Any] | None = None,
    adapter: Any = None,
    **kwargs: Any,
) -> _RegisteredMethod:
    """Construct one registered method without changing the frozen identity."""

    if method_id not in METHODS:
        raise ValueError(f"unknown formal-v4.2 method: {method_id}")
    if method_id == "Perfect-Information-MPC":
        return _RegisteredMethod(method_id, seed=None, parameters=parameters, planner=planner, adapter=adapter, checkpoint=checkpoint, normalization=normalization, deployable=False)
    if method_id == "Official iTransformer-PTO" and adapter is None and kwargs.get("forecaster") is not None:
        adapter = build_formal_v4_method_adapter(method_id, parameters or {}, forecaster=kwargs["forecaster"])
    elif method_id == "Differentiable-LP" and adapter is None and kwargs.get("forecaster") is not None and kwargs.get("layer") is not None:
        adapter = build_formal_v4_method_adapter(method_id, parameters or {}, forecaster=kwargs["forecaster"], layer=kwargs["layer"])
    elif adapter is None and planner is None and parameters:
        try:
            adapter = build_formal_v4_method_adapter(method_id, parameters)
        except (ValueError, TypeError):
            adapter = None
    return _RegisteredMethod(method_id, seed=seed, parameters=parameters, planner=planner, adapter=adapter, checkpoint=checkpoint, normalization=normalization, deployable=True)


def train_v42_baseline(method_id: str, *, seed: int, trainer: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Delegate baseline training while keeping method identity explicit."""

    if method_id not in METHODS:
        raise ValueError(f"unknown formal-v4.2 method: {method_id}")
    if trainer is None:
        return build_v42_method(method_id, seed=seed, **kwargs)
    return trainer(method_id=method_id, seed=int(seed), **kwargs)


def registered_method_rows(contract: FormalV42Contract | Mapping[str, Any], *, gate: str = "gate2") -> tuple[MethodRowV42, ...]:
    if isinstance(contract, FormalV42Contract):
        contract.validate()
        seeds = contract.gate2_seeds if gate == "gate2" else contract.gate3_seeds
    else:
        seeds = tuple(int(value) for value in contract.get("gate2_seeds", (2026, 2027, 2028))) if gate == "gate2" else tuple(int(value) for value in contract.get("gate3_seeds", (2026, 2027, 2028, 2029, 2030)))
    if gate not in {"gate2", "gate3"}:
        raise ValueError("method rows support gate2 or gate3")
    rows = [MethodRowV42(method_id, int(seed), "stochastic", True) for method_id in STOCHASTIC_METHODS for seed in seeds]
    rows.extend((MethodRowV42("Perfect-Information-MPC", None, "oracle_reference", False), MethodRowV42("Seasonal-Naive-PTO", None, "deterministic_pto", True)))
    return tuple(rows)


__all__ = [
    "DETERMINISTIC_METHODS", "DEPLOYABLE_METHODS", "FormalV42Method", "MethodRowV42", "MethodStepV42",
    "STOCHASTIC_METHODS", "build_v42_method", "registered_method_rows", "train_v42_baseline",
]
