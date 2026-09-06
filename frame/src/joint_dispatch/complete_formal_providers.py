"""Provider registry for the complete formal RSC-PF experiment.

The registry is the only place where formal method identities are resolved into
causal action providers. It deliberately accepts only CausalOriginInput;
realized labels and sealed evaluation arrays are not part of the provider API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol, Tuple

import numpy as np

from .complete_formal_contract import CompleteFormalContract, MethodSeedKey, PRIMARY_METHOD_IDS
from .formal_v4_method_adapter import build_formal_v4_method_adapter
from .matched_closed_loop import CausalOriginInput
from .contract import DISPATCH_ORDER


DEPLOYABLE_METHOD_IDS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "State-Conditioned-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
)
REFERENCE_METHOD_IDS = ("Seasonal-Naive-PTO", "Perfect-Information-MPC")


@dataclass(frozen=True)
class CompletePlannedStep:
    forecast: Optional[np.ndarray]
    scheduler_demand: np.ndarray
    renewable_forecast: np.ndarray
    dispatch: np.ndarray
    inference_lp_calls: int

    def __post_init__(self) -> None:
        if self.forecast is not None:
            forecast = np.asarray(self.forecast, dtype=np.float64)
            if forecast.shape != (4, 4) or not np.isfinite(forecast).all():
                raise ValueError("forecast must have finite shape [4,4] or be None")
            object.__setattr__(self, "forecast", forecast)
        for name, shape in (
            ("scheduler_demand", (4, 4)),
            ("renewable_forecast", (4, 2)),
            ("dispatch", (4, len(DISPATCH_ORDER))),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must have finite shape {shape}")
            object.__setattr__(self, name, value)
        calls = int(self.inference_lp_calls)
        if calls < 0:
            raise ValueError("inference_lp_calls must be non-negative")
        object.__setattr__(self, "inference_lp_calls", calls)


class CompleteActionProvider(Protocol):
    method_id: str
    optimizer_role: str
    forecast_metrics_applicable: bool

    def plan(self, origin: CausalOriginInput) -> CompletePlannedStep:
        ...


ProviderFactory = Callable[[MethodSeedKey, Mapping[str, Any]], CompleteActionProvider]


def _origin_window(origin: CausalOriginInput) -> dict[str, Any]:
    """Project exactly the causal fields into legacy adapter shape.

    In particular, this function never adds RealizedOriginLabels or any future
    target to the mapping passed to a provider.
    """

    context = np.asarray(origin.scheduler_context[0], dtype=np.float64)
    return {
        "load_history": np.asarray(origin.load_history[0], dtype=np.float64).copy(),
        "exog_history": np.asarray(origin.exog_history[0], dtype=np.float64).copy(),
        "device_history": np.asarray(origin.device_history[0], dtype=np.float64).copy(),
        "activity_history": np.asarray(origin.activity_history[0], dtype=np.float64).copy(),
        "renewable_forecast": np.asarray(origin.renewable_forecast, dtype=np.float64).copy(),
        "prices_and_weights": context[:, 2:5].copy(),
        "initial_soc": np.asarray([context[0, 5]], dtype=np.float64),
        "previous_chp": np.asarray(origin.previous_chp[0], dtype=np.float64).copy(),
    }


def _four_task_demand(value: Any, forecast: Optional[np.ndarray]) -> np.ndarray:
    if value is None:
        if forecast is None:
            return np.zeros((4, 4), dtype=np.float64)
        return np.asarray(forecast, dtype=np.float64).copy()
    demand = np.asarray(value, dtype=np.float64)
    if demand.shape == (4, 3):
        demand = np.column_stack((demand, np.zeros(4, dtype=np.float64)))
    if demand.shape != (4, 4):
        raise ValueError("scheduler demand must have shape [4,4] or [4,3]")
    return demand


class _AdapterProvider:
    def __init__(self, method_id: str, adapter: Any) -> None:
        self.method_id = method_id
        self.adapter = adapter
        self.optimizer_role = (
            "none at inference"
            if int(getattr(adapter, "online_optimizer_calls_per_window", 0)) == 0
            else "exact optimizer at inference"
        )
        self.forecast_metrics_applicable = bool(
            getattr(adapter, "forecast_metrics_applicable", method_id != "Direct-Policy")
        )

    def plan(self, origin: CausalOriginInput) -> CompletePlannedStep:
        result = self.adapter.predict_and_dispatch(_origin_window(origin))
        if not isinstance(result, Mapping):
            raise TypeError(f"{self.method_id} adapter must return a mapping")
        forecast_value = result.get("forecast")
        forecast = None if forecast_value is None else np.asarray(forecast_value, dtype=np.float64)
        dispatch = result.get("dispatch", result.get("planned_dispatch"))
        if dispatch is None:
            raise ValueError(f"{self.method_id} adapter did not return dispatch")
        calls = int(result.get("optimizer_calls", result.get("inference_lp_calls", getattr(self.adapter, "online_optimizer_calls_per_window", 0))))
        return CompletePlannedStep(
            forecast=forecast,
            scheduler_demand=_four_task_demand(result.get("demand", result.get("scheduler_demand")), forecast),
            renewable_forecast=np.asarray(origin.renewable_forecast, dtype=np.float64).copy(),
            dispatch=np.asarray(dispatch, dtype=np.float64),
            inference_lp_calls=calls,
        )


class _ReferenceProvider:
    """Explicit placeholder for the non-deployable oracle reference."""

    method_id = "Perfect-Information-MPC"
    optimizer_role = "oracle_reference"
    forecast_metrics_applicable = False

    def plan(self, origin: CausalOriginInput) -> CompletePlannedStep:
        raise PermissionError(
            "Perfect-Information-MPC is a non-deployable reference and must be "
            "evaluated through its dedicated realized-information reference path"
        )


def _resource_provider(method_id: str, key: MethodSeedKey, resources: Mapping[str, Any]) -> Optional[CompleteActionProvider]:
    providers = resources.get("providers", {})
    if isinstance(providers, Mapping) and method_id in providers:
        candidate = providers[method_id]
        if callable(candidate) and not hasattr(candidate, "plan"):
            candidate = candidate(key, resources)
        return candidate
    adapters = resources.get("adapters", {})
    if isinstance(adapters, Mapping) and method_id in adapters:
        return _AdapterProvider(method_id, adapters[method_id])
    return None


def _factory_for(method_id: str) -> ProviderFactory:
    def factory(key: MethodSeedKey, resources: Mapping[str, Any]) -> CompleteActionProvider:
        supplied = _resource_provider(method_id, key, resources)
        if supplied is not None:
            return supplied
        if method_id == "Perfect-Information-MPC":
            return _ReferenceProvider()
        if method_id == "Differentiable-LP":
            difflp = resources.get("difflp")
            if isinstance(difflp, Mapping):
                from .complete_formal_difflp import build_difflp_provider
                return build_difflp_provider(**dict(difflp))
        parameters = resources.get("parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{method_id} requires explicit provider resources or parameters")
        all_kwargs = resources.get("adapter_kwargs", {})
        adapter_kwargs = dict(all_kwargs.get(method_id, {})) if isinstance(all_kwargs, Mapping) else {}
        adapter = build_formal_v4_method_adapter(method_id, parameters, **adapter_kwargs)
        return _AdapterProvider(method_id, adapter)

    return factory


class CompleteProviderRegistry:
    def __init__(self, factories: Optional[Mapping[str, ProviderFactory]] = None) -> None:
        self._factories: dict[str, ProviderFactory] = dict(factories or {})

    @classmethod
    def default(cls) -> "CompleteProviderRegistry":
        registry = cls()
        for method_id in PRIMARY_METHOD_IDS:
            registry.register(method_id, _factory_for(method_id))
        return registry

    def register(self, method_id: str, factory: ProviderFactory) -> None:
        if method_id not in PRIMARY_METHOD_IDS:
            raise ValueError(f"{method_id} is not a primary formal method")
        if not callable(factory):
            raise TypeError("provider factory must be callable")
        if method_id in self._factories:
            raise ValueError(f"{method_id} is already registered")
        self._factories[method_id] = factory

    def validate(self, contract: CompleteFormalContract) -> None:
        missing = [method_id for method_id in contract.primary_method_ids if method_id not in self._factories]
        if missing:
            raise ValueError(f"missing complete formal provider(s): {', '.join(missing)}")
        unexpected = sorted(set(self._factories) - set(contract.primary_method_ids))
        if unexpected:
            raise ValueError(f"registry contains non-contract provider(s): {', '.join(unexpected)}")

    def build(self, key: MethodSeedKey, resources: Mapping[str, Any]) -> CompleteActionProvider:
        if key.method_id not in self._factories:
            raise KeyError(f"no complete formal provider registered for {key.method_id}")
        spec = None
        contract_path = resources.get("contract_path")
        if contract_path is not None:
            spec = CompleteFormalContract.from_path(contract_path).method(key.method_id)
        if spec is not None:
            if spec.stochastic and key.seed is None:
                raise ValueError(f"{key.method_id} requires a seed")
            if not spec.stochastic and key.seed is not None:
                raise ValueError(f"{key.method_id} is deterministic and requires seed=None")
        provider = self._factories[key.method_id](key, resources)
        if not hasattr(provider, "plan") or getattr(provider, "method_id", key.method_id) != key.method_id:
            raise TypeError(f"provider for {key.method_id} does not implement the causal provider interface")
        if not hasattr(provider, "optimizer_role") or not hasattr(provider, "forecast_metrics_applicable"):
            raise TypeError(f"provider for {key.method_id} is missing formal metadata")
        return provider


def build_complete_provider_registry() -> CompleteProviderRegistry:
    return CompleteProviderRegistry.default()


__all__ = [
    "CompleteActionProvider",
    "CompletePlannedStep",
    "CompleteProviderRegistry",
    "DEPLOYABLE_METHOD_IDS",
    "REFERENCE_METHOD_IDS",
    "build_complete_provider_registry",
]
