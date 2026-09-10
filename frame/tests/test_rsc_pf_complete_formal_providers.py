from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract, MethodSeedKey
from src.joint_dispatch.complete_formal_providers import (
    CompletePlannedStep,
    DEPLOYABLE_METHOD_IDS,
    CompleteProviderRegistry,
    REFERENCE_METHOD_IDS,
    build_complete_provider_registry,
)
from src.joint_dispatch.matched_closed_loop import CausalOriginInput, PlannedStep
from src.joint_dispatch.contract import DISPATCH_ORDER


FRAME_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"


def _origin() -> CausalOriginInput:
    context = np.zeros((1, 4, 6), dtype=np.float32)
    context[:, :, 5] = 0.5
    return CausalOriginInput(
        load_history=np.zeros((1, 24, 4), dtype=np.float32),
        exog_history=np.zeros((1, 24, 12), dtype=np.float32),
        device_history=np.zeros((1, 24, 17), dtype=np.float32),
        activity_history=np.zeros((1, 24, 6), dtype=np.float32),
        scheduler_context=context,
        previous_chp=np.zeros((1, 1), dtype=np.float32),
        renewable_forecast=np.zeros((4, 2), dtype=np.float32),
        origin_time=np.datetime64("2019-01-01T00"),
        trajectory_id="capacity_bound_causal",
    )


class _CausalProbe:
    def __init__(self, method_id: str) -> None:
        self.method_id = method_id
        self.optimizer_role = "none at inference" if method_id in {"RSC-PF", "Decoupled-RSC-PF", "Direct-Policy"} else "exact optimizer at inference"
        self.forecast_metrics_applicable = method_id != "Direct-Policy"

    def plan(self, origin: CausalOriginInput) -> CompletePlannedStep:
        for forbidden in ("forecast_target", "renewable_realized", "realized_labels", "evaluation_year"):
            with pytest.raises(AttributeError):
                getattr(origin, forbidden)
        _ = origin.load_history, origin.device_history, origin.scheduler_context
        forecast = np.zeros((4, 4), dtype=np.float64)
        return CompletePlannedStep(
            forecast=forecast,
            scheduler_demand=np.zeros((4, 4), dtype=np.float64),
            renewable_forecast=origin.renewable_forecast,
            dispatch=np.zeros((4, len(DISPATCH_ORDER)), dtype=np.float64),
            inference_lp_calls=0 if self.method_id in {"RSC-PF", "Decoupled-RSC-PF", "Direct-Policy"} else 1,
        )


def fixture_resources():
    return {
        "contract_path": CONTRACT_PATH,
        "providers": {method_id: _CausalProbe(method_id) for method_id in DEPLOYABLE_METHOD_IDS},
    }


@pytest.fixture(scope="module")
def contract():
    return CompleteFormalContract.from_path(CONTRACT_PATH)


@pytest.fixture
def registry():
    return build_complete_provider_registry()


def test_registry_requires_every_primary_method(contract, registry):
    registry._factories.pop("Differentiable-LP")
    with pytest.raises(ValueError, match="Differentiable-LP"):
        registry.validate(contract)


@pytest.mark.parametrize("method_id", DEPLOYABLE_METHOD_IDS)
def test_provider_receives_only_causal_origin(method_id, registry):
    registry.validate(CompleteFormalContract.from_path(CONTRACT_PATH))
    provider = registry.build(MethodSeedKey(method_id, 2026), fixture_resources())
    step = provider.plan(_origin())
    assert isinstance(step, PlannedStep)
    assert step.dispatch.shape == (4, 21)
    assert step.forecast.shape == (4, 4)


def test_reference_row_is_registered_but_not_deployable(registry):
    contract = CompleteFormalContract.from_path(CONTRACT_PATH)
    registry.validate(contract)
    provider = registry.build(MethodSeedKey("Perfect-Information-MPC", None), fixture_resources())
    with pytest.raises(PermissionError, match="non-deployable"):
        provider.plan(_origin())


def test_secondary_diagnostics_are_not_aliased_into_primary_registry(registry):
    assert "DecisionFocused-Online" not in registry._factories
    assert "DigitalTwins-Policy" not in registry._factories
    with pytest.raises(ValueError, match="not a primary formal method"):
        registry.register("DigitalTwins-Policy", lambda key, resources: None)
