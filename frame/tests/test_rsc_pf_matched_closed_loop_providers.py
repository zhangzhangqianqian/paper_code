from types import SimpleNamespace

import numpy as np
import torch

from src.joint_dispatch.external_baselines import build_external_baseline
from src.joint_dispatch.matched_closed_loop import CausalOriginInput
from src.joint_dispatch.matched_closed_loop_providers import (
    DigitalTwinsProvider,
    PTOProvider,
    solve_pto_from_forecast,
)
from src.scheduling.dispatch_schema import VARIABLES


def _parameters() -> dict[str, float]:
    return {
        "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0,
        "chp_heat_capacity": 30.0, "gas_boiler_capacity": 40.0,
        "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
        "bess_power_capacity": 10.0, "bess_energy_capacity": 100.0,
        "chp_electric_efficiency": 0.35, "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
        "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9,
        "bess_throughput_cost": 1e-6, "grid_energy_price": 1.0,
        "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25, "carbon_price_default": 0.0,
        "unserved_penalty": 100.0, "chp_ramp_fraction": 0.5,
    }


def _origin() -> CausalOriginInput:
    return CausalOriginInput(
        load_history=np.zeros((1, 24, 4), dtype=np.float32),
        exog_history=np.zeros((1, 24, 12), dtype=np.float32),
        device_history=np.zeros((1, 24, 17), dtype=np.float32),
        activity_history=np.zeros((1, 24, 6), dtype=np.float32),
        scheduler_context=np.zeros((1, 4, 6), dtype=np.float32),
        previous_chp=np.zeros((1, 1), dtype=np.float32),
        renewable_forecast=np.zeros((4, 2), dtype=np.float32),
        origin_time=np.datetime64("2019-01-01T00"),
        trajectory_id="capacity_bound_causal",
    )


def test_predicted_gas_never_enters_pto_balance():
    origin = _origin()
    forecast = np.ones((4, 4), dtype=np.float64)
    changed = forecast.copy()
    changed[:, 3] += 1.0e6
    first = solve_pto_from_forecast("iTransformer-PTO", forecast, origin, _parameters())
    second = solve_pto_from_forecast("iTransformer-PTO", changed, origin, _parameters())
    assert first.scheduler_demand.shape == (4, 4)
    assert np.array_equal(first.dispatch, second.dispatch)


def test_pto_provider_declares_one_inference_lp_and_causal_model_call():
    model = build_external_baseline("iTransformer-PTO")
    normalization = SimpleNamespace(
        load_mean=np.zeros(4, dtype=np.float32), load_scale=np.ones(4, dtype=np.float32),
        exog_mean=np.zeros(12, dtype=np.float32), exog_scale=np.ones(12, dtype=np.float32),
        device_mean=np.zeros(17, dtype=np.float32), device_scale=np.ones(17, dtype=np.float32),
        scheduler_mean=np.zeros(6, dtype=np.float32), scheduler_scale=np.ones(6, dtype=np.float32),
    )
    provider = PTOProvider("iTransformer-PTO", model, normalization, normalized=True).bind_parameters(_parameters())
    planned = provider.plan(_origin())
    assert provider.optimizer_role == "exact optimizer at inference"
    assert planned.inference_lp_calls == 1


def test_digital_twins_provider_has_no_inference_lp():
    model = build_external_baseline("DigitalTwins-Policy", {"decoder_parameters": _parameters()})
    provider = DigitalTwinsProvider("DigitalTwins-Policy", model, None, normalized=False)
    planned = provider.plan(_origin())
    assert provider.optimizer_role == "none at inference"
    assert planned.inference_lp_calls == 0
    assert planned.dispatch.shape == (4, len(VARIABLES))
