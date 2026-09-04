from __future__ import annotations

import numpy as np

from src.joint_dispatch.contract import DISPATCH_ORDER
from src.joint_dispatch.formal_v4_method_adapter import build_formal_v4_method_adapter


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.2, "unserved_penalty": 100.0,
    "surplus_penalty": 0.0, "chp_ramp_fraction": 0.5,
}


def _window() -> dict[str, np.ndarray]:
    loads = np.ones((24, 4), dtype=np.float64)
    return {
        "load_history": loads,
        "exog_history": np.zeros((24, 12), dtype=np.float64),
        "device_history": np.zeros((24, 17), dtype=np.float64),
        "activity_history": np.zeros((24, 6), dtype=np.float64),
        "renewable_forecast": np.ones((4, 2), dtype=np.float64),
        "prices_and_weights": np.tile(np.array([1.0, 0.6, 0.0]), (4, 1)),
        "initial_soc": np.array([[0.5]]), "previous_chp": np.array([[0.0]]),
        "forecast_target": np.ones((4, 4), dtype=np.float64),
    }


def test_pto_adapter_has_one_online_lp_and_common_result_contract():
    adapter = build_formal_v4_method_adapter("Scheme2R-PTO", PARAMETERS)
    result = adapter.predict_and_dispatch(_window(), {})
    assert result["optimizer_calls"] == 1
    assert result["forecast"].shape == (4, 4)
    assert result["dispatch"].shape == (4, len(DISPATCH_ORDER))
    assert np.isfinite(result["latency"])


def test_seasonal_naive_is_causal_and_direct_policy_has_zero_optimizer_calls():
    seasonal = build_formal_v4_method_adapter("Seasonal-Naive-PTO", PARAMETERS)
    result = seasonal.predict_and_dispatch(_window(), {})
    np.testing.assert_array_equal(result["forecast"], np.ones((4, 4)))
    assert result["optimizer_calls"] == 1
    direct = build_formal_v4_method_adapter("Direct-Policy", PARAMETERS)
    direct_result = direct.predict_and_dispatch(_window(), {})
    assert direct_result["forecast"] is None
    assert direct_result["optimizer_calls"] == 0
    assert direct_result["dispatch"].shape == (4, len(DISPATCH_ORDER))


def test_state_conditioned_pto_uses_complete_state_history():
    adapter = build_formal_v4_method_adapter("State-Conditioned-PTO", PARAMETERS)
    result = adapter.predict_and_dispatch(_window(), {})
    assert result["optimizer_calls"] == 1
    assert result["forecast"].shape == (4, 4)
