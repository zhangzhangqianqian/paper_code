from pathlib import Path

import numpy as np

from src.joint_dispatch.external_v46_data import load_external_v46_split
from src.joint_dispatch.pto import PTOForecasts, load_pto_cache, save_pto_cache, seasonal_naive_forecasts, solve_pto_windows


ROOT = Path(__file__).parents[1]
DATA = ROOT / "reports" / "joint_forecast_dispatch_formal_v4_4" / "formal_v4_4_20260905_f" / "pilot" / "data"


def _parameters() -> dict[str, float]:
    return {
        "grid_import_capacity": 1917.0, "chp_electric_capacity": 447.3,
        "chp_heat_capacity": 575.1, "gas_boiler_capacity": 1148.2812,
        "electric_chiller_capacity": 869.115, "absorption_chiller_capacity": 869.115,
        "bess_power_capacity": 255.6, "bess_energy_capacity": 1022.4,
        "chp_electric_efficiency": 0.35, "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
        "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9,
        "bess_throughput_cost": 1e-6, "grid_energy_price": 1.0,
        "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25, "carbon_price_default": 0.0,
        "unserved_penalty": 100.0, "chp_ramp_fraction": 0.5,
    }


def test_pto_runs_one_lp_per_window_and_roundtrips_cache(tmp_path: Path) -> None:
    split = load_external_v46_split(DATA / "early_stop.npz", "validation").take(np.arange(2))
    prediction = split.forecast_target.copy()
    cache = solve_pto_windows(PTOForecasts("test", prediction, split.forecast_target), split, _parameters())
    assert cache.dispatch.shape == (2, 4, 21)
    assert cache.success.shape == (2,)
    assert cache.offline_exact_lp_calls == 2
    path = save_pto_cache(tmp_path / "cache.npz", cache)
    restored = load_pto_cache(path)
    assert np.array_equal(restored.dispatch, cache.dispatch)
    assert restored.messages == cache.messages


def test_seasonal_naive_is_causal() -> None:
    history = np.arange(2 * 24 * 4, dtype=np.float32).reshape(2, 24, 4)
    forecast = seasonal_naive_forecasts(history)
    assert np.array_equal(forecast, history[:, :4, :])
