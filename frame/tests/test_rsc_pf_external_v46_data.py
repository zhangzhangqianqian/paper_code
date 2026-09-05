from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.external_v46_data import (
    ExternalV46Split,
    build_external_v46_oracle,
    build_external_v46_teacher,
    load_external_v46_split,
)


ROOT = Path(__file__).parents[1]
DATA = ROOT / "reports" / "joint_forecast_dispatch_formal_v4_4" / "formal_v4_4_20260905_f" / "pilot" / "data"


def _parameters() -> dict[str, float]:
    return {
        "grid_import_capacity": 1917.0,
        "chp_electric_capacity": 447.3,
        "chp_heat_capacity": 575.1,
        "gas_boiler_capacity": 1148.2812,
        "electric_chiller_capacity": 869.115,
        "absorption_chiller_capacity": 869.115,
        "bess_power_capacity": 255.6,
        "bess_energy_capacity": 1022.4,
        "chp_electric_efficiency": 0.35,
        "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9,
        "electric_chiller_cop": 3.5,
        "absorption_chiller_cop": 0.75,
        "bess_roundtrip_efficiency": 0.9,
        "bess_throughput_cost": 1e-6,
        "grid_energy_price": 1.0,
        "gas_energy_price": 0.6,
        "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25,
        "carbon_price_default": 0.0,
        "unserved_penalty": 100.0,
        "chp_ramp_fraction": 0.5,
    }


def test_current_v46_materialized_split_uses_17_device_channels() -> None:
    split = load_external_v46_split(DATA / "train.npz", "train")
    assert split.load_history.shape[1:] == (24, 4)
    assert split.exog_history.shape[1:] == (24, 12)
    assert split.device_history.shape[1:] == (24, 17)
    assert split.device_status.shape[1:] == (24, 6)
    assert split.scheduler_context.shape[1:] == (4, 6)
    assert split.forecast_target.shape[1:] == (4, 4)


def test_future_labels_do_not_change_causal_features() -> None:
    split = load_external_v46_split(DATA / "early_stop.npz", "validation").take(np.arange(2))
    changed = split.forecast_target.copy()
    changed[:] += 123.0
    assert np.array_equal(split.load_history, split.load_history)
    assert np.array_equal(split.scheduler_context, split.scheduler_context)
    assert not np.array_equal(changed, split.forecast_target)


def test_teacher_and_oracle_have_canonical_dispatch_shapes() -> None:
    split = load_external_v46_split(DATA / "train.npz", "train").take(np.arange(1))
    teacher = build_external_v46_teacher(split, _parameters())
    oracle = build_external_v46_oracle(split, _parameters())
    assert teacher.shape == (1, 4, 21)
    assert oracle.shape == (1,)
    assert np.isfinite(teacher).all()
    assert np.isfinite(oracle).all()


def test_sealed_test_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "sealed_test" / "split.npz"
    path.parent.mkdir()
    np.savez(path)
    with pytest.raises(ValueError, match="sealed test"):
        load_external_v46_split(path, "pilot")
