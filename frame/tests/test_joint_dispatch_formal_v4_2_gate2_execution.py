from __future__ import annotations

from pathlib import Path

import numpy as np

from src.joint_dispatch.formal_v4_2_gate2_execution import Gate2RowKey, execute_gate2_row
from src.joint_dispatch.formal_v4_2_gate2_training import Gate2DataBundle
from src.joint_dispatch.formal_v4_2_data import fit_train_normalization
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit


def _split() -> FormalV4WindowSplit:
    n = 4
    times = np.datetime64("2019-01-01T00:00") + np.arange(n).astype("timedelta64[h]")
    renew_hist = np.ones((n, 24, 2), dtype=np.float64)
    return FormalV4WindowSplit(
        load_history=np.ones((n, 24, 4)), exog_history=np.ones((n, 24, 12)),
        renewable_history=renew_hist, device_history=np.zeros((n, 24, 17)),
        activity_history=np.zeros((n, 24, 6)), forecast_target=np.ones((n, 4, 4)),
        rigid_demand=np.ones((n, 4, 3)), renewable_forecast=np.ones((n, 4, 2)),
        renewable_realized=np.ones((n, 4, 2)), prices_and_weights=np.ones((n, 4, 3)),
        initial_soc=np.full((n, 1), 0.5), previous_chp=np.zeros((n, 1)),
        target_times=times, trajectory_ids=np.asarray(["x"] * n),
        state_hashes=np.asarray([f"s-{i}" for i in range(n)]), split="selection",
    )


def _bundle() -> Gate2DataBundle:
    selection = _split()
    train = FormalV4WindowSplit(**{
        **selection.__dict__,
        "target_times": np.asarray(["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"], dtype="datetime64[ns]"),
        "split": "train",
    })
    normalization = fit_train_normalization(train)
    return Gate2DataBundle(train, selection, selection, normalization, *("1" * 64,) * 4)


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.2, "unserved_penalty": 100.0,
    "surplus_penalty": 0.1, "chp_ramp_fraction": 0.5,
}
LINEAGE = {
    "contract_sha256": "1" * 64, "source_manifest_sha256": "2" * 64,
    "train_manifest_sha256": "3" * 64, "calibration_manifest_sha256": "4" * 64,
    "evaluation_manifest_sha256": "5" * 64, "normalization_sha256": "6" * 64,
}


class _Method:
    method_id = "Seasonal-Naive-PTO"
    forecast_metrics_applicable = True

    def __call__(self, window, state):
        return {"dispatch": np.zeros((4, 21)), "forecast": np.asarray(window["forecast_target"])}


class _Direct(_Method):
    method_id = "Direct-Policy"
    forecast_metrics_applicable = False

    def __call__(self, window, state):
        return {"dispatch": np.zeros((4, 21))}


def test_row_is_computed_from_reopened_rollout(tmp_path: Path) -> None:
    row = execute_gate2_row(
        Gate2RowKey("Seasonal-Naive-PTO", None), _Method(), _bundle(), PARAMETERS,
        tmp_path, LINEAGE,
    )
    with np.load(tmp_path / "ROLLOUT.npz", allow_pickle=False) as saved:
        assert row["settled_hours"] == len(saved["settled_dispatch"])
    assert len(row["metrics_sha256"]) == 64


def test_direct_policy_forecast_is_not_applicable(tmp_path: Path) -> None:
    row = execute_gate2_row(
        Gate2RowKey("Direct-Policy", None), _Direct(), _bundle(), PARAMETERS,
        tmp_path, LINEAGE,
    )
    assert row["forecast_metrics_applicable"] is False
    assert row["forecast_metrics"] is None
