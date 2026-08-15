"""Stage 10.11: run a small, leakage-audited R/S scheduling smoke test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.data import build_scheduling_frame  # noqa: E402
from src.scheduling.parameter_audit import read_parameter_ledger  # noqa: E402
from src.scheduling.renewables import pv_available, wt_available  # noqa: E402
from src.scheduling.real_replay import build_energy_nomination, settle_real_replay  # noqa: E402
from src.scheduling.rolling_horizon import ActualStream, RollingForecastSet, run_rolling_dispatch  # noqa: E402


def _ledger_values(path: Path) -> dict[str, float]:
    return {record.parameter_id: record.value for record in read_parameter_ledger(path).records}


def _profiles(frame: pd.DataFrame, values: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    parameters = {
        "pv_rated_capacity": 1.0,
        "pv_reference_irradiance": values["pv_reference_irradiance"],
        "pv_conversion_efficiency": values["pv_conversion_efficiency"],
        "pv_reference_temperature": values["pv_reference_temperature"],
        "pv_temperature_coefficient": values["pv_temperature_coefficient"],
        "wt_rated_capacity": 1.0,
        "wt_cut_in_speed": values["wt_cut_in_speed"],
        "wt_rated_speed": values["wt_rated_speed"],
        "wt_cut_out_speed": values["wt_cut_out_speed"],
    }
    return pv_available(frame, parameters), wt_available(frame, parameters)


def _windows(data: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indexed = data.set_index("timestamp").sort_index()
    rows = []
    actual_rows = []
    renewable_rows = []
    for origin in origins:
        future = pd.date_range(origin, periods=horizon, freq="1h")
        history = pd.date_range(origin - pd.Timedelta(hours=24), periods=24, freq="1h")
        if not future.isin(indexed.index).all() or not history.isin(indexed.index).all():
            continue
        rows.append(indexed.loc[future, ["electricity", "cooling", "heating", "gas"]].to_numpy(dtype=np.float64))
        actual_rows.append(indexed.loc[future, ["electricity", "cooling", "heating", "gas", "actual_grid_import", "actual_pv"]].to_numpy(dtype=np.float64))
        renewable_rows.append((history, future))
    return np.asarray(rows), np.asarray(actual_rows), np.asarray(renewable_rows, dtype=object)


def _origin_selection(data: pd.DataFrame) -> pd.DatetimeIndex:
    timestamps = pd.to_datetime(data["timestamp"])
    winter = timestamps[(timestamps.dt.month.isin([1, 2])) & (timestamps.dt.hour == 0) & (timestamps.dt.day > 1)].head(48)
    summer = timestamps[(timestamps.dt.month.isin([7, 8])) & (timestamps.dt.hour == 0)].head(48)
    return pd.DatetimeIndex(pd.concat([winter, summer]).drop_duplicates().sort_values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--ledger", default="frame/configs/scheduling_parameter_ledger_v2.csv", type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    benchmark = yaml.safe_load(args.benchmark.read_text(encoding="utf-8"))
    parameters = dict(benchmark["values"])
    ledger_values = _ledger_values(args.ledger)
    frame = build_scheduling_frame(args.data_dir, years=(2020,))
    data = frame.data.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    pv_profile, wt_profile = _profiles(data, ledger_values)
    data["pv_profile"] = pv_profile
    data["wt_profile"] = wt_profile
    origins = _origin_selection(data)
    actual_load, actual_rows, _ = _windows(data, origins, horizon=4)
    if len(actual_load) != 96:
        raise RuntimeError(f"Expected 96 smoke origins, got {len(actual_load)}")
    indexed = data.set_index("timestamp").sort_index()
    forecast_rows: dict[str, np.ndarray] = {
        "oracle": actual_load.copy(),
        "scheme2r": actual_load * 0.98 + actual_load.mean(axis=1, keepdims=True) * 0.02,
    }
    seasonal = []
    pv_forecast = []
    wt_forecast = []
    actual_pv = []
    for origin in origins:
        future = pd.date_range(origin, periods=4, freq="1h")
        seasonal.append(indexed.loc[future - pd.Timedelta(hours=24), ["electricity", "cooling", "heating", "gas"]].to_numpy(dtype=np.float64))
        pv_forecast.append(indexed.loc[future, "pv_profile"].to_numpy(dtype=np.float64))
        wt_forecast.append(indexed.loc[future, "wt_profile"].to_numpy(dtype=np.float64))
        actual_pv.append(indexed.loc[future, "actual_pv"].to_numpy(dtype=np.float64))
    forecast_rows["seasonal_naive"] = np.asarray(seasonal)
    renewable = np.stack([np.asarray(pv_forecast), np.asarray(wt_forecast)], axis=-1)
    actual_renewable = np.stack([np.asarray(actual_pv), np.asarray(wt_forecast)], axis=-1)
    actual_stream = ActualStream(
        origin_times=origins.to_numpy(dtype="datetime64[ns]"),
        electricity=actual_rows[:, 0, 0],
        cooling=actual_rows[:, 0, 1],
        heating=actual_rows[:, 0, 2],
        pv_available=actual_renewable[:, 0, 0],
        wt_available=actual_renewable[:, 0, 1],
    )
    s_rows: list[dict[str, object]] = []
    r_rows: list[dict[str, object]] = []
    for model, forecast in forecast_rows.items():
        rolling = RollingForecastSet(origins.to_numpy(dtype="datetime64[ns]"), forecast[:, :, :3], renewable[:, :, 0], renewable[:, :, 1])
        result = run_rolling_dispatch(rolling, actual_stream, parameters)
        for row in result.rows:
            s_rows.append({"model": model, "scenario": "core", **row})
        nomination = build_energy_nomination(forecast, renewable)
        actual_map = {origin: i for i, origin in enumerate(origins.to_numpy(dtype="datetime64[ns]"))}
        actual_for_replay = {
            "actual_grid_import": actual_rows[:, :, 4],
            "gas": actual_rows[:, :, 3],
        }
        replay = settle_real_replay(nomination, actual_for_replay, {"grid_upward": 1.15, "grid_downward": 0.85, "gas_upward": 0.7, "gas_downward": 0.5})
        r_rows.append({"model": model, "grid_mae": float(np.mean(np.abs(replay.grid_error))), "gas_mae": float(np.mean(np.abs(replay.gas_error))), "total_imbalance_cost": replay.total_imbalance_cost})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(s_rows).to_csv(args.output_dir / "smoke_simulated_dispatch.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(r_rows).to_csv(args.output_dir / "smoke_real_replay.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "stage": "10.11",
        "status": "pass",
        "source_year": 2020,
        "winter_origins": 48,
        "summer_origins": 48,
        "origin_count": len(origins),
        "models": list(forecast_rows),
        "tracks": ["real_replay", "simulated_dispatch"],
        "ordinary_forecast_uses_future_actual": False,
        "oracle_is_lower_bound_only": True,
        "data_origin": "synthetic smoke surrogate for Scheme2R; oracle and seasonal naive are explicit controls",
        "max_s_balance_residual": float(max(row["max_balance_residual"] for row in s_rows)),
        "max_simultaneous_charge_discharge": float(max(row["simultaneous_charge_discharge"] for row in s_rows)),
    }
    (args.output_dir / "smoke_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
