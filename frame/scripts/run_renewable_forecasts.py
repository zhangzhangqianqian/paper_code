"""阶段10.4：生成透明可再生能源预测和独立 oracle。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.contracts import load_scheduling_contract  # noqa: E402
from src.scheduling.data import build_scheduling_frame  # noqa: E402
from src.scheduling.parameter_audit import read_parameter_ledger  # noqa: E402
from src.scheduling.renewable_forecasts import fit_renewable_forecaster  # noqa: E402
from src.scheduling.renewables import pv_available, wt_available  # noqa: E402


def _ledger_values(path: Path) -> dict[str, float]:
    ledger = read_parameter_ledger(path)
    return {record.parameter_id: record.value for record in ledger.records}


def _profile_parameters(values: dict[str, float]) -> dict[str, float]:
    # 容量在阶段10.7按训练期负荷尺度冻结；10.4只生成归一化可用曲线。
    return {
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


def _year_part(frame: pd.DataFrame, year: int) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    return frame.loc[timestamps.dt.year == year].copy().reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--ledger", default="frame/configs/scheduling_parameter_ledger_v2.csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    contract = load_scheduling_contract(args.contract)
    scheduling_frame = build_scheduling_frame(args.data_dir, years=(*contract.train_years, contract.validation_year, contract.test_year))
    values = _ledger_values(args.ledger)
    parameters = _profile_parameters(values)
    weather = scheduling_frame.data
    profiles = pd.DataFrame(
        {
            "timestamp": weather["timestamp"],
            "pv_available": pv_available(weather, parameters),
            "wt_available": wt_available(weather, parameters),
        }
    )
    train = profiles[pd.to_datetime(profiles["timestamp"]).dt.year.isin(contract.train_years)].reset_index(drop=True)
    validation = profiles[pd.to_datetime(profiles["timestamp"]).dt.year == contract.validation_year].reset_index(drop=True)
    test = profiles[pd.to_datetime(profiles["timestamp"]).dt.year == contract.test_year].reset_index(drop=True)
    forecaster = fit_renewable_forecaster(train, validation)

    all_profiles = profiles.reset_index(drop=True)
    timestamps = pd.to_datetime(all_profiles["timestamp"])
    all_values = all_profiles[["pv_available", "wt_available"]].to_numpy(dtype=np.float64)
    origins = []
    predictions = []
    oracle = []
    for index in np.flatnonzero(timestamps.dt.year.to_numpy() == contract.test_year):
        if index + contract.horizon_hours > len(all_profiles):
            continue
        origin = timestamps.iloc[index]
        history_start = index - forecaster.minimum_history_hours
        if history_start < 0:
            continue
        history = all_profiles.iloc[history_start:index].reset_index(drop=True)
        if len(history) < forecaster.minimum_history_hours:
            continue
        history_times = pd.to_datetime(history["timestamp"]).to_numpy()
        future_times = timestamps.iloc[index : index + contract.horizon_hours].to_numpy()
        if not np.all(np.diff(history_times) == np.timedelta64(1, "h")):
            continue
        if len(future_times) != contract.horizon_hours or not np.all(
            np.diff(future_times) == np.timedelta64(1, "h")
        ):
            continue
        # 预测只接收origin之前的历史；真实未来只写入独立oracle文件。
        predictions.append(forecaster.predict(history, contract.horizon_hours))
        oracle.append(all_values[index : index + contract.horizon_hours])
        origins.append(origin.to_datetime64())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "renewable_predictions_test.npz",
        predictions=np.asarray(predictions, dtype=np.float32),
        origin_times=np.asarray(origins, dtype="datetime64[ns]"),
    )
    np.savez_compressed(
        args.output_dir / "renewable_oracle_test.npz",
        actual_available=np.asarray(oracle, dtype=np.float32),
        origin_times=np.asarray(origins, dtype="datetime64[ns]"),
    )
    manifest = {
        "stage": "10.4",
        "status": "pass",
        "selected_method": forecaster.selected_method,
        "validation_scores": dict(forecaster.validation_scores),
        "train_years": list(contract.train_years),
        "validation_year": contract.validation_year,
        "test_year": contract.test_year,
        "test_prediction_shape": list(np.asarray(predictions).shape),
        "oracle_shape": list(np.asarray(oracle).shape),
        "ordinary_prediction_contains_actual_future": False,
        "profile_capacity_scale": "normalized_until_stage_10_7",
    }
    (args.output_dir / "renewable_forecast_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
