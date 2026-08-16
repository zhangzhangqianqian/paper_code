"""Stage 10.13: reproducible formal R/S scheduling execution.

The script consumes frozen prediction artifacts and a generated scheduling
contract. It never trains or selects a forecasting model. Each
model--seed--scenario is written to an isolated temporary directory and
atomically renamed only after its files pass validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.data import build_scheduling_frame  # noqa: E402
from src.scheduling.imbalance_settlement import summarize_replay  # noqa: E402
from src.scheduling.parameter_audit import read_parameter_ledger  # noqa: E402
from src.scheduling.real_replay import (  # noqa: E402
    build_energy_nomination,
    settle_first_step_replay,
    settle_real_replay,
)
from src.scheduling.renewables import pv_available, wt_available  # noqa: E402
from src.scheduling.rolling_horizon import (  # noqa: E402
    ActualStream,
    RollingForecastSet,
    run_rolling_dispatch,
)
from src.scheduling.recourse import evaluate_planned_first_step  # noqa: E402
from src.scheduling.oracle import run_perfect_information_oracle  # noqa: E402


def build_run_matrix(contract: dict[str, object]) -> list[dict[str, object]]:
    """Return the frozen R-first, then S scenario matrix."""

    models = list(contract["models"])
    seeds = [int(value) for value in contract["seeds"]]
    scenarios = list(contract["simulated_scenarios"])
    matrix: list[dict[str, object]] = []
    for model in models:
        for seed in seeds:
            matrix.append(
                {
                    "track": "real_replay",
                    "model": model,
                    "seed": seed,
                    "scenario": "real_replay",
                }
            )
    for model in models:
        for seed in seeds:
            for scenario in scenarios:
                matrix.append(
                    {
                        "track": "simulated_dispatch",
                        "model": model,
                        "seed": seed,
                        "scenario": scenario,
                    }
                )
    return matrix


def filter_run_matrix(
    matrix: list[dict[str, object]],
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    scenarios: list[str] | None = None,
) -> list[dict[str, object]]:
    """Apply explicit smoke filters without changing the frozen contract."""

    model_set = set(models) if models else None
    seed_set = {int(value) for value in seeds} if seeds else None
    scenario_set = set(scenarios) if scenarios else None
    filtered = []
    for row in matrix:
        if model_set is not None and str(row["model"]) not in model_set:
            continue
        if seed_set is not None and int(row["seed"]) not in seed_set:
            continue
        if scenario_set is not None and str(row["scenario"]) not in scenario_set:
            continue
        filtered.append(row)
    if not filtered:
        raise ValueError("run filters selected zero runs")
    return filtered


def _valid_contract(contract: dict[str, object], contract_path: Path) -> tuple[bool, str]:
    if contract.get("ordinary_future_actual_allowed") is not False:
        return False, "ordinary_future_actual_allowed must be false"
    if contract.get("no_test_tuning_after_freeze") is not True:
        return False, "no_test_tuning_after_freeze must be true"
    if int(contract.get("origin_count", 0)) <= 0:
        return False, "origin_count must be positive"
    if tuple(contract.get("task_order", ())) != ("electricity", "cooling", "heating", "gas"):
        return False, "task_order is not the frozen four-task order"
    real_track = contract.get("real_track", {})
    if not isinstance(real_track, Mapping):
        return False, "real_track freeze block is missing"
    try:
        real_pv_scale = float(real_track["pv_scale_train_only"])
    except (KeyError, TypeError, ValueError):
        return False, "real_track.pv_scale_train_only is missing"
    if not np.isfinite(real_pv_scale) or real_pv_scale <= 0:
        return False, "real_track.pv_scale_train_only must be finite and positive"
    if tuple(real_track.get("pv_scale_source_years", ())) != (2015, 2016, 2017, 2018, 2019):
        return False, "real PV scale must be frozen from 2015-2019"
    prices = real_track.get("settlement_prices")
    required_prices = ("grid_upward", "grid_downward", "gas_upward", "gas_downward")
    if not isinstance(prices, Mapping) or any(key not in prices for key in required_prices):
        return False, "real-track settlement prices are missing from the frozen contract"
    if any(not np.isfinite(float(prices[key])) or float(prices[key]) < 0.0 for key in required_prices):
        return False, "real-track settlement prices must be finite and non-negative"
    preflight = contract.get("preflight", {})
    if not isinstance(preflight, Mapping) or preflight.get("formal_dispatch_allowed") is not True:
        return False, "stage-10.11 preflight did not allow formal dispatch"
    sidecar = contract_path.with_suffix(contract_path.suffix + ".sha256")
    if not sidecar.exists():
        return False, "generated frozen contract SHA-256 sidecar is missing"
    digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if digest != sidecar.read_text(encoding="utf-8").strip():
        return False, "frozen contract SHA-256 mismatch"
    return True, "pass"


def _resolve(root: Path, value: str | Path) -> Path:
    """Resolve a configured path without assuming that ``root`` is the repo root.

    This runner lives below ``frame/`` while users normally invoke it from the
    repository root.  Accept both invocation styles for relative paths (for
    example ``frame/configs/...``), and retain the old ``root / path`` fallback
    for paths defined relative to the frame directory.
    """
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, root / path, root.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _load_benchmark(path: Path) -> dict[str, float]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = payload.get("values") if isinstance(payload, Mapping) else None
    if not isinstance(values, Mapping):
        raise ValueError(f"benchmark lacks values: {path}")
    result = {str(key): float(value) for key, value in values.items()}
    required = (
        "grid_import_capacity", "chp_electric_capacity", "chp_heat_capacity",
        "gas_boiler_capacity", "electric_chiller_capacity", "absorption_chiller_capacity",
        "bess_power_capacity", "bess_energy_capacity", "pv_capacity", "wt_capacity",
        "grid_emission_factor", "gas_emission_factor", "carbon_price_sensitivity",
        "carbon_price_default",
    )
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"benchmark lacks required frozen values: {missing}")
    return result


def _ledger_values(path: Path) -> dict[str, float]:
    ledger = read_parameter_ledger(path)
    return {record.parameter_id: float(record.value) for record in ledger.records}


def _profile_parameters(values: Mapping[str, float]) -> dict[str, float]:
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


def _load_renewables(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        predictions = np.asarray(payload["predictions"], dtype=np.float64)
        origins = np.asarray(payload["origin_times"], dtype="datetime64[ns]")
    if predictions.ndim != 3 or predictions.shape[1:] != (4, 2):
        raise ValueError(f"renewable predictions must have shape [N,4,2], got {predictions.shape}")
    if len(origins) != len(predictions) or len(np.unique(origins)) != len(origins):
        raise ValueError("renewable origins are not unique and aligned")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("renewable predictions must be finite and non-negative")
    return predictions, origins


def _load_model_prediction(contract: Mapping[str, object], repo_root: Path, model: str, seed: int) -> tuple[np.ndarray, np.ndarray, Path]:
    artifacts = contract.get("checked_prediction_artifacts", {})
    paths = artifacts.get(model) if isinstance(artifacts, Mapping) else None
    if not isinstance(paths, list):
        raise ValueError(f"contract has no checked artifacts for model={model}")
    needle = f"seed_{int(seed)}"
    matches = [
        _resolve(repo_root, str(value))
        for value in paths
        if Path(str(value)).parent.name == needle
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot locate unique frozen prediction for {model}/{seed}: {matches}")
    prediction_path = matches[0]
    manifest_path = prediction_path.parent / "run_manifest.json"
    if not prediction_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"frozen prediction artifact is incomplete: {prediction_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = tuple(manifest.get("tasks", ()))
    if tasks and tasks != ("electricity", "cooling", "heating", "gas"):
        raise ValueError(f"wrong task order in {manifest_path}: {tasks}")
    with np.load(prediction_path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
        origins = np.asarray(payload["target_times"], dtype="datetime64[ns]")
    if prediction.ndim != 3 or prediction.shape[1:] != (4, 4):
        raise ValueError(f"prediction must have shape [N,4,4], got {prediction.shape}")
    if len(origins) != len(prediction) or len(np.unique(origins)) != len(origins):
        raise ValueError(f"prediction origins are not unique and aligned: {prediction_path}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"prediction contains non-finite values: {prediction_path}")
    return np.maximum(prediction, 0.0), origins, prediction_path


def _window_values(data: pd.DataFrame, origins: np.ndarray, column: str, horizon: int) -> np.ndarray:
    ordered = data.sort_values("timestamp").drop_duplicates("timestamp")
    timestamps = pd.to_datetime(ordered["timestamp"]).to_numpy(dtype="datetime64[ns]")
    values = pd.to_numeric(ordered[column], errors="raise").to_numpy(dtype=np.float64)
    starts = np.searchsorted(timestamps, np.asarray(origins, dtype="datetime64[ns]"))
    offsets = np.arange(horizon, dtype=np.int64)[None, :]
    positions = starts[:, None] + offsets
    if np.any(starts >= len(timestamps)) or np.any(positions[:, -1] >= len(timestamps)):
        raise ValueError(f"{column} lacks a complete future window")
    observed = timestamps[positions]
    expected = np.asarray(origins, dtype="datetime64[ns]")[:, None] + offsets.astype("timedelta64[h]")
    if not np.all(observed == expected):
        raise ValueError(f"{column} has a non-contiguous window")
    return values[positions]


def _season(timestamp: pd.Timestamp) -> str:
    if timestamp.month in (12, 1, 2):
        return "winter"
    if timestamp.month in (3, 4, 5):
        return "spring"
    if timestamp.month in (6, 7, 8):
        return "summer"
    return "autumn"


def _fit_real_pv_scale(data: pd.DataFrame, profile_params: Mapping[str, float], source_years: tuple[int, ...]) -> float:
    timestamps = pd.to_datetime(data["timestamp"])
    train = data.loc[timestamps.dt.year.isin(source_years)].copy()
    profile = pv_available(train, profile_params)
    actual = pd.to_numeric(train["actual_pv"], errors="raise").to_numpy(dtype=np.float64)
    mask = profile > 1e-8
    if not np.any(mask):
        raise ValueError("training period has no positive normalized PV profile")
    scale = float(np.median(actual[mask] / profile[mask]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid training-only real PV scale: {scale}")
    return scale


def _scenario_parameters(base: Mapping[str, float], scenario: str) -> dict[str, float]:
    params = {str(key): float(value) for key, value in base.items()}
    params["carbon_price"] = params["carbon_price_default"]
    if scenario == "no_bess":
        params["bess_power_capacity"] = 0.0
        params["bess_energy_capacity"] = 0.0
    elif scenario == "no_chp":
        params["chp_electric_capacity"] = 0.0
        params["chp_heat_capacity"] = 0.0
    elif scenario == "carbon_price_sensitivity":
        params["carbon_price"] = params["carbon_price_sensitivity"]
    elif scenario not in {"core", "no_renewables", "single_hour"}:
        raise ValueError(f"unknown scheduling scenario: {scenario}")
    return params


def _run_real_replay(
    prediction: np.ndarray,
    origins: np.ndarray,
    renewable_norm: np.ndarray,
    renewable_origins: np.ndarray,
    data: pd.DataFrame,
    pv_scale: float,
    prices: Mapping[str, float],
) -> tuple[pd.DataFrame, dict[str, object]]:
    renewable_map = {origin: renewable_norm[i] for i, origin in enumerate(renewable_origins)}
    keep = np.asarray([origin in renewable_map for origin in origins], dtype=bool)
    if not np.any(keep):
        raise ValueError("no prediction origins overlap renewable forecasts")
    selected_origins = origins[keep]
    selected_prediction = prediction[keep]
    renewable = np.asarray([renewable_map[origin] for origin in selected_origins], dtype=np.float64)
    renewable[:, :, 0] *= pv_scale
    renewable[:, :, 1] = 0.0  # R track has no real WT settlement series.
    actual_grid = _window_values(data, selected_origins, "actual_grid_import", 4)
    actual_gas = _window_values(data, selected_origins, "gas", 4)
    window_result = settle_real_replay(
        build_energy_nomination(selected_prediction, renewable, selected_origins),
        {"actual_grid_import": actual_grid, "gas": actual_gas},
        prices,
    )
    executed_result = settle_first_step_replay(
        build_energy_nomination(selected_prediction, renewable, selected_origins),
        {"actual_grid_import": actual_grid, "gas": actual_gas},
        prices,
    )
    rows = []
    for index, origin in enumerate(selected_origins):
        rows.append(
            {
                "origin": str(origin),
                "season": _season(pd.Timestamp(origin)),
                "grid_mae_window": float(np.mean(np.abs(window_result.grid_error[index]))),
                "gas_mae_window": float(np.mean(np.abs(window_result.gas_error[index]))),
                "grid_mae_executed": float(np.mean(np.abs(executed_result.grid_error[index]))),
                "gas_mae_executed": float(np.mean(np.abs(executed_result.gas_error[index]))),
                "grid_error_executed": float(executed_result.grid_error[index, 0]),
                "gas_error_executed": float(executed_result.gas_error[index, 0]),
                "grid_upward_energy": float(max(executed_result.grid_error[index, 0], 0.0)),
                "grid_downward_energy": float(max(-executed_result.grid_error[index, 0], 0.0)),
                "gas_upward_energy": float(max(executed_result.gas_error[index, 0], 0.0)),
                "gas_downward_energy": float(max(-executed_result.gas_error[index, 0], 0.0)),
                "grid_imbalance_cost": float(max(executed_result.grid_error[index, 0], 0.0) * prices["grid_upward"] + max(-executed_result.grid_error[index, 0], 0.0) * prices["grid_downward"]),
                "gas_imbalance_cost": float(max(executed_result.gas_error[index, 0], 0.0) * prices["gas_upward"] + max(-executed_result.gas_error[index, 0], 0.0) * prices["gas_downward"]),
                "total_imbalance_cost": float(max(executed_result.grid_error[index, 0], 0.0) * prices["grid_upward"] + max(-executed_result.grid_error[index, 0], 0.0) * prices["grid_downward"] + max(executed_result.gas_error[index, 0], 0.0) * prices["gas_upward"] + max(-executed_result.gas_error[index, 0], 0.0) * prices["gas_downward"]),
            }
        )
    metrics = dict(summarize_replay(executed_result))
    metrics.update(
        {
            "grid_mae_window": float(np.mean(np.abs(window_result.grid_error))),
            "gas_mae_window": float(np.mean(np.abs(window_result.gas_error))),
            "grid_mae_executed": float(np.mean(np.abs(executed_result.grid_error))),
            "gas_mae_executed": float(np.mean(np.abs(executed_result.gas_error))),
        }
    )
    metrics.update({"n_windows": len(rows), "pv_scale_train_only": pv_scale})
    return pd.DataFrame(rows), metrics


def _run_simulated_dispatch(
    prediction: np.ndarray,
    origins: np.ndarray,
    renewable_norm: np.ndarray,
    renewable_origins: np.ndarray,
    data: pd.DataFrame,
    profile_params: Mapping[str, float],
    benchmark: Mapping[str, float],
    scenario: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    renewable_map = {origin: renewable_norm[i] for i, origin in enumerate(renewable_origins)}
    valid = np.asarray([origin in renewable_map for origin in origins], dtype=bool)
    if not np.all(valid):
        raise ValueError("model and renewable origins are not exactly aligned")
    horizon = 1 if scenario == "single_hour" else 4
    params = _scenario_parameters(benchmark, scenario)
    pv_capacity = params["pv_capacity"]
    wt_capacity = params["wt_capacity"]
    predicted_renewable = np.asarray([renewable_map[origin] for origin in origins], dtype=np.float64)
    predicted_renewable = predicted_renewable[:, :horizon, :].copy()
    predicted_renewable[:, :, 0] *= pv_capacity
    predicted_renewable[:, :, 1] *= wt_capacity
    test = data.copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    test = test.set_index("timestamp").sort_index()
    missing = [origin for origin in origins if origin not in test.index]
    if missing:
        raise ValueError(f"test data lacks {len(missing)} prediction origins")
    weather_pv = pv_available(data, profile_params) * pv_capacity
    weather_wt = wt_available(data, profile_params) * wt_capacity
    profile_by_time = pd.DataFrame(
        {"pv": weather_pv, "wt": weather_wt}, index=pd.to_datetime(data["timestamp"])
    )
    if scenario == "no_renewables":
        predicted_renewable[:] = 0.0
        profile_by_time.loc[:, ["pv", "wt"]] = 0.0
    actual_e = test.loc[origins, "electricity"].to_numpy(dtype=np.float64)
    actual_c = test.loc[origins, "cooling"].to_numpy(dtype=np.float64)
    actual_h = test.loc[origins, "heating"].to_numpy(dtype=np.float64)
    actual_pv = profile_by_time.loc[origins, "pv"].to_numpy(dtype=np.float64)
    actual_wt = profile_by_time.loc[origins, "wt"].to_numpy(dtype=np.float64)
    if scenario == "no_renewables":
        actual_pv[:] = 0.0
        actual_wt[:] = 0.0
    forecast_set = RollingForecastSet(
        origin_times=origins,
        demand=prediction[:, :horizon, :3],
        pv_available=predicted_renewable[:, :, 0],
        wt_available=predicted_renewable[:, :, 1],
    )
    actual_stream = ActualStream(
        origin_times=origins,
        electricity=actual_e,
        cooling=actual_c,
        heating=actual_h,
        pv_available=actual_pv,
        wt_available=actual_wt,
    )
    started = time.perf_counter()
    result = run_rolling_dispatch(forecast_set, actual_stream, params, initial_soc=0.5)
    elapsed = float(time.perf_counter() - started)
    rows = []
    for row, plan, realized in zip(result.rows, result.plans, result.realized_steps):
        planned_first = evaluate_planned_first_step(plan, params)
        realized_carbon = float(
            realized.realized_grid * params["grid_emission_factor"]
            + realized.realized_gas * params["gas_emission_factor"]
        )
        row_out = dict(row)
        row_out.update(
            {
                "horizon_objective": float(row.get("horizon_objective", plan.objective)),
                "planned_cost_first_step": float(planned_first["planned_cost_first_step"]),
                "planning_deviation_cost": float(realized.realized_cost - planned_first["planned_cost_first_step"]),
                "planned_carbon_first_step": float(planned_first["planned_carbon_first_step"]),
                "realized_carbon_first_step": realized_carbon,
                "planned_curtailment_first_step": float(planned_first["planned_curtailment_first_step"]),
                "realized_curtailment_first_step": float(realized.curtailment),
                "scenario_horizon": horizon,
            }
        )
        rows.append(row_out)
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number])
    metrics: dict[str, object] = {"n_windows": len(frame), "elapsed_seconds": elapsed}
    for column in numeric.columns:
        metrics[f"{column}_mean"] = float(numeric[column].mean())
        metrics[f"{column}_std"] = float(numeric[column].std(ddof=1)) if len(numeric) > 1 else 0.0
        metrics[f"{column}_p95"] = float(numeric[column].quantile(0.95))
    return frame, metrics


def _complete_run(final_dir: Path) -> bool:
    manifest_path = final_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            return False
        for name, digest in manifest.get("file_sha256", {}).items():
            path = final_dir / name
            if not path.exists() or _sha256(path) != digest:
                return False
        return True
    except (OSError, ValueError, TypeError):
        return False


def _write_run_atomically(
    output_root: Path,
    run: Mapping[str, object],
    rows: pd.DataFrame,
    metrics: Mapping[str, object],
    source_path: Path | None,
) -> tuple[Path, dict[str, object]]:
    relative_dir = Path(str(run["track"])) / str(run["model"]) / f"seed_{int(run['seed'])}" / str(run["scenario"])
    final_dir = output_root / relative_dir
    if final_dir.exists():
        raise FileExistsError(f"run output already exists and is not resumable: {final_dir}")
    temporary = output_root / f".tmp_{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        rows_path = temporary / "rows.csv"
        summary_path = temporary / "summary.json"
        rows.to_csv(rows_path, index=False, encoding="utf-8-sig")
        _write_json(summary_path, dict(metrics))
        file_hashes = {"rows.csv": _sha256(rows_path), "summary.json": _sha256(summary_path)}
        manifest = {
            "schema_version": "scheduling-formal-run-v3",
            "status": "complete",
            "track": run["track"],
            "model": run["model"],
            "seed": int(run["seed"]),
            "scenario": run["scenario"],
            "source_prediction": str(source_path) if source_path else None,
            "n_rows": int(len(rows)),
            "file_sha256": file_hashes,
        }
        manifest_path = temporary / "run_manifest.json"
        _write_json(manifest_path, manifest)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(str(temporary), str(final_dir))
        return final_dir, manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_completed_summary(final_dir: Path) -> dict[str, object]:
    manifest = json.loads((final_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
    return {**{key: manifest[key] for key in ("track", "model", "seed", "scenario")}, **summary}


def _oracle_cache(
    output_root: Path,
    scenario: str,
    result: object,
    params: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Write one model-independent oracle table and return it keyed by origin."""

    oracle_dir = output_root / "oracle" / scenario
    rows_path = oracle_dir / "rows.csv"
    if rows_path.exists():
        table = pd.read_csv(rows_path)
        return {
            str(row["origin"]): {str(key): float(value) for key, value in row.items() if key != "origin"}
            for row in table.to_dict(orient="records")
        }
    oracle_rows: list[dict[str, float | str]] = []
    grid_factor = float(params.get("grid_emission_factor", 0.0))
    gas_factor = float(params.get("gas_emission_factor", 0.0))
    for row, realized in zip(result.rows, result.realized_steps):
        oracle_rows.append(
            {
                "origin": str(row["origin"]),
                "oracle_realized_cost": float(row["realized_cost"]),
                "oracle_realized_carbon": float(realized.realized_grid * grid_factor + realized.realized_gas * gas_factor),
                "oracle_unserved_electricity": float(realized.unserved_electricity),
                "oracle_unserved_cooling": float(realized.unserved_cooling),
                "oracle_unserved_heating": float(realized.unserved_heating),
                "oracle_soc": float(realized.soc_after_execution),
                "oracle_grid": float(realized.realized_grid),
                "oracle_gas": float(realized.realized_gas),
                "oracle_electric_chiller_electricity": float(realized.electric_chiller_electricity),
                "oracle_electric_chiller_cooling": float(realized.electric_chiller_cooling),
                "oracle_gas_boiler_heat": float(realized.gas_boiler_heat),
                "oracle_absorption_chiller_cooling": float(realized.absorption_chiller_cooling),
                "oracle_chp_electricity": float(realized.chp_electricity),
                "oracle_chp_heat": float(realized.chp_heat),
                "oracle_bess_charge": float(realized.bess_charge),
                "oracle_bess_discharge": float(realized.bess_discharge),
            }
        )
    oracle_dir.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(oracle_rows)
    table.to_csv(rows_path, index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": "scheduling-oracle-v1",
        "status": "complete",
        "scenario": scenario,
        "model_independent": True,
        "n_rows": int(len(table)),
        "planning_horizon_hours": int(result.plans[0].values["grid"].shape[0]) if result.plans else 0,
    }
    _write_json(oracle_dir / "summary.json", summary)
    _write_json(oracle_dir / "run_manifest.json", summary | {"files": ["rows.csv", "summary.json"]})
    return {
        str(row["origin"]): {str(key): float(value) for key, value in row.items() if key != "origin"}
        for row in oracle_rows
    }


def _attach_oracle_metrics(rows: pd.DataFrame, oracle_by_origin: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """Attach oracle columns and correct first-step regret to S-track rows."""

    output = rows.copy()
    if output.empty:
        return output
    oracle_records = [oracle_by_origin.get(str(origin)) for origin in output["origin"]]
    if any(record is None for record in oracle_records):
        raise ValueError("model and oracle origins are not exactly aligned")
    keys = tuple(oracle_records[0].keys())
    for key in keys:
        output[key] = [float(record[key]) for record in oracle_records]
    output["regret"] = output["realized_cost"] - output["oracle_realized_cost"]
    output["carbon_delta_vs_oracle"] = output["realized_carbon_first_step"] - output["oracle_realized_carbon"]
    return output


def _write_root_reports(output_root: Path, index_rows: list[dict[str, object]], status: str, failures: list[dict[str, object]], contract_path: Path, *, full_matrix: bool, oracle_scenarios: list[str]) -> None:
    pd.DataFrame(index_rows).to_csv(output_root / "metrics_by_run.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "track": row.get("track"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "scenario": row.get("scenario"),
                "run_directory": row.get("run_directory"),
                "status": row.get("status", "complete"),
            }
            for row in index_rows
        ]
    ).to_csv(output_root / "raw_result_index.csv", index=False, encoding="utf-8-sig")
    _write_json(
        output_root / "formal_scheduling_manifest.json",
        {
            "stage": "10.14",
            "schema_version": "scheduling-formal-root-v3",
            "status": status,
            "completed_runs": len(index_rows),
            "full_contract_matrix": bool(full_matrix),
            "oracle_scenarios": list(oracle_scenarios),
            "failure_count": len(failures),
            "failures": failures,
            "contract": str(contract_path),
            "output_files": ["metrics_by_run.csv", "raw_result_index.csv", "formal_scheduling_manifest.json"],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--data-dir", required=False, type=Path)
    parser.add_argument("--benchmark", required=False, type=Path)
    parser.add_argument("--ledger", required=False, type=Path)
    parser.add_argument("--renewable-file", required=False, type=Path)
    parser.add_argument("--repo-root", default=PROJECT_ROOT, type=Path)
    parser.add_argument("--output-dir", default="frame/reports/scheduling_v2/formal", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--models", nargs="+", default=None, help="Smoke-only model filter")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Smoke-only seed filter")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Smoke-only scenario filter")
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    full_matrix = build_run_matrix(contract)
    matrix = filter_run_matrix(full_matrix, args.models, args.seeds, args.scenarios)
    valid, reason = _valid_contract(contract, contract_path)
    if args.dry_run:
        payload = {
            "stage": "10.14",
            "status": "dry_run",
            "contract_valid": valid,
            "contract_reason": reason,
            "run_count": len(matrix),
            "full_contract_run_count": len(full_matrix),
            "tracks": {
                "real_replay": sum(row["track"] == "real_replay" for row in matrix),
                "simulated_dispatch": sum(row["track"] == "simulated_dispatch" for row in matrix),
            },
            "models": contract["models"],
            "seeds": contract["seeds"],
            "scenarios": contract["simulated_scenarios"],
            "resume": args.resume,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if not valid:
        raise RuntimeError(f"Formal scheduling blocked: {reason}")
    repo_root = args.repo_root.resolve()
    data_dir = (args.data_dir or Path(str(contract["data_directory"]))).resolve()
    benchmark_path = _resolve(repo_root, args.benchmark or contract["parameter_freeze"]["benchmark_path"])
    ledger_path = _resolve(repo_root, args.ledger or contract["parameter_freeze"]["ledger_path"])
    renewable_path = (args.renewable_file or (repo_root.parent / "renewable_forecasts_test" / "renewable_predictions_test.npz")).resolve()
    required_paths = {
        "data directory": data_dir,
        "benchmark": benchmark_path,
        "parameter ledger": ledger_path,
        "renewable forecast": renewable_path,
    }
    missing = [f"{label}: {path}" for label, path in required_paths.items() if not path.exists()]
    if missing:
        details = "\n".join(missing)
        raise FileNotFoundError(
            "formal scheduling requires data, benchmark, ledger and renewable forecast files.\n"
            f"Missing paths:\n{details}"
        )
    output_root = args.output_dir.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is non-empty; use a new directory or --resume: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark = _load_benchmark(benchmark_path)
    ledger = _ledger_values(ledger_path)
    profile_params = _profile_parameters(ledger)
    renewable_norm, renewable_origins = _load_renewables(renewable_path)
    source_years = tuple(int(value) for value in contract["parameter_freeze"]["source_years"])
    frame = build_scheduling_frame(data_dir, years=(*source_years, int(contract["test_year"])))
    timestamps = pd.to_datetime(frame.data["timestamp"])
    test_data = frame.data.loc[timestamps.dt.year == int(contract["test_year"])].copy()
    pv_scale = _fit_real_pv_scale(frame.data, profile_params, source_years)
    frozen_real_track = contract.get("real_track", {})
    frozen_pv_scale = float(frozen_real_track["pv_scale_train_only"])
    if not np.isclose(pv_scale, frozen_pv_scale, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            "training-only real PV scale differs from the frozen contract: "
            f"runtime={pv_scale}, frozen={frozen_pv_scale}"
        )
    # Use the frozen value after validation; do not silently re-derive a
    # parameter during the formal run.
    pv_scale = frozen_pv_scale
    frozen_prices = contract["real_track"]["settlement_prices"]
    prices = {key: float(frozen_prices[key]) for key in ("grid_upward", "grid_downward", "gas_upward", "gas_downward")}
    index_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    oracle_by_scenario: dict[str, dict[str, dict[str, float]]] = {}
    for number, run in enumerate(matrix, start=1):
        final_dir = output_root / str(run["track"]) / str(run["model"]) / f"seed_{int(run['seed'])}" / str(run["scenario"])
        if args.resume and _complete_run(final_dir):
            index_rows.append(_load_completed_summary(final_dir) | {"run_directory": str(final_dir), "status": "resumed"})
            print(json.dumps({"stage": "10.13", "status": "run_resumed", "completed": number, "total": len(matrix), "run_id": "/".join(str(run[key]) for key in ("track", "model", "seed", "scenario"))}, ensure_ascii=False), flush=True)
            continue
        if final_dir.exists():
            raise FileExistsError(f"existing incomplete run requires --resume or a new output directory: {final_dir}")
        try:
            prediction, origins, source_path = _load_model_prediction(contract, repo_root, str(run["model"]), int(run["seed"]))
            test_mask = np.asarray(pd.DatetimeIndex(origins).year == int(contract["test_year"]), dtype=bool)
            if int(test_mask.sum()) != int(contract["origin_count"]):
                raise ValueError(f"expected {contract['origin_count']} test origins, got {int(test_mask.sum())}")
            prediction = prediction[test_mask]
            origins = origins[test_mask]
            if run["track"] == "real_replay":
                rows, metrics = _run_real_replay(prediction, origins, renewable_norm, renewable_origins, test_data, pv_scale, prices)
            else:
                scenario = str(run["scenario"])
                if scenario not in oracle_by_scenario:
                    oracle_params = _scenario_parameters(benchmark, scenario)
                    oracle_horizon = 1 if scenario == "single_hour" else 4
                    test_timestamps = pd.to_datetime(test_data["timestamp"])
                    actual_pv = pd.Series(
                        pv_available(test_data, profile_params) * oracle_params["pv_capacity"],
                        index=test_timestamps,
                    )
                    actual_wt = pd.Series(
                        wt_available(test_data, profile_params) * oracle_params["wt_capacity"],
                        index=test_timestamps,
                    )
                    if scenario == "no_renewables":
                        actual_pv[:] = 0.0
                        actual_wt[:] = 0.0
                    oracle_result = run_perfect_information_oracle(
                        test_data,
                        origins,
                        actual_pv,
                        actual_wt,
                        oracle_params,
                        horizon=oracle_horizon,
                        initial_soc=0.5,
                    )
                    oracle_by_scenario[scenario] = _oracle_cache(output_root, scenario, oracle_result, oracle_params)
                rows, metrics = _run_simulated_dispatch(prediction, origins, renewable_norm, renewable_origins, test_data, profile_params, benchmark, str(run["scenario"]))
                rows = _attach_oracle_metrics(rows, oracle_by_scenario[scenario])
            rows.insert(0, "track", run["track"])
            rows.insert(1, "model", run["model"])
            rows.insert(2, "seed", int(run["seed"]))
            rows.insert(3, "scenario", run["scenario"])
            final_dir, run_manifest = _write_run_atomically(output_root, run, rows, metrics, source_path)
            index_rows.append({**run_manifest, **metrics, "run_directory": str(final_dir), "status": "complete"})
            _write_root_reports(output_root, index_rows, "running", failures, contract_path, full_matrix=len(matrix) == len(full_matrix), oracle_scenarios=sorted(oracle_by_scenario))
            print(json.dumps({"stage": "10.13", "status": "run_completed", "completed": number, "total": len(matrix), "run_id": "/".join(str(run[key]) for key in ("track", "model", "seed", "scenario"))}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failure = {**run, "error_type": type(exc).__name__, "error": str(exc)}
            failures.append(failure)
            _write_root_reports(output_root, index_rows, "failed", failures, contract_path, full_matrix=len(matrix) == len(full_matrix), oracle_scenarios=sorted(oracle_by_scenario))
            print(json.dumps({"stage": "10.13", "status": "run_failed", "completed": number - 1, "total": len(matrix), "run": run, "error": str(exc)}, ensure_ascii=False), flush=True)
            return 1
    _write_root_reports(output_root, index_rows, "complete", failures, contract_path, full_matrix=len(matrix) == len(full_matrix), oracle_scenarios=sorted(oracle_by_scenario))
    print(json.dumps({"stage": "10.13", "status": "complete", "completed_runs": len(index_rows), "total_runs": len(matrix), "output_dir": str(output_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
