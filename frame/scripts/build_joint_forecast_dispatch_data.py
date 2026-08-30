"""Build or audit causal data artifacts for joint forecast--dispatch training.

The builder keeps future context origin-keyed: a PV/WT forecast produced at
one origin is never reused as if it were an observation at another origin.
Train/validation artifacts are generated before selection; the 2021 test
split is generated only after a complete selection receipt exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import numpy as np
from pathlib import Path
import platform
import sys
from typing import Sequence

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.contract import load_joint_training_contract


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/joint_forecast_dispatch_contract_v1.json"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--benchmark-solves", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _check_paths(contract) -> dict[str, object]:
    checks = {}
    for name, path in contract.paths.items():
        checks[name] = {"path": str(path), "exists": bool(Path(path).exists())}
    return checks


def _profile_parameters(ledger_values: dict[str, float]) -> dict[str, float]:
    """Return the normalized transparent PV/WT profile parameters."""

    return {
        "pv_rated_capacity": 1.0,
        "pv_reference_irradiance": ledger_values["pv_reference_irradiance"],
        "pv_conversion_efficiency": ledger_values["pv_conversion_efficiency"],
        "pv_reference_temperature": ledger_values["pv_reference_temperature"],
        "pv_temperature_coefficient": ledger_values["pv_temperature_coefficient"],
        "wt_rated_capacity": 1.0,
        "wt_cut_in_speed": ledger_values["wt_cut_in_speed"],
        "wt_rated_speed": ledger_values["wt_rated_speed"],
        "wt_cut_out_speed": ledger_values["wt_cut_out_speed"],
    }


def _read_ledger_values(path: Path) -> dict[str, float]:
    """Read numeric ledger values without applying the legacy track audit."""

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "parameter_id" not in rows.fieldnames or "value" not in rows.fieldnames:
            raise ValueError("parameter ledger must contain parameter_id and value columns")
        values = {}
        for row in rows:
            try:
                values[str(row["parameter_id"])] = float(row["value"])
            except (TypeError, ValueError):
                # Protocol rows such as ``calendar_year`` are intentionally
                # non-numeric and are irrelevant to the profile model.
                continue
    return values


def _first_step_objective(dispatch: dict[str, object], parameters: dict[str, object]) -> float:
    """Evaluate the LP teacher's first-hour objective in the frozen units."""

    def value(name: str) -> float:
        return float(np.asarray(dispatch[name], dtype=np.float64)[0])

    grid = value("grid")
    gas = value("g_chp") + value("g_gb")
    charge = value("p_charge") + value("p_discharge")
    slack = value("slack_e") + value("slack_c") + value("slack_h")
    grid_ef = float(parameters.get("grid_emission_factor", 0.0))
    gas_ef = float(parameters.get("gas_emission_factor", 0.0))
    def parameter_first(name: str, default: float = 0.0) -> float:
        raw = np.asarray(parameters.get(name, default), dtype=np.float64)
        return float(raw.reshape(-1)[0])

    carbon_price = parameter_first("carbon_price", parameter_first("carbon_price_default", 0.0))
    return float(
        grid * parameter_first("grid_energy_price")
        + gas * parameter_first("gas_energy_price")
        + charge * parameter_first("bess_throughput_cost")
        + carbon_price * (grid * grid_ef + gas * gas_ef)
        + slack * parameter_first("unserved_penalty")
    )


def _build_formal_splits(
    contract,
    output_dir: Path,
    *,
    resume: bool,
    smoke_limit: int | None,
    years: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020),
    split_names: tuple[str, ...] = ("train", "validation"),
) -> dict[str, object]:
    """Build requested windows with origin-specific LP teachers."""

    import numpy as np
    import pandas as pd
    import yaml

    from src.joint_dispatch.data import JointNormalization, build_joint_windows, save_joint_split
    from src.joint_dispatch.data import SCHEDULER_CONTEXT_ORDER
    from src.joint_dispatch.contract import DISPATCH_ORDER, EXOG_ORDER, TASK_ORDER
    from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical
    from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
    from src.scheduling.renewable_forecasts import fit_renewable_forecaster
    from src.scheduling.renewables import pv_available, wt_available

    paths = {name: output_dir / f"{name}.npz" for name in split_names}
    if not resume and any(path.exists() for path in paths.values()):
        raise SystemExit(f"refusing to overwrite existing joint data artifacts in {output_dir}")
    if resume and all(path.exists() for path in paths.values()):
        result = {
            "formal": True,
            "complete": True,
            "resumed": True,
            "history_source": "causal_lp",
            "origin_keyed_future_context": True,
        }
        for name, path in paths.items():
            result[f"{name}_windows"] = int(len(np.load(path, allow_pickle=False)["target_times"]))
        return result

    raw, source_metadata = read_kitakyushu_canonical(contract.path("kitakyushu_data_dir"), years=years)
    frame, cleaning = clean_kitakyushu_dataframe(raw)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("canonical frame timestamps must be unique and increasing")

    benchmark_payload = yaml.safe_load(contract.path("benchmark_path").read_text(encoding="utf-8"))
    parameters = dict(benchmark_payload["values"])
    ledger_values = _read_ledger_values(contract.path("parameter_ledger_path"))
    profile_parameters = _profile_parameters(ledger_values)
    profile = pd.DataFrame(
        {
            "timestamp": timestamps,
            "pv_available": pv_available(frame, profile_parameters),
            "wt_available": wt_available(frame, profile_parameters),
        }
    )
    train_profile = profile[timestamps.dt.year.isin((2015, 2016, 2017, 2018, 2019))].reset_index(drop=True)
    validation_profile = profile[timestamps.dt.year == 2020].reset_index(drop=True)
    forecaster = fit_renewable_forecaster(train_profile, validation_profile)

    task_values = frame[list(TASK_ORDER)].to_numpy(dtype=np.float64)
    profile_values = profile[["pv_available", "wt_available"]].to_numpy(dtype=np.float64)
    all_times = timestamps.to_numpy(dtype="datetime64[ns]")
    pv_scale = float(parameters["pv_capacity"])
    wt_scale = float(parameters["wt_capacity"])
    horizon = 4
    state_soc = 0.5
    state_chp = 0.0
    device_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    teacher_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []

    def forecast_at(index: int) -> np.ndarray:
        # The target at ``index`` is generated only from rows before index.
        history = profile.iloc[max(0, index - max(168, forecaster.minimum_history_hours)) : index].reset_index(drop=True)
        if len(history) < forecaster.minimum_history_hours:
            seed = profile_values[max(0, index - 1)] if index > 0 else profile_values[0]
            return np.repeat(seed.reshape(1, 2), horizon, axis=0)
        return np.asarray(forecaster.predict(history, horizon), dtype=np.float64)

    def load_forecast_at(index: int) -> np.ndarray:
        # Seasonal-naive load forecast uses the 24-hour lag and never reads the
        # target row.  Early rows are not used as formal windows but receive a
        # finite fallback so the device trajectory remains complete.
        if index >= 23:
            return task_values[index - 23 : index + 1, :3][:horizon]
        return np.repeat(task_values[max(0, index), :3].reshape(1, 3), horizon, axis=0)

    def solve_teacher(index: int):
        renewable = forecast_at(index)
        demand = load_forecast_at(index)
        dispatch_parameters = {
            **parameters,
            "grid_energy_price": np.full(horizon, float(parameters["grid_energy_price"])),
            "gas_energy_price": np.full(horizon, float(parameters["gas_energy_price"])),
            "carbon_price": np.full(horizon, float(parameters.get("carbon_price_default", 0.0))),
        }
        result = solve_dispatch_lp(
            DispatchInputs(
                demand=demand,
                pv_available=np.maximum(renewable[:, 0], 0.0) * pv_scale,
                wt_available=np.maximum(renewable[:, 1], 0.0) * wt_scale,
                parameters=dispatch_parameters,
                initial_soc=state_soc,
            )
        )
        if not result.success:
            raise RuntimeError(f"causal LP failed at {pd.Timestamp(all_times[index])}: {result.message}")
        return result, renewable, dispatch_parameters

    # Generate one causal LP action per available origin.  Full four-step LP
    # trajectories are retained as teacher labels; only the first action is
    # rolled into the historical device trajectory and SOC state.
    for index in range(max(0, len(frame) - horizon)):
        result, renewable, dispatch_parameters = solve_teacher(index)
        origin = pd.Timestamp(all_times[index])
        future_times = all_times[index : index + horizon]
        for name in DISPATCH_ORDER:
            values = np.asarray(result.values[name], dtype=np.float64)
            if values.shape != (horizon,) or not np.isfinite(values).all():
                raise ValueError(f"LP returned invalid {name} at {origin}")
        device_row = {"timestamp": origin}
        device_row.update({name: float(np.asarray(result.values[name])[0]) for name in DISPATCH_ORDER})
        device_rows.append(device_row)
        for step, target_time in enumerate(future_times):
            context_rows.append(
                {
                    "origin_timestamp": origin,
                    "timestamp": pd.Timestamp(target_time),
                    "pv_available": float(renewable[step, 0] * pv_scale),
                    "wt_available": float(renewable[step, 1] * wt_scale),
                    "grid_price": float(parameters["grid_energy_price"]),
                    "gas_price": float(parameters["gas_energy_price"]),
                    "carbon_price": float(parameters.get("carbon_price_default", 0.0)),
                    "initial_soc": float(state_soc),
                }
            )
            teacher_row = {"origin_timestamp": origin, "timestamp": pd.Timestamp(target_time)}
            teacher_row.update({name: float(np.asarray(result.values[name])[step]) for name in DISPATCH_ORDER})
            teacher_rows.append(teacher_row)
        oracle_rows.append(
            {
                "origin_timestamp": origin,
                "timestamp": origin,
                "oracle_first_step_objective": _first_step_objective(result.values, dispatch_parameters),
            }
        )
        state_soc = float(
            np.clip(float(np.asarray(result.values["soc"], dtype=np.float64)[0]) / float(parameters["bess_energy_capacity"]), 0.0, 1.0)
        )
        state_chp = float(np.asarray(result.values["p_chp"], dtype=np.float64)[0])

    device_frame = pd.DataFrame(device_rows, columns=("timestamp", *DISPATCH_ORDER))
    context_frame = pd.DataFrame(context_rows)
    teacher_frame = pd.DataFrame(teacher_rows)
    oracle_frame = pd.DataFrame(oracle_rows)
    if len(device_frame) < 28:
        raise ValueError("causal device trajectory is too short for 24-to-4 windows")
    common_metadata = {
        "schema_version": contract.schema_version,
        "history_source": "causal_lp",
        "lookback": 24,
        "horizon": 4,
        "renewable_forecaster": forecaster.selected_method,
        "renewable_validation_scores": dict(forecaster.validation_scores),
        "renewable_future_actuals_used_for_prediction": False,
        "origin_keyed_future_context": True,
        "device_trajectory": "offline_lp_first_action_with_soc_carry",
        "source_metadata": source_metadata,
        "cleaning": cleaning,
        "parameters_source": str(contract.path("benchmark_path")),
        "parameter_ledger_source": str(contract.path("parameter_ledger_path")),
        "test_year_loaded": bool(2021 in years),
    }
    results: dict[str, object] = {"formal": True, "complete": True, "resumed": False, **common_metadata}
    for split_name in split_names:
        split = build_joint_windows(
            frame.loc[:, ("timestamp", *TASK_ORDER, *EXOG_ORDER)],
            device_frame,
            context_frame,
            teacher_frame,
            oracle_frame,
            split=split_name,
            history_source="causal_lp",
        )
        if smoke_limit is not None:
            if smoke_limit <= 0:
                raise ValueError("smoke_limit must be positive")
            split = split.take(np.arange(min(smoke_limit, len(split)), dtype=np.int64))
        if len(split) == 0:
            raise ValueError(f"{split_name} produced no causal windows")
        if split_name == "train":
            normalization = JointNormalization.fit(split)
            save_joint_split(split, paths[split_name], normalization, {**common_metadata, "split": split_name})
        elif split_name == "test":
            # Test normalization is inherited from train and is never fitted
            # on 2021 values.  The runner reloads train.npz alongside test.
            save_joint_split(split, paths[split_name], None, {**common_metadata, "split": split_name})
        else:
            save_joint_split(split, paths[split_name], None, {**common_metadata, "split": split_name})
        results[f"{split_name}_windows"] = len(split)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_joint_training_contract(args.contract)
    output_dir = Path(args.output_dir) if args.output_dir is not None else contract.path("output_root") / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = _check_paths(contract)
    missing = [name for name, value in checks.items() if not value["exists"]]
    receipt = {
        "schema_version": contract.schema_version,
        "split": args.split,
        "formal": False,
        "complete": False,
        "dry_run": bool(args.dry_run),
        "smoke_limit": args.smoke_limit,
        "benchmark_requested": args.benchmark_solves,
        "resume": bool(args.resume),
        "python_version": platform.python_version(),
        "path_checks": checks,
        "missing_paths": missing,
    }
    if args.dry_run:
        (output_dir / f"dry_run_{args.split}.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0 if not missing else 2
    if args.benchmark_solves is not None:
        required = int(contract.resource_gate["benchmark_solves"])
        if args.benchmark_solves != required:
            raise SystemExit(
                "resource benchmark is fail-closed: pass exactly "
                f"{required} solves"
            )
        import numpy as np
        import yaml
        from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical
        from src.scheduling.dispatch_lp import DispatchInputs
        from src.joint_dispatch.data import benchmark_lp_generation

        years = (2015, 2016, 2017, 2018, 2019, 2020)
        raw, _ = read_kitakyushu_canonical(contract.path("kitakyushu_data_dir"), years=years)
        frame, _ = clean_kitakyushu_dataframe(raw)
        benchmark_payload = yaml.safe_load(contract.path("benchmark_path").read_text(encoding="utf-8"))
        parameters = dict(benchmark_payload["values"])
        if len(frame) < 28:
            raise SystemExit("not enough canonical hours for the 500-solve benchmark")
        valid_starts = np.arange(0, len(frame) - 4, dtype=int)
        selected = np.linspace(0, len(valid_starts) - 1, required, dtype=int)
        cases = []
        for start_index in valid_starts[selected]:
            demand = frame.loc[int(start_index) : int(start_index) + 3, ["electricity", "cooling", "heating"]].to_numpy(dtype=float)
            cases.append(DispatchInputs(demand=demand, pv_available=np.zeros(4), wt_available=np.zeros(4), parameters=parameters, initial_soc=0.5))
        receipt_obj = benchmark_lp_generation(
            cases,
            projected_total_solves=max(100_000, len(frame)),
            required_solves=required,
            max_projected_hours=float(contract.resource_gate["max_projected_p95_hours"]),
        )
        receipt.update({"formal": False, "complete": True, "resource_gate": receipt_obj.__dict__, "sample_count": required})
        (output_dir / "resource_benchmark_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    if args.split == "test":
        selection_path = output_dir.parent / "selection" / "selection_receipt.json"
        if not selection_path.exists():
            raise SystemExit("test data generation requires a frozen validation selection receipt")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not bool(selection.get("complete", False)) or bool(selection.get("test_set_accessed", True)):
            raise SystemExit("test data generation requires a complete uncontaminated selection receipt")
        result = _build_formal_splits(
            contract,
            output_dir,
            resume=bool(args.resume),
            smoke_limit=args.smoke_limit,
            years=(2015, 2016, 2017, 2018, 2019, 2020, 2021),
            split_names=("test",),
        )
        receipt.update(result)
        receipt["split"] = "test"
        receipt["selection_receipt"] = str(selection_path.resolve())
        receipt["test_year_loaded"] = True
        (output_dir / "data_build_test_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2, default=str))
        return 0
    result = _build_formal_splits(
        contract,
        output_dir,
        resume=bool(args.resume),
        smoke_limit=args.smoke_limit,
    )
    receipt.update(result)
    receipt["split"] = "train_validation"
    (output_dir / "data_build_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
