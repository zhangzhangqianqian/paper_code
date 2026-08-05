"""Stage 7.5: formal external-baseline comparison.

The runner uses the frozen Stage 7 protocol.  Persistence and Seasonal Naive
are deterministic and therefore run once per protocol.  DLinear, MMoE-lite
and the minimal SOFTS adapter are trained for each frozen seed, giving
``2 * 2 + 3 * 2 * 5 = 34`` formal runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage7_3 import (  # noqa: E402
    FORMAL_BATCH_SIZE,
    FORMAL_MAX_EPOCHS,
    FORMAL_PATIENCE,
    FORMAL_PROTOCOLS,
    FORMAL_THREADS,
    HORIZON,
    LOOKBACK,
    _aggregate_rows,
    _read_json,
    _resolve,
    _seasonal_metrics,
    _validate_contract,
    _validate_freeze_and_training_policy,
    _write_csv,
)
from src.baselines import (  # noqa: E402
    persistence_forecast,
    regression_metrics,
    seasonal_naive_forecast,
)
from src.data_pipeline import build_protocol_windows, save_json  # noqa: E402
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.stage7_contract import EXPECTED_SEEDS  # noqa: E402


DETERMINISTIC_MODELS: Tuple[str, ...] = ("persistence", "seasonal_naive")
LEARNED_MODELS: Tuple[str, ...] = ("dlinear", "mmoe-lite", "softs")
ALL_MODELS: Tuple[str, ...] = DETERMINISTIC_MODELS + LEARNED_MODELS
MODEL_DIR_NAMES = {"mmoe-lite": "mmoe_lite"}


def _model_dir_name(model: str) -> str:
    return MODEL_DIR_NAMES.get(model, model)


def build_external_run_plan(
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> Tuple[Dict[str, object], ...]:
    """Return the immutable 34-run Stage 7.5 plan."""

    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != tuple(EXPECTED_SEEDS):
        raise ValueError(
            f"Stage 7.5 requires frozen seeds {EXPECTED_SEEDS}, got {frozen_seeds}"
        )
    rows: List[Dict[str, object]] = []
    for protocol in FORMAL_PROTOCOLS:
        for model in DETERMINISTIC_MODELS:
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "seed": None,
                    "seed_label": "deterministic",
                    "candidate_id": "deterministic",
                    "run_id": f"{protocol}/{model}/deterministic",
                }
            )
        for model in LEARNED_MODELS:
            for seed in frozen_seeds:
                rows.append(
                    {
                        "protocol": protocol,
                        "model": model,
                        "seed": seed,
                        "seed_label": f"seed_{seed}",
                        "candidate_id": "external_fixed",
                        "run_id": f"{protocol}/{model}/seed_{seed}",
                    }
                )
    return tuple(rows)


def _validate_external_scope(contract: Mapping[str, object]) -> None:
    scope = contract.get("stage7_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("stage7_scope is missing")
    expected = ("persistence", "seasonal_naive", "dlinear", "mmoe_lite", "softs")
    if tuple(scope.get("external_baselines", ())) != expected:
        raise ValueError(
            "Stage 7.5 external baseline scope does not match the frozen list"
        )


def _build_raw_windows(frame, protocol: str):
    spec = KITAKYUSHU_SPLIT if protocol == "full" else KITAKYUSHU_SMALL_SAMPLE_SPLIT
    return {
        split_name: build_protocol_windows(
            frame,
            spec,
            split_name=split_name,
            lookback=LOOKBACK,
            horizon=HORIZON,
            exog_columns=(),
            task_columns=KITAKYUSHU_TASKS,
        )
        for split_name in ("train", "validation", "test")
    }


def _write_deterministic_run(
    run: Mapping[str, object],
    run_dir: Path,
    windows: Mapping[str, Mapping[str, np.ndarray]],
    dataset_metadata: Mapping[str, object],
    cleaning_report: Mapping[str, object],
) -> Dict[str, object]:
    model = str(run["model"])
    history = windows["test"]["loads"]
    target = windows["test"]["target"]
    started = time.perf_counter()
    if model == "persistence":
        prediction = persistence_forecast(history, horizon=HORIZON)
    elif model == "seasonal_naive":
        prediction = seasonal_naive_forecast(
            history, horizon=HORIZON, season_length=24
        )
    else:
        raise ValueError(f"unknown deterministic baseline: {model}")
    metrics = regression_metrics(
        target, prediction, task_names=KITAKYUSHU_TASKS
    )
    elapsed = time.perf_counter() - started
    metrics["per_season"] = _seasonal_metrics(
        target, prediction, windows["test"]["target_times"]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "predictions_test.npz",
        target=target,
        prediction=prediction,
        target_times=windows["test"]["target_times"],
    )
    save_json(metrics, run_dir / "metrics_test.json")
    manifest = {
        "stage": "7.5",
        "status": "passed",
        "model": model,
        "baseline_type": "deterministic",
        "protocol": run["protocol"],
        "seed": None,
        "candidate_id": "deterministic",
        "dataset": "kitakyushu_energy_station",
        "dataset_metadata": dataset_metadata,
        "cleaning_report": cleaning_report,
        "tasks": list(KITAKYUSHU_TASKS),
        "input_mode": "loads_only",
        "future_exogenous_used": False,
        "test_set_accessed": True,
        "test_used_for_selection": False,
        "window": {"lookback": LOOKBACK, "horizon": HORIZON},
        "sample_counts": {
            name: int(len(value["target"])) for name, value in windows.items()
        },
        "parameter_count": 0,
        "runtime_seconds": {
            "fit": 0.0,
            "validation_evaluation": 0.0,
            "test_evaluation": float(elapsed),
        },
        "files": ["metrics_test.json", "predictions_test.npz", "run_manifest.json"],
    }
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def _run_learned_baseline(
    run: Mapping[str, object],
    run_dir: Path,
    data_dir: Path,
    training_policy: Mapping[str, object],
) -> Dict[str, object]:
    model = str(run["model"])
    seed = int(run["seed"])
    train_script = PROJECT_ROOT / "scripts" / "train_external_baseline.py"
    command = [
        sys.executable,
        str(train_script),
        "--model",
        model,
        "--dataset",
        "kitakyushu_energy_station",
        "--kitakyushu-data-dir",
        str(data_dir),
        "--protocol",
        str(run["protocol"]),
        "--output-dir",
        str(run_dir),
        "--lookback",
        str(LOOKBACK),
        "--horizon",
        str(HORIZON),
        "--batch-size",
        str(int(training_policy["batch_size"])),
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0.0001",
        "--max-epochs",
        str(int(training_policy["max_epochs"])),
        "--patience",
        str(int(training_policy["early_stopping_patience"])),
        "--threads",
        str(FORMAL_THREADS),
        "--seed",
        str(seed),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"external trainer failed with code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    metrics_path = run_dir / "metrics_test.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"trainer did not write {metrics_path}")
    child_metrics = _read_json(metrics_path)
    metrics = child_metrics["metrics_original_scale"]
    test_samples = int(child_metrics["sample_counts"]["test"])
    child_runtime = child_metrics.get("runtime_seconds", {})
    manifest = {
        "stage": "7.5",
        "status": "passed",
        "model": model,
        "baseline_type": "learned",
        "protocol": run["protocol"],
        "seed": seed,
        "candidate_id": "external_fixed",
        "dataset": "kitakyushu_energy_station",
        "tasks": list(KITAKYUSHU_TASKS),
        "input_mode": child_metrics["input_mode"],
        "future_exogenous_used": False,
        "test_set_accessed": True,
        "test_used_for_selection": False,
        "window": child_metrics["window"],
        "sample_counts": child_metrics["sample_counts"],
        "parameter_count": int(child_metrics["model_parameter_count"]),
        "runtime_seconds": {
            "fit": float(child_runtime.get("fit", 0.0)),
            "validation_evaluation": 0.0,
            "test_evaluation": float(child_runtime.get("test_evaluation", 0.0)),
            "wrapper_elapsed": float(elapsed),
        },
        "trainer_config": child_metrics["trainer_config"],
        "reproducibility": child_metrics["reproducibility"],
        "model_config": child_metrics["model_config"],
        "metrics_test": metrics["overall_equal_task_mean"],
        "child_stdout_tail": completed.stdout[-2000:],
        "files": [
            "best_model.pt",
            "normalization_stats.npz",
            "history.json",
            "metrics_test.json",
            "predictions_test.npz",
            "run_manifest.json",
        ],
    }
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def _summary_row(manifest: Mapping[str, object], run_dir: Path) -> Dict[str, object]:
    if manifest["baseline_type"] == "deterministic":
        metrics = _read_json(run_dir / "metrics_test.json")
    else:
        metrics = _read_json(run_dir / "metrics_test.json")["metrics_original_scale"]
    overall = metrics["overall_equal_task_mean"]
    runtime = manifest["runtime_seconds"]
    return {
        "protocol": manifest["protocol"],
        "model": manifest["model"],
        "baseline_type": manifest["baseline_type"],
        "seed": manifest["seed"] if manifest["seed"] is not None else "deterministic",
        "train_samples": manifest["sample_counts"].get("train", 0),
        "validation_samples": manifest["sample_counts"].get("validation", 0),
        "test_samples": manifest["sample_counts"]["test"],
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "WAPE": overall["WAPE"],
        "MAPE": overall["MAPE"],
        "parameter_count": manifest["parameter_count"],
        "fit_seconds": runtime["fit"],
        "validation_evaluation_seconds": runtime["validation_evaluation"],
        "test_evaluation_seconds": runtime["test_evaluation"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 7.5 formal external baselines"
    )
    parser.add_argument("--kitakyushu-data-dir", default="D:/Paper/Kitakyushu dataset")
    parser.add_argument("--contract", default="frame/configs/stage7_contract.json")
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6_6_kitakyushu/stage6_selected_config.json",
    )
    parser.add_argument(
        "--output-dir", default="frame/reports/stage7_5_kitakyushu_reproducible"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    contract_path = _resolve(args.contract)
    freeze_path = _resolve(args.freeze_config)
    contract = _read_json(contract_path)
    freeze = _read_json(freeze_path)
    _validate_contract(contract)
    _validate_external_scope(contract)
    if (
        freeze.get("freeze_version") != "stage6.6"
        or freeze.get("freeze_status") != "frozen_for_stage7"
    ):
        raise ValueError("Stage 7.5 requires the frozen Stage 6.6 configuration")
    _validate_freeze_and_training_policy(freeze, contract)
    plan = build_external_run_plan()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "7.5",
                    "status": "dry_run",
                    "run_count": len(plan),
                    "deterministic_runs": 4,
                    "learned_runs": 30,
                    "models": list(ALL_MODELS),
                    "protocols": list(FORMAL_PROTOCOLS),
                    "seeds": list(EXPECTED_SEEDS),
                    "sample_limits": None,
                    "max_epochs": FORMAL_MAX_EPOCHS,
                    "early_stopping_patience": FORMAL_PATIENCE,
                    "runs": list(plan),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_dir = _resolve(args.output_dir)
    manifest_path = output_dir / "stage7_5_manifest.json"
    if manifest_path.exists() and not (args.force or args.resume):
        raise FileExistsError(
            f"manifest exists; use --resume or --force: {manifest_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    read_started = time.perf_counter()
    frame, source_metadata = read_kitakyushu_canonical(
        _resolve(args.kitakyushu_data_dir), years=tuple(range(2015, 2022))
    )
    cleaned, cleaning_report = clean_kitakyushu_dataframe(frame)
    read_seconds = time.perf_counter() - read_started
    protocol_windows = {
        protocol: _build_raw_windows(cleaned, protocol) for protocol in FORMAL_PROTOCOLS
    }
    protocol_policies = contract.get("protocol_training_policies") or {
        "full": contract["training_policy"],
        "small_sample": contract["training_policy"],
    }

    completed: List[Dict[str, object]] = []
    failed: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    for run in plan:
        protocol = str(run["protocol"])
        model = str(run["model"])
        run_dir = output_dir / protocol / _model_dir_name(model) / str(run["seed_label"])
        existing_manifest = run_dir / "run_manifest.json"
        if args.resume and existing_manifest.exists():
            existing = _read_json(existing_manifest)
            is_deterministic = existing.get("baseline_type") == "deterministic"
            reproducibility = existing.get("reproducibility", {})
            is_strict_learned = reproducibility.get(
                "model_initialized_after_seed"
            ) is True
            if existing.get("status") == "passed" and (
                is_deterministic or is_strict_learned
            ):
                completed_result = dict(existing)
                completed_result["run_dir"] = str(run_dir)
                completed.append(completed_result)
                rows.append(_summary_row(existing, run_dir))
                print(
                    json.dumps(
                        {
                            "stage": "7.5",
                            "status": "run_reused",
                            "run_id": run["run_id"],
                            "completed": len(completed),
                            "total": len(plan),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            raise ValueError(
                f"cannot resume a non-strict or failed run: {existing_manifest}"
            )
        try:
            if model in DETERMINISTIC_MODELS:
                result = _write_deterministic_run(
                    run,
                    run_dir,
                    protocol_windows[protocol],
                    source_metadata,
                    cleaning_report,
                )
            else:
                result = _run_learned_baseline(
                    run,
                    run_dir,
                    _resolve(args.kitakyushu_data_dir),
                    protocol_policies[protocol],
                )
            result = dict(result)
            result["run_dir"] = str(run_dir)
            completed.append(result)
            rows.append(_summary_row(result, run_dir))
            print(
                json.dumps(
                    {
                        "stage": "7.5",
                        "status": "run_completed",
                        "run_id": run["run_id"],
                        "completed": len(completed),
                        "total": len(plan),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # preserve failures and continue
            error = {
                "stage": "7.5",
                "status": "failed",
                "run_id": run["run_id"],
                "protocol": protocol,
                "model": model,
                "seed": run["seed"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            save_json(error, run_dir / "error.json")
            failed.append(error)
            print(json.dumps(error, ensure_ascii=False), flush=True)

    _write_csv(rows, output_dir / "external_runs.csv")
    _write_csv(_aggregate_rows(rows), output_dir / "external_summary_mean_std.csv")
    manifest = {
        "stage": "7.5",
        "status": "passed" if not failed and len(completed) == len(plan) else "failed",
        "contract": str(contract_path),
        "freeze_config": str(freeze_path),
        "dataset": contract["dataset"],
        "data_source": source_metadata,
        "cleaning_report": cleaning_report,
        "read_seconds": read_seconds,
        "models": list(ALL_MODELS),
        "protocols": list(FORMAL_PROTOCOLS),
        "seeds": list(EXPECTED_SEEDS),
        "deterministic_runs": 4,
        "learned_runs": 30,
        "run_count_expected": len(plan),
        "run_count_completed": len(completed),
        "run_count_failed": len(failed),
        "sample_limits": None,
        "formal_training": True,
        "test_set_accessed": True,
        "test_used_for_selection": False,
        "selection_source": "stage6.6_freeze",
        "reproducibility_revision": "stage7R.1",
        "strict_seed_control": True,
        "model_initialized_after_seed": True,
        "training_dataloader_seeded": True,
        "legacy_results_excluded": True,
        "completed_runs": completed,
        "failed_runs": failed,
        "summary_files": ["external_runs.csv", "external_summary_mean_std.csv"],
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failed or len(completed) != len(plan):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
