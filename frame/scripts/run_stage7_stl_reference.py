"""Run the strict, structure-matched STL reference for Stage 7-R.

Four task models are trained independently for each protocol/seed run.  Each
task keeps Scheme2R's full-window DS-TCN and task-step prediction heads while
removing all cross-task routing, messages and fusion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
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
    _standardize_protocol,
    _validate_contract,
    _validate_freeze_and_training_policy,
    _write_csv,
)
from src.baselines import regression_metrics  # noqa: E402
from src.data_pipeline import save_json  # noqa: E402
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.models import count_trainable_parameters  # noqa: E402
from src.stage7_contract import EXPECTED_SEEDS  # noqa: E402
from src.stage7_reference import MatchedSingleTaskScheme2RModel  # noqa: E402
from src.training import (  # noqa: E402
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    load_checkpoint,
    make_dataloader,
    set_reproducible,
)


MODEL_NAME = "stl_matched"
STAGE_NAME = "7R.STL"


def build_stl_run_plan(
    seeds: Sequence[int] = EXPECTED_SEEDS,
) -> Tuple[Dict[str, object], ...]:
    seeds = tuple(int(seed) for seed in seeds)
    if seeds != tuple(EXPECTED_SEEDS):
        raise ValueError(
            f"Stage 7-R STL requires the frozen seeds {EXPECTED_SEEDS}, got {seeds}"
        )
    return tuple(
        {
            "protocol": protocol,
            "model": MODEL_NAME,
            "candidate_id": "legacy",
            "seed": seed,
            "run_id": f"{protocol}/{MODEL_NAME}/seed_{seed}",
        }
        for protocol in FORMAL_PROTOCOLS
        for seed in seeds
    )


def _task_seed(base_seed: int, task_index: int) -> int:
    return int(base_seed + 1000 * task_index)


def _task_windows(
    windows: Mapping[str, np.ndarray], task_index: int
) -> Dict[str, np.ndarray]:
    return {
        "loads": windows["loads"],
        "exog": windows["exog"],
        "target": windows["target"][:, :, task_index : task_index + 1],
        "target_times": windows["target_times"],
    }


def _build_task_model(
    task_index: int,
    hyperparameters: Mapping[str, object],
) -> MatchedSingleTaskScheme2RModel:
    # The prediction head width is part of the frozen architecture contract.
    # It must not vary with the encoder width (the previous ``hidden_dim // 2``
    # rule made H1 use a different head from Scheme2R's fixed 16-unit head).
    head_hidden_dim = int(hyperparameters["prediction_head_hidden_dim"])
    if head_hidden_dim != 16:
        raise ValueError(
            "structure-matched STL requires prediction_head_hidden_dim=16"
        )
    return MatchedSingleTaskScheme2RModel(
        exog_dim=len(KITAKYUSHU_EXOG_COLUMNS),
        task_index=task_index,
        task_count=len(KITAKYUSHU_TASKS),
        hidden_dim=int(hyperparameters["hidden_dim"]),
        lookback=LOOKBACK,
        kernel_size=int(hyperparameters["scheme2r_kernel_size"]),
        dilations=tuple(hyperparameters["scheme2r_dilations"]),
        dropout=float(hyperparameters["dropout"]),
        horizon=HORIZON,
        head_hidden_dim=head_hidden_dim,
    )


def _run_one(
    run: Mapping[str, object],
    run_dir: Path,
    windows: Mapping[str, Mapping[str, np.ndarray]],
    stats: StandardizationStats,
    hyperparameters: Mapping[str, object],
    training_policy: Mapping[str, object],
) -> Dict[str, object]:
    base_seed = int(run["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    histories: Dict[str, object] = {}
    checkpoints: Dict[str, object] = {}
    task_seeds: Dict[str, int] = {}
    task_parameter_counts: Dict[str, int] = {}
    task_runtime: Dict[str, Dict[str, float]] = {}
    validation_predictions_std: List[np.ndarray] = []
    validation_targets_std: List[np.ndarray] = []
    test_predictions_std: List[np.ndarray] = []
    test_targets_std: List[np.ndarray] = []

    for task_index, task_name in enumerate(KITAKYUSHU_TASKS):
        seed = _task_seed(base_seed, task_index)
        task_seeds[task_name] = seed
        config = TrainerConfig(
            seed=seed,
            torch_threads=FORMAL_THREADS,
            learning_rate=float(hyperparameters["learning_rate"]),
            weight_decay=float(training_policy["weight_decay"]),
            grad_clip_norm=float(training_policy["gradient_clip_norm"]),
            max_epochs=int(training_policy.get("max_epochs", FORMAL_MAX_EPOCHS)),
            early_stopping_patience=int(training_policy.get("early_stopping_patience", FORMAL_PATIENCE)),
        )
        set_reproducible(config)
        model = _build_task_model(task_index, hyperparameters)
        task_parameter_counts[task_name] = count_trainable_parameters(model)
        task_data = {
            split: _task_windows(values, task_index)
            for split, values in windows.items()
        }
        train_loader = make_dataloader(
            task_data["train"], int(training_policy.get("batch_size", FORMAL_BATCH_SIZE)), shuffle=True, seed=seed
        )
        validation_loader = make_dataloader(
            task_data["validation"], int(training_policy.get("batch_size", FORMAL_BATCH_SIZE)), shuffle=False
        )
        test_loader = make_dataloader(
            task_data["test"], int(training_policy.get("batch_size", FORMAL_BATCH_SIZE)), shuffle=False
        )
        checkpoint_path = run_dir / f"best_model_{task_name}.pt"
        fit_started = time.perf_counter()
        history = fit_model(
            model,
            train_loader,
            validation_loader,
            config,
            checkpoint_path,
            input_mode="loads_and_exog",
        )
        fit_seconds = time.perf_counter() - fit_started
        checkpoint = load_checkpoint(model, checkpoint_path, "cpu")

        validation_started = time.perf_counter()
        _, validation_prediction_std, validation_target_std = evaluate_model(
            model,
            validation_loader,
            "cpu",
            input_mode="loads_and_exog",
        )
        validation_seconds = time.perf_counter() - validation_started
        test_started = time.perf_counter()
        _, test_prediction_std, test_target_std = evaluate_model(
            model,
            test_loader,
            "cpu",
            input_mode="loads_and_exog",
        )
        test_seconds = time.perf_counter() - test_started

        histories[task_name] = history
        checkpoints[task_name] = {
            "file": checkpoint_path.name,
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_loss": float(checkpoint["best_validation_loss"]),
            "trainer_config": asdict(config),
        }
        task_runtime[task_name] = {
            "fit": float(fit_seconds),
            "validation_evaluation": float(validation_seconds),
            "test_evaluation": float(test_seconds),
        }
        validation_predictions_std.append(validation_prediction_std)
        validation_targets_std.append(validation_target_std)
        test_predictions_std.append(test_prediction_std)
        test_targets_std.append(test_target_std)

    validation_prediction_std = np.concatenate(validation_predictions_std, axis=2)
    validation_target_std = np.concatenate(validation_targets_std, axis=2)
    test_prediction_std = np.concatenate(test_predictions_std, axis=2)
    test_target_std = np.concatenate(test_targets_std, axis=2)
    validation_prediction = stats.inverse_targets(validation_prediction_std)
    validation_target = stats.inverse_targets(validation_target_std)
    test_prediction = stats.inverse_targets(test_prediction_std)
    test_target = stats.inverse_targets(test_target_std)

    np.savez_compressed(
        run_dir / "predictions_validation.npz",
        prediction=validation_prediction,
        target=validation_target,
        prediction_standardized=validation_prediction_std,
        target_standardized=validation_target_std,
        target_times=windows["validation"]["target_times"],
    )
    np.savez_compressed(
        run_dir / "predictions_test.npz",
        prediction=test_prediction,
        target=test_target,
        prediction_standardized=test_prediction_std,
        target_standardized=test_target_std,
        target_times=windows["test"]["target_times"],
    )
    validation_metrics = regression_metrics(
        validation_target,
        validation_prediction,
        task_names=KITAKYUSHU_TASKS,
    )
    test_metrics = regression_metrics(
        test_target,
        test_prediction,
        task_names=KITAKYUSHU_TASKS,
    )
    validation_metrics["per_season"] = _seasonal_metrics(
        validation_target,
        validation_prediction,
        windows["validation"]["target_times"],
    )
    test_metrics["per_season"] = _seasonal_metrics(
        test_target,
        test_prediction,
        windows["test"]["target_times"],
    )
    stats.save(run_dir / "normalization_stats.npz")
    save_json({"per_task": histories}, run_dir / "history.json")
    save_json(validation_metrics, run_dir / "metrics_validation.json")
    save_json(test_metrics, run_dir / "metrics_test.json")

    fit_seconds = sum(item["fit"] for item in task_runtime.values())
    validation_seconds = sum(
        item["validation_evaluation"] for item in task_runtime.values()
    )
    test_seconds = sum(item["test_evaluation"] for item in task_runtime.values())
    manifest = {
        "stage": STAGE_NAME,
        "status": "passed",
        "protocol": run["protocol"],
        "model": MODEL_NAME,
        "candidate_id": str(run["candidate_id"]),
        "seed": base_seed,
        "task_seeds": task_seeds,
        "tasks": list(KITAKYUSHU_TASKS),
        "window": {"lookback": LOOKBACK, "horizon": HORIZON},
        "input_mode": "loads_and_exog",
        "future_exogenous_used": False,
        "test_set_accessed": True,
        "test_used_for_selection": False,
        "task_models_trained_independently": True,
        "cross_task_parameters": False,
        "matched_components": ["full_window_dstcn", "task_step_forecast_head"],
        "removed_components": [
            "state_encoder",
            "two_level_router",
            "cross_task_message_projection",
            "residual_task_fusion",
        ],
        "sample_counts": {
            name: int(len(value["target"])) for name, value in windows.items()
        },
        "parameter_count": int(sum(task_parameter_counts.values())),
        "task_parameter_counts": task_parameter_counts,
        "runtime_seconds": {
            "fit": float(fit_seconds),
            "validation_evaluation": float(validation_seconds),
            "test_evaluation": float(test_seconds),
        },
        "best_checkpoint_epoch": {
            task: int(value["best_epoch"]) for task, value in checkpoints.items()
        },
        "best_validation_loss": {
            task: float(value["best_validation_loss"])
            for task, value in checkpoints.items()
        },
        "task_checkpoints": checkpoints,
        "reproducibility": {
            "revision": "stage7R.1",
            "model_initialized_after_seed": True,
            "model_initialization_seed": task_seeds,
            "training_dataloader_seed": task_seeds,
            "training_seed_reset_before_fit": True,
        },
        "files": [
            *(f"best_model_{task}.pt" for task in KITAKYUSHU_TASKS),
            "normalization_stats.npz",
            "history.json",
            "metrics_validation.json",
            "metrics_test.json",
            "predictions_validation.npz",
            "predictions_test.npz",
        ],
    }
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def _summary_row(manifest: Mapping[str, object], run_dir: Path) -> Dict[str, object]:
    metrics = _read_json(run_dir / "metrics_test.json")
    overall = metrics["overall_equal_task_mean"]
    runtime = manifest["runtime_seconds"]
    epochs = tuple(int(value) for value in manifest["best_checkpoint_epoch"].values())
    return {
        "protocol": manifest["protocol"],
        "model": manifest["model"],
        "candidate_id": manifest["candidate_id"],
        "seed": manifest["seed"],
        "train_samples": manifest["sample_counts"]["train"],
        "validation_samples": manifest["sample_counts"]["validation"],
        "test_samples": manifest["sample_counts"]["test"],
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "WAPE": overall["WAPE"],
        "MAPE": overall["MAPE"],
        "parameter_count": manifest["parameter_count"],
        "fit_seconds": runtime["fit"],
        "validation_evaluation_seconds": runtime["validation_evaluation"],
        "test_evaluation_seconds": runtime["test_evaluation"],
        "best_checkpoint_epoch": float(np.mean(epochs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Stage 7-R structure-matched STL reference"
    )
    parser.add_argument("--kitakyushu-data-dir", default="D:/Paper/Kitakyushu dataset")
    parser.add_argument("--contract", default="frame/configs/stage7_contract.json")
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6_6_kitakyushu/stage6_selected_config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage7_stl_reference_kitakyushu_formal",
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
    if (
        freeze.get("freeze_version") != "stage6.6"
        or freeze.get("freeze_status") != "frozen_for_stage7"
    ):
        raise ValueError("Stage 7-R STL requires the frozen Stage 6.6 configuration")
    _validate_freeze_and_training_policy(freeze, contract)
    plan = build_stl_run_plan()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": STAGE_NAME,
                    "status": "dry_run",
                    "run_count": len(plan),
                    "model": MODEL_NAME,
                    "protocols": list(FORMAL_PROTOCOLS),
                    "seeds": list(EXPECTED_SEEDS),
                    "independent_tasks_per_run": list(KITAKYUSHU_TASKS),
                    "trained_task_models": len(plan) * len(KITAKYUSHU_TASKS),
                    "sample_limits": None,
                    "training_policies": contract.get("protocol_training_policies", {}),
                    "runs": list(plan),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    output_dir = _resolve(args.output_dir)
    manifest_path = output_dir / "stage7r_stl_manifest.json"
    if manifest_path.exists() and not (args.force or args.resume):
        raise FileExistsError(
            f"manifest exists; use --resume or --force: {manifest_path}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    read_started = time.perf_counter()
    raw_frame, source_metadata = read_kitakyushu_canonical(
        _resolve(args.kitakyushu_data_dir), years=tuple(range(2015, 2022))
    )
    frame, cleaning_report = clean_kitakyushu_dataframe(raw_frame)
    read_seconds = time.perf_counter() - read_started
    protocol_specs = {
        "full": KITAKYUSHU_SPLIT,
        "small_sample": KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    }
    protocol_windows = {}
    protocol_stats = {}
    for protocol, split_spec in protocol_specs.items():
        windows, stats = _standardize_protocol(frame, split_spec)
        protocol_windows[protocol] = windows
        protocol_stats[protocol] = stats
        stats.save(output_dir / protocol / "normalization_stats.npz")

    hyperparameters = freeze["primary_model"]["hyperparameters"]
    primary_candidate = str(freeze["primary_model"]["candidate_id"])
    plan = tuple(
        {**run, "candidate_id": primary_candidate}
        for run in plan
    )
    protocol_policies = contract.get("protocol_training_policies") or {
        "full": contract["training_policy"],
        "small_sample": contract["training_policy"],
    }
    completed: List[Dict[str, object]] = []
    failed: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    for run in plan:
        protocol = str(run["protocol"])
        run_dir = output_dir / protocol / MODEL_NAME / f"seed_{run['seed']}"
        existing_manifest = run_dir / "run_manifest.json"
        if args.resume and existing_manifest.exists():
            existing = _read_json(existing_manifest)
            reproducibility = existing.get("reproducibility", {})
            if existing.get("status") == "passed" and reproducibility.get(
                "model_initialized_after_seed"
            ) is True:
                completed_result = dict(existing)
                completed_result["run_dir"] = str(run_dir)
                completed.append(completed_result)
                rows.append(_summary_row(existing, run_dir))
                print(
                    json.dumps(
                        {
                            "stage": STAGE_NAME,
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
            result = _run_one(
                run,
                run_dir,
                protocol_windows[protocol],
                protocol_stats[protocol],
                hyperparameters,
                protocol_policies[protocol],
            )
            completed_result = dict(result)
            completed_result["run_dir"] = str(run_dir)
            completed.append(completed_result)
            rows.append(_summary_row(result, run_dir))
            print(
                json.dumps(
                    {
                        "stage": STAGE_NAME,
                        "status": "run_completed",
                        "run_id": run["run_id"],
                        "completed": len(completed),
                        "total": len(plan),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            error = {
                "stage": STAGE_NAME,
                "status": "failed",
                "run_id": run["run_id"],
                "protocol": protocol,
                "model": MODEL_NAME,
                "seed": run["seed"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            save_json(error, run_dir / "error.json")
            failed.append(error)
            print(json.dumps(error, ensure_ascii=False), flush=True)

    _write_csv(rows, output_dir / "stl_runs.csv")
    _write_csv(_aggregate_rows(rows), output_dir / "stl_summary_mean_std.csv")
    manifest = {
        "stage": STAGE_NAME,
        "status": "passed" if not failed and len(completed) == len(plan) else "failed",
        "contract": str(contract_path),
        "freeze_config": str(freeze_path),
        "dataset": contract["dataset"],
        "data_source": source_metadata,
        "cleaning_report": cleaning_report,
        "read_seconds": float(read_seconds),
        "models": [MODEL_NAME],
        "protocols": list(FORMAL_PROTOCOLS),
        "seeds": list(EXPECTED_SEEDS),
        "run_count_expected": len(plan),
        "run_count_completed": len(completed),
        "run_count_failed": len(failed),
        "trained_task_model_count_expected": len(plan) * len(KITAKYUSHU_TASKS),
        "sample_limits": None,
        "formal_training": True,
        "test_set_accessed": True,
        "test_used_for_selection": False,
        "selection_source": "stage6.6_frozen_primary_candidate_structure_only",
        "reference_purpose": "task-level negative-transfer measurement",
        "reproducibility_revision": "stage7R.1",
        "strict_seed_control": True,
        "task_models_trained_independently": True,
        "legacy_results_excluded": True,
        "completed_runs": completed,
        "failed_runs": failed,
        "summary_files": ["stl_runs.csv", "stl_summary_mean_std.csv"],
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failed or len(completed) != len(plan):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
