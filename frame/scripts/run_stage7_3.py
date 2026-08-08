"""Stage 7.3: formal main-model and internal-comparison experiments.

The script deliberately does not expose smoke/sample-limit overrides. It
reads the Stage 6.6 freeze, runs the two dynamically selected candidates under
the full and small-sample protocols, and repeats each run for the five frozen
seeds. The test split is evaluated only after the best validation checkpoint
has been restored; it is never used for selection.

Use ``--dry-run`` first to inspect the 20-run plan without reading data or
creating model checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import regression_metrics  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    build_protocol_windows,
    save_json,
    select_training_frame,
)
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.models import build_forecasting_model, count_trainable_parameters  # noqa: E402
from src.external_models import PLELiteBaseline  # noqa: E402
from src.stage7_contract import EXPECTED_SEEDS  # noqa: E402
from src.training import (  # noqa: E402
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    load_checkpoint,
    make_dataloader,
    set_reproducible,
)


FORMAL_MODELS: Tuple[str, ...] = ("scheme2r", "dynamic_symmetric")
FORMAL_PROTOCOLS: Tuple[str, ...] = ("full", "small_sample")
LOOKBACK = 24
HORIZON = 4
FORMAL_BATCH_SIZE = 256
FORMAL_THREADS = 8
FORMAL_MAX_EPOCHS = 100
FORMAL_PATIENCE = 12


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _seasonal_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    target_times: np.ndarray,
) -> Dict[str, Dict[str, object]]:
    timestamps = np.asarray(target_times).astype("datetime64[M]")
    months = (timestamps.astype(int) % 12) + 1
    result: Dict[str, Dict[str, object]] = {}
    for season_name in ("winter", "spring", "summer", "autumn"):
        mask = np.asarray([_season(int(month)) == season_name for month in months])
        if not mask.any():
            result[season_name] = {"sample_count": 0}
            continue
        result[season_name] = regression_metrics(
            target[mask], prediction[mask], task_names=KITAKYUSHU_TASKS
        )
    return result


def _build_model(
    model_name: str,
    hyperparameters: Mapping[str, object],
    exog_dim: int,
):
    if model_name == "ple-lite":
        return PLELiteBaseline(
            lookback=LOOKBACK,
            horizon=HORIZON,
            task_count=len(KITAKYUSHU_TASKS),
            exog_dim=exog_dim,
            shared_expert_count=int(hyperparameters["shared_expert_count"]),
            task_expert_count=int(hyperparameters["task_expert_count"]),
            expert_hidden_dim=int(hyperparameters["expert_hidden_dim"]),
            representation_dim=int(hyperparameters["representation_dim"]),
            head_hidden_dim=int(
                hyperparameters["prediction_head_hidden_dim"]
            ),
            dropout=float(hyperparameters["dropout"]),
        )
    common = {
        "exog_dim": exog_dim,
        "task_count": len(KITAKYUSHU_TASKS),
        "hidden_dim": int(hyperparameters["hidden_dim"]),
        "dropout": float(hyperparameters["dropout"]),
        "horizon": HORIZON,
        "head_hidden_dim": int(hyperparameters["prediction_head_hidden_dim"]),
    }
    if model_name == "scheme2r":
        common.update(
            {
                "lookback": LOOKBACK,
                "kernel_size": int(hyperparameters["scheme2r_kernel_size"]),
                "dilations": tuple(hyperparameters["scheme2r_dilations"]),
                "rank": int(hyperparameters["scheme2r_rank"]),
                "gate_hidden_dim": int(
                    hyperparameters["scheme2r_gate_hidden_dim"]
                ),
                "step_embedding_dim": int(
                    hyperparameters["scheme2r_step_embedding_dim"]
                ),
            }
        )
    elif model_name in {
        "stl_matched",
        "hard_share",
        "static_gate",
        "dynamic_symmetric",
        "dynamic_directed",
    }:
        common.update(
            {
                "lookback": LOOKBACK,
                "kernel_size": int(hyperparameters["kernel_size"]),
                "dilations": tuple(hyperparameters["dilations"]),
            }
        )
    else:
        raise ValueError(f"unsupported formal model: {model_name}")
    return build_forecasting_model(model_name, **common)


def build_run_plan(
    seeds: Sequence[int] = EXPECTED_SEEDS,
    models: Sequence[str] = FORMAL_MODELS,
    candidate_ids: Mapping[str, str] | None = None,
    model_candidates: Sequence[Tuple[str, str]] | None = None,
) -> Tuple[Dict[str, object], ...]:
    """Return the formal plan for the two models selected by Stage 6.6."""

    seeds = tuple(int(seed) for seed in seeds)
    if seeds != tuple(EXPECTED_SEEDS):
        raise ValueError(
            f"Stage 7.3 requires the frozen seeds {EXPECTED_SEEDS}, got {seeds}"
        )
    if model_candidates is None:
        candidates = dict(candidate_ids or {model: "legacy" for model in models})
        if any(model not in candidates for model in models):
            raise ValueError("每个 Stage 7.3 模型必须有冻结候选编号")
        model_candidates = tuple((str(model), str(candidates[model])) for model in models)
    else:
        model_candidates = tuple((str(model), str(candidate)) for model, candidate in model_candidates)
    if len(model_candidates) != 2:
        raise ValueError("Stage 7.3必须冻结两个主/对照候选")
    return tuple(
        {
            "protocol": protocol,
            "model": model,
            "candidate_id": candidate,
            "seed": seed,
            "run_id": f"{protocol}/{model}/{candidate}/seed_{seed}",
        }
        for protocol in FORMAL_PROTOCOLS
        for model, candidate in model_candidates
        for seed in seeds
    )


def _validate_contract(contract: Mapping[str, object]) -> None:
    if contract.get("contract_version") != "stage7.0":
        raise ValueError("Stage 7.3 requires the stage7.0 contract")
    if contract.get("contract_status") != "ready_for_stage7_smoke":
        raise ValueError(
            "Stage 7.3 requires a fresh Stage 7.0 contract generated after Stage 6-R"
        )
    if contract.get("dataset") != "kitakyushu_energy_station":
        raise ValueError("Stage 7.3 is frozen for Kitakyushu Energy Station")
    if tuple(contract.get("tasks", ())) != KITAKYUSHU_TASKS:
        raise ValueError("Stage 7.3 task order does not match Kitakyushu protocol")
    protocol = contract.get("data_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("stage7.0 data_protocol is missing")
    if protocol.get("lookback") != LOOKBACK or protocol.get("horizon") != HORIZON:
        raise ValueError("Stage 7.3 requires the frozen 24-to-4 window")
    if tuple(contract.get("training_policy", {}).get("formal_random_seeds", ())) != EXPECTED_SEEDS:
        raise ValueError("Stage 7.3 seed list is not the frozen 2026-2030 list")
    scope = contract.get("stage7_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("stage7.0 stage7_scope is missing")
    if tuple(scope.get("protocols", ())) != FORMAL_PROTOCOLS:
        raise ValueError("Stage 7.3 protocol list does not match the freeze")
    policy = contract.get("test_set_policy")
    if not isinstance(policy, Mapping) or policy.get("test_reading_allowed_in_stage7") is not True:
        raise ValueError("Stage 7.3 cannot read the test split before the freeze")


def _validate_freeze_and_training_policy(
    freeze: Mapping[str, object],
    contract: Mapping[str, object],
) -> None:
    primary = freeze.get("primary_model")
    comparison = freeze.get("comparison_model")
    if not isinstance(primary, Mapping) or not isinstance(comparison, Mapping):
        raise ValueError("Stage 6.6 freeze is missing primary/comparison models")
    for label, value in (("primary_model", primary), ("comparison_model", comparison)):
        if not isinstance(value.get("model"), str) or not value.get("model"):
            raise ValueError(f"Stage 7.3 {label}.model is missing")
        if not isinstance(value.get("candidate_id"), str) or not value.get("candidate_id"):
            raise ValueError(f"Stage 7.3 {label}.candidate_id is missing")
        if not isinstance(value.get("hyperparameters"), Mapping):
            raise ValueError(f"Stage 7.3 {label}.hyperparameters is missing")
    policy = contract.get("training_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Stage 7.0 training_policy is missing")
    expected = {
        "batch_size": FORMAL_BATCH_SIZE,
        "max_epochs": FORMAL_MAX_EPOCHS,
        "early_stopping_patience": FORMAL_PATIENCE,
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "device": "cpu",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(
                f"Stage 7.3 training policy mismatch for {key}: "
                f"expected {value!r}, got {policy.get(key)!r}"
            )


def _standardize_protocol(
    frame,
    split_spec,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], StandardizationStats]:
    raw_windows = {
        split_name: build_protocol_windows(
            frame,
            split_spec,
            split_name=split_name,
            lookback=LOOKBACK,
            horizon=HORIZON,
            exog_columns=KITAKYUSHU_EXOG_COLUMNS,
            task_columns=KITAKYUSHU_TASKS,
        )
        for split_name in ("train", "validation", "test")
    }
    stats = StandardizationStats.fit(
        select_training_frame(frame, split_spec),
        KITAKYUSHU_EXOG_COLUMNS,
        task_columns=KITAKYUSHU_TASKS,
    )
    standardized = {
        split_name: stats.transform_windows(windows)
        for split_name, windows in raw_windows.items()
    }
    return standardized, stats


def _write_predictions(
    path: Path,
    stats: StandardizationStats,
    prediction_std: np.ndarray,
    target_std: np.ndarray,
    target_times: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    prediction = stats.inverse_targets(prediction_std)
    target = stats.inverse_targets(target_std)
    np.savez_compressed(
        path,
        prediction=prediction,
        target=target,
        prediction_standardized=prediction_std,
        target_standardized=target_std,
        target_times=target_times,
    )
    return prediction, target


def _run_one(
    run: Mapping[str, object],
    run_dir: Path,
    windows: Mapping[str, Mapping[str, np.ndarray]],
    stats: StandardizationStats,
    hyperparameters: Mapping[str, Mapping[str, object]],
    training_policy: Mapping[str, object],
    *,
    stage_label: str = "7.3",
    stage_role: str | None = None,
) -> Dict[str, object]:
    protocol = str(run["protocol"])
    model_name = str(run["model"])
    candidate_id = str(run["candidate_id"])
    hp = hyperparameters.get((model_name, candidate_id)) or hyperparameters.get(model_name)
    if hp is None:
        raise ValueError(f"missing hyperparameters for {model_name}/{candidate_id}")
    if model_name == "stl_matched":
        # Keep the selected STL candidate on the same independent-task path as
        # the dedicated Stage 7-R STL reference; never train it through the
        # joint model wrapper used by the other candidates.
        from run_stage7_stl_reference import _run_one as _run_matched_stl_one

        result = _run_matched_stl_one(
            run,
            run_dir,
            windows,
            stats,
            hp,
            training_policy,
        )
        result = dict(result)
        result["stage"] = stage_label
        result["stage7_role"] = (
            stage_role or "selected_structure_matched_stl"
        )
        save_json(result, run_dir / "run_manifest.json")
        return result
    seed = int(run["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer_config = TrainerConfig(
        seed=seed,
        torch_threads=FORMAL_THREADS,
        learning_rate=float(
            hp["learning_rate"]
        ),
        weight_decay=float(training_policy["weight_decay"]),
        grad_clip_norm=float(training_policy["gradient_clip_norm"]),
        max_epochs=int(training_policy["max_epochs"]),
        early_stopping_patience=int(training_policy["early_stopping_patience"]),
    )
    set_reproducible(trainer_config)
    model = _build_model(
        model_name,
        hp,
        len(KITAKYUSHU_EXOG_COLUMNS),
    )
    train_loader = make_dataloader(
        windows["train"], int(training_policy["batch_size"]), shuffle=True, seed=seed
    )
    validation_loader = make_dataloader(
        windows["validation"], int(training_policy["batch_size"]), shuffle=False
    )
    test_loader = make_dataloader(windows["test"], int(training_policy["batch_size"]), shuffle=False)
    checkpoint_path = run_dir / "best_model.pt"
    started = time.perf_counter()
    history = fit_model(
        model,
        train_loader,
        validation_loader,
        trainer_config,
        checkpoint_path,
        input_mode="loads_and_exog",
    )
    fit_seconds = time.perf_counter() - started
    checkpoint = load_checkpoint(model, checkpoint_path, "cpu")

    validation_started = time.perf_counter()
    validation_loss, validation_prediction_std, validation_target_std = evaluate_model(
        model, validation_loader, "cpu", input_mode="loads_and_exog"
    )
    validation_seconds = time.perf_counter() - validation_started
    test_started = time.perf_counter()
    test_loss, test_prediction_std, test_target_std = evaluate_model(
        model, test_loader, "cpu", input_mode="loads_and_exog"
    )
    test_seconds = time.perf_counter() - test_started

    stats.save(run_dir / "normalization_stats.npz")
    save_json({"history": history}, run_dir / "history.json")
    validation_prediction, validation_target = _write_predictions(
        run_dir / "predictions_validation.npz",
        stats,
        validation_prediction_std,
        validation_target_std,
        windows["validation"]["target_times"],
    )
    test_prediction, test_target = _write_predictions(
        run_dir / "predictions_test.npz",
        stats,
        test_prediction_std,
        test_target_std,
        windows["test"]["target_times"],
    )
    validation_metrics = regression_metrics(
        validation_target, validation_prediction, task_names=KITAKYUSHU_TASKS
    )
    test_metrics = regression_metrics(
        test_target, test_prediction, task_names=KITAKYUSHU_TASKS
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
    save_json(validation_metrics, run_dir / "metrics_validation.json")
    save_json(test_metrics, run_dir / "metrics_test.json")

    manifest = {
        "stage": stage_label,
        "status": "passed",
        "protocol": protocol,
        "model": model_name,
        "candidate_id": candidate_id,
        "seed": seed,
        "tasks": list(KITAKYUSHU_TASKS),
        "window": {"lookback": LOOKBACK, "horizon": HORIZON},
        "input_mode": "loads_and_exog",
        "future_exogenous_used": False,
        "test_set_accessed": True,
        "sample_counts": {
            name: int(len(value["target"])) for name, value in windows.items()
        },
        "parameter_count": count_trainable_parameters(model),
        "runtime_seconds": {
            "fit": float(fit_seconds),
            "validation_evaluation": float(validation_seconds),
            "test_evaluation": float(test_seconds),
        },
        "validation_smooth_l1": float(validation_loss),
        "test_smooth_l1": float(test_loss),
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "trainer_config": asdict(trainer_config),
        "reproducibility": {
            "model_initialized_after_seed": True,
            "model_initialization_seed": seed,
            "training_dataloader_seed": seed,
            "training_seed_reset_before_fit": True,
        },
        "files": [
            "best_model.pt",
            "normalization_stats.npz",
            "history.json",
            "metrics_validation.json",
            "metrics_test.json",
            "predictions_validation.npz",
            "predictions_test.npz",
        ],
    }
    if stage_role is not None:
        manifest["stage7_role"] = stage_role
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def _summary_row(run_manifest: Mapping[str, object]) -> Dict[str, object]:
    test_metrics_path = Path(str(run_manifest["run_dir"])) / "metrics_test.json"
    test_metrics = _read_json(test_metrics_path)
    overall = test_metrics["overall_equal_task_mean"]
    return {
        "protocol": run_manifest["protocol"],
        "model": run_manifest["model"],
        "candidate_id": run_manifest["candidate_id"],
        "seed": run_manifest["seed"],
        "train_samples": run_manifest["sample_counts"]["train"],
        "validation_samples": run_manifest["sample_counts"]["validation"],
        "test_samples": run_manifest["sample_counts"]["test"],
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "WAPE": overall["WAPE"],
        "MAPE": overall["MAPE"],
        "parameter_count": run_manifest["parameter_count"],
        "fit_seconds": run_manifest["runtime_seconds"]["fit"],
        "validation_evaluation_seconds": run_manifest["runtime_seconds"][
            "validation_evaluation"
        ],
        "test_evaluation_seconds": run_manifest["runtime_seconds"]["test_evaluation"],
        "best_checkpoint_epoch": run_manifest["best_checkpoint_epoch"],
    }


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    metric_names = ("MAE", "RMSE", "WAPE", "MAPE")
    groups: Dict[Tuple[str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["protocol"]), str(row["model"])), []).append(row)
    summary = []
    for (protocol, model), group in sorted(groups.items()):
        item: Dict[str, object] = {
            "protocol": protocol,
            "model": model,
            "seed_count": len(group),
        }
        for metric in metric_names:
            values = np.asarray(
                [float(row[metric]) for row in group if row.get(metric) is not None],
                dtype=np.float64,
            )
            item[f"{metric}_mean"] = float(values.mean()) if len(values) else None
            item[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None)
            )
        for field in ("parameter_count", "train_samples", "validation_samples", "test_samples"):
            item[field] = group[0][field]
        for field in ("fit_seconds", "validation_evaluation_seconds", "test_evaluation_seconds"):
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            item[f"{field}_mean"] = float(values.mean())
            item[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 7.3 formal experiments")
    parser.add_argument(
        "--kitakyushu-data-dir",
        default="D:/Paper/Kitakyushu dataset",
    )
    parser.add_argument(
        "--contract",
        default="frame/configs/stage7r_contract.json",
    )
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage7r_3_kitakyushu_formal",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the 20-run plan without reading data or training",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow writing into an existing output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse strict passed runs and continue missing/failed runs",
    )
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
    if freeze.get("freeze_version") != "stage6.6" or freeze.get("freeze_status") != "frozen_for_stage7":
        raise ValueError("Stage 7.3 requires the frozen Stage 6.6 configuration")
    _validate_freeze_and_training_policy(freeze, contract)

    primary = freeze["primary_model"]
    comparison = freeze["comparison_model"]
    selected_models = (str(primary["model"]), str(comparison["model"]))
    model_candidates = (
        (selected_models[0], str(primary["candidate_id"])),
        (selected_models[1], str(comparison["candidate_id"])),
    )
    plan = build_run_plan(model_candidates=model_candidates)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "7.3",
                    "status": "dry_run",
                    "run_count": len(plan),
                    "models": list(selected_models),
                    "protocols": list(FORMAL_PROTOCOLS),
                    "seeds": list(EXPECTED_SEEDS),
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
    manifest_path = output_dir / "stage7_3_manifest.json"
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

    primary_hp = primary["hyperparameters"]
    comparison_hp = comparison["hyperparameters"]
    hyperparameters = {
        (selected_models[0], str(primary["candidate_id"])): primary_hp,
        (selected_models[1], str(comparison["candidate_id"])): comparison_hp,
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
        run_dir = (
            output_dir
            / protocol
            / str(run["model"])
            / str(run["candidate_id"])
            / f"seed_{run['seed']}"
        )
        existing_manifest = run_dir / "run_manifest.json"
        if args.resume and existing_manifest.exists():
            existing = _read_json(existing_manifest)
            reproducibility = existing.get("reproducibility", {})
            if existing.get("status") == "passed" and reproducibility.get(
                "model_initialized_after_seed"
            ) is True:
                existing["run_dir"] = str(run_dir)
                completed.append(existing)
                rows.append(_summary_row(existing))
                print(
                    json.dumps(
                        {
                            "stage": "7.3",
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
            result["run_dir"] = str(run_dir)
            completed.append(result)
            rows.append(_summary_row(result))
            print(
                json.dumps(
                    {
                        "stage": "7.3",
                        "status": "run_completed",
                        "run_id": run["run_id"],
                        "completed": len(completed),
                        "total": len(plan),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # preserve failures and continue independent runs
            error = {
                "stage": "7.3",
                "status": "failed",
                "run_id": run["run_id"],
                "protocol": protocol,
                "model": run["model"],
                "seed": run["seed"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            save_json(error, run_dir / "error.json")
            failed.append(error)
            print(json.dumps(error, ensure_ascii=False), flush=True)

    _write_csv(rows, output_dir / "formal_runs.csv")
    _write_csv(_aggregate_rows(rows), output_dir / "formal_summary_mean_std.csv")
    manifest = {
        "stage": "7.3",
        "status": "passed" if not failed and len(completed) == len(plan) else "failed",
        "contract": str(contract_path),
        "freeze_config": str(freeze_path),
        "dataset": contract["dataset"],
        "data_source": source_metadata,
        "cleaning_report": cleaning_report,
        "read_seconds": read_seconds,
        "models": list(selected_models),
        "protocols": list(FORMAL_PROTOCOLS),
        "seeds": list(EXPECTED_SEEDS),
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
        "summary_files": ["formal_runs.csv", "formal_summary_mean_std.csv"],
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failed or len(completed) != len(plan):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
