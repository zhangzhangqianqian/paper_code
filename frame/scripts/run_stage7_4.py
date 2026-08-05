"""Stage 7.4: formal A0--A4 ablation experiments.

The recursive ablation sequence is evaluated under both frozen protocols and
the five frozen seeds.  This script deliberately has no sample-limit or
short-epoch options: ``--dry-run`` is the only non-training mode.
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
    _write_predictions,
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
from src.stage7_ablations import (  # noqa: E402
    ABLATION_NAMES,
    build_stage7_ablation_model,
)
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


def build_ablation_run_plan(
    seeds: Sequence[int] = EXPECTED_SEEDS,
    include_a4: bool = True,
    candidate_id: str = "legacy",
) -> Tuple[Dict[str, object], ...]:
    """Return the ablation plan; A4 can be omitted when it is reused."""

    seeds = tuple(int(seed) for seed in seeds)
    if seeds != tuple(EXPECTED_SEEDS):
        raise ValueError(
            f"Stage 7.4 requires the frozen seeds {EXPECTED_SEEDS}, got {seeds}"
        )
    ablations = tuple(name for name in ABLATION_NAMES if include_a4 or name != "A4")
    return tuple(
        {
            "protocol": protocol,
            "model": model,
            "seed": seed,
            "candidate_id": str(candidate_id),
            "run_id": f"{protocol}/{model}/seed_{seed}",
        }
        for protocol in FORMAL_PROTOCOLS
        for model in ablations
        for seed in seeds
    )


def _build_ablation_model(
    model_name: str,
    hyperparameters: Mapping[str, object],
    exog_dim: int,
):
    options = {
        "exog_dim": exog_dim,
        "task_count": len(KITAKYUSHU_TASKS),
        "hidden_dim": int(hyperparameters["hidden_dim"]),
        "dropout": float(hyperparameters["dropout"]),
        "horizon": HORIZON,
        "head_hidden_dim": int(hyperparameters["prediction_head_hidden_dim"]),
    }
    if model_name == "A0":
        options.update(
            {
                "kernel_size": int(hyperparameters["kernel_size"]),
                "dilations": tuple(hyperparameters["dilations"]),
            }
        )
    elif model_name in ("A1", "A2", "A3", "A4"):
        options.update(
            {
                "lookback": LOOKBACK,
                "kernel_size": int(hyperparameters["scheme2r_kernel_size"]),
                "dilations": tuple(hyperparameters["scheme2r_dilations"]),
                "gate_hidden_dim": int(
                    hyperparameters["scheme2r_gate_hidden_dim"]
                ),
                "step_embedding_dim": int(
                    hyperparameters["scheme2r_step_embedding_dim"]
                ),
            }
        )
        if model_name == "A4":
            options["rank"] = int(hyperparameters["scheme2r_rank"])
    else:
        raise ValueError(f"unknown Stage 7.4 ablation: {model_name}")
    return build_stage7_ablation_model(model_name, **options)


def _gate_summary(model, loader) -> Dict[str, object]:
    gates = []
    with np.errstate(all="ignore"):
        import torch

        model.eval()
        with torch.no_grad():
            for loads, exog, _ in loader:
                _, details = model.forward_with_details(loads, exog)
                gates.append(details["gates"].cpu().numpy())
    if not gates:
        raise ValueError("cannot summarize an empty gate tensor")
    values = np.concatenate(gates, axis=0).astype(np.float32)
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def _run_one_ablation(
    run: Mapping[str, object],
    run_dir: Path,
    windows: Mapping[str, Mapping[str, np.ndarray]],
    stats: StandardizationStats,
    hyperparameters: Mapping[str, object],
    training_policy: Mapping[str, object],
) -> Dict[str, object]:
    model_name = str(run["model"])
    seed = int(run["seed"])
    run_dir.mkdir(parents=True, exist_ok=True)
    config = TrainerConfig(
        seed=seed,
        torch_threads=FORMAL_THREADS,
        learning_rate=float(hyperparameters["learning_rate"]),
        weight_decay=float(training_policy["weight_decay"]),
        grad_clip_norm=float(training_policy["gradient_clip_norm"]),
        max_epochs=int(training_policy["max_epochs"]),
        early_stopping_patience=int(training_policy["early_stopping_patience"]),
    )
    set_reproducible(config)
    model = _build_ablation_model(
        model_name,
        hyperparameters,
        exog_dim=len(KITAKYUSHU_EXOG_COLUMNS),
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
        config,
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
        "stage": "7.4",
        "status": "passed",
        "protocol": run["protocol"],
        "model": model_name,
        "candidate_id": str(run["candidate_id"]),
        "seed": seed,
        "ablation_description": model.ablation_description,
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
        "gate_summary_validation": _gate_summary(model, validation_loader),
        "trainer_config": asdict(config),
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
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def _summary_row(run_manifest: Mapping[str, object], run_dir: Path) -> Dict[str, object]:
    metrics = _read_json(run_dir / "metrics_test.json")
    overall = metrics["overall_equal_task_mean"]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 7.4 formal A0-A4 ablations")
    parser.add_argument("--kitakyushu-data-dir", default="D:/Paper/Kitakyushu dataset")
    parser.add_argument("--contract", default="frame/configs/stage7_contract.json")
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6_6_kitakyushu/stage6_selected_config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage7_4_kitakyushu_reproducible",
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
    if freeze.get("freeze_version") != "stage6.6" or freeze.get("freeze_status") != "frozen_for_stage7":
        raise ValueError("Stage 7.4 requires the frozen Stage 6.6 configuration")
    _validate_freeze_and_training_policy(freeze, contract)
    if tuple(contract["stage7_scope"]["ablations"]) != ABLATION_NAMES:
        raise ValueError("Stage 7.0 ablation list does not match A0-A4")
    scheme2r_reference = freeze.get("scheme2r_ablation_reference")
    if not isinstance(scheme2r_reference, Mapping):
        raise ValueError("Stage 7.4 requires the frozen Scheme2R ablation reference")
    scheme2r_candidate = str(scheme2r_reference["candidate_id"])
    primary_model = str(freeze["primary_model"]["model"])
    # A4 is the complete Scheme2R endpoint.  Reuse is legal only when the
    # selected primary is Scheme2R; otherwise A4 must be trained explicitly.
    include_a4 = primary_model != "scheme2r"
    plan = build_ablation_run_plan(
        include_a4=include_a4,
        candidate_id=scheme2r_candidate,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "7.4",
                    "status": "dry_run",
                    "run_count": len(plan),
                    "ablations": list(ABLATION_NAMES),
                    "reused_ablations": [] if include_a4 else ["A4"],
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
    manifest_path = output_dir / "stage7_4_manifest.json"
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

    hyperparameters = scheme2r_reference["hyperparameters"]
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
                completed_result = dict(existing)
                completed_result["run_dir"] = str(run_dir)
                completed.append(completed_result)
                rows.append(_summary_row(existing, run_dir))
                print(
                    json.dumps(
                        {
                            "stage": "7.4",
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
            result = _run_one_ablation(
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
                        "stage": "7.4",
                        "status": "run_completed",
                        "run_id": run["run_id"],
                        "completed": len(completed),
                        "total": len(plan),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:  # preserve one-run failures and continue
            error = {
                "stage": "7.4",
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

    _write_csv(rows, output_dir / "ablation_runs.csv")
    _write_csv(_aggregate_rows(rows), output_dir / "ablation_summary_mean_std.csv")
    manifest = {
        "stage": "7.4",
        "status": "passed" if not failed and len(completed) == len(plan) else "failed",
        "contract": str(contract_path),
        "freeze_config": str(freeze_path),
        "dataset": contract["dataset"],
        "data_source": source_metadata,
        "cleaning_report": cleaning_report,
        "read_seconds": read_seconds,
        "ablations": list(ABLATION_NAMES),
        "trained_ablations": [name for name in ABLATION_NAMES if name != "A4"],
        "reused_ablations": {
            "A4": "stage7.3 primary model outputs with identical protocol, candidate, and seed"
        },
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
        "summary_files": ["ablation_runs.csv", "ablation_summary_mean_std.csv"],
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failed or len(completed) != len(plan):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
