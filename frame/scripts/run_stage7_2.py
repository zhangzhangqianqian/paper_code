"""Stage 7.2: small real-data CPU smoke acceptance for the frozen models.

This is an engineering gate, not a performance experiment.  It uses a small
prefix of the train/validation/test windows, one seed and two epochs.  The
test split is read only because Stage 7.0 has frozen model selection already.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch


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
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.models import build_forecasting_model, count_trainable_parameters  # noqa: E402
from src.stage7_ablations import ABLATION_NAMES, build_stage7_ablation_model  # noqa: E402
from src.training import (  # noqa: E402
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    load_checkpoint,
    make_dataloader,
    set_reproducible,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _limit_windows(windows: Mapping[str, np.ndarray], limit: int) -> Dict[str, np.ndarray]:
    if limit <= 0:
        raise ValueError("window sample limits must be positive")
    return {key: value[:limit] for key, value in windows.items()}


def _build_model(
    model_name: str,
    hyperparameters: Mapping[str, object],
    exog_dim: int,
    task_count: int,
    lookback: int,
    horizon: int,
):
    hidden_dim = int(hyperparameters["hidden_dim"])
    dropout = float(hyperparameters["dropout"])
    if model_name in ABLATION_NAMES:
        options = {
            "exog_dim": exog_dim,
            "task_count": task_count,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "horizon": horizon,
        }
        if model_name == "A0":
            options.update(
                {
                    "kernel_size": int(hyperparameters["kernel_size"]),
                    "dilations": (1, 2),
                }
            )
        else:
            options.update(
                {
                    "lookback": lookback,
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
        return build_stage7_ablation_model(model_name, **options)

    if model_name != "dynamic_symmetric":
        raise ValueError(f"unsupported Stage 7.2 model: {model_name}")
    return build_forecasting_model(
        model_name,
        exog_dim=exog_dim,
        task_count=task_count,
        hidden_dim=hidden_dim,
        kernel_size=int(hyperparameters["kernel_size"]),
        dilations=(1, 2),
        dropout=dropout,
        horizon=horizon,
    )


def _write_predictions(
    path: Path,
    stats: StandardizationStats,
    standardized_prediction: np.ndarray,
    standardized_target: np.ndarray,
    target_times: np.ndarray,
) -> None:
    prediction = stats.inverse_targets(standardized_prediction)
    target = stats.inverse_targets(standardized_target)
    np.savez_compressed(
        path,
        prediction=prediction,
        target=target,
        prediction_standardized=standardized_prediction,
        target_standardized=standardized_target,
        target_times=target_times,
    )


def _export_gate_summary(model, model_name: str, loader) -> Dict[str, object]:
    """Export only smoke-sized gate tensors for diagnostics."""

    gates = []
    with torch.no_grad():
        for loads, exog, _ in loader:
            if model_name == "dynamic_symmetric":
                representations = model.encode_tasks(loads, exog)
                state = model.encode_state(exog, representations)
                details = {"gates": model.get_gate_matrix(state)}
            else:
                _, details = model.forward_with_details(loads, exog)
            gates.append(details["gates"].cpu().numpy())
    if not gates:
        raise ValueError("cannot export an empty smoke gate tensor")
    array = np.concatenate(gates, axis=0).astype(np.float32)
    return {"shape": list(array.shape), "min": float(array.min()), "max": float(array.max())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 7.2 CPU smoke acceptance")
    parser.add_argument(
        "--kitakyushu-data-dir",
        default="D:/Paper/Kitakyushu dataset",
        help="directory containing the three official Kitakyushu ZIP packages",
    )
    parser.add_argument(
        "--contract",
        default="frame/configs/stage7r_contract.json",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage7_2_kitakyushu_smoke",
    )
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--max-validation-samples", type=int, default=64)
    parser.add_argument("--max-test-samples", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path = _resolve(args.contract)
    if not contract_path.exists():
        raise FileNotFoundError(f"Stage 7.0 contract not found: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("contract_version") != "stage7.0":
        raise ValueError("Stage 7.2 requires the stage7.0 contract")
    if tuple(contract.get("stage7_scope", {}).get("ablations", ())) != ABLATION_NAMES:
        raise ValueError("Stage 7.0 contract does not contain the frozen A0-A4 list")

    output_dir = _resolve(args.output_dir)
    manifest_path = output_dir / "stage7_2_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"manifest exists; use --force: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    lookback = int(contract["data_protocol"]["lookback"])
    horizon = int(contract["data_protocol"]["horizon"])
    task_count = len(contract["tasks"])
    if tuple(contract["tasks"]) != KITAKYUSHU_TASKS:
        raise ValueError("task order is inconsistent with Kitakyushu protocol")

    read_started = time.perf_counter()
    raw_frame, source_metadata = read_kitakyushu_canonical(
        _resolve(args.kitakyushu_data_dir), years=tuple(range(2015, 2022))
    )
    frame, cleaning_report = clean_kitakyushu_dataframe(raw_frame)
    read_seconds = time.perf_counter() - read_started
    raw_windows = {
        split_name: build_protocol_windows(
            frame,
            KITAKYUSHU_SPLIT,
            split_name=split_name,
            lookback=lookback,
            horizon=horizon,
            exog_columns=KITAKYUSHU_EXOG_COLUMNS,
            task_columns=KITAKYUSHU_TASKS,
        )
        for split_name in ("train", "validation", "test")
    }
    windows = {
        "train": _limit_windows(raw_windows["train"], args.max_train_samples),
        "validation": _limit_windows(
            raw_windows["validation"], args.max_validation_samples
        ),
        "test": _limit_windows(raw_windows["test"], args.max_test_samples),
    }
    stats = StandardizationStats.fit(
        select_training_frame(frame, KITAKYUSHU_SPLIT),
        KITAKYUSHU_EXOG_COLUMNS,
        task_columns=KITAKYUSHU_TASKS,
    )
    standardized = {name: stats.transform_windows(value) for name, value in windows.items()}
    stats.save(output_dir / "normalization_stats.npz")

    primary_hp = contract["models"]["primary"]["hyperparameters"]
    comparison_hp = contract["models"]["comparison"]["hyperparameters"]
    models = [*ABLATION_NAMES, "dynamic_symmetric"]
    hyperparameters = {
        **{name: primary_hp for name in ABLATION_NAMES},
        "dynamic_symmetric": comparison_hp,
    }
    completed = []
    failed = []
    for model_name in models:
        run_dir = output_dir / "runs" / model_name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            trainer_config = TrainerConfig(
                seed=args.seed,
                torch_threads=args.threads,
                learning_rate=float(hyperparameters[model_name]["learning_rate"]),
                weight_decay=float(contract["training_policy"]["weight_decay"]),
                grad_clip_norm=float(contract["training_policy"]["gradient_clip_norm"]),
                max_epochs=args.max_epochs,
                early_stopping_patience=args.patience,
            )
            set_reproducible(trainer_config)
            model = _build_model(
                model_name,
                hyperparameters[model_name],
                exog_dim=len(KITAKYUSHU_EXOG_COLUMNS),
                task_count=task_count,
                lookback=lookback,
                horizon=horizon,
            )
            train_loader = make_dataloader(
                standardized["train"], args.batch_size, shuffle=True, seed=args.seed
            )
            validation_loader = make_dataloader(
                standardized["validation"], args.batch_size, shuffle=False, seed=args.seed + 1
            )
            test_loader = make_dataloader(
                standardized["test"], args.batch_size, shuffle=False, seed=args.seed + 2
            )
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

            eval_started = time.perf_counter()
            validation_loss, validation_pred_std, validation_target_std = evaluate_model(
                model, validation_loader, "cpu", input_mode="loads_and_exog"
            )
            validation_seconds = time.perf_counter() - eval_started
            test_started = time.perf_counter()
            test_loss, test_pred_std, test_target_std = evaluate_model(
                model, test_loader, "cpu", input_mode="loads_and_exog"
            )
            test_seconds = time.perf_counter() - test_started

            validation_prediction = stats.inverse_targets(validation_pred_std)
            validation_target = stats.inverse_targets(validation_target_std)
            test_prediction = stats.inverse_targets(test_pred_std)
            test_target = stats.inverse_targets(test_target_std)
            validation_metrics = regression_metrics(
                validation_target, validation_prediction, task_names=KITAKYUSHU_TASKS
            )
            test_metrics = regression_metrics(
                test_target, test_prediction, task_names=KITAKYUSHU_TASKS
            )
            stats.save(run_dir / "normalization_stats.npz")
            save_json({"history": history}, run_dir / "history.json")
            save_json(validation_metrics, run_dir / "metrics_validation.json")
            save_json(test_metrics, run_dir / "metrics_test.json")
            _write_predictions(
                run_dir / "predictions_validation.npz",
                stats,
                validation_pred_std,
                validation_target_std,
                standardized["validation"]["target_times"],
            )
            _write_predictions(
                run_dir / "predictions_test.npz",
                stats,
                test_pred_std,
                test_target_std,
                standardized["test"]["target_times"],
            )
            gate_summary = _export_gate_summary(model, model_name, validation_loader)
            run_manifest = {
                "stage": "7.2",
                "model": model_name,
                "status": "passed",
                "dataset": contract["dataset"],
                "tasks": list(KITAKYUSHU_TASKS),
                "input_mode": "loads_and_exog",
                "future_exogenous_used": False,
                "test_set_accessed": True,
                "sample_counts": {
                    name: int(len(value["target"])) for name, value in standardized.items()
                },
                "window": {"lookback": lookback, "horizon": horizon},
                "output_shape": [args.batch_size, horizon, task_count],
                "parameter_count": count_trainable_parameters(model),
                "fit_seconds": fit_seconds,
                "validation_evaluation_seconds": validation_seconds,
                "test_evaluation_seconds": test_seconds,
                "validation_smooth_l1": validation_loss,
                "test_smooth_l1": test_loss,
                "best_checkpoint_epoch": int(checkpoint["epoch"]),
                "best_validation_loss": float(checkpoint["best_validation_loss"]),
                "gate_summary_validation": gate_summary,
                "trainer_config": asdict(trainer_config),
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
            save_json(run_manifest, run_dir / "run_manifest.json")
            completed.append(run_manifest)
        except Exception as exc:  # keep a durable record instead of hiding a failed run
            error = {
                "stage": "7.2",
                "model": model_name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            save_json(error, run_dir / "error.json")
            failed.append(error)

    manifest = {
        "stage": "7.2",
        "status": "passed" if not failed else "failed",
        "contract": str(contract_path),
        "dataset": contract["dataset"],
        "data_source": source_metadata,
        "cleaning_report": cleaning_report,
        "read_seconds": read_seconds,
        "protocol": {
            "lookback": lookback,
            "horizon": horizon,
            "task_order": list(KITAKYUSHU_TASKS),
            "sample_limits": {
                "train": args.max_train_samples,
                "validation": args.max_validation_samples,
                "test": args.max_test_samples,
            },
        },
        # The test windows are materialized before the per-model loop.  Keep
        # this true even if a model fails, so a failed run cannot be mistaken
        # for a run that never touched the test split.
        "test_set_accessed": True,
        "formal_training": False,
        "completed_runs": completed,
        "failed_runs": failed,
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
