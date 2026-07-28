"""阶段 6.2：仅使用全年训练集/验证集的候选模型筛选。"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .baselines import regression_metrics
from .data_pipeline import (
    FULL_SPLIT,
    HEEW_EXOG_COLUMNS,
    SMALL_SAMPLE_SPLIT,
    SplitSpec,
    TASKS,
    build_protocol_windows,
    clean_dataframe,
    read_heew_canonical,
    save_json,
    select_training_frame,
)
from .models import count_trainable_parameters, build_forecasting_model
from .stage6_contract import load_stage6_selection_contract
from .training import (
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    make_dataloader,
)


METRICS: Tuple[str, ...] = ("MAE", "RMSE", "WAPE", "MAPE")
EXPECTED_PREDICTION_TAIL = (4, 3)


def _limit_windows(
    windows: Mapping[str, np.ndarray], limit: int | None
) -> Dict[str, np.ndarray]:
    if limit is None:
        return dict(windows)
    if limit <= 0:
        raise ValueError("样本限制必须为正整数")
    return {key: value[:limit] for key, value in windows.items()}


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"不能写入空表：{path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_validation_gates(
    model: torch.nn.Module,
    model_name: str,
    loader,
    target_times: np.ndarray,
    output_path: Path,
) -> Dict[str, object] | None:
    """保存动态模型验证集门控，供阶段 6.4 使用；其他模型不输出。"""

    if model_name == "static_gate":
        static_gate = model.get_gate_matrix().detach().cpu().numpy().astype(np.float32)
        if static_gate.shape != (len(TASKS), len(TASKS)):
            raise ValueError(f"静态门控矩阵形状错误：{static_gate.shape}")
        gate_array = np.broadcast_to(
            static_gate, (len(target_times), len(TASKS), len(TASKS))
        ).copy()
        np.savez_compressed(
            output_path,
            gates=gate_array,
            target_times=target_times,
            gate_kind="static",
        )
        return {
            "file": output_path.name,
            "shape": [int(value) for value in gate_array.shape],
            "kind": "static",
            "row_semantics": "target_task",
            "column_semantics": "source_task",
        }

    if model_name not in {"dynamic_symmetric", "dynamic_directed"}:
        return None
    gates = []
    model.eval()
    with torch.no_grad():
        for loads, exog, _ in loader:
            representations = model.encode_tasks(loads, exog)
            state = model.encode_state(exog, representations)
            if model_name == "dynamic_symmetric":
                batch_gates = model.get_gate_matrix(state)
            else:
                batch_gates = model.get_gate_matrix(state, representations)
            gates.append(batch_gates.cpu().numpy())
    if not gates:
        raise ValueError("验证集门控导出为空")
    gate_array = np.concatenate(gates, axis=0).astype(np.float32)
    if gate_array.shape[1:] != (len(TASKS), len(TASKS)):
        raise ValueError(f"门控矩阵形状错误：{gate_array.shape}")
    np.savez_compressed(
        output_path,
        gates=gate_array,
        target_times=target_times,
        gate_kind="dynamic",
    )
    return {
        "file": output_path.name,
        "shape": [int(value) for value in gate_array.shape],
        "kind": "dynamic",
        "row_semantics": "target_task",
        "column_semantics": "source_task",
    }


def _metric_rows(
    model_name: str,
    candidate_id: str,
    metrics: Mapping[str, object],
) -> Tuple[Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    overall = metrics.get("overall_equal_task_mean")
    per_task = metrics.get("per_task")
    per_horizon = metrics.get("per_horizon_equal_element_mean")
    if not isinstance(overall, Mapping):
        raise ValueError(f"{model_name}/{candidate_id}缺少总体指标")
    if not isinstance(per_task, Mapping) or not isinstance(per_horizon, Mapping):
        raise ValueError(f"{model_name}/{candidate_id}缺少逐任务或逐步长指标")

    overall_row: Dict[str, object] = {
        "model": model_name,
        "candidate_id": candidate_id,
        **{metric: overall.get(metric) for metric in METRICS},
    }
    task_rows: list[Dict[str, object]] = []
    for task in TASKS:
        values = per_task.get(task)
        if not isinstance(values, Mapping):
            raise ValueError(f"{model_name}/{candidate_id}缺少任务指标：{task}")
        task_rows.append(
            {
                "model": model_name,
                "candidate_id": candidate_id,
                "task": task,
                **{metric: values.get(metric) for metric in METRICS},
            }
        )
    horizon_rows: list[Dict[str, object]] = []
    for step in ("step_1", "step_2", "step_3", "step_4"):
        values = per_horizon.get(step)
        if not isinstance(values, Mapping):
            raise ValueError(f"{model_name}/{candidate_id}缺少预测步指标：{step}")
        horizon_rows.append(
            {
                "model": model_name,
                "candidate_id": candidate_id,
                "horizon_step": step,
                **{metric: values.get(metric) for metric in METRICS},
            }
        )
    return overall_row, task_rows, horizon_rows


def write_validation_summaries(
    output_root: str | Path,
    completed_runs: Sequence[Mapping[str, object]],
) -> None:
    """把每个运行的验证 JSON 汇总为阶段 6.2 的三张 CSV 表。"""

    root = Path(output_root)
    overall_rows: list[Dict[str, object]] = []
    task_rows: list[Dict[str, object]] = []
    horizon_rows: list[Dict[str, object]] = []
    for run in completed_runs:
        run_dir = Path(str(run["run_dir"]))
        metrics_path = run_dir / "metrics_validation.json"
        with metrics_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        overall, tasks, horizons = _metric_rows(
            str(payload["model"]), str(payload["candidate_id"]), payload["metrics_original_scale"]
        )
        runtime = payload.get("runtime_seconds", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        overall.update(
            {
                "parameter_count": payload.get("model_parameter_count"),
                "fit_seconds": runtime.get("fit"),
                "validation_evaluation_seconds": runtime.get("validation_evaluation"),
                "validation_samples_per_second": runtime.get(
                    "validation_samples_per_second"
                ),
                "train_samples": payload.get("sample_counts", {}).get("train"),
                "validation_samples": payload.get("sample_counts", {}).get(
                    "validation"
                ),
                "best_validation_loss": payload.get(
                    "best_checkpoint", {}
                ).get("best_validation_loss"),
                "best_checkpoint_epoch": payload.get("best_checkpoint", {}).get(
                    "epoch"
                ),
            }
        )
        overall_rows.append(overall)
        task_rows.extend(tasks)
        horizon_rows.extend(horizons)

    overall_rows.sort(
        key=lambda row: (
            float(row["WAPE"]),
            float(row["RMSE"]),
            int(row["parameter_count"]),
        )
    )
    for rank, row in enumerate(overall_rows, start=1):
        row["validation_rank_by_WAPE"] = rank
    _write_csv(overall_rows, root / "validation_model_comparison.csv")
    _write_csv(task_rows, root / "validation_per_task.csv")
    _write_csv(horizon_rows, root / "validation_per_horizon.csv")


def write_stability_comparison(
    full_output_root: str | Path,
    small_output_root: str | Path,
) -> Dict[str, object]:
    """比较阶段 6.2 全年验证与阶段 6.3 小样本验证的排名和 WAPE。"""

    full_path = Path(full_output_root) / "validation_model_comparison.csv"
    small_root = Path(small_output_root)
    small_path = small_root / "validation_model_comparison.csv"
    if not full_path.exists():
        raise FileNotFoundError(f"找不到阶段 6.2 总体结果：{full_path}")
    if not small_path.exists():
        raise FileNotFoundError(f"找不到阶段 6.3 总体结果：{small_path}")

    def read_rows(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        result: Dict[Tuple[str, str], Dict[str, str]] = {}
        for row in rows:
            key = (str(row["model"]), str(row["candidate_id"]))
            result[key] = row
        return result

    full_rows = read_rows(full_path)
    small_rows = read_rows(small_path)
    if set(full_rows) != set(small_rows):
        raise ValueError("阶段 6.2 与阶段 6.3 的候选模型/超参数组合不一致")

    full_order = [
        key
        for key, _ in sorted(
            full_rows.items(), key=lambda item: int(item[1]["validation_rank_by_WAPE"])
        )
    ]
    small_order = [
        key
        for key, _ in sorted(
            small_rows.items(), key=lambda item: int(item[1]["validation_rank_by_WAPE"])
        )
    ]
    comparison: list[Dict[str, object]] = []
    for key in full_order:
        full_row = full_rows[key]
        small_row = small_rows[key]
        full_wape = float(full_row["WAPE"])
        small_wape = float(small_row["WAPE"])
        full_rank = int(full_row["validation_rank_by_WAPE"])
        small_rank = int(small_row["validation_rank_by_WAPE"])
        comparison.append(
            {
                "model": key[0],
                "candidate_id": key[1],
                "full_validation_WAPE": full_wape,
                "small_sample_validation_WAPE": small_wape,
                "WAPE_delta_small_minus_full": small_wape - full_wape,
                "full_rank": full_rank,
                "small_sample_rank": small_rank,
                "rank_change_small_minus_full": small_rank - full_rank,
            }
        )
    _write_csv(comparison, small_root / "validation_stability_comparison.csv")
    summary = {
        "full_reference": str(full_path),
        "small_sample_reference": str(small_path),
        "candidate_count": len(comparison),
        "rank_order_changed": full_order != small_order,
        "rank_order_completely_reversed": small_order == list(reversed(full_order)),
        "best_full_candidate": {
            "model": full_order[0][0],
            "candidate_id": full_order[0][1],
        },
        "best_small_sample_candidate": {
            "model": small_order[0][0],
            "candidate_id": small_order[0][1],
        },
        "test_set_accessed": False,
    }
    save_json(summary, small_root / "validation_stability_summary.json")
    return summary


def run_protocol_sweep(
    frame,
    output_root: str | Path,
    hyperparameter_candidates: Sequence[Mapping[str, object]],
    model_names: Sequence[str],
    exog_columns: Sequence[str] = HEEW_EXOG_COLUMNS,
    dataset_kind: str = "heew_total",
    split_spec: SplitSpec = FULL_SPLIT,
    protocol_name: str = "full",
    stage_name: str = "6.2",
    manifest_name: str = "stage6_2_manifest.json",
    lookback: int = 24,
    horizon: int = 4,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    early_stopping_patience: int = 12,
    grad_clip_norm: float = 1.0,
    threads: int = 8,
    seed: int = 2026,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> list[Dict[str, object]]:
    """执行阶段 6.2 全年验证集 sweep；函数绝不构造 test 窗口。"""

    if tuple(model_names) != tuple(load_stage6_selection_contract().candidate_models):
        raise ValueError("模型候选必须与阶段 6.1 契约完全一致")
    if lookback != 24 or horizon != 4:
        raise ValueError("阶段 6.2 固定使用 24→4 窗口")
    if not hyperparameter_candidates:
        raise ValueError("至少需要一组超参数候选")

    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    validation_cutoff = pd.Timestamp(split_spec.validation_end) + pd.Timedelta(
        hours=horizon - 1
    )
    # 阶段 6.2 只保留训练集和验证集所需的历史/目标范围，主动丢弃验证截止日之后的数据。
    source = source.loc[source["timestamp"] <= validation_cutoff].copy()
    cleaned, cleaning_report = clean_dataframe(source)
    available_exog = tuple(column for column in exog_columns if column in cleaned.columns)
    if not available_exog:
        raise ValueError("阶段 6.2 需要至少一个外生变量")

    # 只构建 train 和 validation；这里故意不调用 split_name="test"。
    raw_windows = {
        split_name: build_protocol_windows(
            cleaned,
            split_spec,
            split_name=split_name,
            lookback=lookback,
            horizon=horizon,
            exog_columns=available_exog,
        )
        for split_name in ("train", "validation")
    }
    windows = {
        "train": _limit_windows(raw_windows["train"], max_train_samples),
        "validation": _limit_windows(
            raw_windows["validation"], max_validation_samples
        ),
    }
    if any(len(windows[name]["target"]) == 0 for name in windows):
        raise ValueError("训练集或验证集没有可用窗口")

    stats = StandardizationStats.fit(
        select_training_frame(cleaned, split_spec), available_exog
    )
    standardized = {
        name: stats.transform_windows(value) for name, value in windows.items()
    }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stats.save(root / "normalization_stats.npz")

    completed_runs: list[Dict[str, object]] = []
    for model_name in model_names:
        for candidate in hyperparameter_candidates:
            candidate_id = str(candidate["candidate_id"])
            run_dir = root / "runs" / model_name / candidate_id
            run_dir.mkdir(parents=True, exist_ok=True)
            train_loader = make_dataloader(
                standardized["train"], batch_size=batch_size, shuffle=True
            )
            validation_loader = make_dataloader(
                standardized["validation"], batch_size=batch_size, shuffle=False
            )
            model = build_forecasting_model(
                model_name,
                exog_dim=len(available_exog),
                hidden_dim=int(candidate["hidden_dim"]),
                kernel_size=int(candidate["kernel_size"]),
                dilations=(1, 2),
                dropout=float(candidate["dropout"]),
                horizon=horizon,
            )
            trainer_config = TrainerConfig(
                learning_rate=float(candidate["learning_rate"]),
                weight_decay=weight_decay,
                grad_clip_norm=grad_clip_norm,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                torch_threads=threads,
                seed=seed,
            )
            checkpoint_path = run_dir / "best_model.pt"
            fit_started = time.perf_counter()
            history = fit_model(
                model,
                train_loader,
                validation_loader,
                trainer_config,
                checkpoint_path,
                input_mode="loads_and_exog",
            )
            fit_seconds = time.perf_counter() - fit_started
            evaluation_started = time.perf_counter()
            validation_loss, prediction_std, target_std = evaluate_model(
                model,
                validation_loader,
                "cpu",
                input_mode="loads_and_exog",
            )
            evaluation_seconds = time.perf_counter() - evaluation_started
            prediction = stats.inverse_targets(prediction_std)
            target = stats.inverse_targets(target_std)
            metrics = regression_metrics(target, prediction)
            np.savez_compressed(
                run_dir / "predictions_validation.npz",
                target=target,
                prediction=prediction,
                target_standardized=target_std,
                prediction_standardized=prediction_std,
                target_times=windows["validation"]["target_times"],
            )
            gate_export = _write_validation_gates(
                model,
                model_name,
                validation_loader,
                windows["validation"]["target_times"],
                run_dir / "gate_matrix_validation.npz",
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            payload = {
                "stage": stage_name,
                "dataset_kind": dataset_kind,
                "model": model_name,
                "candidate_id": candidate_id,
                "protocol": protocol_name,
                "tasks": list(TASKS),
                "exog_columns": list(available_exog),
                "window": {"lookback": lookback, "horizon": horizon},
                "input_mode": "loads_and_exog",
                "future_exogenous_used": False,
                "test_set_accessed": False,
                "sample_counts": {
                    name: int(len(value["target"])) for name, value in windows.items()
                },
                "metrics_original_scale": metrics,
                "validation_standardized_smooth_l1": float(validation_loss),
                "model_config": dict(candidate),
                "model_parameter_count": int(count_trainable_parameters(model)),
                "runtime_seconds": {
                    "fit": float(fit_seconds),
                    "validation_evaluation": float(evaluation_seconds),
                    "validation_samples_per_second": float(
                        len(windows["validation"]["target"])
                        / max(evaluation_seconds, 1e-9)
                    ),
                },
                "trainer_config": asdict(trainer_config),
                "best_checkpoint": {
                    "epoch": int(checkpoint["epoch"]),
                    "best_validation_loss": float(
                        checkpoint["best_validation_loss"]
                    ),
                    "file": checkpoint_path.name,
                },
                "gate_export": gate_export,
                "cleaning": cleaning_report,
                "normalization": stats.summary(),
            }
            save_json(payload, run_dir / "metrics_validation.json")
            save_json({"history": history}, run_dir / "history.json")
            save_json(
                {"candidate": dict(candidate), "test_set_accessed": False},
                run_dir / "run_manifest.json",
            )
            completed_runs.append({"model": model_name, "candidate_id": candidate_id, "run_dir": str(run_dir)})

    write_validation_summaries(root, completed_runs)
    save_json(
        {
            "stage": stage_name,
            "protocol": protocol_name,
            "candidate_run_count": len(completed_runs),
            "models": list(model_names),
            "hyperparameter_candidates": [dict(value) for value in hyperparameter_candidates],
            "test_set_accessed": False,
            "sample_counts": {
                name: int(len(value["target"])) for name, value in windows.items()
            },
            "cleaning": cleaning_report,
        },
        root / manifest_name,
    )
    return completed_runs


def run_validation_sweep(
    frame,
    output_root: str | Path,
    hyperparameter_candidates: Sequence[Mapping[str, object]],
    model_names: Sequence[str],
    **kwargs,
) -> list[Dict[str, object]]:
    """阶段 6.2 全年验证集 sweep 的兼容包装器。"""

    return run_protocol_sweep(
        frame=frame,
        output_root=output_root,
        hyperparameter_candidates=hyperparameter_candidates,
        model_names=model_names,
        split_spec=FULL_SPLIT,
        protocol_name="full",
        stage_name="6.2",
        manifest_name="stage6_2_manifest.json",
        **kwargs,
    )


def load_heew_for_validation(
    energy_path: str | Path,
    weather_path: str | Path,
):
    """读取并清洗 HEEW；不读取测试指标或测试预测。"""

    frame, resolved = read_heew_canonical(energy_path, weather_path)
    cleaned, cleaning_report = clean_dataframe(frame)
    return cleaned, resolved, cleaning_report
