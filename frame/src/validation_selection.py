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
from .models import (
    MATCHED_STL_MODEL_NAME,
    SCHEME2R_MODEL_NAME,
    count_trainable_parameters,
    build_forecasting_model,
)
from .stage7_reference import MatchedSingleTaskScheme2RModel
from .kitakyushu_pipeline import (
    KITAKYUSHU_SPLIT,
    clean_kitakyushu_dataframe,
)
from .stage6_contract import load_stage6_selection_contract
from .training import (
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    make_dataloader,
    set_reproducible,
)


METRICS: Tuple[str, ...] = ("MAE", "RMSE", "WAPE", "MAPE")


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
    task_names: Sequence[str] = TASKS,
) -> Dict[str, object] | None:
    """保存动态模型验证集门控，供阶段 6.4 使用；其他模型不输出。"""

    if model_name == "static_gate":
        static_gate = model.get_gate_matrix().detach().cpu().numpy().astype(np.float32)
        if static_gate.shape != (len(task_names), len(task_names)):
            raise ValueError(f"静态门控矩阵形状错误：{static_gate.shape}")
        gate_array = np.broadcast_to(
            static_gate, (len(target_times), len(task_names), len(task_names))
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

    if model_name == SCHEME2R_MODEL_NAME:
        gates = []
        rho_values = []
        pi_values = []
        model.eval()
        with torch.no_grad():
            for loads, exog, _ in loader:
                _, details = model.forward_with_details(loads, exog)
                gates.append(details["gates"].cpu().numpy())
                rho_values.append(details["rho"].cpu().numpy())
                pi_values.append(details["pi"].cpu().numpy())
        if not gates:
            raise ValueError("Scheme2R验证集门控导出为空")
        gate_array = np.concatenate(gates, axis=0).astype(np.float32)
        rho_array = np.concatenate(rho_values, axis=0).astype(np.float32)
        pi_array = np.concatenate(pi_values, axis=0).astype(np.float32)
        expected_gate_shape = (
            len(target_times),
            4,
            len(task_names),
            len(task_names),
        )
        if tuple(gate_array.shape) != expected_gate_shape:
            raise ValueError(
                f"Scheme2R门控数组形状错误：{gate_array.shape}，期望{expected_gate_shape}"
            )
        np.savez_compressed(
            output_path,
            gates=gate_array,
            rho=rho_array,
            pi=pi_array,
            target_times=target_times,
            gate_kind="dynamic_step",
        )
        return {
            "file": output_path.name,
            "shape": [int(value) for value in gate_array.shape],
            "kind": "dynamic_step",
            "axes": ["sample", "forecast_step", "target_task", "source_task"],
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
    if gate_array.shape[1:] != (len(task_names), len(task_names)):
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
    task_names: Sequence[str] = TASKS,
) -> Tuple[Dict[str, object], list[Dict[str, object]], list[Dict[str, object]]]:
    overall = metrics.get("overall_equal_task_mean")
    per_task = metrics.get("per_task")
    per_horizon = metrics.get("per_horizon_equal_task_mean")
    if not isinstance(per_horizon, Mapping):
        per_horizon = metrics.get("per_horizon_equal_element_mean")
    if not isinstance(overall, Mapping):
        raise ValueError(f"{model_name}/{candidate_id}缺少总体指标")
    if not isinstance(per_task, Mapping) or not isinstance(per_horizon, Mapping):
        raise ValueError(f"{model_name}/{candidate_id}缺少逐任务或逐步长指标")

    overall_row: Dict[str, object] = {
        "model": model_name,
        "candidate_id": candidate_id,
        **{metric: overall.get(metric) for metric in METRICS},
        "validation_max_per_task_WAPE": max(
            float(values.get("WAPE"))
            for values in per_task.values()
            if isinstance(values, Mapping)
        ),
    }
    task_rows: list[Dict[str, object]] = []
    for task in task_names:
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
    task_names: Sequence[str] = TASKS,
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
            str(payload["model"]),
            str(payload["candidate_id"]),
            payload["metrics_original_scale"],
            task_names=task_names,
        )
        runtime = payload.get("runtime_seconds", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        best_checkpoint = payload.get("best_checkpoint", {})
        if not isinstance(best_checkpoint, Mapping):
            best_checkpoint = {}
        if not best_checkpoint and isinstance(payload.get("best_checkpoints"), Mapping):
            task_checkpoints = payload["best_checkpoints"]
            best_checkpoint = {
                "best_validation_loss": float(
                    np.mean(
                        [
                            float(value["best_validation_loss"])
                            for value in task_checkpoints.values()
                        ]
                    )
                ),
                "epoch": float(
                    np.mean([float(value["epoch"]) for value in task_checkpoints.values()])
                ),
            }
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
                "best_validation_loss": best_checkpoint.get("best_validation_loss"),
                "best_checkpoint_epoch": best_checkpoint.get("epoch"),
            }
        )
        overall_rows.append(overall)
        task_rows.extend(tasks)
        horizon_rows.extend(horizons)

    # The STL rows are the only scientifically valid task-level reference for
    # a transfer-rate tie-breaker.  Compute it after all candidates are read so
    # that each model/candidate is compared against the matching STL candidate.
    stl_task_wape = {
        (str(row["candidate_id"]), str(row["task"])): float(row["WAPE"])
        for row in task_rows
        if str(row["model"]) == "stl_matched"
    }
    for row in overall_rows:
        model = str(row["model"])
        candidate = str(row["candidate_id"])
        task_values = [
            float(task_row["WAPE"])
            for task_row in task_rows
            if str(task_row["model"]) == model
            and str(task_row["candidate_id"]) == candidate
        ]
        reference_values = [
            stl_task_wape[(candidate, task)]
            for task in task_names
            if (candidate, task) in stl_task_wape
        ]
        row["validation_negative_transfer_rate"] = (
            0.0
            if model == "stl_matched"
            else (
                float(sum(value > reference for value, reference in zip(task_values, reference_values)))
                / float(len(reference_values))
                if len(reference_values) == len(task_values) and reference_values
                else None
            )
        )

    def _numeric_or_inf(value: object) -> float:
        try:
            return float(value) if value is not None else float("inf")
        except (TypeError, ValueError):
            return float("inf")

    overall_rows.sort(
        key=lambda row: (
            float(row["WAPE"]),
            _numeric_or_inf(row.get("validation_max_per_task_WAPE")),
            _numeric_or_inf(row.get("validation_negative_transfer_rate")),
            _numeric_or_inf(row.get("parameter_count")),
            _numeric_or_inf(row.get("fit_seconds")),
            str(row["model"]),
            str(row["candidate_id"]),
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
    """比较阶段 6.2 全年验证与阶段 6.3 小样本验证的排名和 WAPE。

    阶段 6.3 只复训阶段 6.2 排名筛选出的候选，因此小样本结果可以是
    全年候选集合的严格子集；对两者的交集进行稳定性比较。
    """

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
    full_keys = set(full_rows)
    small_keys = set(small_rows)
    if not small_keys.issubset(full_keys):
        unknown = sorted(small_keys - full_keys)
        raise ValueError(
            "阶段 6.3 包含阶段 6.2 未筛选的候选模型/超参数组合："
            f"{unknown}"
        )
    compared_keys = full_keys & small_keys
    if not compared_keys:
        raise ValueError("阶段 6.2 与阶段 6.3 没有可比较的候选组合")

    full_order = [
        key
        for key, _ in sorted(
            ((key, full_rows[key]) for key in compared_keys),
            key=lambda item: int(item[1]["validation_rank_by_WAPE"]),
        )
    ]
    small_order = [
        key
        for key, _ in sorted(
            ((key, small_rows[key]) for key in compared_keys),
            key=lambda item: int(item[1]["validation_rank_by_WAPE"]),
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
        "full_candidate_count": len(full_rows),
        "small_sample_candidate_count": len(small_rows),
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


def select_stage6_3_candidates(
    stage6_2_root: str | Path,
    contract=None,
    required_model: str | None = None,
) -> Dict[str, object]:
    """从阶段 6.2 总体 WAPE 表自动选择阶段 6.3 的模型和配置。

    直接按全年验证集总体 WAPE 及契约规定的确定性 tie-breaker 选择前两名。
    ``required_model`` 仅保留为显式兼容参数；正式 Stage 6-R 不传入它，
    因而不会强行保留提出模型。返回的配置来自阶段 6.1 契约。
    """

    root = Path(stage6_2_root)
    comparison_path = root / "validation_model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"找不到阶段6.2总体结果：{comparison_path}")
    if contract is None:
        contract = load_stage6_selection_contract()

    contract_models = tuple(contract.candidate_models)
    contract_model_set = set(contract_models)
    candidate_by_id = {
        str(candidate["candidate_id"]): dict(candidate)
        for candidate in contract.hyperparameter_candidates
    }
    with comparison_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {"model", "candidate_id", "WAPE"}
    if not rows or not required_columns.issubset(rows[0]):
        raise ValueError(
            f"阶段6.2结果缺少选择所需列：{sorted(required_columns)}"
        )

    validated_rows: list[Dict[str, object]] = []
    best_by_model: Dict[str, Dict[str, object]] = {}
    for row in rows:
        model = str(row["model"])
        candidate_id = str(row["candidate_id"])
        if model not in contract_model_set:
            raise ValueError(f"阶段6.2结果包含契约外模型：{model}")
        if candidate_id not in candidate_by_id:
            raise ValueError(f"阶段6.2结果包含契约外配置：{candidate_id}")
        try:
            wape = float(row["WAPE"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"阶段6.2结果中的WAPE不可解析：{model}/{candidate_id}"
            ) from exc
        if not np.isfinite(wape):
            raise ValueError(f"阶段6.2结果中的WAPE不是有限值：{model}/{candidate_id}")
        normalized = {
            **dict(row),
            "model": model,
            "candidate_id": candidate_id,
            "WAPE": wape,
            "full_validation_rank": int(row.get("validation_rank_by_WAPE", 0) or 0),
        }
        validated_rows.append(normalized)
        current = best_by_model.get(model)
        if current is None or (wape, candidate_id) < (
            float(current["WAPE"]), str(current["candidate_id"])
        ):
            best_by_model[model] = normalized

    missing_models = sorted(contract_model_set - set(best_by_model))
    if missing_models:
        raise ValueError(
            "阶段6.2结果缺少候选模型，不能进行阶段6.3筛选："
            f"{missing_models}"
        )
    def numeric_or_inf(value: object) -> float:
        try:
            return float(value) if value is not None else float("inf")
        except (TypeError, ValueError):
            return float("inf")

    ranked_models = sorted(
        validated_rows,
        key=lambda row: (
            float(row["WAPE"]),
            numeric_or_inf(row.get("validation_max_per_task_WAPE")),
            numeric_or_inf(row.get("validation_negative_transfer_rate")),
            numeric_or_inf(row.get("parameter_count")),
            numeric_or_inf(row.get("fit_seconds")),
            str(row["model"]),
            str(row["candidate_id"]),
        ),
    )
    selected_rows = ranked_models[:2]
    if required_model is not None and required_model not in {str(row["model"]) for row in selected_rows}:
        selected_rows.append(best_by_model[required_model])

    selected_models = tuple(str(row["model"]) for row in selected_rows)
    selected_model_candidates: Dict[str, tuple[Mapping[str, object], ...]] = {}
    for row in selected_rows:
        model = str(row["model"])
        selected_model_candidates[model] = selected_model_candidates.get(model, ()) + (
            candidate_by_id[str(row["candidate_id"])],
        )
    selected_candidate_ids = {
        str(candidate["candidate_id"])
        for candidates in selected_model_candidates.values()
        for candidate in candidates
    }
    selected_candidates = tuple(
        candidate
        for candidate_id, candidate in candidate_by_id.items()
        if candidate_id in selected_candidate_ids
    )
    return {
        "models": selected_models,
        "hyperparameter_candidates": selected_candidates,
        "model_hyperparameter_candidates": selected_model_candidates,
        "selected_rows": selected_rows,
        "full_ranking_by_model": ranked_models,
        "required_model": required_model,
    }


def _inverse_single_task(
    values: np.ndarray,
    stats: StandardizationStats,
    task_index: int,
) -> np.ndarray:
    """Inverse-transform one task without borrowing another task's scale."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 1:
        raise ValueError("single-task values must have shape [N, horizon, 1]")
    return (
        values * stats.load_scale[task_index]
        + stats.load_mean[task_index]
    ).astype(np.float32)


def _run_matched_stl_candidate(
    *,
    run_dir: Path,
    candidate: Mapping[str, object],
    windows: Mapping[str, Mapping[str, np.ndarray]],
    stats: StandardizationStats,
    task_names: Sequence[str],
    exog_columns: Sequence[str],
    protocol_name: str,
    stage_name: str,
    batch_size: int,
    weight_decay: float,
    max_epochs: int,
    early_stopping_patience: int,
    grad_clip_norm: float,
    threads: int,
    seed: int,
    lookback: int,
    horizon: int,
    dropout: float | None = None,
) -> Dict[str, object]:
    """Train the strict structure-matched STL candidate task by task.

    Each task receives only its own historical load channel and the shared
    historical exogenous window.  Separate checkpoints and early stopping
    histories prevent the aggregate loss of a wrapper model from masking a
    task-level result.
    """

    if len(task_names) != 4:
        raise ValueError("matched STL currently requires the four Kitakyushu tasks")
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate_id = str(candidate["candidate_id"])
    task_predictions: list[np.ndarray] = []
    task_targets: list[np.ndarray] = []
    task_histories: Dict[str, object] = {}
    task_checkpoints: Dict[str, object] = {}
    task_runtime: Dict[str, Dict[str, float]] = {}
    task_parameter_counts: Dict[str, int] = {}
    task_seeds: Dict[str, int] = {}
    validation_losses: list[float] = []

    for task_index, task_name in enumerate(task_names):
        task_seed = int(seed + 1000 * task_index)
        task_seeds[str(task_name)] = task_seed
        config = TrainerConfig(
            learning_rate=float(candidate["learning_rate"]),
            weight_decay=weight_decay,
            grad_clip_norm=grad_clip_norm,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            torch_threads=threads,
            seed=task_seed,
        )
        set_reproducible(config)
        model = MatchedSingleTaskScheme2RModel(
            exog_dim=len(exog_columns),
            task_index=task_index,
            task_count=len(task_names),
            hidden_dim=int(candidate["hidden_dim"]),
            lookback=lookback,
            kernel_size=int(candidate["scheme2r_kernel_size"]),
            dilations=tuple(int(value) for value in candidate["scheme2r_dilations"]),
            dropout=float(candidate["dropout"] if dropout is None else dropout),
            horizon=horizon,
            head_hidden_dim=int(candidate["prediction_head_hidden_dim"]),
        )
        task_parameter_counts[str(task_name)] = count_trainable_parameters(model)
        task_windows = {
            split_name: {
                "loads": values["loads"],
                "exog": values["exog"],
                "target": values["target"][:, :, task_index : task_index + 1],
                "target_times": values["target_times"],
            }
            for split_name, values in windows.items()
        }
        train_loader = make_dataloader(
            task_windows["train"], batch_size=batch_size, shuffle=True, seed=task_seed
        )
        validation_loader = make_dataloader(
            task_windows["validation"],
            batch_size=batch_size,
            shuffle=False,
            seed=task_seed + 1,
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
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        evaluation_started = time.perf_counter()
        validation_loss, prediction_std, target_std = evaluate_model(
            model,
            validation_loader,
            "cpu",
            input_mode="loads_and_exog",
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        validation_losses.append(float(validation_loss))
        task_predictions.append(_inverse_single_task(prediction_std, stats, task_index))
        task_targets.append(_inverse_single_task(target_std, stats, task_index))
        task_histories[str(task_name)] = history
        task_checkpoints[str(task_name)] = {
            "file": checkpoint_path.name,
            "epoch": int(checkpoint["epoch"]),
            "best_validation_loss": float(checkpoint["best_validation_loss"]),
            "trainer_config": asdict(config),
        }
        task_runtime[str(task_name)] = {
            "fit": float(fit_seconds),
            "validation_evaluation": float(evaluation_seconds),
        }

    prediction = np.concatenate(task_predictions, axis=2)
    target = np.concatenate(task_targets, axis=2)
    metrics = regression_metrics(target, prediction, task_names=task_names)
    np.savez_compressed(
        run_dir / "predictions_validation.npz",
        target=target,
        prediction=prediction,
        target_times=windows["validation"]["target_times"],
    )
    stats.save(run_dir / "normalization_stats.npz")
    save_json({"per_task": task_histories}, run_dir / "history.json")
    save_json(metrics, run_dir / "metrics_validation.json")
    save_json(
        {
            "model": MATCHED_STL_MODEL_NAME,
            "candidate_id": candidate_id,
            "protocol": protocol_name,
            "task_models_trained_independently": True,
            "test_set_accessed": False,
        },
        run_dir / "run_manifest.json",
    )
    total_fit = sum(item["fit"] for item in task_runtime.values())
    total_eval = sum(item["validation_evaluation"] for item in task_runtime.values())
    return {
        "stage": stage_name,
        "dataset_kind": "kitakyushu_energy_station",
        "model": MATCHED_STL_MODEL_NAME,
        "candidate_id": candidate_id,
        "protocol": protocol_name,
        "tasks": list(task_names),
        "exog_columns": list(exog_columns),
        "window": {"lookback": lookback, "horizon": horizon},
        "input_mode": "loads_and_exog",
        "future_exogenous_used": False,
        "test_set_accessed": False,
        "task_models_trained_independently": True,
        "sample_counts": {
            name: int(len(value["target"])) for name, value in windows.items()
        },
        "metrics_original_scale": metrics,
        "validation_standardized_smooth_l1": float(np.mean(validation_losses)),
        "model_config": dict(candidate),
        "model_parameter_count": int(sum(task_parameter_counts.values())),
        "task_parameter_counts": task_parameter_counts,
        "runtime_seconds": {
            "fit": float(total_fit),
            "validation_evaluation": float(total_eval),
            "validation_samples_per_second": float(
                len(windows["validation"]["target"])
                / max(total_eval, 1e-9)
            ),
        },
        "task_seeds": task_seeds,
        "best_checkpoints": task_checkpoints,
        "task_runtime_seconds": task_runtime,
        "cleaning": {},
        "normalization": stats.summary(),
    }


def run_protocol_sweep(
    frame,
    output_root: str | Path,
    hyperparameter_candidates: Sequence[Mapping[str, object]],
    model_names: Sequence[str],
    model_hyperparameter_candidates: Mapping[
        str, Sequence[Mapping[str, object]]
    ] | None = None,
    exog_columns: Sequence[str] = HEEW_EXOG_COLUMNS,
    task_names: Sequence[str] = TASKS,
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

    contract_models = set(load_stage6_selection_contract().candidate_models)
    model_names = tuple(str(name) for name in model_names)
    if not model_names or not set(model_names).issubset(contract_models):
        raise ValueError("模型候选必须来自阶段 6.1 契约")
    if lookback != 24 or horizon != 4:
        raise ValueError("阶段 6.2 固定使用 24→4 窗口")
    if not hyperparameter_candidates:
        raise ValueError("至少需要一组超参数候选")
    if model_hyperparameter_candidates is None:
        candidates_by_model = {
            model_name: tuple(hyperparameter_candidates)
            for model_name in model_names
        }
    else:
        candidates_by_model = {
            str(model_name): tuple(candidates)
            for model_name, candidates in model_hyperparameter_candidates.items()
        }
        if set(candidates_by_model) != set(model_names):
            raise ValueError("模型专属超参数映射必须覆盖且仅覆盖当前模型")
        if any(not candidates for candidates in candidates_by_model.values()):
            raise ValueError("每个模型至少需要一组超参数")

    task_names = tuple(str(name) for name in task_names)
    if not task_names:
        raise ValueError("task_names不能为空")

    source = frame.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    validation_cutoff = pd.Timestamp(split_spec.validation_end)
    # 阶段 6.2 只保留训练集和验证集所需的历史/目标范围，主动丢弃验证截止日之后的数据。
    source = source.loc[source["timestamp"] <= validation_cutoff].copy()
    if dataset_kind == "kitakyushu_energy_station":
        cleaned, cleaning_report = clean_kitakyushu_dataframe(source)
    else:
        cleaned, cleaning_report = clean_dataframe(source, task_columns=task_names)
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
            task_columns=task_names,
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
        select_training_frame(cleaned, split_spec),
        available_exog,
        task_columns=task_names,
    )
    standardized = {
        name: stats.transform_windows(value) for name, value in windows.items()
    }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stats.save(root / "normalization_stats.npz")

    completed_runs: list[Dict[str, object]] = []
    for model_name in model_names:
        for candidate in candidates_by_model[model_name]:
            candidate_id = str(candidate["candidate_id"])
            run_dir = root / "runs" / model_name / candidate_id
            run_dir.mkdir(parents=True, exist_ok=True)
            if model_name == MATCHED_STL_MODEL_NAME:
                payload = _run_matched_stl_candidate(
                    run_dir=run_dir,
                    candidate=candidate,
                    windows=standardized,
                    stats=stats,
                    task_names=task_names,
                    exog_columns=available_exog,
                    protocol_name=protocol_name,
                    stage_name=stage_name,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    max_epochs=max_epochs,
                    early_stopping_patience=early_stopping_patience,
                    grad_clip_norm=grad_clip_norm,
                    threads=threads,
                    seed=seed,
                    lookback=lookback,
                    horizon=horizon,
                )
                save_json(payload, run_dir / "metrics_validation.json")
                completed_runs.append(
                    {
                        "model": model_name,
                        "candidate_id": candidate_id,
                        "run_dir": str(run_dir),
                    }
                )
                continue
            trainer_config = TrainerConfig(
                learning_rate=float(candidate["learning_rate"]),
                weight_decay=weight_decay,
                grad_clip_norm=grad_clip_norm,
                max_epochs=max_epochs,
                early_stopping_patience=early_stopping_patience,
                torch_threads=threads,
                seed=seed,
            )
            # Seed before both DataLoader construction and model initialization.
            # The previous order allowed initialization and batch shuffling to
            # vary while the manifest still reported the same seed.
            set_reproducible(trainer_config)
            train_loader = make_dataloader(
                standardized["train"], batch_size=batch_size, shuffle=True, seed=seed
            )
            validation_loader = make_dataloader(
                standardized["validation"],
                batch_size=batch_size,
                shuffle=False,
                seed=seed + 1,
            )
            model_kwargs = {
                "exog_dim": len(available_exog),
                "hidden_dim": int(candidate["hidden_dim"]),
                "dropout": float(candidate["dropout"]),
                "horizon": horizon,
                "task_count": len(task_names),
            }
            if model_name == SCHEME2R_MODEL_NAME:
                model_kwargs.update(
                    {
                        "lookback": lookback,
                        "kernel_size": int(candidate["scheme2r_kernel_size"]),
                        "dilations": tuple(candidate["scheme2r_dilations"]),
                        "rank": int(candidate["scheme2r_rank"]),
                        "gate_hidden_dim": int(
                            candidate["scheme2r_gate_hidden_dim"]
                        ),
                        "step_embedding_dim": int(
                            candidate["scheme2r_step_embedding_dim"]
                        ),
                    }
                )
            else:
                model_kwargs.update(
                    {
                        "lookback": lookback,
                        "kernel_size": int(candidate["kernel_size"]),
                        "dilations": tuple(
                            int(value) for value in candidate["dilations"]
                        ),
                        "head_hidden_dim": int(
                            candidate["prediction_head_hidden_dim"]
                        ),
                    }
                )
            model = build_forecasting_model(model_name, **model_kwargs)
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
            metrics = regression_metrics(target, prediction, task_names=task_names)
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
                task_names=task_names,
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
                "tasks": list(task_names),
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

    write_validation_summaries(root, completed_runs, task_names=task_names)
    save_json(
        {
            "stage": stage_name,
            "protocol": protocol_name,
            "candidate_run_count": len(completed_runs),
            "models": list(model_names),
            "tasks": list(task_names),
            "hyperparameter_candidates": [dict(value) for value in hyperparameter_candidates],
            **(
                {
                    "model_hyperparameter_candidates": {
                        model: [dict(value) for value in candidates]
                        for model, candidates in candidates_by_model.items()
                    }
                }
                if model_hyperparameter_candidates is not None
                else {}
            ),
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
    dataset_kind: str = "heew_total",
    task_names: Sequence[str] = TASKS,
    split_spec: SplitSpec | None = None,
    **kwargs,
) -> list[Dict[str, object]]:
    """阶段 6.2 全年验证集 sweep 的兼容包装器。"""

    if split_spec is None:
        split_spec = (
            KITAKYUSHU_SPLIT
            if dataset_kind == "kitakyushu_energy_station"
            else FULL_SPLIT
        )
    return run_protocol_sweep(
        frame=frame,
        output_root=output_root,
        hyperparameter_candidates=hyperparameter_candidates,
        model_names=model_names,
        split_spec=split_spec,
        protocol_name="full",
        stage_name="6.2",
        manifest_name="stage6_2_manifest.json",
        dataset_kind=dataset_kind,
        task_names=task_names,
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
