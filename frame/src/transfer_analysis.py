"""阶段 6.5：验证集迁移收益与负迁移率分析。"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


TASKS: Tuple[str, ...] = ("electricity", "cooling", "heating")
METRICS: Tuple[str, ...] = ("MAE", "RMSE", "WAPE", "MAPE")
MTL_MODELS: Tuple[str, ...] = (
    "hard_share",
    "static_gate",
    "dynamic_symmetric",
    "dynamic_directed",
)
SEASONS: Tuple[str, ...] = ("winter", "spring", "summer", "autumn")
SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


@dataclass(frozen=True)
class ValidationRun:
    model: str
    candidate_id: str
    target: np.ndarray
    prediction: np.ndarray
    target_times: np.ndarray


def _safe_mape(actual: np.ndarray, prediction: np.ndarray, epsilon: float = 1e-6) -> float:
    denominator = np.maximum(np.abs(actual), epsilon)
    return float(np.mean(np.abs((actual - prediction) / denominator)) * 100.0)


def calculate_error_metrics(actual: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """计算一个任务/步长/季节单元的原始量纲误差。"""

    y_true = np.asarray(actual, dtype=np.float64)
    y_pred = np.asarray(prediction, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("误差输入形状必须相同且不能为空")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("误差输入不能包含 NaN 或 Inf")
    error = y_true - y_pred
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "WAPE": float(
            np.sum(np.abs(error)) / max(np.sum(np.abs(y_true)), 1e-6) * 100.0
        ),
        "MAPE": _safe_mape(y_true, y_pred),
    }


def calculate_gain(reference_error: float, joint_error: float) -> float:
    """计算 G=(E_STL-E_MTL)/E_STL*100；参照误差过小时返回 NaN。"""

    if not np.isfinite(reference_error) or not np.isfinite(joint_error):
        return float("nan")
    if abs(reference_error) < 1e-12:
        return float("nan")
    return float((reference_error - joint_error) / reference_error * 100.0)


def load_validation_run(
    root: str | Path,
    model: str,
    candidate_id: str,
    task_names: Sequence[str] = TASKS,
) -> ValidationRun:
    run_dir = Path(root) / "runs" / model / candidate_id
    prediction_path = run_dir / "predictions_validation.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"找不到验证预测：{prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as values:
        required = {"target", "prediction", "target_times"}
        missing = sorted(required - set(values.files))
        if missing:
            raise ValueError(f"验证预测缺少字段：{missing}")
        target = np.asarray(values["target"], dtype=np.float64)
        prediction = np.asarray(values["prediction"], dtype=np.float64)
        target_times = np.asarray(values["target_times"])
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError(
            f"{model}/{candidate_id}预测形状必须相同且为[N,H,T]："
            f"target={target.shape}, prediction={prediction.shape}"
        )
    if target.shape[2] != len(task_names) or target.shape[1] != 4:
        raise ValueError(
            f"{model}/{candidate_id}不是约定的[N,4,{len(task_names)}]输出"
        )
    if len(target_times) != target.shape[0]:
        raise ValueError(f"{model}/{candidate_id}时间戳数量与样本数不一致")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError(f"{model}/{candidate_id}验证预测包含 NaN 或 Inf")
    return ValidationRun(model, candidate_id, target, prediction, target_times)


def _season_for_steps(target_times: np.ndarray, horizon: int) -> np.ndarray:
    timestamps = pd.to_datetime(np.asarray(target_times).astype("datetime64[ns]"))
    result = np.empty((len(timestamps), horizon), dtype=object)
    for step in range(horizon):
        months = (timestamps + pd.to_timedelta(step, unit="h")).month.to_numpy()
        result[:, step] = [
            next(season for season, values in SEASON_MONTHS.items() if month in values)
            for month in months
        ]
    return result


def _unit_data(
    run: ValidationRun,
    granularity: str,
    task_index: int | None = None,
    horizon_step: int | None = None,
    season: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """提取一个 task/task×horizon/task×season×horizon 单元。"""

    if granularity == "task":
        if task_index is None:
            raise ValueError("task粒度需要task_index")
        return run.target[:, :, task_index : task_index + 1], run.prediction[:, :, task_index : task_index + 1]
    if granularity == "horizon":
        if horizon_step is None:
            raise ValueError("horizon粒度需要horizon_step")
        return run.target[:, horizon_step : horizon_step + 1, :], run.prediction[:, horizon_step : horizon_step + 1, :]
    if granularity == "season_horizon":
        if horizon_step is None or season is None:
            raise ValueError("season_horizon粒度需要horizon_step和season")
        seasons = _season_for_steps(run.target_times, run.target.shape[1])[:, horizon_step]
        mask = seasons == season
        return (
            run.target[mask, horizon_step : horizon_step + 1, :],
            run.prediction[mask, horizon_step : horizon_step + 1, :],
        )
    raise ValueError(f"未知分析粒度：{granularity}")


def _bootstrap_gain_from_arrays(
    actual: np.ndarray,
    reference_prediction: np.ndarray,
    joint_prediction: np.ndarray,
    metric: str,
    replicates: int,
    rng: np.random.Generator,
    block_size: int = 64,
) -> Tuple[float, float]:
    if replicates <= 0 or actual.shape[0] < 2:
        return float("nan"), float("nan")
    n = actual.shape[0]
    gains = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, block_size):
        count = min(block_size, replicates - start)
        indices = rng.integers(0, n, size=(count, n))
        actual_sample = actual[indices]
        ref_sample = reference_prediction[indices]
        joint_sample = joint_prediction[indices]
        ref_error = actual_sample - ref_sample
        joint_error = actual_sample - joint_sample
        axes = tuple(range(1, actual_sample.ndim))
        if metric == "MAE":
            ref_values = np.mean(np.abs(ref_error), axis=axes)
            joint_values = np.mean(np.abs(joint_error), axis=axes)
        elif metric == "RMSE":
            ref_values = np.sqrt(np.mean(ref_error**2, axis=axes))
            joint_values = np.sqrt(np.mean(joint_error**2, axis=axes))
        elif metric == "WAPE":
            ref_values = np.sum(np.abs(ref_error), axis=axes) / np.maximum(
                np.sum(np.abs(actual_sample), axis=axes), 1e-6
            ) * 100.0
            joint_values = np.sum(np.abs(joint_error), axis=axes) / np.maximum(
                np.sum(np.abs(actual_sample), axis=axes), 1e-6
            ) * 100.0
        elif metric == "MAPE":
            denominator = np.maximum(np.abs(actual_sample), 1e-6)
            ref_values = np.mean(np.abs(ref_error) / denominator, axis=axes) * 100.0
            joint_values = np.mean(np.abs(joint_error) / denominator, axis=axes) * 100.0
        else:
            raise ValueError(f"不支持的 bootstrap 指标：{metric}")
        gains[start : start + count] = (ref_values - joint_values) / np.maximum(
            np.abs(ref_values), 1e-12
        ) * 100.0
    return float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


def _gain_row(
    reference: ValidationRun,
    joint: ValidationRun,
    granularity: str,
    task_index: int | None,
    horizon_step: int | None,
    season: str | None,
    bootstrap_metric: str,
    bootstrap_granularities: Sequence[str],
    bootstrap_replicates: int,
    rng: np.random.Generator,
    task_names: Sequence[str] = TASKS,
) -> Dict[str, object]:
    reference_actual, reference_prediction = _unit_data(
        reference, granularity, task_index, horizon_step, season
    )
    joint_actual, joint_prediction = _unit_data(
        joint, granularity, task_index, horizon_step, season
    )
    if reference_actual.shape != joint_actual.shape:
        raise ValueError("STL与MTL单元形状不一致")
    if not np.allclose(reference_actual, joint_actual, rtol=0.0, atol=1e-6):
        raise ValueError("STL与MTL的验证真实目标不一致")
    reference_metrics = calculate_error_metrics(reference_actual, reference_prediction)
    joint_metrics = calculate_error_metrics(joint_actual, joint_prediction)
    rows = []
    for metric in METRICS:
        gain = calculate_gain(reference_metrics[metric], joint_metrics[metric])
        ci_low = float("nan")
        ci_high = float("nan")
        significance_computed = (
            metric == bootstrap_metric and granularity in bootstrap_granularities
        )
        significant = False if significance_computed else None
        if significance_computed:
            ci_low, ci_high = _bootstrap_gain_from_arrays(
                reference_actual,
                reference_prediction,
                joint_prediction,
                metric,
                bootstrap_replicates,
                rng,
            )
            significant = bool(np.isfinite(ci_high) and ci_high < 0.0)
        rows.append(
            {
                "protocol": "",
                "model": joint.model,
                "candidate_id": joint.candidate_id,
                "reference_model": reference.model,
                "reference_candidate_id": reference.candidate_id,
                "granularity": granularity,
                "task": "" if task_index is None else task_names[task_index],
                "horizon_step": "" if horizon_step is None else horizon_step + 1,
                "season": "" if season is None else season,
                "metric": metric,
                "sample_count": int(reference_actual.shape[0]),
                "reference_error": reference_metrics[metric],
                "joint_error": joint_metrics[metric],
                "gain_pct": gain,
                "gain_ci_low": ci_low,
                "gain_ci_high": ci_high,
                "significant_negative_transfer": significant,
                "significance_computed": significance_computed,
            }
        )
    return rows


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"不能写入空迁移分析表：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_transfer(
    input_root: str | Path,
    output_root: str | Path,
    protocol: str,
    error_metric: str = "WAPE",
    bootstrap_replicates: int = 500,
    bootstrap_granularities: Sequence[str] = ("task",),
    seed: int = 2026,
    task_names: Sequence[str] = TASKS,
) -> Dict[str, object]:
    if error_metric not in METRICS:
        raise ValueError(f"error_metric必须是{METRICS}")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates不能为负数")
    task_names = tuple(str(name) for name in task_names)
    if not task_names:
        raise ValueError("task_names不能为空")
    invalid_granularities = set(bootstrap_granularities) - {
        "task",
        "horizon",
        "season_horizon",
    }
    if invalid_granularities:
        raise ValueError(f"未知 bootstrap 粒度：{sorted(invalid_granularities)}")

    root = Path(input_root)
    out = Path(output_root)
    all_gains: list[Dict[str, object]] = []
    summary_rows: list[Dict[str, object]] = []
    rng = np.random.default_rng(seed)

    candidate_ids = ("H1", "H2", "H3", "H4")
    for candidate_id in candidate_ids:
        reference = load_validation_run(root, "stl", candidate_id, task_names)
        for model in MTL_MODELS:
            joint = load_validation_run(root, model, candidate_id, task_names)
            if not np.array_equal(reference.target_times, joint.target_times):
                raise ValueError(f"{model}/{candidate_id}与STL验证时间戳不一致")
            unit_specs = []
            for task_index in range(len(task_names)):
                unit_specs.append(("task", task_index, None, None))
            for horizon_step in range(reference.target.shape[1]):
                unit_specs.append(("horizon", None, horizon_step, None))
            seasons = _season_for_steps(reference.target_times, reference.target.shape[1])
            present_seasons = [
                season for season in SEASONS if np.any(np.isin(seasons, season))
            ]
            for horizon_step in range(reference.target.shape[1]):
                for season in present_seasons:
                    unit_specs.append(("season_horizon", None, horizon_step, season))

            candidate_rows: list[Dict[str, object]] = []
            for granularity, task_index, horizon_step, season in unit_specs:
                candidate_rows.extend(
                    _gain_row(
                        reference,
                        joint,
                        granularity,
                        task_index,
                        horizon_step,
                        season,
                        error_metric,
                        bootstrap_granularities,
                        bootstrap_replicates,
                        rng,
                        task_names,
                    )
                )
            for row in candidate_rows:
                row["protocol"] = protocol
            all_gains.extend(candidate_rows)

            for granularity in ("task", "horizon", "season_horizon"):
                for metric in METRICS:
                    subset = [
                        row
                        for row in candidate_rows
                        if row["granularity"] == granularity and row["metric"] == metric
                    ]
                    gains = np.asarray(
                        [row["gain_pct"] for row in subset], dtype=np.float64
                    )
                    finite = np.isfinite(gains)
                    negative = int(np.sum(gains[finite] < 0.0))
                    significance_evaluated = (
                        metric == error_metric
                        and granularity in bootstrap_granularities
                    )
                    significant = (
                        int(
                            sum(
                                bool(row["significant_negative_transfer"])
                                for row in subset
                            )
                        )
                        if significance_evaluated
                        else None
                    )
                    summary_rows.append(
                        {
                            "protocol": protocol,
                            "model": model,
                            "candidate_id": candidate_id,
                            "reference_model": "stl",
                            "granularity": granularity,
                            "metric": metric,
                            "unit_count": int(np.sum(finite)),
                            "negative_unit_count": negative,
                            "negative_transfer_rate_pct": float(
                                negative / max(int(np.sum(finite)), 1) * 100.0
                            ),
                            "significant_negative_unit_count": significant,
                            "significant_negative_transfer_rate_pct": (
                                float(significant / max(int(np.sum(finite)), 1) * 100.0)
                                if significance_evaluated
                                else None
                            ),
                            "significance_evaluated": significance_evaluated,
                            "bootstrap_metric": error_metric,
                            "bootstrap_granularities": ",".join(bootstrap_granularities),
                        }
                    )

    out.mkdir(parents=True, exist_ok=True)
    _write_csv(all_gains, out / "transfer_gains.csv")
    _write_csv(summary_rows, out / "negative_transfer_summary.csv")
    manifest = {
        "stage": "6.5",
        "protocol": protocol,
        "input_dir": str(root),
        "output_dir": str(out),
        "reference_model": "stl",
        "joint_models": list(MTL_MODELS),
        "candidate_ids": list(candidate_ids),
        "tasks": list(task_names),
        "error_metric_for_significance": error_metric,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_granularities": list(bootstrap_granularities),
        "random_seed": int(seed),
        "negative_transfer_definition": "gain_pct < 0",
        "significant_negative_transfer_definition": "bootstrap 95% CI upper bound of gain_pct < 0",
        "test_set_accessed": False,
        "gain_row_count": len(all_gains),
        "summary_row_count": len(summary_rows),
    }
    with (out / "stage6_5_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
