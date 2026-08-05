"""阶段2：非学习预测基线和基础误差指标。"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _as_history_array(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("history必须是[样本数,历史长度,任务数]三维数组")
    if values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("history不能包含空维度")
    return values


def persistence_forecast(history: np.ndarray, horizon: int = 4) -> np.ndarray:
    """Persistence：将最后一个已观测值重复到所有未来预测步。"""

    values = _as_history_array(history)
    if horizon <= 0:
        raise ValueError("horizon必须为正整数")
    last_value = values[:, -1:, :]
    return np.repeat(last_value, repeats=horizon, axis=1).astype(np.float32)


def seasonal_naive_forecast(
    history: np.ndarray,
    horizon: int = 4,
    season_length: int = 24,
) -> np.ndarray:
    """Seasonal Naive：使用前一个完整周期中相同相位的值。

    在当前实验中，season_length=24，表示用前一天相同时刻预测未来小时。
    若预测长度超过一个周期，则循环使用上一个周期的序列。
    """

    values = _as_history_array(history)
    if horizon <= 0 or season_length <= 0:
        raise ValueError("horizon和season_length必须为正整数")
    if values.shape[1] < season_length:
        raise ValueError(
            f"历史长度{values.shape[1]}小于季节周期{season_length}，"
            "无法执行Seasonal Naive"
        )
    last_cycle = values[:, -season_length:, :]
    indices = np.arange(horizon) % season_length
    return last_cycle[:, indices, :].astype(np.float32)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> float:
    denominator = np.maximum(np.abs(y_true), epsilon)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1e-6,
    task_names: Sequence[str] = ("electricity", "cooling", "heating"),
) -> Dict[str, object]:
    """返回总体、逐任务和逐预测步的误差指标。

    输入形状固定为[样本数,预测步数,任务数]。总体指标是先逐任务计算，
    再进行等权平均，避免不同负荷量级直接混合。
    """

    actual = np.asarray(y_true, dtype=np.float64)
    forecast = np.asarray(y_pred, dtype=np.float64)
    if actual.shape != forecast.shape or actual.ndim != 3:
        raise ValueError("y_true和y_pred必须形状相同且为[样本,步长,任务]三维数组")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise ValueError("指标输入不能包含NaN或Inf")
    task_names = tuple(str(name) for name in task_names)
    if len(task_names) != actual.shape[2]:
        raise ValueError(
            "task_names数量必须与预测数组任务维度一致："
            f"{len(task_names)} != {actual.shape[2]}"
        )

    errors = actual - forecast
    task_metrics: Dict[str, Dict[str, float]] = {}
    for task_index, task_name in enumerate(task_names):
        task_actual = actual[:, :, task_index]
        task_forecast = forecast[:, :, task_index]
        task_error = task_actual - task_forecast
        task_metrics[task_name] = {
            "MAE": float(np.mean(np.abs(task_error))),
            "RMSE": float(np.sqrt(np.mean(task_error**2))),
            "WAPE": float(
                np.sum(np.abs(task_error))
                / max(np.sum(np.abs(task_actual)), epsilon)
                * 100.0
            ),
            "MAPE": _safe_mape(task_actual, task_forecast, epsilon),
        }

    metric_names = ("MAE", "RMSE", "WAPE", "MAPE")
    horizon_metrics: Dict[str, Dict[str, float]] = {}
    horizon_task_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for horizon_index in range(actual.shape[1]):
        step_name = f"step_{horizon_index + 1}"
        task_values: Dict[str, Dict[str, float]] = {}
        for task_index, task_name in enumerate(task_names):
            step_actual = actual[:, horizon_index, task_index]
            step_forecast = forecast[:, horizon_index, task_index]
            step_error = step_actual - step_forecast
            task_values[task_name] = {
                "MAE": float(np.mean(np.abs(step_error))),
                "RMSE": float(np.sqrt(np.mean(step_error**2))),
                "WAPE": float(
                    np.sum(np.abs(step_error))
                    / max(np.sum(np.abs(step_actual)), epsilon)
                    * 100.0
                ),
                "MAPE": _safe_mape(step_actual, step_forecast, epsilon),
            }
        horizon_task_metrics[step_name] = task_values
        horizon_metrics[step_name] = {
            metric: float(np.mean([values[metric] for values in task_values.values()]))
            for metric in metric_names
        }

    overall = {
        metric: float(np.mean([values[metric] for values in task_metrics.values()]))
        for metric in metric_names
    }
    return {
        "overall_equal_task_mean": overall,
        "per_task": task_metrics,
        "per_horizon_equal_task_mean": horizon_metrics,
        # Kept as an alias for older reports; its semantics are now explicitly
        # equal-task averaging rather than mixing physical units.
        "per_horizon_equal_element_mean": horizon_metrics,
        "sample_count": int(actual.shape[0]),
        "horizon": int(actual.shape[1]),
        "task_count": int(actual.shape[2]),
    }
