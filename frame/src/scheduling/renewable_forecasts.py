"""阶段10.4：仅用训练/验证期选择可再生能源透明预测器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


RENEWABLE_METHODS: Tuple[str, ...] = (
    "persistence",
    "seasonal_naive_24",
    "seasonal_naive_168",
    "dlinear_lite",
)
PROFILE_COLUMNS: Tuple[str, ...] = ("pv_available", "wt_available")


def _validate_profile(frame: pd.DataFrame) -> None:
    missing = [column for column in ("timestamp", *PROFILE_COLUMNS) if column not in frame]
    if missing:
        raise ValueError(f"可再生能源序列缺少字段：{missing}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("可再生能源序列必须按唯一时间戳升序排列")
    if not np.isfinite(frame[list(PROFILE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("可再生能源序列不得包含NaN或无穷值")


def _forecast_series(values: np.ndarray, method: str, horizon: int) -> np.ndarray:
    if len(values) == 0:
        raise ValueError("预测历史不能为空")
    if method == "persistence":
        return np.repeat(values[-1], horizon)
    if method in {"seasonal_naive_24", "seasonal_naive_168"}:
        period = int(method.rsplit("_", 1)[1])
        if len(values) < period:
            raise ValueError(f"{method}至少需要{period}个历史值")
        start = len(values) - period
        indices = start + np.arange(horizon)
        if np.any(indices >= len(values)):
            # 预测长度通常小于季节周期；超出部分按周期循环。
            indices = start + (np.arange(horizon) % period)
        return values[indices]
    if method == "dlinear_lite":
        window = min(24, len(values))
        x = np.arange(window, dtype=np.float64)
        y = values[-window:]
        if window == 1 or np.allclose(y, y[0]):
            return np.repeat(y[-1], horizon)
        slope, intercept = np.polyfit(x, y, deg=1)
        return np.maximum(0.0, intercept + slope * (window + np.arange(horizon)))
    raise ValueError(f"未知可再生预测器：{method}")


@dataclass(frozen=True)
class FrozenRenewableForecaster:
    selected_method: str
    validation_scores: Mapping[str, float]
    minimum_history_hours: int

    def predict(self, history: pd.DataFrame, horizon: int = 4) -> np.ndarray:
        """只根据 history 生成 `[horizon, 2]` 的普通预测。"""

        _validate_profile(history)
        if horizon <= 0:
            raise ValueError("horizon必须为正整数")
        values = history[list(PROFILE_COLUMNS)].to_numpy(dtype=np.float64)
        if len(values) < self.minimum_history_hours:
            raise ValueError(
                f"{self.selected_method}需要至少{self.minimum_history_hours}个历史小时"
            )
        prediction = np.column_stack(
            [
                _forecast_series(values[:, index], self.selected_method, horizon)
                for index in range(values.shape[1])
            ]
        )
        return np.maximum(prediction, 0.0)


def _score_method(
    combined: pd.DataFrame,
    train_length: int,
    method: str,
    horizon: int,
    stride: int,
) -> float:
    actual = combined[list(PROFILE_COLUMNS)].to_numpy(dtype=np.float64)
    timestamps = pd.to_datetime(combined["timestamp"]).to_numpy()
    scores = []
    start = max(train_length, 168)
    last_origin = len(combined) - horizon
    for origin in range(start, last_origin + 1, max(1, stride)):
        window_times = timestamps[origin - 168 : origin + horizon]
        if len(window_times) != 168 + horizon or not np.all(
            np.diff(window_times) == np.timedelta64(1, "h")
        ):
            continue
        history_values = actual[:origin]
        try:
            forecast = np.column_stack(
                [_forecast_series(history_values[:, i], method, horizon) for i in range(2)]
            )
        except ValueError:
            continue
        scores.append(float(np.mean(np.abs(forecast - actual[origin : origin + horizon]))))
    if not scores:
        raise ValueError(f"{method}没有可用的验证窗口")
    return float(np.mean(scores))


def fit_renewable_forecaster(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: Mapping[str, object] | None = None,
) -> FrozenRenewableForecaster:
    """在验证集上选择四个透明候选，不读取测试年。"""

    _validate_profile(train)
    _validate_profile(validation)
    combined = pd.concat([train, validation], ignore_index=True)
    options = RENEWABLE_METHODS
    stride = 24
    if config:
        configured = tuple(str(value) for value in config.get("methods", options))
        unknown = [value for value in configured if value not in RENEWABLE_METHODS]
        if unknown:
            raise ValueError(f"未知可再生候选：{unknown}")
        options = configured
        stride = int(config.get("validation_stride", stride))
    scores = {
        method: _score_method(combined, len(train), method, horizon=4, stride=stride)
        for method in options
    }
    selected = min(options, key=lambda method: (scores[method], options.index(method)))
    required = 168 if "168" in selected else 24 if selected != "persistence" else 1
    return FrozenRenewableForecaster(
        selected_method=selected,
        validation_scores=scores,
        minimum_history_hours=required,
    )
