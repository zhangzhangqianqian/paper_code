"""Perfect-information oracle for the simulated IES scheduling track."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .rolling_horizon import ActualStream, RollingForecastSet, RollingResult, run_rolling_dispatch


def _aligned_series(values: pd.Series | np.ndarray, timestamps: pd.DatetimeIndex, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        series = values.copy()
        series.index = pd.to_datetime(series.index)
        if series.index.has_duplicates:
            raise ValueError(f"{name} contains duplicate timestamps")
        aligned = pd.to_numeric(series.reindex(timestamps), errors="coerce").to_numpy(dtype=float)
    else:
        array = np.asarray(values, dtype=float)
        if array.shape != (len(timestamps),):
            raise ValueError(f"{name} array must align with data rows")
        aligned = array
    if not np.isfinite(aligned).all() or (aligned < 0).any():
        raise ValueError(f"{name} must be finite and non-negative")
    return aligned


def build_oracle_forecast_set(
    data: pd.DataFrame,
    origins: np.ndarray,
    horizon: int,
    actual_pv: pd.Series | np.ndarray,
    actual_wt: pd.Series | np.ndarray,
) -> RollingForecastSet:
    """Build a forecast set from only the true values in each future window.

    The function requires every timestamp in every requested window to exist;
    it never pads, interpolates, or reads beyond ``origin + horizon - 1``.
    """

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    frame = data.copy()
    if "timestamp" not in frame.columns:
        raise ValueError("data must contain timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    if frame["timestamp"].duplicated().any():
        raise ValueError("data contains duplicate timestamps")
    frame = frame.set_index("timestamp").sort_index()
    origins = np.asarray(origins, dtype="datetime64[ns]")
    if origins.ndim != 1 or len(np.unique(origins)) != len(origins):
        raise ValueError("origins must be a unique one-dimensional array")
    required = ("electricity", "cooling", "heating")
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise ValueError(f"data lacks oracle target columns: {missing_columns}")
    timestamps = pd.DatetimeIndex(frame.index)
    pv = _aligned_series(actual_pv, timestamps, "actual_pv")
    wt = _aligned_series(actual_wt, timestamps, "actual_wt")
    pv_series = pd.Series(pv, index=timestamps)
    wt_series = pd.Series(wt, index=timestamps)
    demand_windows: list[np.ndarray] = []
    pv_windows: list[np.ndarray] = []
    wt_windows: list[np.ndarray] = []
    for origin in pd.to_datetime(origins):
        future = pd.date_range(origin, periods=horizon, freq="1h")
        if not future.isin(timestamps).all():
            raise ValueError(f"origin {origin} lacks a complete future window")
        block = frame.loc[future, list(required)]
        values = block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        pv_block = pv_series.loc[future].to_numpy(dtype=float)
        wt_block = wt_series.loc[future].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"oracle target window at {origin} is invalid")
        demand_windows.append(values)
        pv_windows.append(pv_block)
        wt_windows.append(wt_block)
    return RollingForecastSet(
        origin_times=origins,
        demand=np.asarray(demand_windows, dtype=float),
        pv_available=np.asarray(pv_windows, dtype=float),
        wt_available=np.asarray(wt_windows, dtype=float),
    )


def run_perfect_information_oracle(
    data: pd.DataFrame,
    origins: np.ndarray,
    actual_pv: pd.Series | np.ndarray,
    actual_wt: pd.Series | np.ndarray,
    parameters: Mapping[str, float],
    horizon: int = 4,
    initial_soc: float = 0.5,
) -> RollingResult:
    """Run the same rolling protocol using true future windows as forecasts."""

    forecast = build_oracle_forecast_set(data, origins, horizon, actual_pv, actual_wt)
    actual = ActualStream(
        origin_times=forecast.origin_times,
        electricity=forecast.demand[:, 0, 0],
        cooling=forecast.demand[:, 0, 1],
        heating=forecast.demand[:, 0, 2],
        pv_available=forecast.pv_available[:, 0],
        wt_available=forecast.wt_available[:, 0],
    )
    return run_rolling_dispatch(forecast, actual, parameters, initial_soc=initial_soc)

