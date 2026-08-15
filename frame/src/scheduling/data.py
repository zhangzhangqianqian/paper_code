"""阶段10.3：双轨调度数据框架与未来信息权限隔离。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd

from ..kitakyushu_pipeline import (
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_TASKS,
    KITAKYUSHU_YEARS,
    _find_year_member,
    _resolve_column,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from ..data_pipeline import SplitSpec


POWER_ZIP_NAME = "Power.zip"
ACTUAL_COLUMNS: Tuple[str, ...] = ("actual_grid_import", "actual_pv")


@dataclass(frozen=True)
class SchedulingFrame:
    """包含历史输入和仅供事后结算的真实运行字段。"""

    data: pd.DataFrame
    metadata: Mapping[str, Any]
    years: Tuple[int, ...]


@dataclass(frozen=True)
class PlanningInformation:
    """普通计划阶段可见的数据；不含未来真实目标或实际设备出力。"""

    origin: pd.Timestamp
    history: pd.DataFrame
    future_times: pd.DatetimeIndex
    visible_columns: Tuple[str, ...]


@dataclass(frozen=True)
class RealizedInformation:
    """求解完成后才可创建的未来真实值，用于 R 轨结算或 oracle。"""

    origin: pd.Timestamp
    realized: pd.DataFrame
    protected_columns: Tuple[str, ...]


def _read_power_year(data_dir: Path, year: int) -> pd.DataFrame:
    power_zip = data_dir / POWER_ZIP_NAME
    if not power_zip.exists():
        raise FileNotFoundError(f"R轨需要Power.zip：{power_zip}")
    with ZipFile(power_zip) as archive:
        member = _find_year_member(archive, year)
        with archive.open(member) as handle:
            frame = pd.read_excel(handle)
    date_column = _resolve_column(
        frame.columns,
        ("Date", "datetime", "timestamp", "time"),
        token_groups=(("date",), ("time",)),
    )
    pv_column = _resolve_column(
        frame.columns,
        ("Solar energy generation (kW)", "solar energy generation", "pv"),
        token_groups=(("solar", "energy", "generation"), ("solar",)),
    )
    grid_column = _resolve_column(
        frame.columns,
        ("Electricity imported from grid (kW)", "electricity imported from grid", "grid import"),
        token_groups=(("electricity", "imported", "grid"), ("grid", "import")),
    )
    result = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[date_column], errors="raise"),
            "actual_grid_import": pd.to_numeric(frame[grid_column], errors="coerce"),
            "actual_pv": pd.to_numeric(frame[pv_column], errors="coerce"),
        }
    )
    if result["timestamp"].duplicated().any():
        raise ValueError(f"Power.zip {year} 年存在重复时间戳")
    return result


def build_scheduling_frame(
    data_dir: str | Path,
    years: Sequence[int] = KITAKYUSHU_YEARS,
    max_interpolation_hours: int = 3,
) -> SchedulingFrame:
    """读取负荷、燃气、气象和 Power.zip 并构造严格对齐的调度框架。"""

    selected_years = tuple(int(year) for year in years)
    if not selected_years:
        raise ValueError("years不能为空")
    directory = Path(data_dir)
    canonical, metadata = read_kitakyushu_canonical(directory, years=selected_years)
    power_parts = [_read_power_year(directory, year) for year in selected_years]
    power = pd.concat(power_parts, ignore_index=True).sort_values("timestamp")
    if power["timestamp"].duplicated().any():
        raise ValueError("Power.zip年度合并后存在重复时间戳")
    merged = canonical.merge(power, on="timestamp", how="left", validate="one_to_one")
    if merged[list(ACTUAL_COLUMNS)].isna().any().any():
        missing = int(merged[list(ACTUAL_COLUMNS)].isna().any(axis=1).sum())
        raise ValueError(f"Power.zip与规范数据无法完整对齐：{missing}行缺少grid/PV实际值")
    cleaned, cleaning = clean_kitakyushu_dataframe(
        merged, max_interpolation_hours=max_interpolation_hours
    )
    cleaned = cleaned.sort_values("timestamp").reset_index(drop=True)
    metadata = dict(metadata)
    metadata.update(
        {
            "scheduling_power_source": POWER_ZIP_NAME,
            "years": list(selected_years),
            "cleaning": cleaning,
            "actual_columns": list(ACTUAL_COLUMNS),
            "future_actuals_reserved_for": ["settlement", "oracle"],
        }
    )
    return SchedulingFrame(data=cleaned, metadata=metadata, years=selected_years)


def _hourly_history(frame: SchedulingFrame, origin: pd.Timestamp, lookback: int) -> pd.DataFrame:
    timestamps = pd.to_datetime(frame.data["timestamp"], errors="raise")
    history_end = origin - pd.Timedelta(hours=1)
    history_start = origin - pd.Timedelta(hours=lookback)
    history = frame.data[timestamps.between(history_start, history_end, inclusive="both")].copy()
    if len(history) != lookback:
        raise ValueError(f"origin={origin}前历史窗口不是{lookback}个小时")
    history_times = pd.to_datetime(history["timestamp"]).to_numpy()
    if not np.all(np.diff(history_times) == np.timedelta64(1, "h")):
        raise ValueError("历史窗口存在非连续小时，拒绝隐式填充")
    return history.reset_index(drop=True)


def make_plan_view(
    frame: SchedulingFrame,
    origin: str | pd.Timestamp,
    lookback: int = 24,
    horizon: int = 4,
) -> PlanningInformation:
    """只返回历史输入与未来时间索引，严禁返回未来真实字段。"""

    origin_ts = pd.Timestamp(origin)
    history = _hourly_history(frame, origin_ts, lookback)
    visible_columns = ("timestamp", *KITAKYUSHU_TASKS, *KITAKYUSHU_EXOG_COLUMNS)
    history = history.loc[:, list(visible_columns)]
    future_times = pd.date_range(origin_ts, periods=horizon, freq="1h")
    return PlanningInformation(
        origin=origin_ts,
        history=history,
        future_times=future_times,
        visible_columns=visible_columns,
    )


def make_settlement_view(
    frame: SchedulingFrame,
    origin: str | pd.Timestamp,
    horizon: int = 4,
) -> RealizedInformation:
    """返回未来真实值；调用方应在普通计划求解之后使用。"""

    origin_ts = pd.Timestamp(origin)
    future_times = pd.date_range(origin_ts, periods=horizon, freq="1h")
    timestamps = pd.to_datetime(frame.data["timestamp"], errors="raise")
    realized = frame.data[timestamps.isin(future_times)].copy()
    if len(realized) != horizon:
        raise ValueError("结算窗口缺少连续的未来真实记录")
    realized = realized.sort_values("timestamp").reset_index(drop=True)
    if not np.all(
        pd.to_datetime(realized["timestamp"]).to_numpy() == future_times.to_numpy()
    ):
        raise ValueError("结算窗口时间戳未与预测步严格对齐")
    protected = ("timestamp", *KITAKYUSHU_TASKS, *ACTUAL_COLUMNS)
    return RealizedInformation(
        origin=origin_ts,
        realized=realized.loc[:, list(protected)],
        protected_columns=protected,
    )
