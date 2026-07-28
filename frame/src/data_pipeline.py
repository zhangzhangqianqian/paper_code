"""阶段1：多能源负荷数据读取、审计、清洗、切分和滑窗。

设计原则：
    1. 先审计原始数据，再执行清洗；
    2. 不随机切分时间序列；
    3. 标准化参数由上层训练流程只在训练集上拟合；
    4. 滑窗只接受连续小时数据，避免隐式跨缺失时间点取样。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


TASKS: Tuple[str, ...] = ("electricity", "cooling", "heating")
HEEW_WEATHER_COLUMNS: Tuple[str, ...] = (
    "temperature",
    "dew_point",
    "humidity",
    "wind_speed",
    "wind_gust",
    "pressure",
    "precip",
)
HEEW_CALENDAR_COLUMNS: Tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
)
HEEW_EXOG_COLUMNS: Tuple[str, ...] = HEEW_WEATHER_COLUMNS + HEEW_CALENDAR_COLUMNS
DEFAULT_TIMESTAMP_ALIASES: Tuple[str, ...] = (
    "timestamp",
    "datetime",
    "date_time",
    "date",
    "time",
)
DEFAULT_COLUMN_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "electricity": (
        "electricity",
        "electric_load",
        "electric load",
        "power",
        "power_load",
        "elec",
    ),
    "cooling": (
        "cooling",
        "cooling_load",
        "cooling load",
        "cool",
        "cold_load",
    ),
    "heating": (
        "heating",
        "heating_load",
        "heating load",
        "heat_load",
        "heat",
    ),
    "temperature": ("temperature", "temp", "air_temperature", "dry_bulb"),
    "pressure": ("pressure", "air_pressure", "atmospheric_pressure"),
    "dew_point": ("dew_point", "dew point", "dewpoint"),
    "precipitable_water": (
        "precipitable_water",
        "precipitable water",
        "precip_water",
    ),
}


@dataclass(frozen=True)
class SplitSpec:
    """按日历时间定义的时间切分边界。"""

    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


FULL_SPLIT = SplitSpec(
    train_start="2014-01-01 00:00:00",
    train_end="2019-12-31 23:00:00",
    validation_start="2020-01-01 00:00:00",
    validation_end="2020-12-31 23:00:00",
    test_start="2021-01-01 00:00:00",
    test_end="2022-12-31 23:00:00",
)

SMALL_SAMPLE_SPLIT = SplitSpec(
    train_start="2018-07-01 00:00:00",
    train_end="2018-08-17 23:00:00",
    validation_start="2018-08-18 00:00:00",
    validation_end="2018-08-24 23:00:00",
    test_start="2018-08-25 00:00:00",
    test_end="2018-08-31 23:00:00",
)


def _norm_name(value: object) -> str:
    """将字段名规范化，用于大小写、空格和下划线差异的匹配。"""

    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _json_value(value: object) -> object:
    """将NumPy/Pandas值转换为可写入JSON的类型。"""

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def resolve_columns(
    columns: Iterable[object],
    aliases: Optional[Mapping[str, Sequence[str]]] = None,
    timestamp_aliases: Sequence[str] = DEFAULT_TIMESTAMP_ALIASES,
) -> Dict[str, str]:
    """根据别名找到原始CSV中的规范字段。

    返回值的键包括 ``timestamp``、三个任务和已找到的气象变量。三个任务
    和时间戳是必需字段；气象变量在阶段1允许缺失，并由质量报告明确列出。
    """

    aliases = aliases or DEFAULT_COLUMN_ALIASES
    normalized = {_norm_name(column): str(column) for column in columns}
    resolved: Dict[str, str] = {}

    def find(canonical: str, candidates: Sequence[str]) -> None:
        for candidate in candidates:
            original = normalized.get(_norm_name(candidate))
            if original is not None:
                resolved[canonical] = original
                return

    find("timestamp", timestamp_aliases)
    if "timestamp" not in resolved:
        raise ValueError(
            "找不到时间字段。支持的默认名称包括："
            + ", ".join(timestamp_aliases)
            + f"；实际字段为：{list(columns)}"
        )

    for task in TASKS:
        find(task, aliases.get(task, (task,)))
        if task not in resolved:
            raise ValueError(
                f"找不到必需负荷字段 {task!r}；实际字段为：{list(columns)}"
            )

    for feature in (
        "temperature",
        "pressure",
        "dew_point",
        "precipitable_water",
    ):
        find(feature, aliases.get(feature, (feature,)))

    return resolved


def read_csv_canonical(
    path: str | Path,
    aliases: Optional[Mapping[str, Sequence[str]]] = None,
    timestamp_aliases: Sequence[str] = DEFAULT_TIMESTAMP_ALIASES,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """读取CSV并返回规范字段名，但不执行清洗或静默删除。"""

    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{source_path}")

    raw = pd.read_csv(source_path)
    resolved = resolve_columns(raw.columns, aliases, timestamp_aliases)

    selected = raw[list(resolved.values())].copy()
    selected.columns = list(resolved.keys())
    selected["timestamp"] = pd.to_datetime(selected["timestamp"], errors="coerce")
    for column in selected.columns:
        if column != "timestamp":
            selected[column] = pd.to_numeric(selected[column], errors="coerce")

    return selected, resolved


HEEW_ENERGY_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "electricity": ("electricity", "electric load", "electric_load"),
    "cooling": ("cooling", "cooling load", "cooling_load"),
    "heating": ("heat", "heating", "heating load", "heating_load"),
}

HEEW_WEATHER_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "temperature": ("temperature", "temp", "air temperature"),
    "dew_point": ("dew point", "dew_point", "dewpoint"),
    "humidity": ("humidity", "relative humidity", "relative_humidity"),
    "wind_speed": ("wind speed", "wind_speed"),
    "wind_gust": ("wind gust", "wind_gust"),
    "pressure": ("pressure", "air pressure", "air_pressure"),
    "precip": ("precip", "precipitation", "rainfall"),
}


def _resolve_source_columns(
    columns: Iterable[object],
    aliases: Mapping[str, Sequence[str]],
    source_name: str,
) -> Dict[str, str]:
    normalized = {_norm_name(column): str(column) for column in columns}
    resolved: Dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            original = normalized.get(_norm_name(candidate))
            if original is not None:
                resolved[canonical] = original
                break
        if canonical not in resolved:
            raise ValueError(
                f"HEEW {source_name} 文件缺少字段 {canonical!r}；"
                f"实际字段为：{list(columns)}"
            )
    return resolved


def _heew_timestamp(frame: pd.DataFrame, source_name: str) -> pd.Series:
    date_columns = {"year", "month", "day", "hour"}
    normalized = {_norm_name(column): str(column) for column in frame.columns}
    missing = [column for column in date_columns if column not in normalized]
    if missing:
        raise ValueError(
            f"HEEW {source_name} 文件缺少时间字段 {missing}；"
            f"实际字段为：{list(frame.columns)}"
        )
    values = pd.DataFrame(
        {
            "year": pd.to_numeric(frame[normalized["year"]], errors="coerce"),
            "month": pd.to_numeric(frame[normalized["month"]], errors="coerce"),
            "day": pd.to_numeric(frame[normalized["day"]], errors="coerce"),
            "hour": pd.to_numeric(frame[normalized["hour"]], errors="coerce"),
        }
    )
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"HEEW {source_name} 文件存在无法解析的时间字段")
    return timestamps


def _add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    timestamps = pd.to_datetime(working["timestamp"], errors="raise")
    hour = timestamps.dt.hour.to_numpy(dtype=np.float64)
    day_of_week = timestamps.dt.dayofweek.to_numpy(dtype=np.float64)
    month = timestamps.dt.month.to_numpy(dtype=np.float64)
    working["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    working["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    working["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    working["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    working["month_sin"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
    working["month_cos"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
    working["is_weekend"] = (day_of_week >= 5.0).astype(np.float64)
    return working


def read_heew_canonical(
    energy_path: str | Path,
    weather_path: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """读取 HEEW 区域级负荷和气象 CSV，并合并为统一规范格式。

    HEEW 将时间拆分为 ``Year``、``Month``、``Day``、``Hour``，且负荷和气象
    位于两个文件。本函数只负责可靠读取、字段映射和时间对齐，不执行插值、
    异常值替换或标准化。
    """

    energy_source = Path(energy_path)
    weather_source = Path(weather_path)
    if not energy_source.exists():
        raise FileNotFoundError(f"找不到 HEEW 负荷文件：{energy_source}")
    if not weather_source.exists():
        raise FileNotFoundError(f"找不到 HEEW 气象文件：{weather_source}")

    energy_raw = pd.read_csv(energy_source)
    weather_raw = pd.read_csv(weather_source)
    energy_timestamp = _heew_timestamp(energy_raw, "energy")
    weather_timestamp = _heew_timestamp(weather_raw, "weather")
    if energy_timestamp.duplicated().any():
        raise ValueError("HEEW 负荷文件存在重复时间戳")
    if weather_timestamp.duplicated().any():
        raise ValueError("HEEW 气象文件存在重复时间戳")

    energy_resolved = _resolve_source_columns(
        energy_raw.columns, HEEW_ENERGY_ALIASES, "energy"
    )
    weather_resolved = _resolve_source_columns(
        weather_raw.columns, HEEW_WEATHER_ALIASES, "weather"
    )
    energy = pd.DataFrame({"timestamp": energy_timestamp})
    weather = pd.DataFrame({"timestamp": weather_timestamp})
    for canonical, source_column in energy_resolved.items():
        energy[canonical] = pd.to_numeric(
            energy_raw[source_column], errors="coerce"
        )
    for canonical, source_column in weather_resolved.items():
        weather[canonical] = pd.to_numeric(
            weather_raw[source_column], errors="coerce"
        )

    if len(energy) != len(weather) or not energy["timestamp"].equals(
        weather["timestamp"]
    ):
        raise ValueError(
            "HEEW 负荷和气象文件的时间索引不完全一致，已停止合并；"
            "请先检查两个文件的时间范围、缺失小时和排序。"
        )

    merged = energy.merge(
        weather,
        on="timestamp",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp").reset_index(drop=True)
    merged = _add_calendar_features(merged)
    columns = ["timestamp", *TASKS, *HEEW_EXOG_COLUMNS]
    merged = merged[columns]
    resolved = {
        "energy.timestamp": "Year/Month/Day/Hour",
        "weather.timestamp": "Year/Month/Day/Hour",
        **{f"energy.{key}": value for key, value in energy_resolved.items()},
        **{f"weather.{key}": value for key, value in weather_resolved.items()},
    }
    return merged, resolved


def audit_dataframe(
    frame: pd.DataFrame,
    required_columns: Sequence[str] = ("timestamp",) + TASKS,
    frequency: str = "1h",
) -> Dict[str, object]:
    """生成原始或清洗后数据的质量报告。"""

    if "timestamp" not in frame.columns:
        raise ValueError("质量审计需要 timestamp 字段")

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    valid_timestamps = timestamps.dropna()
    ordered = valid_timestamps.is_monotonic_increasing
    duplicate_count = int(valid_timestamps.duplicated().sum())

    gaps = valid_timestamps.sort_values().diff().dropna()
    expected = pd.Timedelta(frequency)
    abnormal_gaps = gaps[gaps != expected]
    missing_hours = sum(
        max(int(gap / expected) - 1, 0) for gap in abnormal_gaps
    )

    column_report: Dict[str, Dict[str, object]] = {}
    for column in frame.columns:
        series = frame[column]
        report: Dict[str, object] = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_ratio": float(series.isna().mean()),
        }
        if column in TASKS:
            numeric = pd.to_numeric(series, errors="coerce")
            report["negative_count"] = int((numeric < 0).sum())
            report["min"] = _json_value(numeric.min())
            report["max"] = _json_value(numeric.max())
        column_report[column] = report

    season_counts: Dict[str, int] = {}
    if not valid_timestamps.empty:
        seasons = valid_timestamps.dt.month.map(
            lambda month: (
                "winter"
                if month in (12, 1, 2)
                else "spring"
                if month in (3, 4, 5)
                else "summer"
                if month in (6, 7, 8)
                else "autumn"
            )
        )
        season_counts = {str(k): int(v) for k, v in seasons.value_counts().items()}

    missing_required = [column for column in required_columns if column not in frame]
    return {
        "row_count": int(len(frame)),
        "column_names": [str(column) for column in frame.columns],
        "required_columns_missing": missing_required,
        "timestamp_null_count": int(timestamps.isna().sum()),
        "timestamp_is_monotonic": bool(ordered),
        "duplicate_timestamp_count": duplicate_count,
        "abnormal_gap_count": int(len(abnormal_gaps)),
        "missing_hours_from_gaps": int(missing_hours),
        "start": _json_value(valid_timestamps.min()) if not valid_timestamps.empty else None,
        "end": _json_value(valid_timestamps.max()) if not valid_timestamps.empty else None,
        "season_counts": season_counts,
        "columns": column_report,
    }


def clean_dataframe(
    frame: pd.DataFrame,
    max_interpolation_hours: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """执行阶段1约定的最小清洗。

    清洗前要求调用方先保存原始审计报告。重复时间戳和无法解析的时间戳
    不自动猜测处理，而是直接报错；负荷负值改为缺失；仅对内部短缺口插值。
    """

    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    if working["timestamp"].isna().any():
        raise ValueError("存在无法解析的 timestamp，已停止清洗")
    if working["timestamp"].duplicated().any():
        raise ValueError("存在重复 timestamp，已停止清洗；请先审计并处理重复记录")

    working = working.sort_values("timestamp").set_index("timestamp")
    for column in working.columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    negative_counts = {}
    for task in TASKS:
        if task in working.columns:
            negative_counts[task] = int((working[task] < 0).sum())
            working.loc[working[task] < 0, task] = np.nan

    before_missing = working.isna().sum().to_dict()
    for column in working.columns:
        working[column] = working[column].interpolate(
            method="time",
            limit=max_interpolation_hours,
            limit_area="inside",
        )

    working = working.reset_index()
    after_missing = working.isna().sum().to_dict()
    report = {
        "negative_values_replaced_by_nan": negative_counts,
        "missing_count_before_interpolation": {
            str(k): int(v) for k, v in before_missing.items()
        },
        "missing_count_after_interpolation": {
            str(k): int(v) for k, v in after_missing.items()
        },
        "max_interpolation_hours": int(max_interpolation_hours),
    }
    return working, report


def split_dataframe(
    frame: pd.DataFrame,
    spec: SplitSpec = FULL_SPLIT,
) -> Dict[str, pd.DataFrame]:
    """按日历边界进行严格的训练/验证/测试切分。"""

    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
    if working["timestamp"].duplicated().any():
        raise ValueError("切分前不允许存在重复 timestamp")

    bounds = {
        "train": (spec.train_start, spec.train_end),
        "validation": (spec.validation_start, spec.validation_end),
        "test": (spec.test_start, spec.test_end),
    }
    result: Dict[str, pd.DataFrame] = {}
    used_indices = set()
    for name, (start, end) in bounds.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = working["timestamp"].between(start_ts, end_ts, inclusive="both")
        piece = working.loc[mask].copy().reset_index(drop=True)
        if piece.empty:
            raise ValueError(f"{name}切分为空：{start} 至 {end}")
        result[name] = piece
        used_indices.update(working.index[mask].tolist())

    if len(used_indices) != sum(len(piece) for piece in result.values()):
        raise ValueError("时间切分存在重叠区间")
    return result


def select_training_frame(
    frame: pd.DataFrame,
    spec: SplitSpec = FULL_SPLIT,
) -> pd.DataFrame:
    """只选择当前协议训练区间，用于拟合标准化参数。

    与 ``build_protocol_windows`` 不同，这里不保留训练区间之外的历史记录，
    避免小样本协议把训练开始日期之前的数据混入标准化统计量。
    """

    working = frame.copy()
    if "timestamp" not in working.columns:
        raise ValueError("训练区间选择需要timestamp字段")
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
    mask = working["timestamp"].between(
        pd.Timestamp(spec.train_start),
        pd.Timestamp(spec.train_end),
        inclusive="both",
    )
    selected = working.loc[mask].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"训练区间为空：{spec.train_start} 至 {spec.train_end}"
        )
    return selected


def build_windows(
    frame: pd.DataFrame,
    lookback: int = 24,
    horizon: int = 4,
    exog_columns: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    """从一个连续时间片段构造24→4监督学习窗口。

    窗口中只使用过去的历史记录；若窗口内部不是连续小时，直接跳过该窗口，
    不通过重采样默默填充未来信息。
    """

    if lookback <= 0 or horizon <= 0:
        raise ValueError("lookback和horizon必须为正整数")
    required = ["timestamp", *TASKS, *exog_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"构造滑窗缺少字段：{missing}")

    working = frame.copy().sort_values("timestamp").reset_index(drop=True)
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
    timestamps = working["timestamp"].to_numpy()
    load_values = working[list(TASKS)].to_numpy(dtype=np.float32)
    exog_values = (
        working[list(exog_columns)].to_numpy(dtype=np.float32)
        if exog_columns
        else np.zeros((len(working), 0), dtype=np.float32)
    )

    loads, exogs, targets, target_times = [], [], [], []
    total = len(working) - lookback - horizon + 1
    one_hour = np.timedelta64(1, "h")
    for start in range(max(total, 0)):
        input_end = start + lookback
        target_end = input_end + horizon
        window_times = timestamps[start:target_end]
        if len(window_times) != lookback + horizon:
            continue
        if not np.all(np.diff(window_times) == one_hour):
            continue
        x_loads = load_values[start:input_end]
        x_exog = exog_values[start:input_end]
        y = load_values[input_end:target_end]
        if not (
            np.isfinite(x_loads).all()
            and np.isfinite(x_exog).all()
            and np.isfinite(y).all()
        ):
            continue
        loads.append(x_loads)
        exogs.append(x_exog)
        targets.append(y)
        target_times.append(window_times[lookback])

    sample_count = len(loads)
    loads_array = np.asarray(loads, dtype=np.float32).reshape(
        sample_count, lookback, len(TASKS)
    )
    if exog_columns:
        exog_array = np.asarray(exogs, dtype=np.float32).reshape(
            sample_count, lookback, len(exog_columns)
        )
    else:
        exog_array = np.empty((sample_count, lookback, 0), dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32).reshape(
        sample_count, horizon, len(TASKS)
    )
    return {
        "loads": loads_array,
        "exog": exog_array,
        "target": target_array,
        "target_times": np.asarray(target_times, dtype="datetime64[ns]"),
    }


def build_protocol_windows(
    frame: pd.DataFrame,
    spec: SplitSpec,
    split_name: str,
    lookback: int = 24,
    horizon: int = 4,
    exog_columns: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    """为指定时间切分构造窗口，同时保留边界前的历史上下文。

    例如，测试集第一个目标时刻可以使用验证集最后24小时作为历史输入，
    但目标值仍然只来自测试区间。这样不会因为切分而人为丢失边界上下文。
    """

    if split_name not in {"train", "validation", "test"}:
        raise ValueError("split_name必须是train、validation或test")
    working = frame.copy().sort_values("timestamp").reset_index(drop=True)
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")

    split_bounds = {
        "train": (spec.train_start, spec.train_end),
        "validation": (spec.validation_start, spec.validation_end),
        "test": (spec.test_start, spec.test_end),
    }
    target_start, target_end = map(pd.Timestamp, split_bounds[split_name])
    context_start = target_start - pd.Timedelta(hours=lookback)
    context_end = target_end + pd.Timedelta(hours=horizon - 1)
    segment = working[
        working["timestamp"].between(context_start, context_end, inclusive="both")
    ].copy()
    windows = build_windows(
        segment,
        lookback=lookback,
        horizon=horizon,
        exog_columns=exog_columns,
    )
    target_times = pd.to_datetime(windows["target_times"])
    mask = (target_times >= target_start) & (target_times <= target_end)
    return {key: value[mask] for key, value in windows.items()}


def save_json(payload: Mapping[str, object], path: str | Path) -> None:
    """以UTF-8写入质量报告或切分摘要。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_value)
