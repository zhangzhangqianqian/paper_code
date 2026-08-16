"""Kitakyushu Energy Station 数据适配、审计和窗口构造。

本模块只负责把 Figshare 发布的年度 Excel 压缩包转换为项目统一的
小时级规范表。它不修改现有 HEEW 数据管线的默认三任务协议；Kitakyushu
使用独立的四任务常量和显式 ``task_columns``，便于后续阶段逐步扩展模型。

数据来源：
    https://figshare.com/articles/dataset/Energy_Station_Data/24978645
    DOI: 10.6084/m9.figshare.24978645
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .data_pipeline import (
    SplitSpec,
    _add_calendar_features,
    audit_dataframe,
    build_protocol_windows,
    build_windows,
)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


KITAKYUSHU_TASKS: Tuple[str, ...] = (
    "electricity",
    "cooling",
    "heating",
    "gas",
)
KITAKYUSHU_WEATHER_COLUMNS: Tuple[str, ...] = (
    "temperature",
    "humidity",
    "solar_irradiance",
    "wind_speed",
    "wind_direction",
)
KITAKYUSHU_CALENDAR_COLUMNS: Tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
)
KITAKYUSHU_EXOG_COLUMNS: Tuple[str, ...] = (
    *KITAKYUSHU_WEATHER_COLUMNS,
    *KITAKYUSHU_CALENDAR_COLUMNS,
)
KITAKYUSHU_YEARS: Tuple[int, ...] = tuple(range(2015, 2022))
# Stage 6-R is validation-only.  It must not even load the sealed 2021
# test-year files; Stage 7 is the first stage allowed to read that year.
KITAKYUSHU_STAGE6_YEARS: Tuple[int, ...] = tuple(range(2015, 2021))
KITAKYUSHU_SPLIT = SplitSpec(
    train_start="2015-01-01 00:00:00",
    train_end="2019-12-31 23:00:00",
    validation_start="2020-01-01 00:00:00",
    validation_end="2020-12-31 23:00:00",
    test_start="2021-01-01 00:00:00",
    test_end="2021-12-31 23:00:00",
)
KITAKYUSHU_SMALL_SAMPLE_SPLIT = SplitSpec(
    train_start="2018-07-01 00:00:00",
    train_end="2018-08-17 23:00:00",
    validation_start="2018-08-18 00:00:00",
    validation_end="2018-08-24 23:00:00",
    test_start="2018-08-25 00:00:00",
    test_end="2018-08-31 23:00:00",
)

LOAD_ZIP_NAME = "Electricity load & Heating load & Cooling load & Hot water load.zip"
GAS_ZIP_NAME = "Gas usage.zip"
WEATHER_ZIP_NAME = "Weather data.zip"

GAS_SOURCE_COLUMNS: Tuple[str, ...] = (
    "boiler",
    "fuel_cell",
    "gas_engine",
    "absorption_chiller_1",
    "absorption_chiller_2",
    "absorption_chiller_3",
)
GAS_SOURCE_ALIASES: Mapping[str, Tuple[str, ...]] = {
    "boiler": ("Boiler (m3)", "boiler"),
    "fuel_cell": ("Fuel cell (m3)", "fuel cell", "fuel_cell"),
    "gas_engine": ("Gas engine (m3)", "gas engine", "gas_engine"),
    "absorption_chiller_1": (
        "Absorption chiller 1 (m3)",
        "absorption chiller 1",
    ),
    "absorption_chiller_2": (
        "Absorption chiller 2 (m3)",
        "absorption chiller 2",
    ),
    "absorption_chiller_3": (
        "Absorption chiller 3 (m3)",
        "absorption chiller 3",
    ),
}


@dataclass(frozen=True)
class KitakyushuPaths:
    """Kitakyushu 三个核心数据包的路径。"""

    root: Path
    load_zip: Path
    gas_zip: Path
    weather_zip: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "KitakyushuPaths":
        directory = Path(root)
        return cls(
            root=directory,
            load_zip=directory / LOAD_ZIP_NAME,
            gas_zip=directory / GAS_ZIP_NAME,
            weather_zip=directory / WEATHER_ZIP_NAME,
        )

    def validate(self) -> None:
        missing = [
            str(path)
            for path in (self.load_zip, self.gas_zip, self.weather_zip)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Kitakyushu 核心数据包缺失：" + "; ".join(missing)
            )


def _norm_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _resolve_column(
    columns: Iterable[object],
    candidates: Sequence[str],
    token_groups: Sequence[Sequence[str]] = (),
) -> str:
    """按精确别名或关键词找到一个 Excel 字段。"""

    normalized = {_norm_name(column): str(column) for column in columns}
    for candidate in candidates:
        original = normalized.get(_norm_name(candidate))
        if original is not None:
            return original
    for original in map(str, columns):
        name = _norm_name(original)
        if any(all(_norm_name(token) in name for token in group) for group in token_groups):
            return original
    raise ValueError(
        f"找不到字段；候选名称={list(candidates)}，实际字段={list(columns)}"
    )


def _find_year_member(zip_file: ZipFile, year: int) -> str:
    suffix = f"__{int(year)}.xlsx".lower()
    candidates = [
        name
        for name in zip_file.namelist()
        if not name.endswith("/") and name.lower().endswith(suffix)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"年度 {year} 在 {zip_file.filename} 中匹配到 {len(candidates)} 个 Excel 文件"
        )
    return candidates[0]


def _read_yearly_excel(zip_path: Path, year: int, source_name: str) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        member = _find_year_member(archive, year)
        with archive.open(member) as handle:
            try:
                frame = pd.read_excel(handle)
            except ImportError as exc:
                raise ImportError(
                    "读取 Kitakyushu Excel 需要 openpyxl，请安装 frame/requirements.txt 中的依赖"
                ) from exc
    if frame.empty:
        raise ValueError(f"Kitakyushu {source_name} 文件为空：{member}")
    frame.attrs["source_member"] = member
    return frame


def _parse_timestamp(frame: pd.DataFrame, source_name: str) -> pd.Series:
    date_column = _resolve_column(
        frame.columns,
        ("Date", "datetime", "timestamp", "time"),
        token_groups=(("date",), ("time",)),
    )
    timestamps = pd.to_datetime(frame[date_column], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(
            f"Kitakyushu {source_name} 存在无法解析的时间戳："
            f"{int(timestamps.isna().sum())} 条"
        )
    return timestamps


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _read_load_year(year: int, paths: KitakyushuPaths) -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw = _read_yearly_excel(paths.load_zip, year, "load")
    timestamp = _parse_timestamp(raw, "load")
    resolved = {
        "timestamp": _resolve_column(raw.columns, ("Date", "datetime", "timestamp")),
        "electricity": _resolve_column(
            raw.columns,
            ("Electricity load (kW)", "electricity", "electric load"),
            token_groups=(("electricity", "load"),),
        ),
        "cooling": _resolve_column(
            raw.columns,
            ("Cooling load (kW)", "cooling", "cooling load"),
            token_groups=(("cooling", "load"),),
        ),
        "heating": _resolve_column(
            raw.columns,
            ("Heating load (kW)", "heating", "heating load"),
            token_groups=(("heating", "load"),),
        ),
    }
    result = pd.DataFrame({"timestamp": timestamp})
    for canonical in KITAKYUSHU_TASKS[:3]:
        result[canonical] = _numeric(raw, resolved[canonical])
    return result, resolved


def _read_gas_components_year(
    year: int, paths: KitakyushuPaths
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw = _read_yearly_excel(paths.gas_zip, year, "gas")
    timestamp = _parse_timestamp(raw, "gas")
    resolved = {
        "timestamp": _resolve_column(raw.columns, ("Date", "datetime", "timestamp"))
    }
    for canonical, candidates in GAS_SOURCE_ALIASES.items():
        resolved[canonical] = _resolve_column(
            raw.columns,
            candidates,
            token_groups=((canonical.replace("_", " "),),),
        )
    result = pd.DataFrame({"timestamp": timestamp})
    for column in GAS_SOURCE_COLUMNS:
        result[column] = _numeric(raw, resolved[column])
    return result, resolved


def _read_gas_year(year: int, paths: KitakyushuPaths) -> Tuple[pd.DataFrame, Dict[str, str]]:
    components, resolved = _read_gas_components_year(year, paths)
    # Preserve the original aggregation semantics: a row is usable only when
    # all six source columns are present.  Partial component missingness must
    # remain missing rather than being silently ignored.
    gas = components[list(GAS_SOURCE_COLUMNS)].sum(
        axis=1, min_count=len(GAS_SOURCE_COLUMNS)
    )
    result = components[["timestamp"]].copy()
    result["gas"] = gas
    return result, resolved


def _read_weather_year(
    year: int, paths: KitakyushuPaths
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw = _read_yearly_excel(paths.weather_zip, year, "weather")
    timestamp = _parse_timestamp(raw, "weather")
    aliases = {
        "solar_irradiance": (
            "Horizontal solar irradition (W)",
            "Horizontal solar irradiation (W)",
            "solar irradiance",
        ),
        "temperature": (
            "Outdoor air temperature (Ąć)",
            "Outdoor air temperature (°C)",
            "temperature",
        ),
        "humidity": ("Outdoor air humidity (%)", "humidity"),
        "wind_speed": ("Wind speed (m/s)", "wind speed"),
        "wind_direction": ("Wind direction", "wind direction"),
    }
    token_groups = {
        "solar_irradiance": (("solar", "irrad"),),
        "temperature": (("outdoor", "air", "temperature"), ("temperature",)),
        "humidity": (("outdoor", "air", "humidity"), ("humidity",)),
        "wind_speed": (("wind", "speed"),),
        "wind_direction": (("wind", "direction"),),
    }
    resolved = {
        "timestamp": _resolve_column(raw.columns, ("Date", "datetime", "timestamp"))
    }
    for canonical, candidates in aliases.items():
        resolved[canonical] = _resolve_column(
            raw.columns, candidates, token_groups=token_groups[canonical]
        )
    result = pd.DataFrame({"timestamp": timestamp})
    for canonical in KITAKYUSHU_WEATHER_COLUMNS:
        result[canonical] = _numeric(raw, resolved[canonical])
    return result, resolved


def read_kitakyushu_canonical(
    data_dir: str | Path,
    years: Sequence[int] = KITAKYUSHU_YEARS,
    add_calendar: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """读取年度压缩包并生成四任务小时级规范表。

    返回的气负荷是六类能源站天然气设备消耗量之和。三个来源文件必须在
    每个年份拥有完全一致的小时索引；若不一致则直接报错，避免静默错位。
    """

    paths = KitakyushuPaths.from_root(data_dir)
    paths.validate()
    selected_years = tuple(int(year) for year in years)
    if not selected_years:
        raise ValueError("years不能为空")

    pieces = []
    mappings: Dict[str, Dict[str, str]] = {}
    for year in selected_years:
        load, load_map = _read_load_year(year, paths)
        gas, gas_map = _read_gas_year(year, paths)
        weather, weather_map = _read_weather_year(year, paths)
        for name, frame in (("load", load), ("gas", gas), ("weather", weather)):
            if frame["timestamp"].duplicated().any():
                raise ValueError(f"{year} 年 {name} 文件存在重复时间戳")
        if not load["timestamp"].equals(gas["timestamp"]) or not load[
            "timestamp"
        ].equals(weather["timestamp"]):
            raise ValueError(f"{year} 年负荷、燃气和气象时间索引不完全一致")
        merged = load.merge(gas, on="timestamp", how="inner", validate="one_to_one")
        merged = merged.merge(
            weather, on="timestamp", how="inner", validate="one_to_one"
        )
        pieces.append(merged)
        mappings[str(year)] = {
            **{f"load.{key}": value for key, value in load_map.items()},
            **{f"gas.{key}": value for key, value in gas_map.items()},
            **{f"weather.{key}": value for key, value in weather_map.items()},
        }

    canonical = pd.concat(pieces, ignore_index=True)
    canonical = canonical.sort_values("timestamp").reset_index(drop=True)
    if canonical["timestamp"].duplicated().any():
        raise ValueError("年度合并后存在重复时间戳")
    if add_calendar:
        canonical = _add_calendar_features(canonical)
    ordered_columns = [
        "timestamp",
        *KITAKYUSHU_TASKS,
        *KITAKYUSHU_WEATHER_COLUMNS,
        *(KITAKYUSHU_CALENDAR_COLUMNS if add_calendar else ()),
    ]
    canonical = canonical[ordered_columns]
    metadata: Dict[str, object] = {
        "dataset": "Kitakyushu Energy Station Data",
        "source_url": "https://figshare.com/articles/dataset/Energy_Station_Data/24978645",
        "doi": "10.6084/m9.figshare.24978645",
        "years": list(selected_years),
        "tasks": list(KITAKYUSHU_TASKS),
        "gas_definition": "sum of six energy-station gas-consumption columns in m3",
        "resolved_columns": mappings,
        "source_files": {
            label: {
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256_file(path),
            }
            for label, path in (
                ("load_zip", paths.load_zip),
                ("gas_zip", paths.gas_zip),
                ("weather_zip", paths.weather_zip),
            )
        },
    }
    return canonical, metadata


def read_kitakyushu_gas_components(
    data_dir: str | Path,
    years: Sequence[int],
) -> pd.DataFrame:
    """读取指定年份的六类能源站燃气设备分量，不读取负荷或气象文件。

    返回 ``timestamp`` 与 :data:`GAS_SOURCE_COLUMNS`，单位沿用原始数据的
    ``m3``。该接口用于设备运行阶段审计；常规预测任务仍使用聚合后的 ``gas``。
    """

    paths = KitakyushuPaths.from_root(data_dir)
    paths.validate()
    selected_years = tuple(int(year) for year in years)
    if not selected_years:
        raise ValueError("years不能为空")
    pieces = []
    for year in selected_years:
        components, _ = _read_gas_components_year(year, paths)
        if components["timestamp"].duplicated().any():
            raise ValueError(f"{year} 年 gas 文件存在重复时间戳")
        pieces.append(components)
    result = pd.concat(pieces, ignore_index=True).sort_values("timestamp")
    result = result.reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("年度 gas 分量合并后存在重复时间戳")
    return result[["timestamp", *GAS_SOURCE_COLUMNS]]


def clean_kitakyushu_dataframe(
    frame: pd.DataFrame,
    max_interpolation_hours: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """按协议清洗 Kitakyushu 数据。

    目标负荷从不插值：目标缺失、负值或无法转换的记录会被删除。只对气象
    外生变量做有限长度的内部时间插值；仍缺失的记录也会被删除，以保证
    后续滑窗不会跨越隐式缺口。
    """

    if max_interpolation_hours < 0:
        raise ValueError("max_interpolation_hours不能为负数")
    missing = [column for column in ("timestamp", *KITAKYUSHU_TASKS) if column not in frame]
    if missing:
        raise ValueError(f"Kitakyushu 清洗缺少字段：{missing}")
    working = frame.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    if working["timestamp"].isna().any():
        raise ValueError("Kitakyushu 数据存在无法解析的 timestamp")
    if working["timestamp"].duplicated().any():
        raise ValueError("Kitakyushu 数据存在重复 timestamp")
    working = working.sort_values("timestamp").set_index("timestamp")
    for column in working.columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    negative_counts: Dict[str, int] = {}
    for task in KITAKYUSHU_TASKS:
        negative_counts[task] = int((working[task] < 0).sum())
        working.loc[working[task] < 0, task] = np.nan

    target_missing_before = int(working[list(KITAKYUSHU_TASKS)].isna().any(axis=1).sum())
    exog_before = working[list(KITAKYUSHU_EXOG_COLUMNS)].isna().sum().to_dict()
    for column in KITAKYUSHU_WEATHER_COLUMNS:
        working[column] = working[column].interpolate(
            method="time",
            limit=max_interpolation_hours,
            limit_area="inside",
        )
    exog_missing_after_interp = int(
        working[list(KITAKYUSHU_WEATHER_COLUMNS)].isna().any(axis=1).sum()
    )
    invalid = working[list(KITAKYUSHU_TASKS)].isna().any(axis=1) | working[
        list(KITAKYUSHU_EXOG_COLUMNS)
    ].isna().any(axis=1)
    cleaned = working.loc[~invalid].reset_index()
    report = {
        "negative_values_replaced_by_nan": negative_counts,
        "target_rows_removed": target_missing_before,
        "weather_missing_before_interpolation": {
            str(key): int(value) for key, value in exog_before.items()
        },
        "rows_with_weather_missing_after_interpolation": exog_missing_after_interp,
        "rows_removed_total": int(invalid.sum()),
        "rows_after_cleaning": int(len(cleaned)),
        "max_interpolation_hours": int(max_interpolation_hours),
        "targets_were_interpolated": False,
    }
    return cleaned, report


def audit_kitakyushu_dataframe(frame: pd.DataFrame) -> Dict[str, object]:
    """生成包含四个目标任务的质量报告。"""

    required = ("timestamp", *KITAKYUSHU_TASKS, *KITAKYUSHU_EXOG_COLUMNS)
    return audit_dataframe(
        frame,
        required_columns=required,
        task_columns=KITAKYUSHU_TASKS,
    )


def build_kitakyushu_windows(
    frame: pd.DataFrame,
    lookback: int = 24,
    horizon: int = 4,
    exog_columns: Sequence[str] = KITAKYUSHU_EXOG_COLUMNS,
) -> Dict[str, np.ndarray]:
    """构造 Kitakyushu 四任务窗口，默认使用历史气象和日历变量。"""

    return build_windows(
        frame,
        lookback=lookback,
        horizon=horizon,
        exog_columns=exog_columns,
        task_columns=KITAKYUSHU_TASKS,
    )


def build_kitakyushu_protocol_windows(
    frame: pd.DataFrame,
    split_name: str,
    lookback: int = 24,
    horizon: int = 4,
    exog_columns: Sequence[str] = KITAKYUSHU_EXOG_COLUMNS,
    split_spec: SplitSpec = KITAKYUSHU_SPLIT,
) -> Dict[str, np.ndarray]:
    """按指定时间协议构造四任务窗口，默认使用全年协议。"""

    return build_protocol_windows(
        frame,
        split_spec,
        split_name=split_name,
        lookback=lookback,
        horizon=horizon,
        exog_columns=exog_columns,
        task_columns=KITAKYUSHU_TASKS,
    )
