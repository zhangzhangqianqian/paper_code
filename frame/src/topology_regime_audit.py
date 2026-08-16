"""Audit Kitakyushu equipment-regime evidence before the topology pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .data_pipeline import SplitSpec
from .kitakyushu_pipeline import (
    GAS_SOURCE_COLUMNS,
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_STAGE6_YEARS,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
    build_kitakyushu_protocol_windows,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
    read_kitakyushu_gas_components,
)


POST_GE_SPLIT = SplitSpec(
    train_start="2017-01-01 00:00:00",
    train_end="2019-12-31 23:00:00",
    validation_start="2020-01-01 00:00:00",
    validation_end="2020-12-31 23:00:00",
    test_start="2021-01-01 00:00:00",
    test_end="2021-12-31 23:00:00",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def summarize_yearly_frame(
    frame: pd.DataFrame,
    years: Sequence[int],
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return row, missingness and hourly-gap statistics for selected years."""

    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    rows = []
    for year in (int(item) for item in years):
        mask = timestamps.dt.year == year
        part = frame.loc[mask].copy()
        part_timestamps = timestamps.loc[mask].sort_values()
        gaps = part_timestamps.diff().dt.total_seconds().div(3600.0)
        record: dict[str, Any] = {
            "year": year,
            "rows": int(len(part)),
            "timestamp_min": part_timestamps.min(),
            "timestamp_max": part_timestamps.max(),
            "timestamp_duplicates": int(part_timestamps.duplicated().sum()),
            "hourly_gaps_gt_1": int((gaps > 1).sum()),
        }
        for column in columns:
            if column not in part:
                raise ValueError(f"missing audit column: {column}")
            record[f"{column}_missing"] = int(part[column].isna().sum())
            record[f"{column}_missing_rate"] = float(part[column].isna().mean()) if len(part) else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_task_statistics(
    frame: pd.DataFrame,
    years: Sequence[int],
    *,
    task_columns: Sequence[str] = KITAKYUSHU_TASKS,
) -> pd.DataFrame:
    rows = []
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    for year in (int(item) for item in years):
        part = frame.loc[timestamps.dt.year == year]
        for task in task_columns:
            values = pd.to_numeric(part[task], errors="coerce")
            finite = values[np.isfinite(values)]
            rows.append(
                {
                    "year": year,
                    "task": task,
                    "valid_count": int(finite.size),
                    "mean": float(finite.mean()) if finite.size else None,
                    "std": float(finite.std(ddof=1)) if finite.size > 1 else None,
                    "q01": float(finite.quantile(0.01)) if finite.size else None,
                    "q50": float(finite.quantile(0.50)) if finite.size else None,
                    "q99": float(finite.quantile(0.99)) if finite.size else None,
                    "zero_ratio": float((finite == 0).mean()) if finite.size else None,
                }
            )
    return pd.DataFrame(rows)


def summarize_gas_components(
    components: pd.DataFrame,
    years: Sequence[int],
) -> pd.DataFrame:
    timestamps = pd.to_datetime(components["timestamp"], errors="coerce")
    rows = []
    for year in (int(item) for item in years):
        part = components.loc[timestamps.dt.year == year]
        for column in GAS_SOURCE_COLUMNS:
            values = pd.to_numeric(part[column], errors="coerce")
            finite = values[np.isfinite(values)]
            nonzero = finite[finite != 0]
            last_nonzero = part.loc[values.fillna(0).ne(0), "timestamp"]
            rows.append(
                {
                    "year": year,
                    "component": column,
                    "valid_count": int(finite.size),
                    "nonzero_hours": int(nonzero.size),
                    "sum_m3": float(nonzero.sum()) if nonzero.size else 0.0,
                    "max_m3": float(nonzero.max()) if nonzero.size else 0.0,
                    "last_nonzero_timestamp": (
                        pd.to_datetime(last_nonzero).max().isoformat()
                        if len(last_nonzero)
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def _transition_monthly(components: pd.DataFrame) -> pd.DataFrame:
    working = components.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
    working = working.loc[working["timestamp"].dt.year.isin((2015, 2016, 2017))]
    working["month"] = working["timestamp"].dt.to_period("M").astype(str)
    records = []
    for month, part in working.groupby("month", sort=True):
        record: dict[str, Any] = {"month": month, "rows": int(len(part))}
        for column in GAS_SOURCE_COLUMNS:
            values = pd.to_numeric(part[column], errors="coerce")
            record[f"{column}_nonzero_hours"] = int((values.fillna(0) != 0).sum())
            record[f"{column}_sum_m3"] = float(values.fillna(0).sum())
        records.append(record)
    return pd.DataFrame(records)


def _protocol_sample_counts(
    cleaned: pd.DataFrame,
    protocols: Mapping[str, SplitSpec],
) -> pd.DataFrame:
    records = []
    for name, split in protocols.items():
        for split_name in ("train", "validation"):
            windows = build_kitakyushu_protocol_windows(
                cleaned,
                split_name=split_name,
                split_spec=split,
                lookback=24,
                horizon=4,
            )
            records.append(
                {
                    "protocol": name,
                    "split": split_name,
                    "window_count": int(windows["loads"].shape[0]),
                    "lookback": 24,
                    "horizon": 4,
                }
            )
    return pd.DataFrame(records)


def build_topology_regime_audit(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    years: Sequence[int] = KITAKYUSHU_STAGE6_YEARS,
) -> dict[str, Any]:
    """Create the pre-branch audit using only data through 2020."""

    selected_years = tuple(int(year) for year in years)
    if selected_years != KITAKYUSHU_STAGE6_YEARS:
        raise ValueError(
            "pre-branch topology audit must use exactly 2015-2020; audit 2021 after freeze"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    canonical, metadata = read_kitakyushu_canonical(data_dir, years=selected_years)
    components = read_kitakyushu_gas_components(data_dir, selected_years)
    cleaned, cleaning_report = clean_kitakyushu_dataframe(canonical)

    data_quality = summarize_yearly_frame(
        canonical,
        selected_years,
        columns=(*KITAKYUSHU_TASKS, *KITAKYUSHU_EXOG_COLUMNS),
    )
    task_statistics = summarize_task_statistics(canonical, selected_years)
    gas_statistics = summarize_gas_components(components, selected_years)
    monthly_transition = _transition_monthly(components)
    sample_counts = _protocol_sample_counts(
        cleaned,
        {
            "cross_topology": KITAKYUSHU_SPLIT,
            "post_ge_regular_operation": POST_GE_SPLIT,
        },
    )

    gas_engine_2018_2020 = gas_statistics.loc[
        (gas_statistics["component"] == "gas_engine")
        & gas_statistics["year"].isin((2018, 2019, 2020))
    ]
    gas_engine_2017 = gas_statistics.loc[
        (gas_statistics["component"] == "gas_engine")
        & (gas_statistics["year"] == 2017)
    ]
    if int(gas_engine_2018_2020["nonzero_hours"].sum()) != 0:
        raise ValueError("pre-branch audit found gas-engine nonzero records in 2018-2020")
    if not gas_engine_2017.empty and float(gas_engine_2017["max_m3"].max()) > 4.0:
        raise ValueError("2017 gas-engine residual values exceed the frozen audit bound")

    for name, table in (
        ("yearly_data_quality.csv", data_quality),
        ("yearly_task_statistics.csv", task_statistics),
        ("yearly_gas_component_statistics.csv", gas_statistics),
        ("gas_engine_transition_monthly.csv", monthly_transition),
        ("protocol_sample_counts.csv", sample_counts),
    ):
        _write_csv(table, output / name)

    report: dict[str, Any] = {
        "status": "passed",
        "dataset": "kitakyushu_energy_station",
        "audited_years": list(selected_years),
        "latest_audited_year": 2020,
        "test_year_accessed": False,
        "official_timeline": {
            "fuel_cell": "regular operation ended after 2011",
            "gas_engine": "regular operation reported through July 2016",
        },
        "local_findings": {
            "gas_engine_2017_residual_records_are_disclosed": True,
            "gas_engine_2018_2020_nonzero_hours": int(gas_engine_2018_2020["nonzero_hours"].sum()),
        },
        "protocols": {
            "cross_topology": {
                "train_years": [2015, 2016, 2017, 2018, 2019],
                "validation_years": [2020],
            },
            "post_ge_regular_operation": {
                "train_years": [2017, 2018, 2019],
                "validation_years": [2020],
            },
        },
        "source_metadata": metadata,
        "cleaning_report": cleaning_report,
        "gas_source_columns": list(GAS_SOURCE_COLUMNS),
        "component_sum_definition": "gas is the sum of six energy-station gas-consumption components",
        "artifacts": {
            "data_quality": "yearly_data_quality.csv",
            "task_statistics": "yearly_task_statistics.csv",
            "gas_component_statistics": "yearly_gas_component_statistics.csv",
            "transition_monthly": "gas_engine_transition_monthly.csv",
            "sample_counts": "protocol_sample_counts.csv",
        },
    }
    with (output / "topology_regime_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_ready)
    return report


__all__ = [
    "POST_GE_SPLIT",
    "build_topology_regime_audit",
    "summarize_gas_components",
    "summarize_task_statistics",
    "summarize_yearly_frame",
]
