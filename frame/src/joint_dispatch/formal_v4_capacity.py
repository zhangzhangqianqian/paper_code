"""Deterministic, training-only origin selection for capacity certification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .formal_v4_data import FormalV4BaseSeries
from ..scheduling.dispatch_lp import DispatchInputs, DispatchResult, solve_dispatch_lp


ORIGIN_SELECTION_SCHEMA = "formal-v4.1-capacity-origin-manifest-v1"
TRAIN_YEARS = (2015, 2016, 2017, 2018)
SEASONS = ("winter", "spring", "summer", "autumn")
DEFAULT_QUOTAS = {
    "total": 500,
    "year_season": 320,
    "cooling_top_decile": 60,
    "heating_top_decile": 40,
    "electricity_top_decile": 40,
    "uniform_remaining": 40,
}


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _base_arrays(base: FormalV4BaseSeries) -> tuple[np.ndarray, np.ndarray]:
    loads = np.asarray(base.load_and_exog, dtype=np.float64)
    timestamps = np.asarray(base.timestamps, dtype="datetime64[ns]")
    if loads.ndim != 2 or loads.shape[1] < 4 or timestamps.ndim != 1 or loads.shape[0] != timestamps.size:
        raise ValueError("capacity origins require a valid four-task base series")
    return loads[:, :4], timestamps


def _source_hash(base: FormalV4BaseSeries) -> str:
    digest = hashlib.sha256()
    for array in (
        np.ascontiguousarray(base.load_and_exog),
        np.ascontiguousarray(base.renewable_forecast),
        np.ascontiguousarray(base.renewable_realized),
        np.ascontiguousarray(base.prices_and_weights),
        np.ascontiguousarray(base.timestamps.astype("datetime64[ns]")),
    ):
        digest.update(array.tobytes())
    return digest.hexdigest()


def _valid_origins(timestamps: np.ndarray) -> np.ndarray:
    valid: list[int] = []
    for origin in range(24, len(timestamps) - 3):
        window = timestamps[origin - 24 : origin + 4]
        if np.all(np.diff(window) == np.timedelta64(1, "h")):
            valid.append(origin)
    return np.asarray(valid, dtype=np.int64)


def _choose_ranked(ordered: np.ndarray, count: int, selected: set[int]) -> list[int]:
    if count <= 0:
        return []
    if ordered.size == 0:
        raise ValueError("capacity origin stratum is empty")
    positions = np.rint(np.linspace(0, ordered.size - 1, count)).astype(np.int64)
    chosen: list[int] = []
    for position in positions:
        found = None
        for offset in range(ordered.size):
            candidate = int(ordered[(int(position) + offset) % ordered.size])
            if candidate not in selected:
                found = candidate
                break
        if found is None:
            raise ValueError("capacity origin stratum cannot fill its unique quota")
        selected.add(found)
        chosen.append(found)
    return chosen


@dataclass(frozen=True)
class CapacityOriginManifest:
    origin_indices: np.ndarray
    origin_timestamps: np.ndarray
    strata: np.ndarray
    demand_summaries: Mapping[str, np.ndarray]
    source_base_sha256: str
    selection_config: Mapping[str, int]
    schema_version: str = ORIGIN_SELECTION_SCHEMA
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        indices = np.asarray(self.origin_indices, dtype=np.int64)
        timestamps = np.asarray(self.origin_timestamps, dtype="datetime64[ns]")
        strata = np.asarray(self.strata, dtype=str)
        if indices.ndim != 1 or timestamps.shape != indices.shape or strata.shape != indices.shape:
            raise ValueError("capacity origin arrays must have the same one-dimensional shape")
        if indices.size != int(self.selection_config.get("total", 500)):
            raise ValueError("capacity origin manifest does not contain the frozen total")
        if np.unique(indices).size != indices.size or not self.source_base_sha256:
            raise ValueError("capacity origins must be unique and source-bound")
        summary = {str(key): np.asarray(value, dtype=np.float64) for key, value in self.demand_summaries.items()}
        required = {"electricity_sum", "cooling_sum", "heating_sum", "gas_sum", "cooling_max", "weekday", "hour_bin"}
        if set(summary) != required or any(value.shape != indices.shape for value in summary.values()):
            raise ValueError("capacity demand summaries are incomplete")
        if np.any(summary["weekday"] < 0) or np.any(summary["hour_bin"] < 0):
            raise ValueError("capacity calendar summaries are invalid")
        object.__setattr__(self, "origin_indices", indices)
        object.__setattr__(self, "origin_timestamps", timestamps)
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "demand_summaries", summary)
        expected = self._identity_payload()
        computed = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if self.manifest_sha256 and self.manifest_sha256 != computed:
            raise ValueError("capacity origin manifest hash does not match payload")
        object.__setattr__(self, "manifest_sha256", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "origin_indices": self.origin_indices.tolist(),
            "origin_timestamps": [str(value) for value in self.origin_timestamps],
            "strata": self.strata.tolist(),
            "demand_summaries": {key: value.tolist() for key, value in sorted(self.demand_summaries.items())},
            "source_base_sha256": self.source_base_sha256,
            "selection_config": dict(sorted(self.selection_config.items())),
        }

    def to_payload(self) -> dict[str, Any]:
        return self._identity_payload() | {"manifest_sha256": self.manifest_sha256}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite capacity origin manifest: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class CapacityAuditReceipt:
    status: str
    selected: Mapping[str, Any] | None
    stage_one: tuple[Mapping[str, Any], ...]
    stage_two: tuple[Mapping[str, Any], ...]
    origin_manifest_sha256: str
    source_base_sha256: str
    candidate_multipliers: tuple[float, ...]
    thresholds: Mapping[str, float]
    diagnostic_initial_state: Mapping[str, float]
    full_timestamp_start: str
    full_timestamp_end: str
    full_state_reset_count: int
    solver_identity: str
    capacity_scenario_hash: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"pass", "fail"}:
            raise ValueError("capacity audit status must be pass or fail")
        if tuple(sorted(self.candidate_multipliers)) != self.candidate_multipliers or len(set(self.candidate_multipliers)) != len(self.candidate_multipliers):
            raise ValueError("capacity multipliers must be sorted and unique")
        if self.diagnostic_initial_state != {"initial_soc": 0.5, "previous_chp": 0.0}:
            raise ValueError("capacity diagnostic initial state is not frozen")
        computed = hashlib.sha256(json.dumps(self._identity_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if self.capacity_scenario_hash and self.capacity_scenario_hash != computed:
            raise ValueError("capacity scenario hash does not match scientific payload")
        object.__setattr__(self, "capacity_scenario_hash", computed)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selected": self.selected,
            "stage_one": [dict(row) for row in self.stage_one],
            "stage_two": [dict(row) for row in self.stage_two],
            "origin_manifest_sha256": self.origin_manifest_sha256,
            "source_base_sha256": self.source_base_sha256,
            "candidate_multipliers": list(self.candidate_multipliers),
            "thresholds": dict(self.thresholds),
            "diagnostic_initial_state": dict(self.diagnostic_initial_state),
            "full_timestamp_start": self.full_timestamp_start,
            "full_timestamp_end": self.full_timestamp_end,
            "full_state_reset_count": self.full_state_reset_count,
            "solver_identity": self.solver_identity,
        }

    def to_payload(self) -> dict[str, Any]:
        return self._identity_payload() | {"capacity_scenario_hash": self.capacity_scenario_hash}


def _metric_row(
    multiplier: float,
    *,
    cooling_shortage: float,
    cooling_demand: float,
    shortage_hours: int,
    total_hours: int,
    solves: int,
    strata: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "multiplier": float(multiplier),
        "cooling_shortage_energy": float(cooling_shortage),
        "cooling_demand_energy": float(cooling_demand),
        "cooling_shortage_energy_ratio": float(cooling_shortage / max(cooling_demand, 1.0e-12)),
        "cooling_shortage_hours": int(shortage_hours),
        "cooling_shortage_hour_rate": float(shortage_hours / max(total_hours, 1)),
        "solves": int(solves),
        "meets_threshold": False,
    }
    if strata is not None:
        row["by_stratum"] = {str(key): dict(value) for key, value in strata.items()}
    return row


def _meets(row: Mapping[str, Any], thresholds: Mapping[str, float]) -> bool:
    return float(row["cooling_shortage_energy_ratio"]) <= float(thresholds["cooling_shortage_energy_ratio_max"]) and float(row["cooling_shortage_hour_rate"]) <= float(thresholds["cooling_shortage_hour_rate_max"])


def _solve_metrics(
    base: FormalV4BaseSeries,
    parameters: Mapping[str, Any],
    origins: np.ndarray,
    multiplier: float,
    *,
    reset_each_origin: bool,
    solver: Any,
    strata: np.ndarray | None = None,
) -> tuple[dict[str, Any], int]:
    loads = np.asarray(base.load_and_exog, dtype=np.float64)[:, :3]
    renewable = np.asarray(base.renewable_realized, dtype=np.float64)
    timestamps = np.asarray(base.timestamps, dtype="datetime64[ns]")
    context = dict(parameters)
    for key in ("electric_chiller_capacity", "absorption_chiller_capacity"):
        if key not in context:
            raise ValueError(f"capacity audit parameters missing {key}")
        context[key] = float(context[key]) * float(multiplier)
    soc, previous_chp = 0.5, 0.0
    shortage = demand = 0.0
    shortage_hours = solves = 0
    resets = 0
    by_stratum: dict[str, dict[str, float]] = {}
    for position, raw_origin in enumerate(origins.tolist()):
        origin = int(raw_origin)
        if reset_each_origin or position == 0 or origin - int(origins[position - 1]) != 1 or timestamps[origin] - timestamps[origin - 1] != np.timedelta64(1, "h"):
            soc, previous_chp = 0.5, 0.0
            resets += 1
        demand_window = loads[origin : origin + 4]
        renewable_window = renewable[origin : origin + 4]
        if demand_window.shape != (4, 3) or renewable_window.shape != (4, 2):
            raise ValueError("capacity audit origin does not have a complete four-hour horizon")
        result: DispatchResult = solver(DispatchInputs(
            demand=demand_window,
            pv_available=renewable_window[:, 0],
            wt_available=renewable_window[:, 1],
            parameters=context,
            initial_soc=soc,
            previous_chp=previous_chp,
        ))
        solves += 1
        if not result.success:
            raise RuntimeError(f"capacity LP failed at origin {origin}: {result.message}")
        slack = np.asarray(result.values["slack_c"], dtype=np.float64)
        shortage += float(slack.sum())
        demand += float(demand_window[:, 1].sum())
        shortage_hours += int(np.count_nonzero(slack > 1.0e-8))
        if strata is not None:
            label = str(strata[position])
            bucket = by_stratum.setdefault(label, {"cooling_shortage_energy": 0.0, "cooling_demand_energy": 0.0, "cooling_shortage_hours": 0.0, "hours": 0.0})
            bucket["cooling_shortage_energy"] += float(slack.sum())
            bucket["cooling_demand_energy"] += float(demand_window[:, 1].sum())
            bucket["cooling_shortage_hours"] += float(np.count_nonzero(slack > 1.0e-8))
            bucket["hours"] += 4.0
        soc = float(np.clip(result.values["soc"][0] / float(context["bess_energy_capacity"]), 0.0, 1.0))
        previous_chp = float(result.values["p_chp"][0])
    for bucket in by_stratum.values():
        bucket["cooling_shortage_energy_ratio"] = bucket["cooling_shortage_energy"] / max(bucket["cooling_demand_energy"], 1.0e-12)
        bucket["cooling_shortage_hour_rate"] = bucket["cooling_shortage_hours"] / max(bucket["hours"], 1.0)
    return _metric_row(
        multiplier,
        cooling_shortage=shortage,
        cooling_demand=demand,
        shortage_hours=shortage_hours,
        total_hours=int(len(origins) * 4),
        solves=solves,
        strata=by_stratum or None,
    ), resets


def run_capacity_audit(
    base: FormalV4BaseSeries,
    parameters: Mapping[str, Any],
    origin_manifest: CapacityOriginManifest,
    candidate_multipliers: tuple[float, ...] | list[float],
    *,
    cooling_shortage_energy_ratio_max: float = 0.005,
    cooling_shortage_hour_rate_max: float = 0.01,
    solver: Any = solve_dispatch_lp,
) -> CapacityAuditReceipt:
    """Run the 500-origin diagnostic and full chronological certificate."""

    candidates = tuple(float(value) for value in candidate_multipliers)
    if not candidates or tuple(sorted(candidates)) != candidates or len(set(candidates)) != len(candidates):
        raise ValueError("capacity candidate multipliers must be sorted and unique")
    thresholds = {
        "cooling_shortage_energy_ratio_max": float(cooling_shortage_energy_ratio_max),
        "cooling_shortage_hour_rate_max": float(cooling_shortage_hour_rate_max),
    }
    if any(value < 0.0 for value in thresholds.values()):
        raise ValueError("capacity shortage thresholds must be non-negative")
    if len(origin_manifest.origin_indices) != 500:
        raise ValueError("capacity audit requires the frozen 500-origin manifest")
    loads, timestamps = _base_arrays(base)
    valid = _valid_origins(timestamps)
    if not np.array_equal(np.unique(origin_manifest.origin_indices), origin_manifest.origin_indices):
        raise ValueError("capacity origin manifest must be unique and chronological")
    if not set(origin_manifest.origin_indices.tolist()).issubset(set(valid.tolist())):
        raise ValueError("capacity origin manifest contains an invalid or discontinuous origin")
    stage_one: list[Mapping[str, Any]] = []
    stage_two: list[Mapping[str, Any]] = []
    selected: Mapping[str, Any] | None = None
    full_resets = 0
    for multiplier in candidates:
        row, resets = _solve_metrics(base, parameters, origin_manifest.origin_indices, multiplier, reset_each_origin=True, solver=solver, strata=origin_manifest.strata)
        row = dict(row)
        row["meets_threshold"] = _meets(row, thresholds)
        stage_one.append(row)
        full_origins = valid
        full_row, full_row_resets = _solve_metrics(base, parameters, full_origins, multiplier, reset_each_origin=False, solver=solver)
        full_row = dict(full_row)
        full_row["meets_threshold"] = _meets(full_row, thresholds)
        stage_two.append(full_row)
        full_resets = full_row_resets
        if selected is None and row["meets_threshold"] and full_row["meets_threshold"]:
            selected = dict(full_row)
            selected["selection_stage"] = "both_stage_one_and_full_chronology"
            selected["diagnostic_row"] = row
    return CapacityAuditReceipt(
        status="pass" if selected is not None else "fail",
        selected=selected,
        stage_one=tuple(stage_one),
        stage_two=tuple(stage_two),
        origin_manifest_sha256=origin_manifest.manifest_sha256,
        source_base_sha256=origin_manifest.source_base_sha256,
        candidate_multipliers=candidates,
        thresholds=thresholds,
        diagnostic_initial_state={"initial_soc": 0.5, "previous_chp": 0.0},
        full_timestamp_start=str(timestamps[valid[0] - 24]),
        full_timestamp_end=str(timestamps[valid[-1] + 3]),
        full_state_reset_count=full_resets,
        solver_identity=f"{getattr(solver, '__module__', '')}.{getattr(solver, '__qualname__', repr(solver))}",
    )


def _origin_config(config: Mapping[str, Any]) -> dict[str, int]:
    capacity = config.get("capacity", config)
    raw = capacity.get("origin_selection", {}) if isinstance(capacity, Mapping) else {}
    quotas = {key: int(raw.get(key, value)) for key, value in DEFAULT_QUOTAS.items()}
    if quotas != DEFAULT_QUOTAS:
        raise ValueError("capacity origin quotas are not frozen")
    return quotas


def select_capacity_origins(base: FormalV4BaseSeries, config: Mapping[str, Any]) -> CapacityOriginManifest:
    """Select exactly 500 deterministic, unique, training-only origins."""

    quotas = _origin_config(config)
    loads, timestamps = _base_arrays(base)
    all_years = pd.DatetimeIndex(timestamps).year.to_numpy(dtype=np.int64)
    if np.any(~np.isin(all_years, TRAIN_YEARS)):
        raise PermissionError("capacity origin base timestamps must be within 2015-2018")
    if set(int(year) for year in all_years) != set(TRAIN_YEARS):
        raise ValueError("capacity origin base must include every training year 2015-2018")
    valid = _valid_origins(timestamps)
    if valid.size < quotas["total"]:
        raise ValueError("training base has fewer than 500 valid continuous origins")
    index_times = pd.DatetimeIndex(timestamps.astype("datetime64[ns]"))
    future = np.stack([loads[index : index + 4, :].sum(axis=0) for index in valid], axis=0)
    cooling_max = np.asarray([loads[index : index + 4, 1].max() for index in valid], dtype=np.float64)
    years = index_times[valid].year.to_numpy()
    seasons = np.asarray([_season(int(month)) for month in index_times[valid].month], dtype=str)
    weekdays = index_times[valid].weekday.to_numpy(dtype=np.int64)
    hour_bins = (index_times[valid].hour.to_numpy(dtype=np.int64) // 6) * 6
    selected: set[int] = set()
    selected_with_label: list[tuple[int, str]] = []
    positions = {int(origin): position for position, origin in enumerate(valid.tolist())}

    for year in TRAIN_YEARS:
        for season in SEASONS:
            cell = valid[(years == year) & (seasons == season)]
            chosen = _choose_ranked(cell, quotas["year_season"] // 16, selected)
            selected_with_label.extend((origin, f"year_{year}_{season}") for origin in chosen)

    def add_top_decile(column: int, count: int, label: str) -> None:
        metric = future[:, column]
        threshold = float(np.quantile(metric, 0.90))
        candidates = valid[metric >= threshold]
        ordered = candidates[np.argsort(-metric[[positions[int(item)] for item in candidates]], kind="stable")]
        chosen = _choose_ranked(ordered, count, selected)
        selected_with_label.extend((origin, label) for origin in chosen)

    add_top_decile(1, quotas["cooling_top_decile"], "cooling_top_decile")
    add_top_decile(2, quotas["heating_top_decile"], "heating_top_decile")
    add_top_decile(0, quotas["electricity_top_decile"], "electricity_top_decile")
    remaining = np.asarray([origin for origin in valid if int(origin) not in selected], dtype=np.int64)
    chosen = _choose_ranked(remaining, quotas["uniform_remaining"], selected)
    selected_with_label.extend((origin, "uniform_remaining") for origin in chosen)
    if len(selected_with_label) != quotas["total"]:
        raise ValueError("capacity origin manifest has an incorrect total")

    selected_with_label.sort(key=lambda item: item[0])
    origins = np.asarray([item[0] for item in selected_with_label], dtype=np.int64)
    labels = np.asarray([item[1] for item in selected_with_label], dtype=str)
    selected_positions = np.asarray([positions[int(origin)] for origin in origins], dtype=np.int64)
    if not ({0, 1, 2, 3, 4} & set(weekdays[selected_positions])) or not ({5, 6} & set(weekdays[selected_positions])):
        raise ValueError("capacity origin manifest lacks weekday/weekend coverage")
    if set(hour_bins[selected_positions]) != {0, 6, 12, 18}:
        raise ValueError("capacity origin manifest lacks six-hour time-of-day coverage")
    if float(future[selected_positions, 1].sum()) <= 0.0:
        raise ValueError("capacity origin manifest has zero audited cooling demand")
    summaries = {
        "electricity_sum": future[selected_positions, 0],
        "cooling_sum": future[selected_positions, 1],
        "heating_sum": future[selected_positions, 2],
        "gas_sum": future[selected_positions, 3],
        "cooling_max": cooling_max[selected_positions],
        "weekday": weekdays[selected_positions].astype(np.float64),
        "hour_bin": hour_bins[selected_positions].astype(np.float64),
    }
    return CapacityOriginManifest(
        origin_indices=origins,
        origin_timestamps=timestamps[origins],
        strata=labels,
        demand_summaries=summaries,
        source_base_sha256=_source_hash(base),
        selection_config=quotas,
    )


__all__ = ["CapacityOriginManifest", "ORIGIN_SELECTION_SCHEMA", "select_capacity_origins"]
