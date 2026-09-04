"""Causal data containers and window construction for joint training.

The module deliberately keeps data preparation independent of PyTorch.  Every
window is formed from a single aligned hourly index and stores the full device
trajectory used by the state-aware forecaster, the future scheduling context,
and the LP teacher labels.  Future observations are never copied into a
history tensor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import platform
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..data_pipeline import SplitSpec
from ..kitakyushu_pipeline import KITAKYUSHU_SPLIT
from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER


SCHEDULER_CONTEXT_ORDER = (
    "pv_available",
    "wt_available",
    "grid_price",
    "gas_price",
    "carbon_price",
    "initial_soc",
)
DEVICE_STATUS_ORDER = STATUS_ORDER

_STATUS_SOURCE = {
    "chp_on": "p_chp",
    "gas_boiler_on": "q_gb",
    "electric_chiller_on": "q_ec",
    "absorption_chiller_on": "q_ac",
    "bess_charge_on": "p_charge",
    "bess_discharge_on": "p_discharge",
}


def _as_float_array(value: object, *, name: str, ndim: int | None = None, dtype: np.dtype | type = np.float32) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def derive_device_status(dispatch: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Derive six auxiliary on/off indicators from continuous dispatch values."""

    values = _as_float_array(dispatch, name="dispatch", ndim=3)
    if values.shape[-1] != len(DISPATCH_ORDER):
        raise ValueError(f"dispatch last dimension must be {len(DISPATCH_ORDER)}")
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and non-negative")
    columns = []
    for status_name in STATUS_ORDER:
        source = values[..., DISPATCH_ORDER.index(_STATUS_SOURCE[status_name])]
        columns.append((source > float(epsilon)).astype(np.float32))
    return np.stack(columns, axis=-1)


def _prepare_table(table: pd.DataFrame, required: Sequence[str], name: str) -> pd.DataFrame:
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    working = table.loc[:, list(required)].copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
    if working["timestamp"].duplicated().any():
        raise ValueError(f"{name} contains duplicate timestamps")
    working = working.sort_values("timestamp").reset_index(drop=True)
    return working


def _prepare_origin_table(
    table: pd.DataFrame,
    required: Sequence[str],
    name: str,
) -> pd.DataFrame:
    """Prepare a table whose future rows are keyed by forecast origin.

    The ordinary hourly tables are keyed by target timestamp only.  Forecast
    driven dispatch needs a second key because two different origins can
    produce different PV/WT forecasts for the same target hour.  Keeping the
    origin explicit prevents an accidental many-to-one merge or a future
    value from being reused for the wrong window.
    """

    missing = [column for column in ("origin_timestamp", *required) if column not in table.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    working = table.loc[:, ["origin_timestamp", *required]].copy()
    working["origin_timestamp"] = pd.to_datetime(working["origin_timestamp"], errors="raise")
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="raise")
    if working[["origin_timestamp", "timestamp"]].duplicated().any():
        raise ValueError(f"{name} contains duplicate origin/target pairs")
    working = working.sort_values(["origin_timestamp", "timestamp"]).reset_index(drop=True)
    return working


def _assert_same_timestamps(tables: Iterable[pd.DataFrame]) -> np.ndarray:
    iterator = iter(tables)
    first = next(iterator)["timestamp"].to_numpy(dtype="datetime64[ns]")
    for table in iterator:
        current = table["timestamp"].to_numpy(dtype="datetime64[ns]")
        if current.shape != first.shape or not np.array_equal(current, first):
            raise ValueError("timestamp alignment failed across data artifacts")
    return first


def _window_bounds(split: str, spec: SplitSpec) -> tuple[pd.Timestamp, pd.Timestamp]:
    bounds = {
        "train": (spec.train_start, spec.train_end),
        "validation": (spec.validation_start, spec.validation_end),
        "test": (spec.test_start, spec.test_end),
    }
    if split not in bounds:
        raise ValueError("split must be train, validation, or test")
    return pd.Timestamp(bounds[split][0]), pd.Timestamp(bounds[split][1])


@dataclass(frozen=True)
class JointWindowSplit:
    """Aligned windows consumed by the joint forecaster and scheduler."""

    load_history: np.ndarray
    exog_history: np.ndarray
    device_history: np.ndarray
    device_status: np.ndarray
    forecast_target: np.ndarray
    scheduler_context: np.ndarray
    previous_chp: np.ndarray
    teacher_dispatch: np.ndarray
    oracle_first_step_objective: np.ndarray
    target_times: np.ndarray
    split: str
    history_source: str = "causal_lp"
    realized_renewables: np.ndarray | None = None
    oracle_dispatch: np.ndarray | None = None
    oracle_four_step_objective: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {
            "load_history": self.load_history,
            "exog_history": self.exog_history,
            "device_history": self.device_history,
            "device_status": self.device_status,
            "forecast_target": self.forecast_target,
            "scheduler_context": self.scheduler_context,
            "previous_chp": self.previous_chp,
            "teacher_dispatch": self.teacher_dispatch,
            "oracle_first_step_objective": self.oracle_first_step_objective,
        }
        # V3 integrity fields are kept in float64 so objective-parity and
        # physical-constraint audits are not masked by float32 quantisation.
        # Legacy v1/v2 artifacts retain their historical float32 storage.
        v3_present = any(value is not None for value in (self.realized_renewables, self.oracle_dispatch, self.oracle_four_step_objective))
        converted = {
            key: _as_float_array(value, name=key, dtype=np.float64 if v3_present else np.float32)
            for key, value in arrays.items()
        }
        object.__setattr__(self, "target_times", np.asarray(self.target_times, dtype="datetime64[ns]"))
        for key, value in converted.items():
            object.__setattr__(self, key, value)
        n = self.load_history.shape[0]
        if self.load_history.ndim != 3 or self.load_history.shape[1:] != (24, len(TASK_ORDER)):
            raise ValueError("load_history must have shape [N,24,4]")
        expected = {
            "exog_history": (24, len(EXOG_ORDER)),
            "device_history": (24, len(DISPATCH_ORDER)),
            "device_status": (24, len(STATUS_ORDER)),
            "forecast_target": (4, len(TASK_ORDER)),
            "scheduler_context": (4, len(SCHEDULER_CONTEXT_ORDER)),
            "teacher_dispatch": (4, len(DISPATCH_ORDER)),
        }
        for key, tail in expected.items():
            value = getattr(self, key)
            if value.ndim != 3 or value.shape[1:] != tail:
                raise ValueError(f"{key} must have shape [N,{tail[0]},{tail[1]}]")
            if value.shape[0] != n:
                raise ValueError(f"{key} has a different sample count")
        value = self.oracle_first_step_objective
        if value.shape != (n,):
            raise ValueError("oracle_first_step_objective must have shape [N]")
        if self.previous_chp.shape != (n, 1):
            raise ValueError("previous_chp must have shape [N,1]")
        if self.target_times.shape != (n,):
            raise ValueError("target_times must have shape [N]")
        if n > 1 and not np.all(np.diff(self.target_times) > np.timedelta64(0, "s")):
            raise ValueError("target_times must be strictly chronological")
        if not np.isin(self.device_status, (0.0, 1.0)).all():
            raise ValueError("device_status must be binary")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.history_source not in {"causal_lp", "joint_policy_rollin"}:
            raise ValueError("history_source must be causal_lp or joint_policy_rollin")
        if self.history_source == "joint_policy_rollin" and self.split != "train":
            raise ValueError("joint_policy_rollin histories are legal only for train")
        v3_fields = (self.realized_renewables, self.oracle_dispatch, self.oracle_four_step_objective)
        if any(value is not None for value in v3_fields):
            if not all(value is not None for value in v3_fields):
                raise ValueError("v3 split fields must be supplied together")
            realized = np.asarray(self.realized_renewables, dtype=np.float64)
            oracle_dispatch = np.asarray(self.oracle_dispatch, dtype=np.float64)
            oracle_objective = np.asarray(self.oracle_four_step_objective, dtype=np.float64)
            if realized.shape != (n, 4, 2):
                raise ValueError("realized_renewables must have shape [N,4,2]")
            if oracle_dispatch.shape != (n, 4, len(DISPATCH_ORDER)):
                raise ValueError("oracle_dispatch must have shape [N,4,21]")
            if oracle_objective.shape != (n,):
                raise ValueError("oracle_four_step_objective must have shape [N]")
            if not np.isfinite(realized).all() or not np.isfinite(oracle_dispatch).all() or not np.isfinite(oracle_objective).all():
                raise ValueError("v3 split fields must be finite")
            if (realized < 0.0).any():
                raise ValueError("realized_renewables must be non-negative")
            object.__setattr__(self, "realized_renewables", realized)
            object.__setattr__(self, "oracle_dispatch", oracle_dispatch)
            object.__setattr__(self, "oracle_four_step_objective", oracle_objective)

    def __len__(self) -> int:
        return int(self.load_history.shape[0])

    def validate(self) -> None:
        """Re-run constructor validation for callers loading mutable arrays."""

        type(self)(
            load_history=self.load_history, exog_history=self.exog_history,
            device_history=self.device_history, device_status=self.device_status,
            forecast_target=self.forecast_target, scheduler_context=self.scheduler_context,
            previous_chp=self.previous_chp, teacher_dispatch=self.teacher_dispatch,
            oracle_first_step_objective=self.oracle_first_step_objective,
            target_times=self.target_times, split=self.split, history_source=self.history_source,
            realized_renewables=self.realized_renewables,
            oracle_dispatch=self.oracle_dispatch,
            oracle_four_step_objective=self.oracle_four_step_objective,
        )

    def take(self, indices: Sequence[int]) -> "JointWindowSplit":
        index = np.asarray(indices, dtype=np.int64)
        fields = {
            name: getattr(self, name)[index]
            for name in (
                "load_history", "exog_history", "device_history", "device_status",
                "forecast_target", "scheduler_context", "previous_chp",
                "teacher_dispatch", "oracle_first_step_objective", "target_times",
            )
        }
        if self.realized_renewables is not None:
            fields.update({
                "realized_renewables": self.realized_renewables[index],
                "oracle_dispatch": self.oracle_dispatch[index],
                "oracle_four_step_objective": self.oracle_four_step_objective[index],
            })
        return JointWindowSplit(split=self.split, history_source=self.history_source, **fields)


def build_joint_windows(
    frame: pd.DataFrame,
    device_trajectory: pd.DataFrame,
    scheduler_context: pd.DataFrame,
    teacher_dispatch: pd.DataFrame,
    oracle_first_step_objective: pd.DataFrame,
    *,
    split: str,
    split_spec: SplitSpec = KITAKYUSHU_SPLIT,
    lookback: int = 24,
    horizon: int = 4,
    history_source: str = "causal_lp",
) -> JointWindowSplit:
    """Build aligned 24→4 windows while enforcing strict temporal causality."""

    if lookback != 24 or horizon != 4:
        raise ValueError("the joint contract fixes lookback=24 and horizon=4")
    base = _prepare_table(frame, ("timestamp", *TASK_ORDER, *EXOG_ORDER), "forecast frame")
    device = _prepare_table(device_trajectory, ("timestamp", *DISPATCH_ORDER), "device trajectory")
    origin_columns = (
        "origin_timestamp" in scheduler_context.columns,
        "origin_timestamp" in teacher_dispatch.columns,
        "origin_timestamp" in oracle_first_step_objective.columns,
    )
    if any(origin_columns) and not all(origin_columns):
        raise ValueError("origin_timestamp must be supplied for context, teacher, and oracle together")
    origin_specific = all(origin_columns)
    if origin_specific:
        context = _prepare_origin_table(
            scheduler_context, ("timestamp", *SCHEDULER_CONTEXT_ORDER), "scheduler context"
        )
        teacher = _prepare_origin_table(
            teacher_dispatch, ("timestamp", *DISPATCH_ORDER), "teacher dispatch"
        )
        oracle = _prepare_origin_table(
            oracle_first_step_objective, ("timestamp", "oracle_first_step_objective"), "oracle objective"
        )
        # Only the historical streams have one row per timestamp.  A causal
        # device trajectory may legitimately stop three hours before the end
        # of the raw frame because no four-hour LP horizon remains.  Align it
        # to the base index with NaN tail sentinels; windows touching a missing
        # history row are rejected below rather than silently imputed.
        base_timestamps = base["timestamp"].to_numpy(dtype="datetime64[ns]")
        device_timestamps = device["timestamp"].to_numpy(dtype="datetime64[ns]")
        if not np.all(np.isin(device_timestamps, base_timestamps)):
            raise ValueError("device trajectory contains timestamps outside forecast frame")
        timestamps = base_timestamps
    else:
        context = _prepare_table(scheduler_context, ("timestamp", *SCHEDULER_CONTEXT_ORDER), "scheduler context")
        teacher = _prepare_table(teacher_dispatch, ("timestamp", *DISPATCH_ORDER), "teacher dispatch")
        oracle = _prepare_table(oracle_first_step_objective, ("timestamp", "oracle_first_step_objective"), "oracle objective")
        timestamps = _assert_same_timestamps((base, device, context, teacher, oracle))
    if len(timestamps) < lookback + horizon:
        raise ValueError("aligned data is shorter than one joint window")
    lower, upper = _window_bounds(split, split_spec)
    # Keep source values in float64 while constructing v3 references.  This
    # avoids quantising demand/dispatch before the strict objective and
    # physical-constraint audit; callers may still cast batches to float32.
    task_values = base[list(TASK_ORDER)].to_numpy(dtype=np.float64)
    exog_values = base[list(EXOG_ORDER)].to_numpy(dtype=np.float64)
    raw_device_values = device[list(DISPATCH_ORDER)].to_numpy(dtype=np.float64)
    if origin_specific and not np.array_equal(
        device["timestamp"].to_numpy(dtype="datetime64[ns]"),
        timestamps,
    ):
        device_values = np.full((len(timestamps), len(DISPATCH_ORDER)), np.nan, dtype=np.float64)
        device_position = {time: index for index, time in enumerate(device["timestamp"].to_numpy(dtype="datetime64[ns]"))}
        for position, time in enumerate(timestamps):
            source_index = device_position.get(time)
            if source_index is not None:
                device_values[position] = raw_device_values[source_index]
    else:
        device_values = raw_device_values
    context_values = None if origin_specific else context[list(SCHEDULER_CONTEXT_ORDER)].to_numpy(dtype=np.float32)
    teacher_values = None if origin_specific else teacher[list(DISPATCH_ORDER)].to_numpy(dtype=np.float32)
    oracle_values = None if origin_specific else oracle["oracle_first_step_objective"].to_numpy(dtype=np.float32)
    if origin_specific:
        context_groups = {
            np.datetime64(origin, "ns"): group.sort_values("timestamp")
            for origin, group in context.groupby("origin_timestamp", sort=False)
        }
        teacher_groups = {
            np.datetime64(origin, "ns"): group.sort_values("timestamp")
            for origin, group in teacher.groupby("origin_timestamp", sort=False)
        }
        oracle_groups = {
            np.datetime64(origin, "ns"): group.sort_values("timestamp")
            for origin, group in oracle.groupby("origin_timestamp", sort=False)
        }
    one_hour = np.timedelta64(1, "h")
    loads: list[np.ndarray] = []
    exogs: list[np.ndarray] = []
    devices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    previous_chp: list[float] = []
    teachers: list[np.ndarray] = []
    oracles: list[float] = []
    target_times: list[np.datetime64] = []
    chp_idx = DISPATCH_ORDER.index("p_chp")
    for start in range(len(timestamps) - lookback - horizon + 1):
        end = start + lookback
        target_end = end + horizon
        times = timestamps[start:target_end]
        if not np.all(np.diff(times) == one_hour):
            continue
        target_time = pd.Timestamp(times[lookback])
        target_last = pd.Timestamp(times[-1])
        if target_time < lower or target_last > upper:
            continue
        if not np.isfinite(device_values[start:end]).all():
            continue
        origin_key = np.datetime64(times[lookback], "ns")
        if origin_specific:
            target_times_for_window = np.asarray(times[lookback:target_end], dtype="datetime64[ns]")
            context_group = context_groups.get(origin_key)
            teacher_group = teacher_groups.get(origin_key)
            oracle_group = oracle_groups.get(origin_key)
            if context_group is None or teacher_group is None or oracle_group is None:
                continue
            context_target_times = context_group["timestamp"].to_numpy(dtype="datetime64[ns]")
            teacher_target_times = teacher_group["timestamp"].to_numpy(dtype="datetime64[ns]")
            oracle_target_times = oracle_group["timestamp"].to_numpy(dtype="datetime64[ns]")
            if not np.array_equal(context_target_times, target_times_for_window):
                continue
            if not np.array_equal(teacher_target_times, target_times_for_window):
                continue
            if oracle_target_times.shape != (1,) or oracle_target_times[0] != target_times_for_window[0]:
                continue
            contexts.append(context_group[list(SCHEDULER_CONTEXT_ORDER)].to_numpy(dtype=np.float64))
            teachers.append(teacher_group[list(DISPATCH_ORDER)].to_numpy(dtype=np.float64))
            oracles.append(float(oracle_group["oracle_first_step_objective"].iloc[0]))
        else:
            contexts.append(context_values[end:target_end])
            teachers.append(teacher_values[end:target_end])
            oracles.append(float(oracle_values[end]))
        loads.append(task_values[start:end])
        exogs.append(exog_values[start:end])
        devices.append(device_values[start:end])
        targets.append(task_values[end:target_end])
        previous_chp.append(float(device_values[end - 1, chp_idx]))
        target_times.append(times[lookback])
    n = len(loads)
    empty = lambda tail, dtype=np.float64: np.empty((0, *tail), dtype=dtype)
    return JointWindowSplit(
        load_history=np.asarray(loads, dtype=np.float64).reshape((n, lookback, len(TASK_ORDER))) if n else empty((lookback, len(TASK_ORDER))),
        exog_history=np.asarray(exogs, dtype=np.float64).reshape((n, lookback, len(EXOG_ORDER))) if n else empty((lookback, len(EXOG_ORDER))),
        device_history=np.asarray(devices, dtype=np.float64).reshape((n, lookback, len(DISPATCH_ORDER))) if n else empty((lookback, len(DISPATCH_ORDER))),
        device_status=derive_device_status(np.asarray(devices, dtype=np.float64).reshape((n, lookback, len(DISPATCH_ORDER))) if n else empty((lookback, len(DISPATCH_ORDER)))),
        forecast_target=np.asarray(targets, dtype=np.float64).reshape((n, horizon, len(TASK_ORDER))) if n else empty((horizon, len(TASK_ORDER))),
        scheduler_context=np.asarray(contexts, dtype=np.float64).reshape((n, horizon, len(SCHEDULER_CONTEXT_ORDER))) if n else empty((horizon, len(SCHEDULER_CONTEXT_ORDER))),
        previous_chp=np.asarray(previous_chp, dtype=np.float64).reshape((n, 1)),
        teacher_dispatch=np.asarray(teachers, dtype=np.float64).reshape((n, horizon, len(DISPATCH_ORDER))) if n else empty((horizon, len(DISPATCH_ORDER))),
        oracle_first_step_objective=np.asarray(oracles, dtype=np.float64),
        target_times=np.asarray(target_times, dtype="datetime64[ns]"),
        split=split,
        history_source=history_source,
    )


def build_joint_windows_v3(
    frame: pd.DataFrame,
    device_trajectory: pd.DataFrame,
    scheduler_context: pd.DataFrame,
    teacher_dispatch: pd.DataFrame,
    realized_renewables: pd.DataFrame,
    oracle_dispatch: pd.DataFrame,
    oracle_four_step_objective: pd.DataFrame,
    *,
    split: str,
    split_spec: SplitSpec = KITAKYUSHU_SPLIT,
    history_source: str = "causal_lp",
) -> JointWindowSplit:
    """Build v3 windows with separate realized renewables and oracle fields.

    The established v1/v2 alignment routine remains the source of the
    forecast/history/teacher window indices. V3-only origin-keyed artifacts
    are then joined using the resulting target origins, preventing accidental
    positional alignment when a history row is missing.
    """

    realized = _prepare_origin_table(
        realized_renewables,
        ("timestamp", "pv_realized", "wt_realized"),
        "realized renewables",
    )
    oracle_values = _prepare_origin_table(
        oracle_dispatch,
        ("timestamp", *DISPATCH_ORDER),
        "oracle dispatch",
    )
    objective = _prepare_origin_table(
        oracle_four_step_objective,
        ("timestamp", "oracle_four_step_objective"),
        "oracle four-hour objective",
    )
    legacy_objective = objective.rename(
        columns={"oracle_four_step_objective": "oracle_first_step_objective"}
    )
    base = build_joint_windows(
        frame,
        device_trajectory,
        scheduler_context,
        teacher_dispatch,
        legacy_objective,
        split=split,
        split_spec=split_spec,
        history_source=history_source,
    )
    realized_groups = {
        np.datetime64(origin, "ns"): group.sort_values("timestamp")
        for origin, group in realized.groupby("origin_timestamp", sort=False)
    }
    oracle_groups = {
        np.datetime64(origin, "ns"): group.sort_values("timestamp")
        for origin, group in oracle_values.groupby("origin_timestamp", sort=False)
    }
    realized_rows: list[np.ndarray] = []
    oracle_rows: list[np.ndarray] = []
    for origin in base.target_times:
        key = np.datetime64(origin, "ns")
        realized_group = realized_groups.get(key)
        oracle_group = oracle_groups.get(key)
        if realized_group is None or oracle_group is None:
            raise ValueError(f"v3 reference rows missing for origin {pd.Timestamp(origin)}")
        realized_times = realized_group["timestamp"].to_numpy(dtype="datetime64[ns]")
        oracle_times = oracle_group["timestamp"].to_numpy(dtype="datetime64[ns]")
        expected_times = key + np.arange(4, dtype=np.int64).astype("timedelta64[h]")
        if not np.array_equal(realized_times, expected_times) or not np.array_equal(oracle_times, expected_times):
            raise ValueError(f"v3 reference horizon alignment failed for origin {pd.Timestamp(origin)}")
        realized_rows.append(realized_group[["pv_realized", "wt_realized"]].to_numpy(dtype=np.float64))
        oracle_rows.append(oracle_group[list(DISPATCH_ORDER)].to_numpy(dtype=np.float64))
    n = len(base)
    realized_array = np.asarray(realized_rows, dtype=np.float64).reshape((n, 4, 2))
    oracle_array = np.asarray(oracle_rows, dtype=np.float64).reshape((n, 4, len(DISPATCH_ORDER)))
    objective_by_origin = {
        np.datetime64(origin, "ns"): float(group["oracle_four_step_objective"].iloc[0])
        for origin, group in objective.groupby("origin_timestamp", sort=False)
    }
    try:
        objective_array = np.asarray(
            [objective_by_origin[np.datetime64(origin, "ns")] for origin in base.target_times],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(f"v3 objective row missing for origin {exc.args[0]}") from exc
    # ``build_joint_windows`` intentionally preserves legacy float32 storage.
    # Rejoin the v3 target/context rows directly from their float64 source
    # tables so strict objective and SOC/ramp audits use the same values that
    # generated the LP references.
    frame_lookup = frame.copy()
    frame_lookup["timestamp"] = pd.to_datetime(frame_lookup["timestamp"], errors="raise")
    frame_lookup = frame_lookup.set_index("timestamp")
    target_exact = np.asarray(
        [frame_lookup.loc[pd.to_datetime(origin) + pd.to_timedelta(np.arange(4), unit="h"), list(TASK_ORDER)].to_numpy(dtype=np.float64) for origin in base.target_times],
        dtype=np.float64,
    ).reshape((n, 4, len(TASK_ORDER)))
    context_by_origin = {
        np.datetime64(origin, "ns"): group.sort_values("timestamp")
        for origin, group in scheduler_context.groupby("origin_timestamp", sort=False)
    }
    teacher_by_origin = {
        np.datetime64(origin, "ns"): group.sort_values("timestamp")
        for origin, group in teacher_dispatch.groupby("origin_timestamp", sort=False)
    }
    device_by_time = {
        np.datetime64(row["timestamp"], "ns"): float(row["p_chp"])
        for _, row in device_trajectory.iterrows()
    }
    context_exact = np.asarray(
        [context_by_origin[np.datetime64(origin, "ns")][list(SCHEDULER_CONTEXT_ORDER)].to_numpy(dtype=np.float64) for origin in base.target_times],
        dtype=np.float64,
    ).reshape((n, 4, len(SCHEDULER_CONTEXT_ORDER)))
    previous_exact = np.asarray(
        [device_by_time[np.datetime64(origin, "ns") - np.timedelta64(1, "h")] for origin in base.target_times],
        dtype=np.float64,
    ).reshape((n, 1))
    teacher_exact = np.asarray(
        [teacher_by_origin[np.datetime64(origin, "ns")][list(DISPATCH_ORDER)].to_numpy(dtype=np.float64) for origin in base.target_times],
        dtype=np.float64,
    ).reshape((n, 4, len(DISPATCH_ORDER)))
    return JointWindowSplit(
        load_history=base.load_history,
        exog_history=base.exog_history,
        device_history=base.device_history,
        device_status=base.device_status,
        forecast_target=target_exact,
        scheduler_context=context_exact,
        previous_chp=previous_exact,
        teacher_dispatch=teacher_exact,
        oracle_first_step_objective=objective_array,
        target_times=base.target_times,
        split=base.split,
        history_source=base.history_source,
        realized_renewables=realized_array,
        oracle_dispatch=oracle_array,
        oracle_four_step_objective=objective_array,
    )


def _mean_scale(array: np.ndarray, axis: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(array.mean(axis=axis), dtype=np.float32)
    scale = np.asarray(array.std(axis=axis), dtype=np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return mean, scale


@dataclass(frozen=True)
class JointNormalization:
    """Train-only affine normalization shared by the connected model."""

    load_mean: np.ndarray
    load_scale: np.ndarray
    exog_mean: np.ndarray
    exog_scale: np.ndarray
    device_mean: np.ndarray
    device_scale: np.ndarray
    scheduler_mean: np.ndarray
    scheduler_scale: np.ndarray
    fitted_split: str = "train"

    @classmethod
    def fit(cls, split: JointWindowSplit, *, expected_split: str = "train") -> "JointNormalization":
        if split.split != expected_split:
            raise ValueError(f"normalization must be fitted on {expected_split}")
        if len(split) == 0:
            raise ValueError("cannot fit normalization on an empty split")
        return cls(
            *_mean_scale(split.load_history, (0, 1)),
            *_mean_scale(split.exog_history, (0, 1)),
            *_mean_scale(split.device_history, (0, 1)),
            *_mean_scale(split.scheduler_context, (0, 1)),
            fitted_split=expected_split,
        )

    @staticmethod
    def _apply(array: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return ((array - mean.reshape((1, 1, -1))) / scale.reshape((1, 1, -1))).astype(np.float32)

    def transform(self, split: JointWindowSplit) -> JointWindowSplit:
        return replace(
            split,
            load_history=self._apply(split.load_history, self.load_mean, self.load_scale),
            exog_history=self._apply(split.exog_history, self.exog_mean, self.exog_scale),
            device_history=self._apply(split.device_history, self.device_mean, self.device_scale),
            forecast_target=self._apply(split.forecast_target, self.load_mean, self.load_scale),
            scheduler_context=self._apply(split.scheduler_context, self.scheduler_mean, self.scheduler_scale),
        )


def fit_joint_normalization(split: JointWindowSplit) -> JointNormalization:
    """Public contract helper: fit statistics on the train split only."""

    return JointNormalization.fit(split, expected_split="train")


@dataclass(frozen=True)
class LPSolveBenchmark:
    """Fail-closed timing receipt for the offline LP generation budget."""

    solve_count: int
    success_count: int
    median_seconds: float
    p95_seconds: float
    projected_total_solves: int
    projected_p95_hours: float
    gate_passed: bool
    solver: str = "scipy.optimize.linprog(method='highs')"
    python_version: str = platform.python_version()
    cpu: str = platform.processor() or platform.uname().processor or platform.machine() or platform.uname().machine or "unknown"

    def __post_init__(self) -> None:
        integer_fields = ("solve_count", "success_count", "projected_total_solves")
        for name in integer_fields:
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.success_count > self.solve_count:
            raise ValueError("success_count cannot exceed solve_count")
        for name in ("median_seconds", "p95_seconds", "projected_p95_hours"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def benchmark_lp_generation(
    cases: Sequence[DispatchInputs],
    *,
    projected_total_solves: int | None = None,
    required_solves: int = 500,
    max_projected_hours: float = 24.0,
) -> LPSolveBenchmark:
    """Benchmark exactly ``required_solves`` offline LP cases.

    No case is silently discarded: a failed solve or a projected duration over
    the contract budget raises immediately after returning the receipt.
    ``required_solves`` is configurable only for deterministic unit tests; the
    formal contract passes 500.
    """

    if required_solves <= 0 or len(cases) != required_solves:
        raise ValueError(f"benchmark requires exactly {required_solves} cases")
    durations: list[float] = []
    success = 0
    for case in cases:
        started = time.perf_counter()
        result = solve_dispatch_lp(case)
        durations.append(float(time.perf_counter() - started))
        if result.success:
            success += 1
    p95 = float(np.percentile(np.asarray(durations, dtype=np.float64), 95))
    median = float(np.median(np.asarray(durations, dtype=np.float64)))
    total = int(projected_total_solves if projected_total_solves is not None else required_solves)
    if total <= 0:
        raise ValueError("projected_total_solves must be positive")
    projected_hours = float(p95 * total / 3600.0)
    receipt = LPSolveBenchmark(
        solve_count=required_solves,
        success_count=success,
        median_seconds=median,
        p95_seconds=p95,
        projected_total_solves=total,
        projected_p95_hours=projected_hours,
        gate_passed=bool(success == required_solves and projected_hours <= max_projected_hours),
    )
    if not receipt.gate_passed:
        raise RuntimeError(
            "LP resource gate failed: "
            f"success={success}/{required_solves}, projected_p95_hours={projected_hours:.3f}"
        )
    return receipt


def _price_at_origin(
    renewable_frame: pd.DataFrame,
    index: int,
    name: str,
    parameters: Mapping[str, object],
) -> float:
    if name in renewable_frame.columns:
        value = float(renewable_frame.iloc[index][name])
    else:
        value = float(parameters.get(name, parameters.get(f"{name}_default", 0.0)))
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def build_causal_device_trajectory(
    hourly_frame: pd.DataFrame,
    renewable_frame: pd.DataFrame,
    parameters: Mapping[str, object],
    *,
    initial_soc: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Generate historical continuous device actions with an offline LP.

    At origin ``t`` the LP sees only a seasonal-naive (24-hour lag) forecast
    built from rows at or before ``t``.  PV/WT availability is persisted from
    ``t``.  Only the first LP step is executed and its SOC is carried forward.
    """

    required_load = ["timestamp", "electricity", "cooling", "heating"]
    base = _prepare_table(hourly_frame, required_load, "hourly frame")
    renewable_required = ["timestamp", "pv_available", "wt_available"]
    renewable_required.extend(
        name for name in ("grid_price", "gas_price", "carbon_price")
        if name in renewable_frame.columns
    )
    renew = _prepare_table(renewable_frame, renewable_required, "renewable frame")
    timestamps = _assert_same_timestamps((base, renew))
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if end_ts < start_ts:
        raise ValueError("end must not precede start")
    selected = np.flatnonzero((timestamps >= np.datetime64(start_ts)) & (timestamps <= np.datetime64(end_ts)))
    if selected.size == 0:
        raise ValueError("trajectory interval contains no timestamps")
    load_values = base[["electricity", "cooling", "heating"]].to_numpy(dtype=np.float64)
    pv = renew["pv_available"].to_numpy(dtype=np.float64)
    wt = renew["wt_available"].to_numpy(dtype=np.float64)
    for array_name, values in (("loads", load_values), ("pv_available", pv), ("wt_available", wt)):
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError(f"{array_name} must be finite and non-negative")
    soc = float(initial_soc)
    if not np.isfinite(soc) or not 0.0 <= soc <= 1.0:
        raise ValueError("initial_soc must be in [0,1]")
    rows: list[dict[str, float | str]] = []
    for index in selected:
        history_start = max(0, int(index) - 23)
        history = load_values[history_start : int(index) + 1]
        if len(history) >= 24:
            forecast = load_values[int(index) - 23 : int(index) + 1]
            # Forecast t+1:t+4 from the corresponding 24-hour-lag values.
            demand = forecast[:4]
        else:
            demand = np.repeat(load_values[int(index)][None, :], 4, axis=0)
        if demand.shape != (4, 3):
            raise ValueError("causal seasonal-naive forecast has an invalid shape")
        origin = int(index)
        lp = solve_dispatch_lp(
            DispatchInputs(
                demand=demand,
                pv_available=np.repeat(pv[origin], 4),
                wt_available=np.repeat(wt[origin], 4),
                parameters={
                    **dict(parameters),
                    "grid_energy_price": np.repeat(_price_at_origin(renew, origin, "grid_price", parameters), 4),
                    "gas_energy_price": np.repeat(_price_at_origin(renew, origin, "gas_price", parameters), 4),
                    "carbon_price": np.repeat(_price_at_origin(renew, origin, "carbon_price", parameters), 4),
                },
                initial_soc=soc,
            )
        )
        if not lp.success:
            raise RuntimeError(f"causal LP failed at {timestamps[origin]}: {lp.message}")
        for name in DISPATCH_ORDER:
            value = np.asarray(lp.values[name], dtype=np.float64)
            if value.shape != (4,) or not np.isfinite(value[0]):
                raise ValueError(f"LP returned invalid {name} trajectory")
        first = {"timestamp": str(pd.Timestamp(timestamps[origin]))}
        first.update({name: float(lp.values[name][0]) for name in DISPATCH_ORDER})
        rows.append(first)
        capacity = float(parameters.get("bess_energy_capacity", 1.0))
        if capacity <= 0.0 or not np.isfinite(capacity):
            raise ValueError("bess_energy_capacity must be positive")
        soc = float(np.clip(float(lp.values["soc"][0]) / capacity, 0.0, 1.0))
    return pd.DataFrame(rows, columns=("timestamp", *DISPATCH_ORDER))


def save_joint_split(
    split: JointWindowSplit,
    path: str | Path,
    normalization: JointNormalization | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Atomically persist arrays and optional train-only normalization."""

    split_path = Path(path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = split_path.with_suffix(split_path.suffix + ".tmp")
    payload: dict[str, object] = {
        "load_history": split.load_history,
        "exog_history": split.exog_history,
        "device_history": split.device_history,
        "device_status": split.device_status,
        "forecast_target": split.forecast_target,
        "scheduler_context": split.scheduler_context,
        "previous_chp": split.previous_chp,
        "teacher_dispatch": split.teacher_dispatch,
        "oracle_first_step_objective": split.oracle_first_step_objective,
        "target_times": split.target_times,
        "split": np.asarray(split.split),
        "history_source": np.asarray(split.history_source),
        "metadata_json": np.asarray(json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True)),
    }
    if split.realized_renewables is not None:
        payload.update({
            "realized_renewables": split.realized_renewables,
            "oracle_dispatch": split.oracle_dispatch,
            "oracle_four_step_objective": split.oracle_four_step_objective,
        })
    if normalization is not None:
        for name in (
            "load_mean", "load_scale", "exog_mean", "exog_scale", "device_mean", "device_scale",
            "scheduler_mean", "scheduler_scale",
        ):
            payload[f"normalization_{name}"] = getattr(normalization, name)
        payload["normalization_fitted_split"] = np.asarray(normalization.fitted_split)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(split_path)


def load_joint_split(path: str | Path) -> tuple[JointWindowSplit, JointNormalization | None, Mapping[str, object]]:
    """Load an aligned split and optional normalization with shape validation."""

    with np.load(Path(path), allow_pickle=False) as payload:
        required = {
            "load_history", "exog_history", "device_history", "device_status", "forecast_target",
            "scheduler_context", "previous_chp", "teacher_dispatch", "oracle_first_step_objective",
            "target_times", "split", "history_source",
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"joint split artifact is missing fields: {missing}")
        split = JointWindowSplit(
            load_history=payload["load_history"], exog_history=payload["exog_history"],
            device_history=payload["device_history"], device_status=payload["device_status"],
            forecast_target=payload["forecast_target"], scheduler_context=payload["scheduler_context"],
            previous_chp=payload["previous_chp"], teacher_dispatch=payload["teacher_dispatch"],
            oracle_first_step_objective=payload["oracle_first_step_objective"],
            target_times=payload["target_times"], split=str(np.asarray(payload["split"]).item()),
            history_source=str(np.asarray(payload["history_source"]).item()),
            realized_renewables=payload["realized_renewables"] if "realized_renewables" in payload else None,
            oracle_dispatch=payload["oracle_dispatch"] if "oracle_dispatch" in payload else None,
            oracle_four_step_objective=payload["oracle_four_step_objective"] if "oracle_four_step_objective" in payload else None,
        )
        norm_fields = (
            "load_mean", "load_scale", "exog_mean", "exog_scale", "device_mean", "device_scale",
            "scheduler_mean", "scheduler_scale",
        )
        present = [f"normalization_{name}" in payload for name in norm_fields]
        if any(present) and not all(present):
            raise ValueError("normalization artifact is incomplete")
        normalization = None
        if all(present):
            if "normalization_fitted_split" not in payload:
                raise ValueError("normalization artifact is missing fitted_split")
            normalization = JointNormalization(
                **{name: np.asarray(payload[f"normalization_{name}"], dtype=np.float32) for name in norm_fields},
                fitted_split=str(np.asarray(payload["normalization_fitted_split"]).item()),
            )
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item())) if "metadata_json" in payload else {}
    return split, normalization, metadata


__all__ = [
    "SCHEDULER_CONTEXT_ORDER",
    "DEVICE_STATUS_ORDER",
    "STATUS_ORDER",
    "JointNormalization",
    "JointWindowSplit",
    "LPSolveBenchmark",
    "benchmark_lp_generation",
    "build_causal_device_trajectory",
    "build_joint_windows",
    "build_joint_windows_v3",
    "derive_device_status",
    "fit_joint_normalization",
    "load_joint_split",
    "save_joint_split",
]
