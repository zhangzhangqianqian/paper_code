"""Formal-v4 causal data contracts and capacity-bound window materialization."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER
from .data import derive_device_status


CONTINUOUS_DEVICE_FEATURE_NAMES: tuple[str, ...] = DISPATCH_ORDER[:17]
ACTIVITY_FEATURE_NAMES: tuple[str, ...] = STATUS_ORDER
EXCLUDED_DEVICE_FEATURE_NAMES: tuple[str, ...] = ("slack_e", "slack_c", "slack_h", "q_dump")


def _array(value: object, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _hash_arrays(*arrays: np.ndarray, extra: str = "") -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(extra.encode("utf-8"))
    return digest.hexdigest()


def _scale(array: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(array.mean(axis=axes), dtype=np.float32)
    std = np.asarray(array.std(axis=axes), dtype=np.float32)
    return mean, np.where(std < 1.0e-6, 1.0, std).astype(np.float32)


@dataclass(frozen=True)
class FormalV4BaseSeries:
    load_and_exog: np.ndarray
    renewable_forecast: np.ndarray
    renewable_realized: np.ndarray
    prices_and_weights: np.ndarray
    timestamps: np.ndarray
    split: str

    def __post_init__(self) -> None:
        load = _array(self.load_and_exog, "load_and_exog", 2)
        if load.shape[1] != len(TASK_ORDER) + len(EXOG_ORDER):
            raise ValueError("load_and_exog must contain 4 tasks followed by 12 exogenous features")
        forecast = _array(self.renewable_forecast, "renewable_forecast", 2)
        realized = _array(self.renewable_realized, "renewable_realized", 2)
        prices = _array(self.prices_and_weights, "prices_and_weights", 2)
        n = load.shape[0]
        if forecast.shape != (n, 2) or realized.shape != (n, 2):
            raise ValueError("renewable arrays must have shape [T,2]")
        if prices.shape != (n, 3):
            raise ValueError("prices_and_weights must have shape [T,3]")
        timestamps = np.asarray(self.timestamps, dtype="datetime64[ns]")
        if timestamps.shape != (n,) or n == 0:
            raise ValueError("timestamps must have one non-empty row per observation")
        # Preserve source timestamps, including documented short gaps.  Window
        # materialization below rejects any 24->4 window crossing a gap rather
        # than imputing targets or silently changing the time scale.
        deltas = np.diff(timestamps)
        if np.any(deltas <= np.timedelta64(0, "s")):
            raise ValueError("base timestamps must be strictly increasing")
        if self.split not in {"train", "selection", "evaluation"}:
            raise ValueError("split must be train, selection, or evaluation")
        if (forecast < 0.0).any() or (realized < 0.0).any():
            raise ValueError("renewable values must be non-negative")
        object.__setattr__(self, "load_and_exog", load.astype(np.float64))
        object.__setattr__(self, "renewable_forecast", forecast.astype(np.float64))
        object.__setattr__(self, "renewable_realized", realized.astype(np.float64))
        object.__setattr__(self, "prices_and_weights", prices.astype(np.float64))
        object.__setattr__(self, "timestamps", timestamps)


@dataclass(frozen=True)
class FormalV4WindowSplit:
    load_history: np.ndarray
    exog_history: np.ndarray
    renewable_history: np.ndarray
    device_history: np.ndarray
    activity_history: np.ndarray
    forecast_target: np.ndarray
    rigid_demand: np.ndarray
    renewable_forecast: np.ndarray
    renewable_realized: np.ndarray
    prices_and_weights: np.ndarray
    initial_soc: np.ndarray
    previous_chp: np.ndarray
    target_times: np.ndarray
    trajectory_ids: np.ndarray
    state_hashes: np.ndarray
    device_feature_names: tuple[str, ...] = CONTINUOUS_DEVICE_FEATURE_NAMES
    activity_feature_names: tuple[str, ...] = ACTIVITY_FEATURE_NAMES
    split: str = "train"
    history_source: str = "causal_simulation"

    def __post_init__(self) -> None:
        fields = {
            "load_history": (self.load_history, (24, 4)),
            "exog_history": (self.exog_history, (24, 12)),
            "renewable_history": (self.renewable_history, (24, 2)),
            "device_history": (self.device_history, (24, 17)),
            "activity_history": (self.activity_history, (24, 6)),
            "forecast_target": (self.forecast_target, (4, 4)),
            "rigid_demand": (self.rigid_demand, (4, 3)),
            "renewable_forecast": (self.renewable_forecast, (4, 2)),
            "renewable_realized": (self.renewable_realized, (4, 2)),
            "prices_and_weights": (self.prices_and_weights, (4, 3)),
        }
        converted: dict[str, np.ndarray] = {}
        n: int | None = None
        for name, (value, tail) in fields.items():
            array = _array(value, name, 3)
            if tuple(array.shape[1:]) != tail:
                raise ValueError(f"{name} must have shape [N,{tail[0]},{tail[1]}]")
            n = array.shape[0] if n is None else n
            if array.shape[0] != n:
                raise ValueError(f"{name} has a different sample count")
            converted[name] = array.astype(np.float64)
        assert n is not None
        initial_soc = _array(self.initial_soc, "initial_soc", 2)
        previous_chp = _array(self.previous_chp, "previous_chp", 2)
        if initial_soc.shape != (n, 1) or previous_chp.shape != (n, 1):
            raise ValueError("initial_soc and previous_chp must have shape [N,1]")
        target_times = np.asarray(self.target_times, dtype="datetime64[ns]")
        trajectory_ids = np.asarray(self.trajectory_ids, dtype=str)
        state_hashes = np.asarray(self.state_hashes, dtype=str)
        if target_times.shape != (n,) or trajectory_ids.shape != (n,) or state_hashes.shape != (n,):
            raise ValueError("target_times, trajectory_ids and state_hashes must have shape [N]")
        if n > 1 and not np.all(np.diff(target_times) > np.timedelta64(0, "s")):
            raise ValueError("target_times must be strictly chronological")
        if not np.isin(self.activity_history, (0.0, 1.0)).all():
            raise ValueError("activity_history must be binary")
        if tuple(self.device_feature_names) != CONTINUOUS_DEVICE_FEATURE_NAMES:
            raise ValueError("device_feature_names must be the frozen 17-field ledger")
        if tuple(self.activity_feature_names) != ACTIVITY_FEATURE_NAMES:
            raise ValueError("activity_feature_names must match STATUS_ORDER")
        if self.split not in {"train", "selection", "evaluation"}:
            raise ValueError("split must be train, selection, or evaluation")
        if self.history_source not in {"causal_simulation", "joint_policy_rollin"}:
            raise ValueError("unsupported history_source")
        expected_forecast = np.repeat(converted["renewable_history"][:, -1:, :], 4, axis=1)
        if not np.array_equal(converted["renewable_forecast"], expected_forecast):
            raise ValueError("renewable_forecast must repeat the last causal availability")
        for name, array in converted.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "initial_soc", initial_soc.astype(np.float64))
        object.__setattr__(self, "previous_chp", previous_chp.astype(np.float64))
        object.__setattr__(self, "target_times", target_times)
        object.__setattr__(self, "trajectory_ids", trajectory_ids)
        object.__setattr__(self, "state_hashes", state_hashes)

    def __len__(self) -> int:
        return int(self.load_history.shape[0])

    @property
    def gas_context_target(self) -> np.ndarray:
        return self.forecast_target[..., 3:4]


@dataclass(frozen=True)
class FormalV4Normalization:
    load_mean: np.ndarray
    load_scale: np.ndarray
    exog_mean: np.ndarray
    exog_scale: np.ndarray
    device_mean: np.ndarray
    device_scale: np.ndarray
    activity_mean: np.ndarray
    activity_scale: np.ndarray
    scheduler_mean: np.ndarray
    scheduler_scale: np.ndarray
    fitted_split: str = "train"

    @classmethod
    def fit(cls, split: FormalV4WindowSplit) -> "FormalV4Normalization":
        if split.split != "train":
            raise ValueError("formal-v4 normalization must be fitted on train")
        if len(split) == 0:
            raise ValueError("cannot fit normalization on an empty split")
        return cls(
            *_scale(split.load_history, (0, 1)), *_scale(split.exog_history, (0, 1)),
            *_scale(split.device_history, (0, 1)), *_scale(split.activity_history, (0, 1)),
            *_scale(split.prices_and_weights, (0, 1)), fitted_split="train",
        )


@dataclass(frozen=True)
class SameInformationTeacherOverlay:
    dispatch: np.ndarray
    target_times: np.ndarray
    state_hashes: np.ndarray
    stage_p_checkpoint_sha256: str
    split: str

    def __post_init__(self) -> None:
        dispatch = _array(self.dispatch, "dispatch", 3)
        if dispatch.shape[-1] != len(DISPATCH_ORDER) or dispatch.shape[1] != 4:
            raise ValueError("teacher dispatch must have shape [N,4,21]")
        target_times = np.asarray(self.target_times, dtype="datetime64[ns]")
        state_hashes = np.asarray(self.state_hashes, dtype=str)
        if target_times.shape != (dispatch.shape[0],) or state_hashes.shape != (dispatch.shape[0],):
            raise ValueError("teacher alignment fields must have shape [N]")
        if not self.stage_p_checkpoint_sha256:
            raise ValueError("teacher overlay must record the Stage P checkpoint hash")
        object.__setattr__(self, "dispatch", dispatch.astype(np.float64))
        object.__setattr__(self, "target_times", target_times)
        object.__setattr__(self, "state_hashes", state_hashes)


def _capacity_receipt(value: Mapping[str, Any] | str | Path | None) -> Mapping[str, Any]:
    if value is None:
        raise PermissionError("state materialization requires a frozen capacity receipt")
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise PermissionError("state materialization requires a frozen capacity receipt")
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = value
    if not isinstance(payload, Mapping) or payload.get("gate0_authorized") is not True:
        raise PermissionError("state materialization requires a Gate 0 capacity receipt")
    return payload


def materialize_state_windows(
    base: FormalV4BaseSeries,
    device_trajectory: np.ndarray,
    *,
    capacity_receipt: Mapping[str, Any] | str | Path | None,
    split: str | None = None,
    bess_energy_capacity: float | None = None,
    settled_mask: np.ndarray | None = None,
) -> FormalV4WindowSplit:
    """Create capacity-bound 24→4 windows only after Gate 0 is authorized."""

    receipt = _capacity_receipt(capacity_receipt)
    requested_split = str(split or base.split)
    if requested_split == "evaluation" and receipt.get("gate3_authorized") is not True:
        raise PermissionError("evaluation data require Gate 3 authorization")
    trajectory = _array(device_trajectory, "device_trajectory", 2)
    if trajectory.shape != (len(base.timestamps), len(DISPATCH_ORDER)):
        raise ValueError("device_trajectory must have shape [T,21]")
    if len(base.timestamps) < 28:
        raise ValueError("base series must contain at least 28 hourly rows")
    mask = None if settled_mask is None else np.asarray(settled_mask, dtype=bool)
    if mask is not None and mask.shape != (len(base.timestamps),):
        raise ValueError("settled_mask must have shape [T]")
    energy_capacity = float(bess_energy_capacity if bess_energy_capacity is not None else receipt.get("bess_energy_capacity", 1.0))
    if not np.isfinite(energy_capacity) or energy_capacity <= 0.0:
        raise ValueError("bess_energy_capacity must be positive")
    loads = base.load_and_exog[:, :4]
    exog = base.load_and_exog[:, 4:]
    statuses = derive_device_status(trajectory.reshape(1, len(trajectory), -1))[0]
    histories_load: list[np.ndarray] = []; histories_exog: list[np.ndarray] = []; histories_renew: list[np.ndarray] = []
    histories_device: list[np.ndarray] = []; histories_activity: list[np.ndarray] = []
    targets: list[np.ndarray] = []; rigid: list[np.ndarray] = []; forecasts: list[np.ndarray] = []
    realized: list[np.ndarray] = []; prices: list[np.ndarray] = []; socs: list[list[float]] = []; previous: list[list[float]] = []
    target_times: list[np.datetime64] = []; state_hashes: list[str] = []
    for origin in range(24, len(base.timestamps) - 3):
        if mask is not None and (not bool(mask[origin]) or not bool(mask[origin - 24:origin].all())):
            continue
        window_times = base.timestamps[origin - 24:origin + 4]
        if not np.all(np.diff(window_times) == np.timedelta64(1, "h")):
            continue
        history_device = trajectory[origin - 24:origin, :17]
        history_renew = base.renewable_realized[origin - 24:origin]
        history_load = loads[origin - 24:origin]
        history_exog = exog[origin - 24:origin]
        future_renew = np.repeat(history_renew[-1:, :], 4, axis=0)
        histories_load.append(history_load); histories_exog.append(history_exog); histories_renew.append(history_renew)
        histories_device.append(history_device); histories_activity.append(statuses[origin - 24:origin])
        targets.append(loads[origin:origin + 4]); rigid.append(loads[origin:origin + 4, :3]); forecasts.append(future_renew)
        realized.append(base.renewable_realized[origin:origin + 4]); prices.append(base.prices_and_weights[origin:origin + 4])
        socs.append([float(np.clip(trajectory[origin - 1, _I["soc"]] / energy_capacity, 0.0, 1.0))])
        previous.append([float(max(trajectory[origin - 1, _I["p_chp"]], 0.0))])
        target_times.append(base.timestamps[origin])
        state_hashes.append(_hash_arrays(history_load, history_exog, history_device, histories_activity[-1], extra=str(base.timestamps[origin])))
    return FormalV4WindowSplit(
        np.asarray(histories_load), np.asarray(histories_exog), np.asarray(histories_renew), np.asarray(histories_device),
        np.asarray(histories_activity), np.asarray(targets), np.asarray(rigid), np.asarray(forecasts), np.asarray(realized),
        np.asarray(prices), np.asarray(socs), np.asarray(previous), np.asarray(target_times),
        np.asarray([str(receipt.get("trajectory_id", "capacity_bound_causal"))] * len(target_times)), np.asarray(state_hashes),
        split=requested_split, history_source="causal_simulation",
    )


def build_same_information_teacher(
    split: FormalV4WindowSplit,
    teacher_dispatch: np.ndarray,
    *,
    stage_p_checkpoint_sha256: str,
) -> SameInformationTeacherOverlay:
    """Bind Stage-S labels to the exact deployment state, timestamps and hash."""

    dispatch = _array(teacher_dispatch, "teacher_dispatch", 3)
    if dispatch.shape != (len(split), 4, len(DISPATCH_ORDER)):
        raise ValueError("teacher_dispatch must match the split [N,4,21]")
    return SameInformationTeacherOverlay(
        dispatch=dispatch, target_times=split.target_times, state_hashes=split.state_hashes,
        stage_p_checkpoint_sha256=stage_p_checkpoint_sha256, split=split.split,
    )


_I = {name: index for index, name in enumerate(DISPATCH_ORDER)}


__all__ = [
    "ACTIVITY_FEATURE_NAMES", "CONTINUOUS_DEVICE_FEATURE_NAMES", "EXCLUDED_DEVICE_FEATURE_NAMES",
    "FormalV4BaseSeries", "FormalV4Normalization", "FormalV4WindowSplit", "SameInformationTeacherOverlay",
    "build_same_information_teacher", "materialize_state_windows",
]
