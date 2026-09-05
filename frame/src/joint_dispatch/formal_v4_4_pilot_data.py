"""Version-isolated Pilot indices and batches for formal-v4.4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
import torch

from .formal_v4_2_data import NormalizationReceiptV42, apply_normalization
from .formal_v4_4_regime import derive_last_observed_regime, derive_thermal_regimes, thermal_transition_mask


def _years(times: np.ndarray) -> np.ndarray:
    return times.astype("datetime64[Y]").astype(int) + 1970


def _hash_indices(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _season_id(times: np.ndarray) -> np.ndarray:
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    return np.select([np.isin(months, [12, 1, 2]), np.isin(months, [6, 7, 8])], [0, 2], default=1).astype(np.int64)


def _dominant_regime(future: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.bincount(row, minlength=3).argmax()) for row in future], dtype=np.int64)


def _transition_flag(last: np.ndarray, future: np.ndarray) -> np.ndarray:
    return thermal_transition_mask(future, last)


@dataclass(frozen=True)
class PilotSplitReceiptV44:
    train: np.ndarray
    early_stop: np.ndarray
    selection_full: np.ndarray
    selection_stress: np.ndarray
    train_hash: str
    early_stop_hash: str
    selection_full_hash: str
    selection_stress_hash: str
    purge_hours: int
    stratum_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        values = []
        for name in ("train", "early_stop", "selection_full", "selection_stress"):
            array = np.asarray(getattr(self, name), dtype=np.int64)
            if array.ndim != 1 or len(array) == 0 or len(np.unique(array)) != len(array):
                raise ValueError(f"{name} must be a non-empty unique index array")
            object.__setattr__(self, name, array)
            values.append(array)
        if int(self.purge_hours) != 28:
            raise ValueError("formal-v4.4 purge must be 28 hours")
        for name in ("train_hash", "early_stop_hash", "selection_full_hash", "selection_stress_hash"):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if any(int(v) <= 0 for v in self.stratum_counts.values()):
            raise ValueError("every Pilot stratum must be represented")

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.tolist(), "early_stop": self.early_stop.tolist(),
            "selection_full": self.selection_full.tolist(), "selection_stress": self.selection_stress.tolist(),
            "train_hash": self.train_hash, "early_stop_hash": self.early_stop_hash,
            "selection_full_hash": self.selection_full_hash, "selection_stress_hash": self.selection_stress_hash,
            "purge_hours": self.purge_hours, "stratum_counts": dict(self.stratum_counts),
        }


def _check_years(times: np.ndarray, expected: set[int], name: str) -> None:
    actual = set(int(value) for value in _years(times))
    if actual != expected:
        raise ValueError(f"{name} must contain exactly years {sorted(expected)}")


def _minimum_time_distance(a: np.ndarray, b: np.ndarray) -> np.timedelta64:
    values = np.asarray(a, dtype="datetime64[ns]")
    other = np.asarray(b, dtype="datetime64[ns]")
    return np.min(np.abs(values[:, None] - other[None, :]))


def _deterministic_sample(candidate: np.ndarray, strata: np.ndarray, size: int, seed: int) -> np.ndarray:
    if len(candidate) < size:
        raise ValueError("candidate training pool is smaller than the requested Pilot size")
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(["-".join(str(int(v)) for v in row) for row in strata], dtype=str)
    groups = {label: candidate[labels == label] for label in np.unique(labels)}
    selected: list[int] = []
    for label in sorted(groups):
        group = groups[label].copy()
        rng.shuffle(group)
        selected.append(int(group[0]))
    remaining = np.asarray([index for index in candidate.tolist() if index not in set(selected)], dtype=np.int64)
    rng.shuffle(remaining)
    selected.extend(int(value) for value in remaining[: max(0, size - len(selected))])
    if len(selected) < size:
        raise ValueError("stratified Pilot sampling could not fill the requested size")
    return np.sort(np.asarray(selected[:size], dtype=np.int64))


def build_pilot_indices(
    train_origins: Any,
    train_last_regime: Any,
    train_future_regime: Any,
    selection_origins: Any,
    selection_last_regime: Any,
    selection_future_regime: Any,
    *,
    n_train: int = 4096,
    seed: int = 2026,
) -> PilotSplitReceiptV44:
    train_times = np.asarray(train_origins, dtype="datetime64[ns]")
    selection_times = np.asarray(selection_origins, dtype="datetime64[ns]")
    train_last = np.asarray(train_last_regime, dtype=np.int64)
    train_future = np.asarray(train_future_regime, dtype=np.int64)
    selection_last = np.asarray(selection_last_regime, dtype=np.int64)
    selection_future = np.asarray(selection_future_regime, dtype=np.int64)
    if train_times.ndim != 1 or train_last.shape != (len(train_times),) or train_future.shape != (len(train_times), 4):
        raise ValueError("training Pilot arrays have incompatible shapes")
    if selection_times.ndim != 1 or selection_last.shape != (len(selection_times),) or selection_future.shape != (len(selection_times), 4):
        raise ValueError("selection Pilot arrays have incompatible shapes")
    _check_years(train_times, {2015, 2016, 2017, 2018}, "training origins")
    _check_years(selection_times, {2019}, "selection origins")
    if n_train != 4096 or seed != 2026:
        raise ValueError("formal-v4.4 Pilot size and seed are frozen")
    if not np.isin(train_last, (0, 1, 2)).all() or not np.isin(train_future, (0, 1, 2)).all():
        raise ValueError("training regimes are invalid")
    if not np.isin(selection_last, (0, 1, 2)).all() or not np.isin(selection_future, (0, 1, 2)).all():
        raise ValueError("selection regimes are invalid")

    validation_mask = np.zeros(len(train_times), dtype=bool)
    for month in (1, 4, 7, 10):
        validation_mask |= (
            (train_times >= np.datetime64(f"2018-{month:02d}-08"))
            & (train_times < np.datetime64(f"2018-{month:02d}-22"))
        )
    early_stop = np.flatnonzero(validation_mask)
    if len(early_stop) == 0:
        raise ValueError("fixed early-stop blocks are empty")
    candidate_mask = ~validation_mask
    for validation_time in train_times[early_stop]:
        candidate_mask &= np.abs(train_times - validation_time) > np.timedelta64(28, "h")
    candidate = np.flatnonzero(candidate_mask)
    strata = np.stack((_season_id(train_times), _dominant_regime(train_future), _transition_flag(train_last, train_future)), axis=1)
    selected = _deterministic_sample(candidate, strata[candidate], n_train, seed)
    selected_labels = np.asarray(["-".join(str(int(v)) for v in row) for row in strata[selected]], dtype=str)
    counts = {str(label): int(np.sum(selected_labels == label)) for label in np.unique(selected_labels)}
    transition = _transition_flag(selection_last, selection_future)
    active = (selection_future[:, 0] == 1) | (selection_future[:, 0] == 2)
    transition_indices = np.flatnonzero(transition)
    non_transition_active = np.flatnonzero((~transition) & active)
    rng = np.random.default_rng(seed)
    rng.shuffle(non_transition_active)
    stress = np.unique(np.concatenate((transition_indices, non_transition_active[: len(transition_indices)])))
    if len(stress) == 0:
        raise ValueError("selection stress sample is empty")
    return PilotSplitReceiptV44(
        train=selected, early_stop=early_stop,
        selection_full=np.arange(len(selection_times), dtype=np.int64),
        selection_stress=np.sort(stress),
        train_hash=_hash_indices(selected), early_stop_hash=_hash_indices(early_stop),
        selection_full_hash=_hash_indices(np.arange(len(selection_times), dtype=np.int64)),
        selection_stress_hash=_hash_indices(np.sort(stress)),
        purge_hours=28, stratum_counts=counts,
    )


def validate_pilot_split(receipt: PilotSplitReceiptV44, train_times: Any, selection_times: Any) -> None:
    train = np.asarray(train_times, dtype="datetime64[ns]")
    selection = np.asarray(selection_times, dtype="datetime64[ns]")
    if np.any(receipt.train >= len(train)) or np.any(receipt.early_stop >= len(train)):
        raise ValueError("training Pilot index is out of range")
    if np.any(receipt.selection_full >= len(selection)) or np.any(receipt.selection_stress >= len(selection)):
        raise ValueError("selection Pilot index is out of range")
    if set(int(v) for v in _years(selection[receipt.selection_full])) != {2019}:
        raise ValueError("selection Pilot contains a forbidden year")
    if set(receipt.train.tolist()) & set(receipt.early_stop.tolist()):
        raise ValueError("Pilot train and early-stop indices overlap")
    if _minimum_time_distance(train[receipt.train], train[receipt.early_stop]) <= np.timedelta64(28, "h"):
        raise ValueError("Pilot train and early-stop indices violate the purge")


def build_v44_batches(
    materialized: Any,
    normalization: NormalizationReceiptV42,
    indices: Any,
    batch_size: int,
    teacher_dispatch: Any | None = None,
) -> list[dict[str, torch.Tensor]]:
    index = np.asarray(indices, dtype=np.int64)
    if index.ndim != 1 or len(index) == 0 or np.any(index < 0) or np.any(index >= len(materialized)):
        raise ValueError("indices are invalid for materialized windows")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    normalized = apply_normalization(materialized, normalization)
    raw_history = np.asarray(materialized.load_history[index], dtype=np.float64)
    target = np.asarray(materialized.forecast_target[index], dtype=np.float64)
    last = derive_last_observed_regime(raw_history)
    future = derive_thermal_regimes(target)
    transition = thermal_transition_mask(future, last)
    teacher = None if teacher_dispatch is None else np.asarray(teacher_dispatch)[index]
    result: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(index), int(batch_size)):
        stop = min(start + int(batch_size), len(index))
        local = index[start:stop]
        item = {
            "load_history": torch.from_numpy(normalized.load_history[local]),
            "exog_history": torch.from_numpy(normalized.exog_history[local]),
            "device_history": torch.from_numpy(normalized.device_history[local]),
            "activity_history": torch.from_numpy(normalized.activity_history[local]),
            "scheduler_context": torch.from_numpy(normalized.scheduler_context[local]),
            "previous_chp": torch.as_tensor(materialized.previous_chp[local], dtype=torch.float32),
            "initial_soc": torch.as_tensor(materialized.initial_soc[local], dtype=torch.float32),
            "target_normalized": torch.from_numpy(normalized.target_normalized[local]),
            "target_physical": torch.as_tensor(materialized.forecast_target[local], dtype=torch.float32),
            "realized_renewables": torch.as_tensor(materialized.renewable_realized[local], dtype=torch.float32),
            "last_thermal_regime": torch.as_tensor(last[start:stop], dtype=torch.long),
            "thermal_regime_target": torch.as_tensor(future[start:stop], dtype=torch.long),
            "thermal_transition_mask": torch.as_tensor(transition[start:stop], dtype=torch.float32),
            "sample_indices": torch.as_tensor(local, dtype=torch.long),
        }
        if teacher is not None:
            item["teacher_dispatch"] = torch.as_tensor(teacher[start:stop], dtype=torch.float32)
        result.append(item)
    return result


__all__ = ["PilotSplitReceiptV44", "build_pilot_indices", "build_v44_batches", "validate_pilot_split"]
