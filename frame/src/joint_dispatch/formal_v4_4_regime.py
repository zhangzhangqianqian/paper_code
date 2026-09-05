"""Causal thermal regimes, transition priors, and train-only statistics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


EPSILON = 1.0e-9
TRAIN_YEARS = (2015, 2016, 2017, 2018)


def _array(value: Any, ndim: int, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with ndim={ndim}")
    return result


def _validate_target(target: Any) -> np.ndarray:
    value = _array(target, 3, "target").astype(np.float64, copy=False)
    if value.shape[-1] != 4:
        raise ValueError("target must have shape [N,4,4]")
    if np.any(value[..., 1:3] < 0.0):
        raise ValueError("cooling and heating targets must be non-negative")
    return value


def _thermal_labels(target: np.ndarray, epsilon: float) -> np.ndarray:
    cooling = target[..., 1] > epsilon
    heating = target[..., 2] > epsilon
    if np.any(cooling & heating):
        raise ValueError("simultaneous cooling and heating targets are not representable")
    labels = np.zeros(cooling.shape, dtype=np.int64)
    labels[cooling] = 1
    labels[heating] = 2
    return labels


def derive_thermal_regimes(target: Any, epsilon: float = EPSILON) -> np.ndarray:
    if not np.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    return _thermal_labels(_validate_target(target), float(epsilon))


def derive_last_observed_regime(load_history: Any, epsilon: float = EPSILON) -> np.ndarray:
    history = _array(load_history, 3, "load_history").astype(np.float64, copy=False)
    if history.shape[1:] != (24, 4):
        raise ValueError("load_history must have shape [N,24,4]")
    return _thermal_labels(history[:, -1:, :], float(epsilon))[:, 0]


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass(frozen=True)
class ThermalPriorReceiptV44:
    transition_probability: np.ndarray
    transition_count: np.ndarray
    active_mean: np.ndarray
    active_scale: np.ndarray
    active_count: np.ndarray
    class_count: np.ndarray
    years: tuple[int, ...]
    target_sha256: str
    history_sha256: str
    timestamp_sha256: str

    def __post_init__(self) -> None:
        if np.asarray(self.transition_probability).shape != (4, 3, 3):
            raise ValueError("transition_probability must have shape [4,3,3]")
        if np.asarray(self.transition_count).shape != (4, 3, 3):
            raise ValueError("transition_count must have shape [4,3,3]")
        for name in ("active_mean", "active_scale", "active_count"):
            value = np.asarray(getattr(self, name))
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite with shape [2]")
        if np.any(np.asarray(self.active_scale) <= 0.0) or np.any(np.asarray(self.active_count) <= 0):
            raise ValueError("active statistics must be positive")
        if np.asarray(self.class_count).shape != (3,) or np.any(np.asarray(self.class_count) <= 0):
            raise ValueError("every thermal class must occur in training data")
        probability = np.asarray(self.transition_probability, dtype=np.float64)
        if not np.isfinite(probability).all() or np.any(probability <= 0.0):
            raise ValueError("transition probabilities must be finite and positive")
        np.testing.assert_allclose(probability.sum(axis=-1), 1.0, rtol=0.0, atol=1.0e-12)
        if tuple(self.years) != TRAIN_YEARS:
            raise ValueError("training years must be exactly 2015-2018")
        for name in ("target_sha256", "history_sha256", "timestamp_sha256"):
            digest = getattr(self, name)
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_probability": np.asarray(self.transition_probability).tolist(),
            "transition_count": np.asarray(self.transition_count).tolist(),
            "active_mean": np.asarray(self.active_mean).tolist(),
            "active_scale": np.asarray(self.active_scale).tolist(),
            "active_count": np.asarray(self.active_count).tolist(),
            "class_count": np.asarray(self.class_count).tolist(),
            "years": list(self.years),
            "target_sha256": self.target_sha256,
            "history_sha256": self.history_sha256,
            "timestamp_sha256": self.timestamp_sha256,
        }


def _validate_training_times(times: Any, n: int) -> np.ndarray:
    value = np.asarray(times, dtype="datetime64[ns]")
    if value.shape != (n,):
        raise ValueError("training timestamps must have shape [N]")
    years = value.astype("datetime64[Y]").astype(int) + 1970
    if set(int(y) for y in years) != set(TRAIN_YEARS):
        raise ValueError("timestamps must cover exactly the training years 2015-2018")
    if any(int(y) not in TRAIN_YEARS for y in years):
        raise ValueError("timestamps contain a non-training year")
    return value


def fit_thermal_prior(
    train_target: Any,
    train_history: Any,
    train_times: Any,
    *,
    alpha: float = 1.0,
    epsilon: float = EPSILON,
) -> ThermalPriorReceiptV44:
    target = _validate_target(train_target)
    history = _array(train_history, 3, "train_history").astype(np.float64, copy=False)
    if history.shape != (target.shape[0], 24, 4):
        raise ValueError("train_history must have shape [N,24,4] aligned with train_target")
    if not np.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("Laplace alpha must be positive and finite")
    times = _validate_training_times(train_times, target.shape[0])
    future = derive_thermal_regimes(target, epsilon)
    last = derive_last_observed_regime(history, epsilon)
    counts = np.full((4, 3, 3), float(alpha), dtype=np.float64)
    for horizon in range(4):
        np.add.at(counts[horizon], (last, future[:, horizon]), 1.0)
    probability = counts / counts.sum(axis=-1, keepdims=True)
    class_count = np.bincount(future.reshape(-1), minlength=3)
    active_values = []
    active_counts = []
    scales = []
    for class_id, task_index in ((1, 1), (2, 2)):
        values = target[..., task_index][future == class_id]
        if values.size == 0:
            raise ValueError(f"thermal class {class_id} has no active training values")
        active_values.append(float(values.mean()))
        scale = float(values.std())
        scales.append(scale if scale > 1.0e-12 else 1.0)
        active_counts.append(int(values.size))
    return ThermalPriorReceiptV44(
        transition_probability=probability,
        transition_count=counts,
        active_mean=np.asarray(active_values, dtype=np.float64),
        active_scale=np.asarray(scales, dtype=np.float64),
        active_count=np.asarray(active_counts, dtype=np.int64),
        class_count=class_count.astype(np.int64),
        years=TRAIN_YEARS,
        target_sha256=_hash_array(target),
        history_sha256=_hash_array(history),
        timestamp_sha256=_hash_array(times.astype("datetime64[ns]").astype(np.int64)),
    )


def thermal_transition_mask(future_regime: Any, last_regime: Any) -> np.ndarray:
    future = np.asarray(future_regime)
    previous = np.asarray(last_regime)
    if future.ndim != 2 or future.shape[1] != 4 or previous.shape != (future.shape[0],):
        raise ValueError("regime arrays have incompatible shapes")
    if not np.isin(future, (0, 1, 2)).all() or not np.isin(previous, (0, 1, 2)).all():
        raise ValueError("regimes must contain only 0, 1, and 2")
    return np.any(future != previous[:, None], axis=1)


__all__ = [
    "EPSILON", "ThermalPriorReceiptV44", "derive_last_observed_regime",
    "derive_thermal_regimes", "fit_thermal_prior", "thermal_transition_mask",
]
