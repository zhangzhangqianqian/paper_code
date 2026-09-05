"""Target-derived thermal regimes and train-only magnitude statistics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import numpy as np


THERMAL_EPSILON = 1.0e-9


@dataclass(frozen=True)
class ThermalMagnitudeReceiptV43:
    mean: np.ndarray
    scale: np.ndarray
    active_count: np.ndarray
    class_count: np.ndarray
    target_sha256: str

    def __post_init__(self) -> None:
        for name in ("mean", "scale", "active_count"):
            value = np.asarray(getattr(self, name))
            if value.shape != (2,):
                raise ValueError(f"{name} must have shape [2]")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        if np.any(self.scale <= 0.0) or np.any(self.active_count <= 0.0):
            raise ValueError("active magnitude statistics must be positive")
        if np.asarray(self.class_count).shape != (3,) or not np.isfinite(self.class_count).all() or np.any(self.class_count <= 0.0):
            raise ValueError("class_count must contain three positive class counts")
        if not isinstance(self.target_sha256, str) or len(self.target_sha256) != 64:
            raise ValueError("target_sha256 must be a SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64).tolist(),
            "scale": np.asarray(self.scale, dtype=np.float64).tolist(),
            "active_count": np.asarray(self.active_count, dtype=np.int64).tolist(),
            "class_count": np.asarray(self.class_count, dtype=np.int64).tolist(),
            "target_sha256": self.target_sha256,
        }


def _validate_target(target: np.ndarray) -> np.ndarray:
    value = np.asarray(target, dtype=np.float64)
    if value.ndim != 3 or value.shape[-1] != 4:
        raise ValueError("target must have shape [N,horizon,4]")
    if not np.isfinite(value).all():
        raise ValueError("target must be finite")
    if np.any(value[..., 1:3] < 0.0):
        raise ValueError("cooling and heating targets must be non-negative")
    return value


def derive_thermal_regimes(target: np.ndarray, epsilon: float = THERMAL_EPSILON) -> np.ndarray:
    """Return 0=off, 1=cooling, 2=heating for every forecast horizon."""

    if not np.isfinite(float(epsilon)) or float(epsilon) < 0.0:
        raise ValueError("epsilon must be finite and non-negative")
    value = _validate_target(target)
    cooling = value[..., 1] > float(epsilon)
    heating = value[..., 2] > float(epsilon)
    if np.any(cooling & heating):
        raise ValueError("simultaneous cooling and heating targets are not representable")
    result = np.zeros(cooling.shape, dtype=np.int64)
    result[cooling] = 1
    result[heating] = 2
    return result


def fit_thermal_magnitude_statistics(
    train_target: np.ndarray,
    train_regimes: np.ndarray,
) -> ThermalMagnitudeReceiptV43:
    """Fit active cooling/heating mean and scale using training values only."""

    target = _validate_target(train_target)
    regimes = np.asarray(train_regimes)
    if regimes.shape != target.shape[:2] or not np.isin(regimes, (0, 1, 2)).all():
        raise ValueError("train_regimes must have shape [N,horizon] and values 0,1,2")
    values = []
    counts = []
    class_count = np.bincount(regimes.reshape(-1), minlength=3)
    for class_id, task_index in ((1, 1), (2, 2)):
        active = target[..., task_index][regimes == class_id]
        if active.size == 0:
            raise ValueError(f"thermal class {class_id} has no active training values")
        values.append(float(active.mean()))
        scale = float(active.std())
        counts.append(int(active.size))
        if scale <= 1.0e-12:
            scale = 1.0
        if not np.isfinite(scale):
            raise ValueError("active magnitude scale must be finite")
        if len(values) == 1:
            scales = [scale]
        else:
            scales.append(scale)
    raw = np.ascontiguousarray(target, dtype=np.float64).tobytes()
    return ThermalMagnitudeReceiptV43(
        mean=np.asarray(values, dtype=np.float64),
        scale=np.asarray(scales, dtype=np.float64),
        active_count=np.asarray(counts, dtype=np.int64),
        class_count=np.asarray(class_count, dtype=np.int64),
        target_sha256=hashlib.sha256(raw).hexdigest(),
    )


def thermal_transition_mask(
    future_regimes: np.ndarray,
    last_observed_regime: np.ndarray,
) -> np.ndarray:
    """Mark windows whose first four-hour path changes from its last state."""

    future = np.asarray(future_regimes)
    previous = np.asarray(last_observed_regime)
    if future.ndim != 2 or future.shape[1] != 4:
        raise ValueError("future_regimes must have shape [N,4]")
    if previous.shape != (future.shape[0],) or not np.isin(previous, (0, 1, 2)).all():
        raise ValueError("last_observed_regime must have shape [N] and values 0,1,2")
    return np.any(future != previous[:, None], axis=1)


__all__ = [
    "THERMAL_EPSILON", "ThermalMagnitudeReceiptV43", "derive_thermal_regimes",
    "fit_thermal_magnitude_statistics", "thermal_transition_mask",
]
