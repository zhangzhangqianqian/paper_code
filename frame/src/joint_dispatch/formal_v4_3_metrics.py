"""Full-year and regime-aware forecast metrics for formal-v4.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _validate_arrays(prediction: np.ndarray, target: np.ndarray, probabilities: np.ndarray, regimes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    prob = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(regimes, dtype=np.int64)
    if pred.ndim != 3 or tuple(pred.shape[1:]) != (4, 4) or truth.shape != pred.shape:
        raise ValueError("prediction and target must have shape [N,4,4]")
    if prob.shape != (pred.shape[0], 4, 3) or labels.shape != (pred.shape[0], 4):
        raise ValueError("regime arrays have invalid shape")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or not np.isfinite(prob).all():
        raise ValueError("metric arrays must be finite")
    if not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("regime labels must be 0, 1, or 2")
    return pred, truth, prob, labels


def _wape(pred: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.abs(target).sum())
    return float("nan") if denominator <= 0.0 else float(np.abs(pred - target).sum() / denominator)


def _regime_scores(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for truth, pred in zip(target.reshape(-1), predicted.reshape(-1)):
        confusion[int(truth), int(pred)] += 1
    true_positive = np.diag(confusion).astype(np.float64)
    precision = true_positive / confusion.sum(axis=0).clip(min=1)
    recall = true_positive / confusion.sum(axis=1).clip(min=1)
    f1 = 2.0 * precision * recall / (precision + recall).clip(min=1.0e-12)
    return {
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "f1_by_class": f1.tolist(),
        "recall_by_class": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


@dataclass(frozen=True)
class ForecastMetricsV43:
    all_hour: dict[str, Any]
    active_only: dict[str, Any]
    inactive_leakage: dict[str, Any]
    regime: dict[str, Any]
    transition_by_horizon: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_hour": self.all_hour,
            "active_only": self.active_only,
            "inactive_leakage": self.inactive_leakage,
            "regime": self.regime,
            "transition_by_horizon": self.transition_by_horizon,
        }


def compute_forecast_metrics_v43(
    prediction: np.ndarray,
    target: np.ndarray,
    regime_probability: np.ndarray,
    target_regime: np.ndarray,
    target_times: np.ndarray | None = None,
) -> ForecastMetricsV43:
    pred, truth, probability, labels = _validate_arrays(prediction, target, regime_probability, target_regime)
    if target_times is not None:
        times = np.asarray(target_times, dtype="datetime64[ns]")
        if times.shape != (pred.shape[0], 4):
            raise ValueError("target_times must have shape [N,4]")
        if np.any(np.diff(times.astype("int64"), axis=1) <= 0):
            raise ValueError("target_times must be strictly increasing within each window")

    errors = np.abs(pred - truth)
    all_hour = {
        "mae": errors.mean(axis=(0, 1)).tolist(),
        "rmse": np.sqrt(np.square(pred - truth).mean(axis=(0, 1))).tolist(),
        "wape": [_wape(pred[..., task], truth[..., task]) for task in range(4)],
    }
    active_only: dict[str, Any] = {}
    for name, task_index, class_id in (("cooling", 1, 1), ("heating", 2, 2)):
        mask = labels == class_id
        actual = truth[..., task_index][mask]
        estimate = pred[..., task_index][mask]
        active_only[f"{name}_count"] = int(mask.sum())
        active_only[f"{name}_mae"] = float(np.abs(estimate - actual).mean()) if actual.size else float("nan")
        active_only[f"{name}_wape"] = _wape(estimate, actual) if actual.size else float("nan")
    inactive_cooling = labels != 1
    inactive_heating = labels != 2
    inactive_leakage = {
        "cooling_mae": float(np.abs(pred[..., 1][inactive_cooling]).mean()) if inactive_cooling.any() else float("nan"),
        "heating_mae": float(np.abs(pred[..., 2][inactive_heating]).mean()) if inactive_heating.any() else float("nan"),
        "cooling_total": float(np.abs(pred[..., 1][inactive_cooling]).sum()),
        "heating_total": float(np.abs(pred[..., 2][inactive_heating]).sum()),
        "cooling_p95": float(np.percentile(np.abs(pred[..., 1][inactive_cooling]), 95)) if inactive_cooling.any() else float("nan"),
        "heating_p95": float(np.percentile(np.abs(pred[..., 2][inactive_heating]), 95)) if inactive_heating.any() else float("nan"),
    }
    predicted_regime = probability.argmax(axis=-1)
    regime = _regime_scores(predicted_regime, labels)
    transition_by_horizon: dict[str, Any] = {}
    previous = labels[:, 0]
    for horizon in range(4):
        if horizon > 0:
            previous = labels[:, horizon - 1]
        changed = labels[:, horizon] != previous
        transition_by_horizon[f"t+{horizon + 1}"] = {
            "transition_count": int(changed.sum()),
            "regime_accuracy": float((predicted_regime[:, horizon] == labels[:, horizon]).mean()),
            "transition_accuracy": float((predicted_regime[:, horizon][changed] == labels[:, horizon][changed]).mean()) if changed.any() else float("nan"),
        }
    return ForecastMetricsV43(all_hour, active_only, inactive_leakage, regime, transition_by_horizon)


__all__ = ["ForecastMetricsV43", "compute_forecast_metrics_v43"]
