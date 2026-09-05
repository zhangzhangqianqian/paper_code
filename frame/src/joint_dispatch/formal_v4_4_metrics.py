"""Chronology-preserving forecast and regime metrics for formal-v4.4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


TASKS = ("electricity", "cooling", "heating", "gas")


@dataclass(frozen=True)
class ForecastMetricsV44:
    task: Mapping[str, Mapping[str, float]]
    active_only: Mapping[str, Mapping[str, float]]
    inactive_leakage: Mapping[str, float]
    transition: Mapping[str, float]
    confusion: tuple[tuple[int, ...], ...]
    macro_f1: float
    balanced_accuracy: float
    prior_macro_f1: float
    four_task_score: float
    sample_count: int
    horizon: int


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with {ndim} dimensions")
    return result.astype(np.float64, copy=False)


def _wape(error: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.abs(target).sum())
    if denominator <= 0.0:
        return float("nan")
    return float(np.abs(error).sum() / denominator)


def _summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())), "wape": _wape(error, target)}


def _classification_metrics(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, tuple[tuple[int, ...], ...]]:
    confusion = np.zeros((3, 3), dtype=np.int64)
    for actual, estimate in zip(truth.reshape(-1), predicted.reshape(-1)):
        if 0 <= int(actual) <= 2 and 0 <= int(estimate) <= 2:
            confusion[int(actual), int(estimate)] += 1
    f1_values: list[float] = []; recalls: list[float] = []
    for index in range(3):
        tp = float(confusion[index, index]); fp = float(confusion[:, index].sum() - confusion[index, index]); fn = float(confusion[index, :].sum() - confusion[index, index])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if tp + fn: recalls.append(recall)
    return float(np.mean(f1_values)), float(np.mean(recalls)) if recalls else float("nan"), tuple(tuple(int(v) for v in row) for row in confusion)


def compute_forecast_metrics_v44(
    prediction: Any,
    target: Any,
    probability: Any,
    prior_probability: Any,
    regimes: Any,
    times: Any,
) -> ForecastMetricsV44:
    """Compute metrics without dropping inactive windows or changing chronology."""

    pred = _array(prediction, "prediction", 3); truth = _array(target, "target", 3)
    if pred.shape != truth.shape or tuple(pred.shape[2:]) != (4,):
        raise ValueError("prediction and target must both have shape [N,H,4]")
    prob = _array(probability, "probability", 3); prior = _array(prior_probability, "prior_probability", 3)
    if prob.shape != prior.shape or tuple(prob.shape[:2]) != tuple(pred.shape[:2]) or prob.shape[2] != 3:
        raise ValueError("probability arrays must have shape [N,H,3]")
    if regimes is None:
        regimes_array = np.where(truth[..., 1] > 1.0e-9, 1, np.where(truth[..., 2] > 1.0e-9, 2, 0)).astype(np.int64)
    else:
        regimes_array = np.asarray(regimes)
        if regimes_array.shape != pred.shape[:2] or not np.isfinite(regimes_array).all():
            raise ValueError("regimes must have shape [N,H]")
        regimes_array = regimes_array.astype(np.int64, copy=False)
    time_array = np.asarray(times)
    if time_array.shape not in {pred.shape[:2], (pred.shape[0],)} and time_array.size != pred.shape[0] * pred.shape[1]:
        raise ValueError("times must align with windows")
    predicted_regime = np.argmax(prob, axis=-1); prior_regime = np.argmax(prior, axis=-1)
    task_metrics = {name: _summary(pred[..., index], truth[..., index]) for index, name in enumerate(TASKS)}
    active_metrics: dict[str, dict[str, float]] = {}
    leakage: dict[str, float] = {}
    for index, name in ((1, "cooling"), (2, "heating")):
        active = regimes_array == index; inactive = ~active
        active_pred = pred[..., index][active]; active_true = truth[..., index][active]
        if active_pred.size == 0:
            active_metrics[name] = {"mae": float("nan"), "rmse": float("nan"), "wape": float("nan"), "count": 0.0}
        else:
            values = _summary(active_pred, active_true); values["count"] = float(active_pred.size); active_metrics[name] = values
        leakage_values = np.abs(pred[..., index][inactive])
        leakage[name] = float(leakage_values.mean()) if leakage_values.size else float("nan")
        leakage[f"{name}_p95"] = float(np.percentile(leakage_values, 95.0)) if leakage_values.size else float("nan")
        leakage[f"{name}_total"] = float(leakage_values.sum())
    macro_f1, balanced_accuracy, confusion = _classification_metrics(predicted_regime, regimes_array)
    prior_macro_f1, prior_balanced_accuracy, _ = _classification_metrics(prior_regime, regimes_array)
    transition_mask = np.zeros_like(regimes_array, dtype=bool); transition_mask[:, 1:] = regimes_array[:, 1:] != regimes_array[:, :-1]
    if transition_mask.any():
        transition_balanced, _transition_ba, _ = _classification_metrics(predicted_regime[transition_mask], regimes_array[transition_mask])
        prior_transition_balanced, _prior_transition_ba, _ = _classification_metrics(prior_regime[transition_mask], regimes_array[transition_mask])
        # The classification helper's second return is balanced accuracy; use
        # it for the transition criterion while retaining the F1 diagnostics.
        _, transition_ba, _ = _classification_metrics(predicted_regime[transition_mask], regimes_array[transition_mask])
        _, prior_transition_ba, _ = _classification_metrics(prior_regime[transition_mask], regimes_array[transition_mask])
    else:
        transition_balanced = macro_f1; prior_transition_balanced = prior_macro_f1; transition_ba = balanced_accuracy; prior_transition_ba = prior_balanced_accuracy
    transition = {"balanced_accuracy": float(transition_ba), "prior_balanced_accuracy": float(prior_transition_ba), "balanced_accuracy_gain": float(transition_ba - prior_transition_ba), "count": float(transition_mask.sum())}
    task_wapes = [task_metrics[name]["wape"] for name in TASKS]
    four_task = float(np.mean(task_wapes)) if np.isfinite(task_wapes).all() else float("nan")
    return ForecastMetricsV44(task_metrics, active_metrics, leakage, transition, confusion, macro_f1, balanced_accuracy, prior_macro_f1, four_task, pred.shape[0], pred.shape[1])


__all__ = ["ForecastMetricsV44", "compute_forecast_metrics_v44"]
