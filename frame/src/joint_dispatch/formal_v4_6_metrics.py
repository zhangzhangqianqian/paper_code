"""Stable nominal-forecast and risk-adjustment metrics for formal-v4.6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


TASKS = ("electricity", "cooling", "heating", "gas")
THERMAL_TASKS = ("cooling", "heating")


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def normalized_inactive_leakage_v46(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, float]:
    prediction = _array(prediction, "prediction")
    target = _array(target, "target")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (4, 4):
        raise ValueError("prediction and target must have shape [N,4,4]")
    result: dict[str, float] = {}
    for index, name in ((1, "cooling"), (2, "heating")):
        inactive = np.abs(prediction[..., index][target[..., index] <= 1.0e-9]).sum()
        active = np.abs(target[..., index][target[..., index] > 1.0e-9]).sum()
        result[name] = float(inactive / max(float(active), 1.0e-12))
    return result


@dataclass(frozen=True)
class ForecastMetricsV46:
    mae: Mapping[str, float]
    rmse: Mapping[str, float]
    wape: Mapping[str, float]
    macro_f1: float
    transition_balanced_accuracy: float
    inactive_leakage: Mapping[str, float]


@dataclass(frozen=True)
class RiskMetricsV46:
    adjustment_mean: Mapping[str, tuple[float, ...]]
    adjustment_p95: Mapping[str, tuple[float, ...]]
    regime_adjustment_mean: Mapping[str, tuple[float, ...]]
    regime_adjustment_p95: Mapping[str, tuple[float, ...]]
    cap_utilization_mean: float
    cap_utilization_p95: float
    cap_utilization_max: float
    maximum_cap_violation: float
    gas_adjustment: float = 0.0


def _regimes(target: np.ndarray) -> np.ndarray:
    cooling = target[..., 1] > 1.0e-9
    heating = target[..., 2] > 1.0e-9
    if np.any(cooling & heating):
        raise ValueError("simultaneous cooling and heating targets are not representable")
    return np.where(cooling, 1, np.where(heating, 2, 0)).astype(np.int64)


def forecast_metrics_v46(prediction: np.ndarray, target: np.ndarray, regime_probabilities: np.ndarray | None = None) -> ForecastMetricsV46:
    prediction = _array(prediction, "prediction")
    target = _array(target, "target")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (4, 4):
        raise ValueError("prediction and target must have shape [N,4,4]")
    mae = {name: float(np.abs(prediction[..., i] - target[..., i]).mean()) for i, name in enumerate(TASKS)}
    rmse = {name: float(np.sqrt(np.square(prediction[..., i] - target[..., i]).mean())) for i, name in enumerate(TASKS)}
    wape = {name: float(np.abs(prediction[..., i] - target[..., i]).sum() / max(float(np.abs(target[..., i]).sum()), 1.0e-12)) for i, name in enumerate(TASKS)}
    true = _regimes(target)
    if regime_probabilities is None:
        predicted = np.where(prediction[..., 1] > 1.0e-9, 1, np.where(prediction[..., 2] > 1.0e-9, 2, 0))
    else:
        probs = _array(regime_probabilities, "regime_probabilities")
        if probs.shape != (prediction.shape[0], 4, 3):
            raise ValueError("regime_probabilities must have shape [N,4,3]")
        predicted = probs.argmax(axis=-1)
    f1_values: list[float] = []
    for label in (0, 1, 2):
        tp = float(np.sum((predicted == label) & (true == label)))
        fp = float(np.sum((predicted == label) & (true != label)))
        fn = float(np.sum((predicted != label) & (true == label)))
        f1_values.append(0.0 if 2.0 * tp + fp + fn <= 0.0 else 2.0 * tp / (2.0 * tp + fp + fn))
    macro_f1 = float(np.mean(f1_values))
    transition_true = true[:, 1:] != true[:, :-1]
    transition_pred = predicted[:, 1:] != predicted[:, :-1]
    balanced_parts = []
    for label in (False, True):
        positives = transition_true == label
        balanced_parts.append(float(np.sum(transition_pred[positives] == label) / max(int(positives.sum()), 1)))
    transition_balanced_accuracy = float(np.mean(balanced_parts))
    return ForecastMetricsV46(mae, rmse, wape, macro_f1, transition_balanced_accuracy, normalized_inactive_leakage_v46(prediction, target))


def risk_metrics_v46(
    risk_adjustment: np.ndarray,
    risk_cap: np.ndarray,
    target: np.ndarray,
    regime_probabilities: np.ndarray | None = None,
) -> RiskMetricsV46:
    adjustment = _array(risk_adjustment, "risk_adjustment")
    cap = _array(risk_cap, "risk_cap")
    target = _array(target, "target")
    if adjustment.ndim != 3 or adjustment.shape[1:] != (4, 3) or cap.shape != adjustment.shape:
        raise ValueError("risk arrays must have shape [N,4,3]")
    if target.shape != (adjustment.shape[0], 4, 4) or np.any(adjustment < -1.0e-8) or np.any(cap <= 0.0):
        raise ValueError("risk arrays and target dimensions are invalid")
    fraction = adjustment / np.maximum(cap, 1.0e-12)
    regime = _regimes(target)
    names = ("electricity", "cooling", "heating")
    mean = {name: tuple(float(value) for value in adjustment[..., i].mean(axis=0)) for i, name in enumerate(names)}
    p95 = {name: tuple(float(value) for value in np.quantile(adjustment[..., i], 0.95, axis=0)) for i, name in enumerate(names)}
    regime_mean: dict[str, tuple[float, ...]] = {}
    regime_p95: dict[str, tuple[float, ...]] = {}
    for label, name in ((0, "off"), (1, "cooling"), (2, "heating")):
        means: list[float] = []
        p95s: list[float] = []
        for horizon in range(4):
            values = adjustment[:, horizon, :][regime[:, horizon] == label]
            if values.size == 0:
                values = np.zeros((1, 3), dtype=np.float64)
            means.extend(float(value) for value in values.mean(axis=0))
            p95s.extend(float(value) for value in np.quantile(values, 0.95, axis=0))
        regime_mean[name] = tuple(means)
        regime_p95[name] = tuple(p95s)
    maximum_violation = float(np.maximum(adjustment - cap, 0.0).max())
    return RiskMetricsV46(
        mean, p95, regime_mean, regime_p95,
        float(fraction.mean()), float(np.quantile(fraction, 0.95)), float(fraction.max()), maximum_violation, 0.0,
    )


__all__ = ["ForecastMetricsV46", "RiskMetricsV46", "forecast_metrics_v46", "normalized_inactive_leakage_v46", "risk_metrics_v46"]
