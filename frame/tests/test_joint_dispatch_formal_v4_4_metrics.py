from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_4_metrics import compute_forecast_metrics_v44


def _fixture():
    target = np.zeros((2, 3, 4), dtype=float)
    target[..., 0] = 10.0; target[..., 3] = 4.0
    target[0, 1:, 1] = 10.0; target[1, (0, 2), 2] = 10.0
    regimes = np.array([[0, 1, 1], [2, 0, 2]])
    prediction = target.copy(); prediction[..., 1] = 2.0; prediction[..., 2] = 2.0
    prediction[0, 1:, 1] = 11.0; prediction[1, (0, 2), 2] = 11.0
    probability = np.full((2, 3, 3), 0.05); prior = probability.copy()
    for index, value in np.ndenumerate(regimes): probability[index + (int(value),)] = 0.90
    prior[:] = 0.05
    prior[..., 0] = 0.90
    return prediction, target, probability, prior, regimes


def test_inactive_leakage_and_active_wape_are_not_zero_dominated() -> None:
    prediction, target, probability, prior, regimes = _fixture()
    metrics = compute_forecast_metrics_v44(prediction, target, probability, prior, regimes, np.arange(6))
    assert metrics.inactive_leakage["cooling"] == pytest.approx(2.0)
    assert metrics.active_only["cooling"]["wape"] == pytest.approx(0.10)


def test_transition_accuracy_compares_with_prior() -> None:
    prediction, target, probability, prior, regimes = _fixture()
    metrics = compute_forecast_metrics_v44(prediction, target, probability, prior, regimes, np.arange(6))
    assert metrics.transition["balanced_accuracy_gain"] > 0.0
    assert metrics.transition["count"] == 3.0
