from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_3_metrics import compute_forecast_metrics_v43


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.zeros((1, 4, 4), dtype=np.float64)
    target[0, 0, 3] = 10.0
    target[0, 1, 1] = 100.0
    target[0, 2, 2] = 50.0
    target[0, 3, 3] = 10.0
    prediction = target.copy()
    prediction[0, 0, 1] = 50.0
    probability = np.zeros((1, 4, 3), dtype=np.float64)
    probability[..., 0] = 1.0
    probability[0, 1, 1] = 1.0; probability[0, 1, 0] = 0.0
    probability[0, 2, 2] = 1.0; probability[0, 2, 0] = 0.0
    regimes = np.asarray([[0, 1, 2, 0]], dtype=np.int64)
    return prediction, target, probability, regimes


def test_inactive_leakage_is_not_hidden_by_active_wape() -> None:
    prediction, target, probability, regimes = _arrays()
    metrics = compute_forecast_metrics_v43(prediction, target, probability, regimes)
    assert metrics.active_only["cooling_wape"] == pytest.approx(0.0)
    assert metrics.inactive_leakage["cooling_mae"] == pytest.approx(50.0 / 3.0)


def test_regime_confusion_matrix_counts_all_three_classes() -> None:
    prediction, target, probability, regimes = _arrays()
    metrics = compute_forecast_metrics_v43(prediction, target, probability, regimes)
    assert np.asarray(metrics.regime["confusion_matrix"]).shape == (3, 3)
    assert metrics.regime["macro_f1"] == pytest.approx(1.0)


def test_non_increasing_target_times_within_window_are_rejected() -> None:
    prediction, target, probability, regimes = _arrays()
    times = np.asarray([["2019-01-01T00", "2019-01-01T01", "2019-01-01T01", "2019-01-01T03"]])
    with pytest.raises(ValueError, match="strictly increasing"):
        compute_forecast_metrics_v43(prediction, target, probability, regimes, times)
