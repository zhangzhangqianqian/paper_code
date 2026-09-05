import numpy as np
import pytest

from src.joint_dispatch.closed_loop_metrics import (
    physical_feasible,
    recourse_distance,
    regime_aware_forecast_metrics,
    shortage_free,
    terminal_stock_adjustment,
)


def test_physical_feasibility_and_shortage_are_independent():
    assert physical_feasible(np.zeros(8)) is True
    assert shortage_free(np.array([0.0, 0.0, 2.0])) is False
    assert physical_feasible(np.array([0.0, 0.0, 2.0e-6])) is False


def test_recourse_distance_uses_realized_three_carrier_demand():
    planned = np.zeros(21)
    settled = np.ones(21)
    absolute, normalized, by_channel = recourse_distance(planned, settled, np.array([10.0, 20.0, 30.0]))
    assert absolute == 9.0
    assert np.isclose(normalized, 9.0 / 60.0)
    assert by_channel.shape == (9,)


def test_regime_aware_metrics_use_frozen_training_scales():
    target = np.array([[[1.0, 0.0, 2.0, 1.0], [1.0, 2.0, 0.0, 1.0]]])
    prediction = target.copy()
    prediction[0, 0, 1] = 0.5
    prediction[0, 1, 2] = 0.25
    metrics = regime_aware_forecast_metrics(prediction, target, {"cooling": 2.0, "heating": 2.0})
    assert metrics["thermal"]["cooling"]["active"]["count"] == 1
    assert np.isclose(metrics["thermal"]["cooling"]["inactive_leakage"]["mean"], 0.25)
    assert metrics["ordinary"]["electricity"]["by_horizon"][0]["mae"] == 0.0


@pytest.mark.parametrize(
    ("initial", "final", "expected"),
    [(0.5, 0.25, 0.25 / np.sqrt(0.9)), (0.5, 0.75, -(0.25 * np.sqrt(0.9))), (0.5, 0.5, 0.0)],
)
def test_terminal_stock_adjustment_directions(initial, final, expected):
    assert np.isclose(terminal_stock_adjustment(initial, final, 1.0, 0.9, 1.0, 0.0, 0.5), expected)


def test_terminal_stock_adjustment_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        terminal_stock_adjustment(0.5, 0.5, -1.0, 0.9, 1.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        terminal_stock_adjustment(0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.5)
