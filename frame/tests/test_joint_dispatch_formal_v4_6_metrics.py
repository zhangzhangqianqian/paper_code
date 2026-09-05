from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_6_metrics import normalized_inactive_leakage_v46, risk_metrics_v46


def test_normalized_inactive_leakage_uses_active_target_mass():
    prediction = np.array([[[0., 4., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
    target = np.array([[[0., 0., 0., 0.], [0., 8., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]])
    result = normalized_inactive_leakage_v46(prediction, target)
    assert result["cooling"] == pytest.approx(4.0 / 8.0)


def test_risk_metrics_reports_horizon_cells_and_zero_gas():
    adjustment = np.ones((2, 4, 3))
    cap = np.full((2, 4, 3), 2.0)
    target = np.zeros((2, 4, 4))
    target[:, :, 0] = 1.0
    metrics = risk_metrics_v46(adjustment, cap, target)
    assert len(metrics.adjustment_mean["electricity"]) == 4
    assert len(metrics.regime_adjustment_mean["off"]) == 12
    assert metrics.gas_adjustment == 0.0
    assert metrics.maximum_cap_violation == 0.0
