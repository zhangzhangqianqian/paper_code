from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.evaluation import (
    benjamini_hochberg,
    dispatch_mae_by_variable,
    mae_by_task_horizon,
    paired_moving_block_bootstrap,
    realized_dispatch_summary,
    rmse_by_task_horizon,
    wape_by_task_horizon,
)


def test_forecast_metrics_are_task_horizon_aware():
    target = np.ones((2, 2, 4))
    prediction = target + np.array([[[1.0, 2.0, 0.0, 4.0], [1.0, 2.0, 0.0, 4.0]]])
    np.testing.assert_allclose(mae_by_task_horizon(prediction, target)[0], [1.0, 2.0, 0.0, 4.0])
    np.testing.assert_allclose(rmse_by_task_horizon(prediction, target)[1], [1.0, 2.0, 0.0, 4.0])
    np.testing.assert_allclose(wape_by_task_horizon(prediction, target)[0], [1.0, 2.0, 0.0, 4.0])


def test_dispatch_summary_and_variable_mae():
    dispatch = np.zeros((1, 1, 21))
    dispatch[0, 0, 0] = 3.0
    demand = np.array([[[3.0, 0.0, 0.0]]])
    summary = realized_dispatch_summary(dispatch, demand)
    assert summary["shortage"] == 0.0
    assert summary["operating_cost"] == 3.0
    assert dispatch_mae_by_variable(dispatch, dispatch + 1.0).shape == (21,)


def test_moving_block_bootstrap_preserves_pairing_and_rejects_invalid_block():
    first = np.arange(336, dtype=float).reshape(2, 168)
    second = first - 1.0
    result = paired_moving_block_bootstrap(first, second, block_hours=168, replicates=25, seed=1)
    assert result["observed_difference"] == 1.0
    assert result["dependence_model"] == "paired_contiguous_moving_block"
    with pytest.raises(ValueError, match="invalid moving-block"):
        paired_moving_block_bootstrap(first, second, block_hours=0)


def test_benjamini_hochberg_returns_raw_and_adjusted_values():
    result = benjamini_hochberg([0.001, 0.02, 0.9])
    assert result["raw_p"].shape == (3,)
    assert result["fdr_adjusted_p"][0] <= result["fdr_adjusted_p"][1]
    assert bool(result["reject"][0])
