"""阶段2非学习基线测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import (  # noqa: E402
    persistence_forecast,
    regression_metrics,
    seasonal_naive_forecast,
)


class BaselineTest(unittest.TestCase):
    def setUp(self):
        # 2个样本、24小时、3类负荷；数值递增便于手工核验预测结果。
        self.history = np.arange(2 * 24 * 3, dtype=np.float32).reshape(2, 24, 3)

    def test_persistence_repeats_last_observation(self):
        prediction = persistence_forecast(self.history, horizon=4)
        expected = np.repeat(self.history[:, -1:, :], repeats=4, axis=1)
        np.testing.assert_array_equal(prediction, expected)

    def test_seasonal_naive_uses_previous_daily_phase(self):
        prediction = seasonal_naive_forecast(
            self.history, horizon=4, season_length=24
        )
        expected = self.history[:, :4, :]
        np.testing.assert_array_equal(prediction, expected)

    def test_invalid_season_length_is_rejected(self):
        with self.assertRaises(ValueError):
            seasonal_naive_forecast(self.history[:, :12, :], season_length=24)

    def test_metrics_are_zero_for_perfect_forecast(self):
        metrics = regression_metrics(self.history[:, :4, :], self.history[:, :4, :])
        for task_values in metrics["per_task"].values():
            self.assertEqual(task_values["MAE"], 0.0)
            self.assertEqual(task_values["RMSE"], 0.0)
            self.assertEqual(task_values["WAPE"], 0.0)
            self.assertEqual(task_values["MAPE"], 0.0)

    def test_metrics_detect_constant_error(self):
        actual = np.ones((2, 4, 3), dtype=np.float32) * 10
        prediction = np.ones((2, 4, 3), dtype=np.float32) * 8
        metrics = regression_metrics(actual, prediction)
        self.assertAlmostEqual(metrics["overall_equal_task_mean"]["MAE"], 2.0)
        self.assertAlmostEqual(metrics["overall_equal_task_mean"]["RMSE"], 2.0)
        self.assertAlmostEqual(metrics["overall_equal_task_mean"]["WAPE"], 20.0)
        self.assertAlmostEqual(metrics["overall_equal_task_mean"]["MAPE"], 20.0)

    def test_metrics_accept_four_task_names(self):
        actual = np.ones((2, 4, 4), dtype=np.float32)
        prediction = np.zeros_like(actual)
        names = ("electricity", "cooling", "heating", "gas")
        metrics = regression_metrics(actual, prediction, task_names=names)
        self.assertEqual(tuple(metrics["per_task"]), names)
        self.assertEqual(metrics["task_count"], 4)
        with self.assertRaises(ValueError):
            regression_metrics(actual, prediction)


if __name__ == "__main__":
    unittest.main(verbosity=2)
