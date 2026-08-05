import unittest

import numpy as np

from src.transfer_analysis import (
    _benjamini_hochberg,
    _moving_block_indices,
    calculate_error_metrics,
    calculate_gain,
)


class TransferAnalysisTest(unittest.TestCase):
    def test_gain_sign_convention(self):
        self.assertAlmostEqual(calculate_gain(10.0, 8.0), 20.0)
        self.assertAlmostEqual(calculate_gain(10.0, 12.0), -20.0)
        self.assertTrue(np.isnan(calculate_gain(0.0, 1.0)))

    def test_error_metrics_are_on_original_scale(self):
        actual = np.ones((2, 4, 3), dtype=np.float32) * 10.0
        prediction = np.ones_like(actual) * 8.0
        metrics = calculate_error_metrics(actual, prediction)
        self.assertAlmostEqual(metrics["MAE"], 2.0)
        self.assertAlmostEqual(metrics["RMSE"], 2.0)
        self.assertAlmostEqual(metrics["WAPE"], 20.0)
        self.assertAlmostEqual(metrics["MAPE"], 20.0)

    def test_moving_block_bootstrap_preserves_local_successors(self):
        indices = _moving_block_indices(
            n=12,
            replicates=4,
            block_size=3,
            rng=np.random.default_rng(2026),
        )
        self.assertEqual(indices.shape, (4, 12))
        for row in indices:
            successors = (row[:-1] + 1) % 12 == row[1:]
            self.assertGreaterEqual(int(successors.sum()), 2)

    def test_moving_block_bootstrap_rejects_invalid_block_size(self):
        with self.assertRaises(ValueError):
            _moving_block_indices(12, 4, 0, np.random.default_rng(2026))

    def test_benjamini_hochberg_preserves_order_and_bounds(self):
        adjusted = _benjamini_hochberg([0.01, 0.04, 0.20])
        self.assertEqual(adjusted.shape, (3,))
        self.assertLessEqual(float(adjusted[0]), float(adjusted[1]))
        self.assertTrue(np.all((adjusted >= 0.0) & (adjusted <= 1.0)))


if __name__ == "__main__":
    unittest.main()
