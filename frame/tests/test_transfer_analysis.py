import unittest

import numpy as np

from src.transfer_analysis import calculate_error_metrics, calculate_gain


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


if __name__ == "__main__":
    unittest.main()
