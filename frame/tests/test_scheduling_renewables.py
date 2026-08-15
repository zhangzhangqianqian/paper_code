"""阶段10.4透明 PV/WT 与验证期预测选择测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.renewable_forecasts import fit_renewable_forecaster  # noqa: E402
from src.scheduling.renewables import pv_available, wt_available  # noqa: E402


class RenewableModelTest(unittest.TestCase):
    def setUp(self):
        self.parameters = {
            "pv_rated_capacity": 100.0,
            "pv_reference_irradiance": 1000.0,
            "pv_conversion_efficiency": 0.2,
            "pv_reference_temperature": 25.0,
            "pv_temperature_coefficient": -0.004,
            "wt_rated_capacity": 80.0,
            "wt_cut_in_speed": 3.0,
            "wt_rated_speed": 12.0,
            "wt_cut_out_speed": 25.0,
        }

    def test_pv_and_wt_physical_boundaries(self):
        weather = pd.DataFrame(
            {
                "solar_irradiance": [-20.0, 0.0, 1000.0, 1500.0],
                "temperature": [25.0, 25.0, 25.0, 25.0],
                "wind_speed": [0.0, 3.0, 12.0, 30.0],
            }
        )
        pv = pv_available(weather, self.parameters)
        wt = wt_available(weather, self.parameters)
        self.assertEqual(pv[0], 0.0)
        self.assertEqual(pv[1], 0.0)
        self.assertTrue(np.all((pv >= 0.0) & (pv <= 100.0)))
        self.assertEqual(wt[0], 0.0)
        self.assertEqual(wt[-1], 0.0)
        self.assertTrue(np.all((wt >= 0.0) & (wt <= 80.0)))

    def test_validation_selection_does_not_use_test_data(self):
        train_times = pd.date_range("2020-01-01", periods=192, freq="1h")
        validation_times = pd.date_range("2020-01-09", periods=96, freq="1h")
        train = pd.DataFrame(
            {
                "timestamp": train_times,
                "pv_available": np.arange(192, dtype=float),
                "wt_available": np.arange(192, dtype=float) * 0.5,
            }
        )
        validation = pd.DataFrame(
            {
                "timestamp": validation_times,
                "pv_available": np.arange(96, dtype=float),
                "wt_available": np.arange(96, dtype=float) * 0.5,
            }
        )
        forecaster = fit_renewable_forecaster(
            train, validation, {"methods": ["persistence", "seasonal_naive_24"], "validation_stride": 24}
        )
        self.assertIn(forecaster.selected_method, {"persistence", "seasonal_naive_24"})
        prediction = forecaster.predict(train.tail(168), horizon=4)
        self.assertEqual(prediction.shape, (4, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
