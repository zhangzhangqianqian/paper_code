from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.data import PlanningInformation, SchedulingFrame, make_plan_view  # noqa: E402


class SchedulingLeakageTest(unittest.TestCase):
    def test_plan_view_excludes_actual_columns(self):
        times = pd.date_range("2021-01-01", periods=30, freq="1h")
        data = pd.DataFrame(
            {
                "timestamp": times,
                "electricity": np.ones(30),
                "cooling": np.ones(30),
                "heating": np.ones(30),
                "gas": np.ones(30),
                "temperature": np.ones(30),
                "humidity": np.ones(30),
                "wind_speed": np.ones(30),
                "wind_direction": np.ones(30),
                "irradiance": np.ones(30),
                "actual_grid_import": np.ones(30),
                "actual_pv": np.ones(30),
            }
        )
        # The production frame uses the canonical exogenous columns. For this
        # unit test, patch the canonical names expected by the view.
        data["solar_irradiance"] = data.pop("irradiance")
        data["hour_sin"] = 0.0
        data["hour_cos"] = 1.0
        data["dow_sin"] = 0.0
        data["dow_cos"] = 1.0
        data["month_sin"] = 0.0
        data["month_cos"] = 1.0
        data["is_weekend"] = 0
        data["wind_direction"] = 0.0
        frame = SchedulingFrame(data=data, metadata={}, years=(2021,))
        plan = make_plan_view(frame, times[24], lookback=24, horizon=4)
        self.assertNotIn("actual_grid_import", plan.history.columns)
        self.assertNotIn("actual_pv", plan.history.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
