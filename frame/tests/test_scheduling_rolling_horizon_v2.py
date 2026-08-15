from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.dispatch_lp import DispatchInputs  # noqa: E402
from src.scheduling.rolling_horizon import ActualStream, RollingForecastSet, run_rolling_dispatch  # noqa: E402
from frame.tests.test_dispatch_lp import PARAMETERS  # noqa: E402


class RollingHorizonTest(unittest.TestCase):
    def test_nine_origins_have_continuous_soc_and_single_settlement(self):
        n = 9
        origins = np.arange("2021-01-01T00", "2021-01-01T09", dtype="datetime64[h]")
        demand = np.full((n, 4, 3), [12.0, 5.0, 4.0])
        zeros = np.zeros((n, 4))
        actual = ActualStream(origins, np.full(n, 12.0), np.full(n, 5.0), np.full(n, 4.0), np.zeros(n), np.zeros(n))
        result = run_rolling_dispatch(RollingForecastSet(origins, demand, zeros, zeros), actual, PARAMETERS)
        self.assertEqual(len(result.rows), n)
        self.assertEqual(len({row["origin"] for row in result.rows}), n)
        self.assertTrue(all(row["solver_status"] == "optimal" for row in result.rows))
        self.assertTrue(np.isfinite(result.final_soc))

    def test_future_prediction_changes_first_plan_in_coupled_case(self):
        origins = np.array(["2021-01-01T00"], dtype="datetime64[h]")
        base = np.full((1, 4, 3), [20.0, 5.0, 4.0])
        changed = base.copy()
        changed[0, 1, 0] = 40.0
        zeros = np.zeros((1, 4))
        actual = ActualStream(origins, np.array([20.0]), np.array([5.0]), np.array([4.0]), np.zeros(1), np.zeros(1))
        a = run_rolling_dispatch(RollingForecastSet(origins, base, zeros, zeros), actual, PARAMETERS)
        b = run_rolling_dispatch(RollingForecastSet(origins, changed, zeros, zeros), actual, PARAMETERS)
        self.assertNotEqual(a.rows[0]["planned_cost"], b.rows[0]["planned_cost"])

    def test_checkpoint_can_resume_completed_prefix(self):
        n = 2
        origins = np.arange("2021-01-01T00", "2021-01-01T02", dtype="datetime64[h]")
        demand = np.full((n, 4, 3), [12.0, 5.0, 4.0])
        zeros = np.zeros((n, 4))
        actual = ActualStream(origins, np.full(n, 12.0), np.full(n, 5.0), np.full(n, 4.0), np.zeros(n), np.zeros(n))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "rolling.json"
            first = run_rolling_dispatch(RollingForecastSet(origins[:1], demand[:1], zeros[:1], zeros[:1]), actual_stream=ActualStream(origins[:1], actual.electricity[:1], actual.cooling[:1], actual.heating[:1], actual.pv_available[:1], actual.wt_available[:1]), parameters=PARAMETERS, checkpoint_path=checkpoint)
            self.assertEqual(len(first.rows), 1)
            resumed = run_rolling_dispatch(RollingForecastSet(origins, demand, zeros, zeros), actual, PARAMETERS, checkpoint_path=checkpoint)
            self.assertEqual(len(resumed.rows), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
