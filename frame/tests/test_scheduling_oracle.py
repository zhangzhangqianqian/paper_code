from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.oracle import build_oracle_forecast_set, run_perfect_information_oracle  # noqa: E402
from frame.tests.test_dispatch_lp import PARAMETERS  # noqa: E402


class SchedulingOracleTest(unittest.TestCase):
    def _frame(self, count: int = 8) -> pd.DataFrame:
        timestamps = pd.date_range("2021-01-01", periods=count, freq="1h")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "electricity": np.arange(count, dtype=float) + 10.0,
                "cooling": np.full(count, 5.0),
                "heating": np.full(count, 4.0),
            }
        )

    def test_oracle_uses_exact_future_windows_without_padding(self):
        frame = self._frame()
        origins = np.asarray(["2021-01-01T00", "2021-01-01T01"], dtype="datetime64[ns]")
        pv = pd.Series(np.arange(8, dtype=float), index=pd.to_datetime(frame["timestamp"]))
        wt = pd.Series(np.ones(8), index=pd.to_datetime(frame["timestamp"]))
        result = build_oracle_forecast_set(frame, origins, 4, pv, wt)
        self.assertEqual(result.demand.shape, (2, 4, 3))
        self.assertEqual(result.pv_available[0].tolist(), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(result.demand[1, :, 0].tolist(), [11.0, 12.0, 13.0, 14.0])
        with self.assertRaises(ValueError):
            build_oracle_forecast_set(frame, np.asarray(["2021-01-01T05"], dtype="datetime64[ns]"), 4, pv, wt)

    def test_oracle_runs_same_first_step_protocol(self):
        frame = self._frame()
        origins = np.asarray(["2021-01-01T00", "2021-01-01T01"], dtype="datetime64[ns]")
        timestamps = pd.to_datetime(frame["timestamp"])
        pv = pd.Series(np.ones(len(frame)), index=timestamps)
        wt = pd.Series(np.zeros(len(frame)), index=timestamps)
        result = run_perfect_information_oracle(frame, origins, pv, wt, PARAMETERS, horizon=4)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(len(result.realized_steps), 2)
        self.assertTrue(all(row["solver_status"] == "optimal" for row in result.rows))
        self.assertTrue(all(np.isfinite(row["realized_cost"]) for row in result.rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
