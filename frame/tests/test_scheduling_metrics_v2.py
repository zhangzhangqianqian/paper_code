from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.metrics import summarize_real_replay, summarize_simulated_dispatch  # noqa: E402
from src.scheduling.statistics import benjamini_hochberg, paired_block_bootstrap  # noqa: E402


class SchedulingMetricsTest(unittest.TestCase):
    def test_r_and_s_summaries_are_separate(self):
        r = pd.DataFrame({"model": ["a", "a"], "candidate": ["H1", "H1"], "seed": [1, 1], "protocol": ["full", "full"], "grid_mae_window": [1.0, 3.0], "gas_mae_window": [2.0, 4.0]})
        s = pd.DataFrame({"model": ["a", "a"], "scenario": ["core", "core"], "seed": [1, 1], "planned_cost": [10.0, 12.0], "realized_cost": [11.0, 13.0], "regret": [1.0, 1.0]})
        self.assertEqual(float(summarize_real_replay(r).iloc[0]["grid_mae_window"]), 2.0)
        self.assertEqual(float(summarize_simulated_dispatch(s).iloc[0][("planned_cost", "mean")]), 11.0)

    def test_bootstrap_and_bh(self):
        timestamps = pd.date_range("2021-01-01", periods=20, freq="12h")
        a = pd.DataFrame({"timestamp": timestamps, "value": np.arange(20, dtype=float)})
        b = pd.DataFrame({"timestamp": timestamps, "value": np.zeros(20)})
        result = paired_block_bootstrap(a, b, replicates=100, seed=7)
        self.assertEqual(result.replicates, 100)
        self.assertTrue(result.ci_low <= result.estimate <= result.ci_high)
        correction = benjamini_hochberg([0.001, 0.03, 0.8])
        self.assertTrue(bool(correction["reject"][0]))
        self.assertFalse(bool(correction["reject"][2]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
