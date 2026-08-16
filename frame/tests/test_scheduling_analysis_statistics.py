from __future__ import annotations

import unittest
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.analysis_statistics import apply_bh, compare_models, sign_flip_p_value


class AnalysisStatisticsTest(unittest.TestCase):
    def _daily(self):
        rows = []
        for day in range(6):
            for model, offset in (("scheme2r", 0.0), ("baseline", 1.0)):
                for seed in (2026, 2027):
                    rows.append({"track": "simulated_dispatch", "scenario": "core", "date": f"2021-01-{day+1:02d}", "model": model, "seed": seed, "realized_cost": 10.0 + offset + day * 0.1})
        return pd.DataFrame(rows)

    def test_paired_comparison_is_day_level_and_reproducible(self):
        daily = self._daily()
        result = compare_models(daily, "scheme2r", "baseline", "simulated_dispatch", "core", "realized_cost", replicates=200, seed=2026)
        self.assertEqual(result["n_days"], 6)
        self.assertEqual(result["n_seeds"], 2)
        self.assertLess(result["paired_difference"], 0.0)
        self.assertEqual(result["p_value"], sign_flip_p_value([-1.0] * 6, 200, 2026))
        adjusted = apply_bh(pd.DataFrame([result]))
        self.assertIn("adjusted_p_value", adjusted.columns)

    def test_missing_day_is_not_silently_intersected(self):
        daily = self._daily()
        daily = daily[~((daily["model"] == "baseline") & (daily["date"] == "2021-01-06"))]
        with self.assertRaises(ValueError):
            compare_models(daily, "scheme2r", "baseline", "simulated_dispatch", "core", "realized_cost")


if __name__ == "__main__":
    unittest.main(verbosity=2)
