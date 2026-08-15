"""阶段10.6真实运行回放测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.imbalance_settlement import summarize_replay  # noqa: E402
from src.scheduling.real_replay import build_energy_nomination, settle_real_replay  # noqa: E402


class RealStationReplayTest(unittest.TestCase):
    def test_nomination_and_asymmetric_settlement(self):
        forecasts = np.zeros((1, 4, 4), dtype=float)
        forecasts[:, :, 0] = 10.0
        forecasts[:, :, 3] = 5.0
        pv = np.zeros((1, 4, 2), dtype=float)
        pv[:, :, 0] = 2.0
        nomination = build_energy_nomination(forecasts, pv)
        self.assertTrue(np.all(nomination.grid == 8.0))
        self.assertTrue(np.all(nomination.gas == 5.0))
        result = settle_real_replay(
            nomination,
            {"actual_grid_import": np.full((1, 4), 10.0), "gas": np.full((1, 4), 4.0)},
            {"grid_upward": 3.0, "grid_downward": 1.0, "gas_upward": 2.0, "gas_downward": 0.5},
        )
        self.assertAlmostEqual(result.grid_upward_energy, 8.0)
        self.assertAlmostEqual(result.gas_downward_energy, 4.0)
        # Grid upward: 8 * 3 = 24; gas downward: 4 * 0.5 = 2.
        self.assertAlmostEqual(result.total_imbalance_cost, 26.0)

    def test_oracle_nomination_has_zero_deviation_cost(self):
        forecasts = np.ones((2, 4, 4), dtype=float)
        pv = np.zeros((2, 4, 2), dtype=float)
        nomination = build_energy_nomination(forecasts, pv)
        result = settle_real_replay(
            nomination,
            {"actual_grid_import": nomination.grid, "gas": nomination.gas},
            {"grid_upward": 3.0, "grid_downward": 1.0, "gas_upward": 2.0, "gas_downward": 0.5},
        )
        self.assertEqual(result.total_imbalance_cost, 0.0)
        self.assertEqual(summarize_replay(result)["grid_mae"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
