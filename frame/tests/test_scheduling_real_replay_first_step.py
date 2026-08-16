from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.real_replay import (  # noqa: E402
    EnergyNomination,
    settle_first_step_replay,
    settle_real_replay,
)


PRICES = {
    "grid_upward": 3.0,
    "grid_downward": 1.0,
    "gas_upward": 2.0,
    "gas_downward": 0.5,
}


class FirstStepReplayTest(unittest.TestCase):
    def test_window_diagnostics_can_differ_from_executed_settlement(self):
        nomination = EnergyNomination(
            grid=np.asarray([[8.0, 80.0, 80.0, 80.0]], dtype=float),
            gas=np.asarray([[5.0, 50.0, 50.0, 50.0]], dtype=float),
        )
        actual = {
            "actual_grid_import": np.asarray([[10.0, 100.0, 100.0, 100.0]], dtype=float),
            "gas": np.asarray([[4.0, 40.0, 40.0, 40.0]], dtype=float),
        }
        window = settle_real_replay(nomination, actual, PRICES)
        first = settle_first_step_replay(nomination, actual, PRICES)

        self.assertAlmostEqual(window.grid_upward_energy, 62.0)
        self.assertAlmostEqual(window.gas_downward_energy, 31.0)
        self.assertAlmostEqual(first.grid_upward_energy, 2.0)
        self.assertAlmostEqual(first.gas_downward_energy, 1.0)
        self.assertAlmostEqual(first.total_imbalance_cost, 6.5)

    def test_first_step_preserves_batch_alignment_and_origin_times(self):
        origins = np.asarray(["2021-01-01T00", "2021-01-01T01"], dtype="datetime64[ns]")
        nomination = EnergyNomination(
            grid=np.asarray([[8.0, 8.0], [9.0, 9.0]], dtype=float),
            gas=np.asarray([[5.0, 5.0], [6.0, 6.0]], dtype=float),
            origin_times=origins,
        )
        actual = {
            "actual_grid_import": np.asarray([[8.0, 8.0], [10.0, 10.0]], dtype=float),
            "gas": np.asarray([[5.0, 5.0], [5.0, 5.0]], dtype=float),
        }
        result = settle_first_step_replay(nomination, actual, PRICES)
        self.assertEqual(result.grid_error.shape, (2, 1))
        self.assertEqual(result.gas_error.shape, (2, 1))
        self.assertTrue(np.array_equal(result.grid_error[:, 0], np.asarray([0.0, 1.0])))
        self.assertTrue(np.array_equal(result.gas_error[:, 0], np.asarray([0.0, -1.0])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
