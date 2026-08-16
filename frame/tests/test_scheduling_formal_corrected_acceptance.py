from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_scheduling_formal_corrected import validate_run_table  # noqa: E402


class CorrectedAcceptanceTest(unittest.TestCase):
    def test_simulated_table_checks_first_step_and_oracle_identities(self):
        frame = pd.DataFrame(
            {
                "origin": ["2021-01-01 00:00:00"],
                "horizon_objective": [10.0],
                "planned_cost_first_step": [2.0],
                "realized_cost": [3.0],
                "planning_deviation_cost": [1.0],
                "planned_carbon_first_step": [4.0],
                "realized_carbon_first_step": [5.0],
                "planned_curtailment_first_step": [0.0],
                "realized_curtailment_first_step": [0.0],
                "regret": [0.5],
                "oracle_realized_cost": [2.5],
                "oracle_realized_carbon": [4.5],
                "oracle_unserved_electricity": [0.0],
                "oracle_unserved_cooling": [0.0],
                "oracle_unserved_heating": [0.0],
                "oracle_soc": [20.0],
            }
        )
        report = validate_run_table(frame, "simulated_dispatch", {"2021-01-01 00:00:00"})
        self.assertTrue(report["finite"])

    def test_invalid_regret_identity_is_rejected(self):
        frame = pd.DataFrame(
            {
                "origin": ["2021-01-01 00:00:00"],
                "horizon_objective": [10.0], "planned_cost_first_step": [2.0], "realized_cost": [3.0],
                "planning_deviation_cost": [1.0], "planned_carbon_first_step": [4.0], "realized_carbon_first_step": [5.0],
                "planned_curtailment_first_step": [0.0], "realized_curtailment_first_step": [0.0], "regret": [9.0],
                "oracle_realized_cost": [2.5], "oracle_realized_carbon": [4.5],
                "oracle_unserved_electricity": [0.0], "oracle_unserved_cooling": [0.0], "oracle_unserved_heating": [0.0], "oracle_soc": [20.0],
            }
        )
        with self.assertRaises(ValueError):
            validate_run_table(frame, "simulated_dispatch", {"2021-01-01 00:00:00"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
