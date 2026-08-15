from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.benchmark_parameters import derive_benchmark_parameters


LEDGER = {
    "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45,
    "gas_boiler_efficiency": 0.9,
    "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75,
    "bess_roundtrip_efficiency": 0.9,
    "grid_energy_price": 1.0,
    "gas_energy_price": 0.6,
    "unserved_penalty": 100.0,
    "chp_ramp_fraction": 0.5,
    "pv_rated_capacity": 0.15,
    "wt_rated_capacity": 0.10,
}


class BenchmarkParametersTest(unittest.TestCase):
    def test_capacity_rules_use_only_passed_training_frame(self):
        train = pd.DataFrame(
            {
                "electricity": [10.0, 20.0, 30.0, 40.0],
                "cooling": [4.0, 5.0, 6.0, 7.0],
                "heating": [3.0, 4.0, 5.0, 6.0],
                "pv_profile": [0.0, 0.5, 1.0, 0.5],
                "wt_profile": [0.2, 0.4, 0.3, 0.1],
            }
        )
        base = derive_benchmark_parameters(train, LEDGER)
        contaminated = pd.concat(
            [train, pd.DataFrame({"electricity": [1e9], "cooling": [1e9], "heating": [1e9], "pv_profile": [1.0], "wt_profile": [1.0]})],
            ignore_index=True,
        )
        changed = derive_benchmark_parameters(contaminated.iloc[: len(train)], LEDGER)
        self.assertEqual(base.values, changed.values)
        self.assertAlmostEqual(base.values["chp_electric_capacity"], 0.35 * np.percentile(train["electricity"], 95))

    def test_efficiencies_and_capacities_are_valid(self):
        train = pd.DataFrame({"electricity": [10.0, 20.0], "cooling": [4.0, 8.0], "heating": [3.0, 6.0]})
        result = derive_benchmark_parameters(train, LEDGER)
        self.assertGreater(result.values["bess_energy_capacity"], result.values["bess_power_capacity"])
        for key in ("chp_electric_efficiency", "chp_heat_efficiency", "gas_boiler_efficiency", "bess_roundtrip_efficiency"):
            self.assertGreater(result.values[key], 0.0)
            self.assertLessEqual(result.values[key], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
