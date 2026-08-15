"""阶段10.0双轨调度研究契约测试。"""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.contracts import (  # noqa: E402
    TRACK_ORDER,
    load_scheduling_contract,
    validate_scheduling_contract,
)


class SchedulingContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_scheduling_contract()

    def test_fixed_years_tasks_and_tracks(self):
        self.assertEqual(self.contract.train_years, (2015, 2016, 2017, 2018, 2019))
        self.assertEqual(self.contract.validation_year, 2020)
        self.assertEqual(self.contract.test_year, 2021)
        self.assertEqual(self.contract.history_hours, 24)
        self.assertEqual(self.contract.horizon_hours, 4)
        self.assertEqual(self.contract.task_order, ("electricity", "cooling", "heating", "gas"))
        self.assertEqual(self.contract.tracks, TRACK_ORDER)

    def test_gas_is_station_side_consumption(self):
        self.assertEqual(self.contract.gas_role, "station_side_device_consumption")
        self.assertFalse(self.contract.raw["gas"]["main_dispatch_balance"])
        self.assertFalse(self.contract.track_config("simulated_dispatch")["gas_demand_balance"])

    def test_rejects_gas_demand_balance(self):
        invalid = copy.deepcopy(self.contract.raw)
        invalid["track_configs"]["simulated_dispatch"]["gas_demand_balance"] = True
        with self.assertRaises(ValueError):
            validate_scheduling_contract(invalid)

    def test_rejects_future_actuals_and_shared_outputs(self):
        invalid = copy.deepcopy(self.contract.raw)
        invalid["leakage"]["test_year_used_for_scaling"] = True
        with self.assertRaises(ValueError):
            validate_scheduling_contract(invalid)

        invalid = copy.deepcopy(self.contract.raw)
        invalid["outputs"]["simulated_dispatch_root"] = invalid["outputs"]["real_replay_root"]
        with self.assertRaises(ValueError):
            validate_scheduling_contract(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
