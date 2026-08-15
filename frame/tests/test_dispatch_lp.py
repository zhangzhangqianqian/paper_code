from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp  # noqa: E402


PARAMETERS = {
    "grid_import_capacity": 100.0,
    "chp_electric_capacity": 20.0,
    "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0,
    "electric_chiller_capacity": 30.0,
    "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0,
    "bess_energy_capacity": 40.0,
    "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45,
    "gas_boiler_efficiency": 0.9,
    "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75,
    "bess_roundtrip_efficiency": 0.9,
    "bess_throughput_cost": 0.000001,
    "grid_energy_price": 1.0,
    "gas_energy_price": 0.6,
    "unserved_penalty": 100.0,
    "chp_ramp_fraction": 0.5,
}


class DispatchLPTest(unittest.TestCase):
    def test_feasible_window_has_small_residuals(self):
        inputs = DispatchInputs(
            demand=np.array([[10.0, 5.0, 4.0], [12.0, 5.0, 4.0], [10.0, 5.0, 4.0], [8.0, 4.0, 3.0]]),
            pv_available=np.array([2.0, 3.0, 1.0, 0.0]),
            wt_available=np.array([1.0, 1.0, 1.0, 1.0]),
            parameters=PARAMETERS,
        )
        result = solve_dispatch_lp(inputs)
        self.assertTrue(result.success, result.message)
        self.assertLessEqual(max(result.balance_residuals.values()), 1e-7)
        self.assertLessEqual(result.simultaneous_charge_discharge, 1e-7)
        self.assertTrue(np.all(result.values["grid"] >= -1e-9))

    def test_forecast_step_perturbation_changes_plan(self):
        base = DispatchInputs(np.full((4, 3), [20.0, 10.0, 8.0]), np.zeros(4), np.zeros(4), PARAMETERS)
        changed_demand = np.full((4, 3), [20.0, 10.0, 8.0])
        changed_demand[1, 0] = 40.0
        changed = DispatchInputs(changed_demand, np.zeros(4), np.zeros(4), PARAMETERS)
        first = solve_dispatch_lp(base)
        second = solve_dispatch_lp(changed)
        self.assertTrue(first.success and second.success)
        self.assertNotAlmostEqual(first.objective, second.objective)


if __name__ == "__main__":
    unittest.main(verbosity=2)
