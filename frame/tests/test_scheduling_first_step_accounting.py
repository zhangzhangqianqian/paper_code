from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp  # noqa: E402
from src.scheduling.recourse import evaluate_planned_first_step, settle_first_step  # noqa: E402
from frame.tests.test_dispatch_lp import PARAMETERS  # noqa: E402


class FirstStepAccountingTest(unittest.TestCase):
    def test_plan_and_realized_values_share_the_first_hour_basis(self):
        parameters = dict(PARAMETERS, grid_emission_factor=0.4, gas_emission_factor=0.7, carbon_price=0.2)
        inputs = DispatchInputs(
            demand=np.array([[20.0, 8.0, 6.0], [35.0, 8.0, 6.0], [20.0, 8.0, 6.0], [20.0, 8.0, 6.0]]),
            pv_available=np.array([2.0, 2.0, 2.0, 2.0]),
            wt_available=np.zeros(4),
            parameters=parameters,
        )
        plan = solve_dispatch_lp(inputs)
        self.assertTrue(plan.success, plan.message)
        planned = evaluate_planned_first_step(plan, parameters)
        actual = {"electricity": 21.0, "cooling": 8.0, "heating": 6.0, "pv_available": 2.0, "wt_available": 0.0}
        realized = settle_first_step(plan, actual, parameters)
        self.assertNotAlmostEqual(planned["planned_cost_first_step"], plan.objective)
        expected_cost = (
            plan.values["grid"][0] * (parameters["grid_energy_price"] + parameters["carbon_price"] * parameters["grid_emission_factor"])
            + (plan.values["g_chp"][0] + plan.values["g_gb"][0]) * (parameters["gas_energy_price"] + parameters["carbon_price"] * parameters["gas_emission_factor"])
            + parameters["unserved_penalty"] * sum(plan.values[name][0] for name in ("slack_e", "slack_c", "slack_h"))
            + parameters["bess_throughput_cost"] * (plan.values["p_charge"][0] + plan.values["p_discharge"][0])
        )
        self.assertAlmostEqual(planned["planned_cost_first_step"], expected_cost, places=7)
        self.assertAlmostEqual(
            planned["planned_carbon_first_step"],
            plan.values["grid"][0] * parameters["grid_emission_factor"]
            + (plan.values["g_chp"][0] + plan.values["g_gb"][0]) * parameters["gas_emission_factor"],
            places=7,
        )
        self.assertAlmostEqual(planned["planned_curtailment_first_step"], plan.values["pv_curt"][0] + plan.values["wt_curt"][0], places=7)
        self.assertAlmostEqual(realized.actual_electricity, actual["electricity"])
        self.assertAlmostEqual(realized.soc_after_execution, plan.values["soc"][0])
        self.assertAlmostEqual(realized.chp_electricity, plan.values["p_chp"][0])

    def test_rolling_rows_expose_corrected_accounting_columns(self):
        from src.scheduling.rolling_horizon import ActualStream, RollingForecastSet, run_rolling_dispatch

        origins = np.array(["2021-01-01T00"], dtype="datetime64[h]")
        demand = np.full((1, 4, 3), [12.0, 5.0, 4.0])
        zeros = np.zeros((1, 4))
        actual = ActualStream(origins, np.array([12.0]), np.array([5.0]), np.array([4.0]), np.zeros(1), np.zeros(1))
        result = run_rolling_dispatch(RollingForecastSet(origins, demand, zeros, zeros), actual, PARAMETERS)
        row = result.rows[0]
        for key in ("horizon_objective", "planned_cost_first_step", "realized_cost", "planning_deviation_cost", "planned_carbon_first_step", "realized_carbon_first_step", "planned_curtailment_first_step", "realized_curtailment_first_step", "planned_grid_first_step", "planned_gas_first_step", "planned_chp_electricity", "planned_gas_boiler_heat"):
            self.assertIn(key, row)
        self.assertAlmostEqual(row["planning_deviation_cost"], row["realized_cost"] - row["planned_cost_first_step"], places=7)
        self.assertNotIn("regret", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
