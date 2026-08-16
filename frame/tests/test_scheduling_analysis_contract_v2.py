from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CONTRACT_PATH = PROJECT_ROOT / "configs" / "scheduling_analysis_contract_v2.json"


class SchedulingAnalysisContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_schema_and_frozen_run_matrix(self):
        self.assertEqual(self.contract["schema_version"], "scheduling-analysis-contract-v2")
        self.assertEqual(
            self.contract["models"],
            ["scheme2r", "dynamic_symmetric", "hard_share", "mmoe_lite", "ple_lite"],
        )
        self.assertEqual(self.contract["seeds"], [2026, 2027, 2028, 2029, 2030])
        self.assertEqual(len(self.contract["simulated_scenarios"]), 6)
        self.assertEqual(self.contract["expected_runs"]["total_model_runs"], 175)
        self.assertEqual(self.contract["expected_runs"]["origins_per_run"], 8757)

    def test_tracks_and_accounting_are_explicitly_separated(self):
        self.assertEqual(self.contract["tracks"], ["real_replay", "simulated_dispatch"])
        self.assertTrue(self.contract["tracks_must_remain_separate"])
        accounting = self.contract["accounting"]
        self.assertEqual(accounting["executed_forecast_step"], 0)
        self.assertEqual(accounting["r_energy_and_imbalance_settlement"], "executed_first_step_only")
        self.assertEqual(accounting["s_regret"], "realized_cost_minus_same_scenario_perfect_information_oracle_cost")
        self.assertFalse(accounting["horizon_objective_is_primary_cost"])
        self.assertFalse(accounting["real_currency_claim_allowed"])

    def test_statistics_use_complete_days_and_prespecified_correction(self):
        statistics = self.contract["statistics"]
        self.assertEqual(statistics["inference_unit"], "complete_calendar_day")
        self.assertEqual(statistics["complete_days_in_2021"], 364)
        self.assertEqual(statistics["partial_final_day_hours"], 21)
        self.assertFalse(statistics["partial_final_day_included_in_day_level_inference"])
        self.assertEqual(statistics["bootstrap"]["replicates"], 2000)
        self.assertEqual(statistics["bootstrap"]["seed"], 2026)
        self.assertEqual(statistics["multiple_comparison"], "benjamini_hochberg")
        self.assertEqual(statistics["alpha"], 0.05)

    def test_typical_day_rule_cannot_use_test_model_performance(self):
        typical_day = self.contract["typical_day"]
        self.assertFalse(typical_day["selection_may_use_model_results"])
        self.assertEqual(typical_day["trajectory_model"], "scheme2r")
        self.assertEqual(typical_day["trajectory_seed"], 2026)
        self.assertEqual(typical_day["trajectory_scenario"], "core")

    def test_outputs_and_feasibility_gate_are_frozen(self):
        self.assertEqual(self.contract["outputs"]["root"], "frame/reports/scheduling_v2/analysis")
        self.assertEqual(self.contract["feasibility_screen"]["max_balance_residual"], 1e-7)
        self.assertEqual(self.contract["feasibility_screen"]["unserved_hour_rate_review_threshold"], 0.01)
        self.assertEqual(self.contract["feasibility_screen"]["unserved_demand_fraction_review_threshold"], 0.001)
        self.assertEqual(self.contract["feasibility_screen"]["failure_action"], "capacity_feasibility_review_required_and_no_cost_ranking")


if __name__ == "__main__":
    unittest.main(verbosity=2)
