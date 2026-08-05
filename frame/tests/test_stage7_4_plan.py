"""Contract tests for the Stage 7.4 formal ablation runner."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stage7_4.py"


class Stage74PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(SCRIPT), run_name="stage7_4_test_module")

    def test_frozen_plan_has_50_runs(self):
        plan = self.module["build_ablation_run_plan"]()
        self.assertEqual(len(plan), 50)
        self.assertEqual(
            {row["model"] for row in plan}, {"A0", "A1", "A2", "A3", "A4"}
        )
        self.assertEqual(
            {row["protocol"] for row in plan}, {"full", "small_sample"}
        )
        self.assertEqual(
            {row["seed"] for row in plan}, {2026, 2027, 2028, 2029, 2030}
        )
        self.assertTrue(all(row["candidate_id"] == "legacy" for row in plan))

    def test_a4_can_be_reused_without_training_duplicate(self):
        plan = self.module["build_ablation_run_plan"](include_a4=False, candidate_id="H2")
        self.assertEqual(len(plan), 40)
        self.assertNotIn("A4", {row["model"] for row in plan})
        self.assertTrue(all(row["candidate_id"] == "H2" for row in plan))

    def test_non_frozen_seed_list_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module["build_ablation_run_plan"]((2026,))


if __name__ == "__main__":
    unittest.main()
