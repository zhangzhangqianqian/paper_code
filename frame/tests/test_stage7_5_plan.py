"""Contract tests for the Stage 7.5 external-baseline runner."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stage7_5.py"


class Stage75PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(SCRIPT), run_name="stage7_5_test_module")

    def test_frozen_plan_has_34_runs(self):
        plan = self.module["build_external_run_plan"]()
        self.assertEqual(len(plan), 34)
        deterministic = [row for row in plan if row["seed"] is None]
        learned = [row for row in plan if row["seed"] is not None]
        self.assertEqual(len(deterministic), 4)
        self.assertEqual(len(learned), 30)
        self.assertEqual(
            {row["model"] for row in deterministic},
            {"persistence", "seasonal_naive"},
        )
        self.assertEqual(
            {row["model"] for row in learned},
            {"dlinear", "mmoe-lite", "softs"},
        )
        self.assertEqual(
            {row["protocol"] for row in plan}, {"full", "small_sample"}
        )
        self.assertEqual(
            {row["seed"] for row in learned}, {2026, 2027, 2028, 2029, 2030}
        )

    def test_non_frozen_seed_list_is_rejected(self):
        with self.assertRaises(ValueError):
            self.module["build_external_run_plan"]((2026,))


if __name__ == "__main__":
    unittest.main()
