"""Contract tests for the Stage 7.6 aggregation runner."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stage7_6.py"


class Stage76PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(SCRIPT), run_name="stage7_6_test_module")

    def test_source_plan_tracks_trained_runs_and_reused_a4(self):
        plan = self.module["build_stage7_6_source_plan"]()
        self.assertEqual([item["stage"] for item in plan], ["7.3", "7.4", "7.5", "7R.STL"])
        self.assertEqual(sum(item["expected_runs"] for item in plan), 104)
        self.assertEqual(self.module["REUSED_A4_RUNS"], 10)
        self.assertEqual(self.module["EXPECTED_TOTAL_RUNS"], 114)


if __name__ == "__main__":
    unittest.main()
