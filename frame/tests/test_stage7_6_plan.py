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

    def test_source_plan_tracks_current_non_scheme2r_freeze(self):
        freeze = {"primary_model": {"model": "stl_matched"}}
        plan = self.module["build_stage7_6_source_plan"](freeze)
        self.assertEqual([item["stage"] for item in plan], ["7.3", "7.4", "7.5", "7R.STL"])
        self.assertEqual(sum(item["expected_runs"] for item in plan), 124)
        self.assertEqual(self.module["stage7_6_expected_counts"](freeze)["reused_a4"], 0)
        self.assertEqual(
            sum(item["expected_runs"] for item in plan)
            + self.module["stage7_6_expected_counts"](freeze)["reused_a4"],
            124,
        )

    def test_non_scheme2r_freeze_trains_a4_instead_of_reusing(self):
        freeze = {"primary_model": {"model": "dynamic_directed"}}
        counts = self.module["stage7_6_expected_counts"](freeze)
        self.assertEqual(counts["7.4"], 50)
        self.assertEqual(counts["reused_a4"], 0)


if __name__ == "__main__":
    unittest.main()
