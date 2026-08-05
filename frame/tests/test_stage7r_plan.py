"""Contract tests for the complete Stage 7-R revision."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR = ROOT / "scripts" / "run_stage7r_reproducible.py"
ACCEPTANCE = ROOT / "scripts" / "run_stage7r_acceptance.py"


class Stage7RPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.orchestrator = runpy.run_path(
            str(ORCHESTRATOR), run_name="stage7r_orchestrator_test"
        )
        cls.acceptance = runpy.run_path(
            str(ACCEPTANCE), run_name="stage7r_acceptance_test"
        )

    def test_revision_plan_contains_104_materialized_runs(self):
        plan = self.orchestrator["build_revision_plan"]()
        self.assertEqual([item["run_count"] for item in plan], [20, 40, 34, 10])
        self.assertEqual(sum(item["run_count"] for item in plan), 104)

    def test_acceptance_sources_match_revision_plan(self):
        sources = self.acceptance["build_acceptance_sources"](Path("results"))
        self.assertEqual(sum(item["expected_runs"] for item in sources), 104)
        self.assertEqual(sources[-1]["source_group"], "matched_stl_reference")


if __name__ == "__main__":
    unittest.main()
