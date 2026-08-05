"""Tests for the frozen Stage 7.3 formal run plan (no data access)."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Stage7FormalPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = runpy.run_path(
            str(PROJECT_ROOT / "scripts" / "run_stage7_3.py"),
            run_name="stage7_3_test_module",
        )

    def test_formal_plan_has_twenty_runs(self) -> None:
        plan = self.module["build_run_plan"]()
        self.assertEqual(len(plan), 20)
        self.assertEqual(
            {(item["protocol"], item["model"]) for item in plan},
            {
                ("full", "scheme2r"),
                ("full", "dynamic_symmetric"),
                ("small_sample", "scheme2r"),
                ("small_sample", "dynamic_symmetric"),
            },
        )
        self.assertEqual(
            sorted({int(item["seed"]) for item in plan}),
            [2026, 2027, 2028, 2029, 2030],
        )

    def test_non_frozen_seed_list_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.module["build_run_plan"]((2026,))


if __name__ == "__main__":
    unittest.main()

