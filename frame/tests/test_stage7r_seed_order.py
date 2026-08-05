"""Regression tests for the Stage 7-R seed-order correction."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Stage7RSeedOrderTests(unittest.TestCase):
    def _assert_seed_before_model(self, relative_path: str, model_marker: str) -> None:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        seed_position = source.index("    set_reproducible(")
        model_position = source.index(model_marker, seed_position)
        self.assertLess(seed_position, model_position)
        self.assertIn("shuffle=True, seed=", source)

    def test_stage7_3_seeds_before_model_construction(self):
        self._assert_seed_before_model("scripts/run_stage7_3.py", "model = _build_model(")

    def test_stage7_4_seeds_before_model_construction(self):
        self._assert_seed_before_model(
            "scripts/run_stage7_4.py", "model = _build_ablation_model("
        )

    def test_external_trainer_seeds_before_model_construction(self):
        self._assert_seed_before_model(
            "scripts/train_external_baseline.py", "model = DLinearBaseline("
        )

    def test_main_training_entry_seeds_before_model_construction(self):
        self._assert_seed_before_model(
            "scripts/train_models.py", "model = _make_model("
        )

    def test_stage7_2_seeds_before_model_construction(self):
        self._assert_seed_before_model(
            "scripts/run_stage7_2.py", "model = _build_model("
        )

    def test_matched_stl_reference_seeds_before_model_construction(self):
        self._assert_seed_before_model(
            "scripts/run_stage7_stl_reference.py", "model = _build_task_model("
        )


if __name__ == "__main__":
    unittest.main()
