"""Pure-function checks for the frozen Stage 8 analysis entry point."""

from __future__ import annotations

import runpy
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stage8_analysis.py"


class Stage8AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(SCRIPT), run_name="stage8_analysis_test_module")

    def test_block_bootstrap_preserves_sample_count(self):
        indices = self.module["_block_indices"](
            17, 4, 5, np.random.default_rng(20260806)
        )
        self.assertEqual(indices.shape, (5, 17))
        self.assertTrue(np.all((indices >= 0) & (indices < 17)))

    def test_gain_positive_means_joint_error_is_lower(self):
        self.assertGreater(self.module["_gain"](10.0, 8.0), 0.0)
        self.assertLess(self.module["_gain"](10.0, 12.0), 0.0)

    def test_bh_adjust_marks_only_significant_negative_gain(self):
        rows = [
            {"bootstrap_p_value": 0.001, "gain_MAE_pct": -2.0},
            {"bootstrap_p_value": 0.8, "gain_MAE_pct": -1.0},
        ]
        self.module["_bh_adjust"](rows)
        self.assertTrue(rows[0]["significant_negative_transfer"])
        self.assertFalse(rows[1]["significant_negative_transfer"])

    def test_zero_safe_mape_does_not_create_huge_zero_denominator_error(self):
        metrics = self.module["error_metrics"](
            np.asarray([0.0, 2.0]), np.asarray([1.0, 3.0])
        )
        self.assertEqual(metrics["MAPE_valid_count"], 1)
        self.assertAlmostEqual(float(metrics["MAPE_valid_fraction"]), 0.5)
        self.assertAlmostEqual(float(metrics["MAPE"]), 50.0)

    def test_hierarchical_bootstrap_is_reproducible(self):
        actual = np.tile(np.arange(24, dtype=float), (5, 1))
        reference = actual + 1.0
        joint = actual + 0.5
        first = self.module["hierarchical_bootstrap_mae_gain"](
            actual, reference, joint, block_size=4, replicates=20,
            rng=np.random.default_rng(11),
        )
        second = self.module["hierarchical_bootstrap_mae_gain"](
            actual, reference, joint, block_size=4, replicates=20,
            rng=np.random.default_rng(11),
        )
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.0)

    def test_paired_error_bootstrap_preserves_time_axis_and_is_reproducible(self):
        reference = np.ones((5, 48), dtype=float)
        joint = np.full((5, 48), 0.75, dtype=float)
        first = self.module["paired_block_bootstrap_error_gain"](
            reference, joint, block_size=24, replicates=20,
            rng=np.random.default_rng(19),
        )
        second = self.module["paired_block_bootstrap_error_gain"](
            reference, joint, block_size=24, replicates=20,
            rng=np.random.default_rng(19),
        )
        self.assertEqual(first, second)
        self.assertGreater(first[0], 0.0)

    def test_context_labels_use_target_steps(self):
        times = np.asarray(["2021-02-28T23:00:00"], dtype="datetime64[ns]")
        seasons = self.module["season_labels"](times)
        weekdays = self.module["weekday_labels"](times)
        self.assertEqual(seasons.shape, (1, 4))
        self.assertEqual(weekdays.shape, (1, 4))
        self.assertEqual(seasons[0, 0], "winter")

    def test_spearman_constant_vector_is_undefined(self):
        self.assertIsNone(
            self.module["spearman_correlation"](
                np.ones(8), np.arange(8, dtype=float)
            )
        )

    def test_stage8_roles_separate_joint_models_and_stl_references(self):
        roles = self.module["stage8_role_spec"]()
        self.assertEqual(
            set(roles["joint_main_models"]),
            {"hard_share", "dynamic_symmetric", "mmoe-lite", "ple-lite", "scheme2r"},
        )
        self.assertEqual(
            roles["selection_audit_reference"]["role"],
            "validation_selection_audit_only",
        )
        self.assertEqual(
            roles["matched_transfer_reference"]["role"],
            "matched_transfer_reference_only",
        )

    def test_stage8_output_contract_excludes_best_stl_main_comparison(self):
        outputs = set(self.module["stage8_expected_tables"]())
        self.assertIn("joint_model_comparison.csv", outputs)
        self.assertIn("joint_model_significance.csv", outputs)
        self.assertIn("general_baseline_comparison.csv", outputs)
        self.assertNotIn("best_stl_reference_comparison.csv", outputs)

    def test_equal_task_metrics_do_not_mix_physical_scales(self):
        actual = np.zeros((2, 1, 2), dtype=float)
        prediction = np.zeros_like(actual)
        actual[:, :, 0] = 1.0
        actual[:, :, 1] = 1000.0
        prediction[:, :, 0] = 2.0
        prediction[:, :, 1] = 1000.0
        metrics = self.module["equal_task_error_metrics"](actual, prediction)
        self.assertAlmostEqual(float(metrics["MAE"]), 0.5)
        self.assertAlmostEqual(float(metrics["WAPE"]), 50.0)
        self.assertEqual(metrics["task_count"], 2)
        self.assertEqual(metrics["WAPE_valid_task_count"], 2)

    def test_joint_record_matrix_requires_every_protocol_seed(self):
        roles = self.module["stage8_role_spec"]()
        records = {}
        for model, candidate in roles["joint_main_models"].items():
            records[model] = [
                {
                    "manifest": {
                        "model": model,
                        "candidate_id": candidate,
                        "protocol": protocol,
                        "seed": seed,
                        "tasks": ["electricity", "cooling", "heating", "gas"],
                        "test_used_for_selection": False,
                    }
                }
                for protocol in ("full", "small_sample")
                for seed in (2026, 2027, 2028, 2029, 2030)
            ]
        self.module["validate_joint_record_matrix"](records)
        records["ple-lite"].pop()
        with self.assertRaises(ValueError):
            self.module["validate_joint_record_matrix"](records)


if __name__ == "__main__":
    unittest.main()
