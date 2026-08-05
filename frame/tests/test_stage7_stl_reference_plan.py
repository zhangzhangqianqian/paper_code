"""Contract tests for the Stage 7-R matched-STL formal runner."""

from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.kitakyushu_pipeline import KITAKYUSHU_EXOG_COLUMNS, KITAKYUSHU_TASKS
from src.training import StandardizationStats


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stage7_stl_reference.py"


class Stage7STLReferencePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(SCRIPT), run_name="stage7_stl_test_module")

    def test_plan_contains_ten_protocol_seed_runs(self):
        plan = self.module["build_stl_run_plan"]()
        self.assertEqual(len(plan), 10)
        self.assertEqual({item["protocol"] for item in plan}, {"full", "small_sample"})
        self.assertEqual({item["model"] for item in plan}, {"stl_matched"})

    def test_task_seed_mapping_is_stable_and_unique(self):
        task_seed = self.module["_task_seed"]
        values = [task_seed(2026, task_index) for task_index in range(4)]
        self.assertEqual(values, [2026, 3026, 4026, 5026])
        self.assertEqual(len(set(values)), 4)

    def test_four_independent_tasks_complete_one_epoch_smoke(self):
        self.module["FORMAL_BATCH_SIZE"] = 4
        self.module["FORMAL_THREADS"] = 1
        self.module["FORMAL_MAX_EPOCHS"] = 1
        self.module["FORMAL_PATIENCE"] = 1
        rng = np.random.default_rng(2026)
        exog_dim = len(KITAKYUSHU_EXOG_COLUMNS)

        def windows(sample_count: int, start: str):
            return {
                "loads": rng.normal(size=(sample_count, 24, 4)).astype(np.float32),
                "exog": rng.normal(size=(sample_count, 24, exog_dim)).astype(np.float32),
                "target": rng.normal(size=(sample_count, 4, 4)).astype(np.float32),
                "target_times": np.datetime64(start) + np.arange(sample_count).astype(
                    "timedelta64[h]"
                ),
            }

        split_windows = {
            "train": windows(12, "2019-01-01"),
            "validation": windows(8, "2020-01-01"),
            "test": windows(8, "2021-01-01"),
        }
        stats = StandardizationStats(
            load_mean=np.zeros(4, dtype=np.float32),
            load_scale=np.ones(4, dtype=np.float32),
            exog_mean=np.zeros(exog_dim, dtype=np.float32),
            exog_scale=np.ones(exog_dim, dtype=np.float32),
            exog_columns=KITAKYUSHU_EXOG_COLUMNS,
            task_columns=KITAKYUSHU_TASKS,
        )
        hyperparameters = {
            "hidden_dim": 8,
            "prediction_head_hidden_dim": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "scheme2r_kernel_size": 5,
            "scheme2r_dilations": [1, 2, 4],
        }
        training_policy = {"weight_decay": 0.0, "gradient_clip_norm": 1.0}
        run = {
            "protocol": "full",
            "model": "stl_matched",
            "candidate_id": "H3",
            "seed": 2026,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.module["_run_one"](
                run,
                output,
                split_windows,
                stats,
                hyperparameters,
                training_policy,
            )
            with np.load(output / "predictions_test.npz") as archive:
                self.assertEqual(tuple(archive["prediction"].shape), (8, 4, 4))
            self.assertEqual(
                len(list(output.glob("best_model_*.pt"))), len(KITAKYUSHU_TASKS)
            )
        self.assertEqual(manifest["status"], "passed")
        self.assertTrue(manifest["task_models_trained_independently"])

    def test_matched_stl_head_width_is_fixed(self):
        build = self.module["_build_task_model"]
        model = build(
            0,
            {
                "hidden_dim": 16,
                "prediction_head_hidden_dim": 16,
                "dropout": 0.0,
                "scheme2r_kernel_size": 5,
                "scheme2r_dilations": [1, 2, 4],
            },
        )
        self.assertEqual(model.head.head_hidden_dim, 16)
        with self.assertRaises(ValueError):
            build(
                0,
                {
                    "hidden_dim": 32,
                    "prediction_head_hidden_dim": 8,
                    "dropout": 0.0,
                    "scheme2r_kernel_size": 5,
                    "scheme2r_dilations": [1, 2, 4],
                },
            )


if __name__ == "__main__":
    unittest.main()
