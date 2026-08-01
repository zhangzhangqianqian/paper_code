import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage6_contract import load_stage6_selection_contract
from src.validation_selection import (
    write_stability_comparison,
    write_validation_summaries,
)


class ValidationSelectionTest(unittest.TestCase):
    def test_stage6_contract_expands_to_twenty_four_runs(self):
        contract = load_stage6_selection_contract()
        combinations = [
            (model, candidate["candidate_id"])
            for model in contract.candidate_models
            for candidate in contract.hyperparameter_candidates
        ]
        self.assertEqual(len(combinations), 24)
        self.assertEqual(len(set(combinations)), 24)

    def test_validation_summary_has_overall_task_and_horizon_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "stl" / "H1"
            run_dir.mkdir(parents=True)
            metrics = {
                "model": "stl",
                "candidate_id": "H1",
                "metrics_original_scale": {
                    "overall_equal_task_mean": {
                        "MAE": 1.0,
                        "RMSE": 2.0,
                        "WAPE": 3.0,
                        "MAPE": 4.0,
                    },
                    "per_task": {
                        task: {"MAE": 1.0, "RMSE": 2.0, "WAPE": 3.0, "MAPE": 4.0}
                        for task in ("electricity", "cooling", "heating")
                    },
                    "per_horizon_equal_element_mean": {
                        f"step_{index}": {
                            "MAE": 1.0,
                            "RMSE": 2.0,
                            "WAPE": 3.0,
                            "MAPE": 4.0,
                        }
                        for index in range(1, 5)
                    },
                },
                "sample_counts": {"train": 10, "validation": 5},
                "model_parameter_count": 123,
                "runtime_seconds": {
                    "fit": 1.5,
                    "validation_evaluation": 0.5,
                    "validation_samples_per_second": 10.0,
                },
                "best_checkpoint": {
                    "epoch": 2,
                    "best_validation_loss": 0.25,
                },
            }
            (run_dir / "metrics_validation.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            write_validation_summaries(
                root,
                [{"model": "stl", "candidate_id": "H1", "run_dir": str(run_dir)}],
            )
            overall = (root / "validation_model_comparison.csv").read_text(
                encoding="utf-8-sig"
            )
            per_task = (root / "validation_per_task.csv").read_text(
                encoding="utf-8-sig"
            )
            per_horizon = (root / "validation_per_horizon.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("validation_rank_by_WAPE", overall)
            self.assertEqual(len(per_task.splitlines()), 4)
            self.assertEqual(len(per_horizon.splitlines()), 5)

    def test_summary_does_not_create_test_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "stl" / "H1"
            run_dir.mkdir(parents=True)
            metrics = {
                "model": "stl",
                "candidate_id": "H1",
                "metrics_original_scale": {
                    "overall_equal_task_mean": {key: 1.0 for key in ("MAE", "RMSE", "WAPE", "MAPE")},
                    "per_task": {
                        task: {key: 1.0 for key in ("MAE", "RMSE", "WAPE", "MAPE")}
                        for task in ("electricity", "cooling", "heating")
                    },
                    "per_horizon_equal_element_mean": {
                        f"step_{index}": {key: 1.0 for key in ("MAE", "RMSE", "WAPE", "MAPE")}
                        for index in range(1, 5)
                    },
                },
                "sample_counts": {"train": 1, "validation": 1},
                "model_parameter_count": 1,
                "runtime_seconds": {},
                "best_checkpoint": {},
            }
            (run_dir / "metrics_validation.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            write_validation_summaries(
                root,
                [{"model": "stl", "candidate_id": "H1", "run_dir": str(run_dir)}],
            )
            self.assertFalse((root / "metrics_test.json").exists())
            self.assertFalse((run_dir / "predictions_test.npz").exists())

    def test_stability_comparison_tracks_rank_and_wape_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            small = root / "small"
            full.mkdir()
            small.mkdir()
            fieldnames = [
                "model",
                "candidate_id",
                "WAPE",
                "validation_rank_by_WAPE",
            ]
            rows_full = [
                {"model": "stl", "candidate_id": "H1", "WAPE": "1.0", "validation_rank_by_WAPE": "1"},
                {"model": "hard_share", "candidate_id": "H1", "WAPE": "2.0", "validation_rank_by_WAPE": "2"},
            ]
            rows_small = [
                {"model": "stl", "candidate_id": "H1", "WAPE": "3.0", "validation_rank_by_WAPE": "2"},
                {"model": "hard_share", "candidate_id": "H1", "WAPE": "1.5", "validation_rank_by_WAPE": "1"},
            ]
            for path, rows in (
                (full / "validation_model_comparison.csv", rows_full),
                (small / "validation_model_comparison.csv", rows_small),
            ):
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
            summary = write_stability_comparison(full, small)
            self.assertTrue(summary["rank_order_changed"])
            self.assertTrue(summary["rank_order_completely_reversed"])
            comparison = (small / "validation_stability_comparison.csv").read_text(
                encoding="utf-8-sig"
            )
            self.assertIn("WAPE_delta_small_minus_full", comparison)


if __name__ == "__main__":
    unittest.main()
