import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.stage6_contract import load_stage6_selection_contract
from src.validation_selection import (
    rank_validation_rows,
    select_stage6_3_candidates,
    write_stability_comparison,
    write_validation_summaries,
)


class ValidationSelectionTest(unittest.TestCase):
    @staticmethod
    def _ranking_row(
        model,
        wape,
        max_task_wape,
        negative_transfer_rate=0.0,
        parameter_count=10,
        fit_seconds=2.0,
    ):
        return {
            "model": model,
            "candidate_id": "H1",
            "WAPE": wape,
            "validation_max_per_task_WAPE": max_task_wape,
            "validation_negative_transfer_rate": negative_transfer_rate,
            "parameter_count": parameter_count,
            "fit_seconds": fit_seconds,
        }

    def test_rank_validation_rows_uses_max_task_wape_inside_tolerance(self):
        rows = [
            self._ranking_row("a", 10.00, 20.0),
            self._ranking_row("b", 10.05, 19.0),
        ]

        ranked = rank_validation_rows(
            rows, tie_tolerance_percentage_points=0.1
        )

        self.assertEqual([row["model"] for row in ranked], ["b", "a"])

    def test_rank_validation_rows_keeps_wape_order_outside_tolerance(self):
        rows = [
            self._ranking_row("a", 10.00, 20.0),
            self._ranking_row("b", 10.11, 1.0),
        ]

        ranked = rank_validation_rows(
            rows, tie_tolerance_percentage_points=0.1
        )

        self.assertEqual([row["model"] for row in ranked], ["a", "b"])

    def test_rank_validation_rows_applies_all_tie_breakers_in_order(self):
        rows = [
            self._ranking_row("fit", 10.00, 20.0, 0.5, 100, 1.0),
            self._ranking_row("parameters", 10.00, 20.0, 0.5, 90, 9.0),
            self._ranking_row("transfer", 10.00, 20.0, 0.25, 500, 9.0),
            self._ranking_row("max_task", 10.09, 19.0, 1.0, 500, 9.0),
        ]

        ranked = rank_validation_rows(
            rows, tie_tolerance_percentage_points=0.1
        )

        self.assertEqual(
            [row["model"] for row in ranked],
            ["max_task", "transfer", "parameters", "fit"],
        )

    def test_rank_validation_rows_rejects_negative_tolerance(self):
        with self.assertRaises(ValueError):
            rank_validation_rows([], tie_tolerance_percentage_points=-0.1)

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

    def test_stage6_3_selection_uses_model_best_config_and_top_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fieldnames = ["model", "candidate_id", "WAPE"]
            rows = [
                {"model": "stl_matched", "candidate_id": "H1", "WAPE": "30.0"},
                {"model": "stl_matched", "candidate_id": "H2", "WAPE": "20.0"},
                {"model": "dynamic_symmetric", "candidate_id": "H3", "WAPE": "11.0"},
                {"model": "dynamic_directed", "candidate_id": "H1", "WAPE": "15.0"},
                {"model": "scheme2r", "candidate_id": "H4", "WAPE": "10.0"},
                {"model": "hard_share", "candidate_id": "H2", "WAPE": "25.0"},
                {"model": "static_gate", "candidate_id": "H1", "WAPE": "22.0"},
            ]
            with (root / "validation_model_comparison.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            selection = select_stage6_3_candidates(root)
            self.assertEqual(selection["models"], ("scheme2r", "dynamic_symmetric"))
            self.assertEqual(
                [row["candidate_id"] for row in selection["selected_rows"]],
                ["H4", "H3"],
            )
            self.assertEqual(
                {
                    model: [candidate["candidate_id"] for candidate in candidates]
                    for model, candidates in selection[
                        "model_hyperparameter_candidates"
                    ].items()
                },
                {"scheme2r": ["H4"], "dynamic_symmetric": ["H3"]},
            )

    def test_stage6_3_selection_adds_scheme2r_when_absent_from_top_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fieldnames = ["model", "candidate_id", "WAPE"]
            rows = [
                {"model": "stl_matched", "candidate_id": "H1", "WAPE": "1.0"},
                {"model": "hard_share", "candidate_id": "H2", "WAPE": "2.0"},
                {"model": "scheme2r", "candidate_id": "H4", "WAPE": "9.0"},
                {"model": "dynamic_symmetric", "candidate_id": "H3", "WAPE": "10.0"},
                {"model": "dynamic_directed", "candidate_id": "H1", "WAPE": "11.0"},
                {"model": "static_gate", "candidate_id": "H1", "WAPE": "12.0"},
            ]
            with (root / "validation_model_comparison.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            selection = select_stage6_3_candidates(root)
            self.assertEqual(
                selection["models"], ("stl_matched", "hard_share", "scheme2r")
            )
            self.assertEqual(len(selection["selected_rows"]), 3)

    def test_stability_comparison_accepts_selected_subset(self):
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
            full_rows = [
                {"model": "scheme2r", "candidate_id": "H3", "WAPE": "1.0", "validation_rank_by_WAPE": "1"},
                {"model": "dynamic_symmetric", "candidate_id": "H3", "WAPE": "2.0", "validation_rank_by_WAPE": "2"},
                {"model": "stl", "candidate_id": "H4", "WAPE": "3.0", "validation_rank_by_WAPE": "3"},
            ]
            small_rows = [
                {"model": "scheme2r", "candidate_id": "H3", "WAPE": "2.5", "validation_rank_by_WAPE": "2"},
                {"model": "dynamic_symmetric", "candidate_id": "H3", "WAPE": "2.0", "validation_rank_by_WAPE": "1"},
            ]
            for path, rows in (
                (full / "validation_model_comparison.csv", full_rows),
                (small / "validation_model_comparison.csv", small_rows),
            ):
                with path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

            summary = write_stability_comparison(full, small)
            self.assertEqual(summary["full_candidate_count"], 3)
            self.assertEqual(summary["small_sample_candidate_count"], 2)
            self.assertEqual(summary["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
