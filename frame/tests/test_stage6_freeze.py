import json
import tempfile
import unittest
from pathlib import Path

from src.stage6_freeze import (
    FreezeValidationError,
    _rank_full_candidates,
    _validate_stage6_4,
    freeze_stage6,
)


class Stage6FreezeTest(unittest.TestCase):
    def test_ranks_candidates_without_hardcoded_model_or_h3(self):
        rows = [
            {"model": "model_b", "candidate_id": "H4", "WAPE": "2.0", "RMSE": "5", "parameter_count": "20", "fit_seconds": "4"},
            {"model": "model_a", "candidate_id": "H2", "WAPE": "1.0", "RMSE": "9", "parameter_count": "40", "fit_seconds": "4"},
            {"model": "model_c", "candidate_id": "H1", "WAPE": "1.0", "RMSE": "9", "parameter_count": "30", "fit_seconds": "4"},
        ]
        ranked = _rank_full_candidates(rows)
        self.assertEqual([(row["model"], row["candidate_id"]) for row in ranked], [
            ("model_c", "H1"), ("model_a", "H2"), ("model_b", "H4")
        ])
        self.assertEqual([row["selection_rank"] for row in ranked], [1, 2, 3])

    def test_rejects_smoke_input_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = root / "stage6_2_smoke"
            smoke.mkdir()
            with self.assertRaises(FreezeValidationError):
                freeze_stage6(
                    root / "out",
                    smoke,
                    root / "small",
                    root / "gate",
                    root / "transfer",
                )

    def test_rejects_missing_input_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FreezeValidationError):
                freeze_stage6(
                    root / "out",
                    root / "full",
                    root / "small",
                    root / "gate",
                    root / "transfer",
                )

    def test_rejects_test_artifact_before_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            full.mkdir(parents=True)
            (full / "predictions_test.npz").write_bytes(b"blocked")
            with self.assertRaises(FreezeValidationError):
                freeze_stage6(
                    root / "out",
                    full,
                    root / "small",
                    root / "gate",
                    root / "transfer",
                )

    def test_stage6_4_count_matches_selected_gated_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "gate"
            root.mkdir()
            analyzed = [
                {"model": "dynamic_symmetric", "candidate_id": "H2"},
                *[
                    {"model": "scheme2r", "candidate_id": f"H2_step{step}"}
                    for step in range(1, 5)
                ],
            ]
            manifest = {
                "stage": "6.4",
                "analyzed_run_count": 5,
                "analyzed_runs": analyzed,
                "missing_runs": ["static_gate/H1", "scheme2r/H1"],
                "test_set_accessed": False,
            }
            (root / "stage6_4_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "gate_diagnostics_summary.csv").write_text("x\n", encoding="utf-8")
            (root / "gate_asymmetry.csv").write_text("x\n", encoding="utf-8")
            result = _validate_stage6_4(
                root,
                {("dynamic_symmetric", "H2"), ("scheme2r", "H2")},
            )
            self.assertEqual(result["analyzed_run_count"], 5)

    def test_freeze_output_contains_required_files(self):
        repo = Path(__file__).resolve().parents[2]
        required = [
            repo / "frame" / "reports" / "stage6_2" / "kitakyushu_full" / "stage6_2_manifest.json",
            repo / "frame" / "reports" / "stage6_3" / "kitakyushu_small_sample" / "stage6_3_manifest.json",
            repo / "frame" / "reports" / "stage6_4" / "kitakyushu_full" / "stage6_4_manifest.json",
            repo / "frame" / "reports" / "stage6_5" / "kitakyushu_full" / "stage6_5_manifest.json",
        ]
        if not all(path.exists() for path in required):
            self.skipTest("正式阶段 6 结果尚未生成")
        stage6_manifest = json.loads(required[0].read_text(encoding="utf-8"))
        if stage6_manifest.get("contract_version") != "stage6.1-r1":
            self.skipTest("当前目录为旧版阶段 6 结果，等待 Stage 6-R 重跑")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "freeze"
            result = freeze_stage6(
                output,
                *[path.parent for path in required],
            )
            self.assertTrue(result["primary_model"])
            self.assertFalse(result["test_set_accessed"])
            config = json.loads((output / "stage6_selected_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["primary_model"]["candidate_id"], config["primary_model"]["hyperparameters"]["candidate_id"])
            self.assertEqual(config["data_protocol"]["task_order"], ["electricity", "cooling", "heating", "gas"])
            self.assertEqual(config["training_policy"]["formal_random_seeds"], [2026, 2027, 2028, 2029, 2030])
