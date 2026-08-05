"""Tests for the Stage 7.3 -> 7.4 sequential launcher."""

from __future__ import annotations

import json
import runpy
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_stage7_3_then_7_4.py"


class Stage734OrchestratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(SCRIPT), run_name="stage7_3_then_7_4_test")

    def test_commands_keep_output_directories_independent(self):
        build = self.module["build_stage_command"]
        command = build(
            Path("stage.py"),
            kitakyushu_data_dir="data",
            contract="contract.json",
            freeze_config="freeze.json",
            output_dir="stage7_3",
        )
        self.assertIn("stage7_3", command)
        self.assertNotIn("stage7_4", command)
        self.assertNotIn("predictions_test.npz", command)

    def test_manifest_validation_requires_all_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "stage7_3_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "run_count_expected": 20,
                        "run_count_completed": 20,
                        "run_count_failed": 0,
                    }
                ),
                encoding="utf-8",
            )
            manifest = self.module["validate_stage_manifest"](root, "7.3")
            self.assertEqual(manifest["run_count_completed"], 20)

    def test_manifest_validation_rejects_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "stage7_3_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "passed",
                        "run_count_expected": 20,
                        "run_count_completed": 19,
                        "run_count_failed": 1,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                self.module["validate_stage_manifest"](root, "7.3")

    def test_failed_stage73_blocks_stage74(self):
        self.assertFalse(self.module["should_start_stage74"](False))
        self.assertTrue(self.module["should_start_stage74"](True))


if __name__ == "__main__":
    unittest.main()
