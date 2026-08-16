from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_scheduling_analysis_v2 import build_analysis_manifest  # noqa: E402


class AnalysisRunnerTest(unittest.TestCase):
    def test_manifest_records_provenance_and_no_tuning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root / "source.csv"
            payload.write_text("x\n1\n", encoding="utf-8")
            manifest = build_analysis_manifest(root, root / "contract.json", root, [payload], 175, 6)
            self.assertEqual(manifest["run_count"], 175)
            self.assertFalse(manifest["test_set_used_for_tuning"])
            self.assertIn("source.csv", manifest["output_sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
