from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_stage10_14 import audit  # noqa: E402


class Stage1014AuditTest(unittest.TestCase):
    def test_audit_reports_missing_evidence_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            formal = root / "formal_corrected"
            analysis = root / "analysis"
            analysis.mkdir()
            report = audit(formal, analysis, root / "audit.json")
            self.assertEqual(report["status"], "fail")
            self.assertTrue((root / "audit.json").exists())
            self.assertEqual(json.loads((root / "audit.json").read_text(encoding="utf-8"))["status"], "fail")


if __name__ == "__main__":
    unittest.main(verbosity=2)
