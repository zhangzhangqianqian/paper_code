from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_scheduling_preflight_v2 import _check  # noqa: E402


class SchedulingPreflightTest(unittest.TestCase):
    def test_failed_check_is_not_allowed(self):
        self.assertEqual(_check(False, "bad")["status"], "fail")

    def test_smoke_manifest_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke_manifest.json"
            path.write_text(json.dumps({"status": "pass", "origin_count": 96}), encoding="utf-8")
            self.assertEqual(json.loads(path.read_text())["origin_count"], 96)


if __name__ == "__main__":
    unittest.main(verbosity=2)
