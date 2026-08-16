from __future__ import annotations

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AnalysisIoContractTest(unittest.TestCase):
    def test_module_exposes_read_only_loading_contract(self):
        from src.scheduling.analysis_io import FormalSchedulingData, load_formal_runs, materialize_source_data

        self.assertTrue(callable(load_formal_runs))
        self.assertTrue(callable(materialize_source_data))
        self.assertTrue(hasattr(FormalSchedulingData, "runs_for"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
