from __future__ import annotations

import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AnalysisTablesTest(unittest.TestCase):
    def test_table_module_exposes_fixed_generation_entrypoint(self):
        from src.scheduling.analysis_tables import MODEL_ORDER, generate_tables

        self.assertEqual(MODEL_ORDER[-1], "scheme2r")
        self.assertTrue(callable(generate_tables))


if __name__ == "__main__":
    unittest.main(verbosity=2)
