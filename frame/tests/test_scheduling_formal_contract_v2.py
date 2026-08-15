from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.freeze_scheduling_contract_v2 import SEEDS, _model_sources  # noqa: E402


class FormalContractTest(unittest.TestCase):
    def test_five_models_and_five_seeds_are_fixed(self):
        sources = _model_sources(Path("D:/Paper/github_work/paper-code"))
        self.assertEqual(len(sources), 5)
        self.assertEqual(SEEDS, [2026, 2027, 2028, 2029, 2030])
        self.assertIn("scheme2r", sources)
        self.assertIn("mmoe_lite", sources)


if __name__ == "__main__":
    unittest.main(verbosity=2)
