from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class FiguresTest(unittest.TestCase):
    def test_figure_module_uses_python_public_entrypoints(self):
        from src.scheduling.figures import generate_figures, plot_r1, plot_s1

        self.assertTrue(callable(generate_figures))
        self.assertTrue(callable(plot_r1))
        self.assertTrue(callable(plot_s1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
