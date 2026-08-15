"""阶段10.2：SciPy/HiGHS 最小连续线性规划测试。"""

from __future__ import annotations

import unittest


class HighsSolverTest(unittest.TestCase):
    def test_minimal_lp(self):
        from scipy.optimize import linprog

        result = linprog(
            c=[1.0, 2.0],
            A_ub=[[-1.0, -1.0]],
            b_ub=[-1.0],
            bounds=[(0.0, None), (0.0, None)],
            method="highs",
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(float(result.fun), 1.0, places=9)
        self.assertLessEqual(float(result.ineqlin.residual[0]), 1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
