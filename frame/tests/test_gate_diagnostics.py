import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gate_diagnostics import compute_gate_diagnostics, load_gate_array


def _uniform_gates(sample_count=4):
    gates = np.zeros((sample_count, 3, 3), dtype=np.float32)
    for target in range(3):
        for source in range(3):
            if target != source:
                gates[:, target, source] = 0.1
    return gates


class GateDiagnosticsTest(unittest.TestCase):
    def test_symmetric_gate_uses_three_unique_pairs_and_zero_asymmetry(self):
        gates = _uniform_gates()
        summary, edges, matrices = compute_gate_diagnostics(
            gates, "dynamic_symmetric", "H1"
        )
        self.assertEqual(summary["edge_count_for_entropy"], 3)
        self.assertAlmostEqual(summary["entropy_mean"], 1.0, places=5)
        self.assertAlmostEqual(summary["asymmetry_mean_abs"], 0.0, places=7)
        self.assertEqual(len(edges), 3)
        np.testing.assert_allclose(matrices["asymmetry"], 0.0)

    def test_directed_gate_reports_asymmetry(self):
        gates = _uniform_gates()
        gates[:, 0, 1] = 0.4
        summary, edges, _ = compute_gate_diagnostics(
            gates, "dynamic_directed", "H1"
        )
        self.assertEqual(summary["edge_count_for_entropy"], 6)
        self.assertGreater(summary["asymmetry_mean_abs"], 0.0)
        self.assertEqual(len(edges), 3)

    def test_static_matrix_is_promoted_to_sample_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gates.npz"
            gates = _uniform_gates(1)[0]
            np.savez_compressed(path, gates=gates)
            loaded = load_gate_array(path)
            self.assertEqual(loaded.shape, (1, 3, 3))

    def test_gate_loader_rejects_nonzero_diagonal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gates.npz"
            gates = _uniform_gates(1)
            gates[0, 0, 0] = 0.1
            np.savez_compressed(path, gates=gates)
            with self.assertRaises(ValueError):
                load_gate_array(path)

    def test_four_task_gate_diagnostics(self):
        gates = np.zeros((5, 4, 4), dtype=np.float32)
        for target in range(4):
            for source in range(4):
                if target != source:
                    gates[:, target, source] = 0.1
        names = ("electricity", "cooling", "heating", "gas")
        summary, edges, matrices = compute_gate_diagnostics(
            gates,
            "dynamic_directed",
            "H1",
            task_names=names,
        )
        self.assertEqual(summary["edge_count_for_entropy"], 12)
        self.assertEqual(len(edges), 6)
        self.assertEqual(matrices["mean"].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
