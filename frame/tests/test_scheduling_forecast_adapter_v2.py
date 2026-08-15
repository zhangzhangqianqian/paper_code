"""阶段10.5预测适配器测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.forecast_adapter import align_dispatch_inputs, load_forecast_run  # noqa: E402


class ForecastAdapterTest(unittest.TestCase):
    def test_load_and_align_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "full" / "scheme2r" / "H4" / "seed_2026"
            root.mkdir(parents=True)
            times = np.arange("2021-01-01T00", "2021-01-05T00", dtype="datetime64[h]")[:5]
            prediction = np.ones((5, 4, 4), dtype=np.float64)
            prediction[0, 0, 0] = -1.0
            np.savez(root / "predictions_test.npz", prediction=prediction, target_times=times)
            (root / "run_manifest.json").write_text(
                json.dumps({"protocol": "full", "tasks": ["electricity", "cooling", "heating", "gas"]}),
                encoding="utf-8",
            )
            run = load_forecast_run("scheme2r", "H4", 2026, Path(directory))
            renewable = {"predictions": np.ones((5, 4, 2)), "origin_times": times}
            aligned = align_dispatch_inputs(run, renewable)
            self.assertEqual(aligned.demand.shape, (5, 4, 3))
            self.assertEqual(aligned.gas.shape, (5, 4))
            self.assertEqual(aligned.renewable.shape, (5, 4, 2))
            self.assertEqual(aligned.clipped_negative_count, 1)

    def test_alignment_rejects_mismatched_origins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "full" / "scheme2r" / "H4" / "seed_2026"
            root.mkdir(parents=True)
            times = np.arange("2021-01-01T00", "2021-01-05T00", dtype="datetime64[h]")[:5]
            np.savez(root / "predictions_test.npz", prediction=np.ones((5, 4, 4)), target_times=times)
            (root / "run_manifest.json").write_text(json.dumps({"protocol": "full"}), encoding="utf-8")
            run = load_forecast_run("scheme2r", "H4", 2026, Path(directory))
            renewable = {"predictions": np.ones((5, 4, 2)), "origin_times": times + np.timedelta64(1, "h")}
            with self.assertRaises(ValueError):
                align_dispatch_inputs(run, renewable)


if __name__ == "__main__":
    unittest.main(verbosity=2)
