import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.external_report import MODEL_ORDER, load_and_validate_runs, write_unified_report


def _fake_metrics(model: str) -> dict:
    tasks = {task: {metric: 1.0 for metric in ("MAE", "RMSE", "WAPE", "MAPE")}
             for task in ("electricity", "cooling", "heating")}
    horizon = {f"step_{index}": {metric: 1.0 for metric in ("MAE", "RMSE", "WAPE", "MAPE")}
               for index in range(1, 5)}
    return {
        "model": model,
        "baseline_name": model,
        "protocol": "small_sample",
        "tasks": ["electricity", "cooling", "heating"],
        "window": {"lookback": 24, "horizon": 4},
        "input_mode": "loads_and_exog" if model == "mmoe-lite" else "loads_only",
        "exog_used": model == "mmoe-lite",
        "future_exogenous_used": False,
        "model_config": {},
        "sample_counts": {"train": 4, "validation": 2, "test": 2},
        "model_parameter_count": 10,
        "runtime_seconds": {
            "fit": 1.0,
            "test_evaluation": 0.1,
            "test_samples_per_second": 20.0,
        },
        "metrics_original_scale": {
            "overall_equal_task_mean": {metric: 1.0 for metric in ("MAE", "RMSE", "WAPE", "MAPE")},
            "per_task": tasks,
            "per_horizon_equal_element_mean": horizon,
        },
    }


class ExternalReportTest(unittest.TestCase):
    def test_unified_report_validates_and_writes_all_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = np.zeros((2, 4, 3), dtype=np.float32)
            for model in MODEL_ORDER:
                run_dir = root / model.replace("-", "_")
                run_dir.mkdir()
                (run_dir / "metrics_test.json").write_text(
                    json.dumps(_fake_metrics(model)), encoding="utf-8"
                )
                np.savez_compressed(
                    run_dir / "predictions_test.npz",
                    prediction=prediction,
                    target=prediction,
                )
            runs = load_and_validate_runs(root, "small_sample")
            summary = write_unified_report(root, "small_sample", runs)
            self.assertTrue(summary["consistency_checks"]["same_output_tail"])
            for filename in (
                "comparison_overall.csv",
                "comparison_per_task.csv",
                "comparison_per_horizon.csv",
                "fairness_table.md",
                "summary.json",
            ):
                self.assertTrue((root / filename).exists())


if __name__ == "__main__":
    unittest.main()
