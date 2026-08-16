from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import json

import numpy as np

from frame.src.topology_scheduling_runner import (
    build_scheduling_run_plan,
    route_prediction_channels,
    validate_prediction_artifact,
)
from frame.src.topology_protocol_analysis import validate_phase_b_result_artifacts

from frame.tests.test_topology_test_runner import _freeze


class TopologySchedulingRunnerTests(unittest.TestCase):
    def test_branch_specific_scheduling_matrix(self) -> None:
        core = build_scheduling_run_plan(_freeze("core_prediction_changed"))
        self.assertEqual(len(core), 175)
        gas = build_scheduling_run_plan(_freeze("gas_only_changed"))
        self.assertEqual(len(gas), 31)
        stable = build_scheduling_run_plan(_freeze("core_conclusion_stable"))
        self.assertEqual(len(stable), 12)
        self.assertTrue(all(row["track"] in {"real_replay", "simulated_dispatch"} for row in core))

    def test_s_track_excludes_gas_from_rigid_demand(self) -> None:
        prediction = np.ones((3, 4, 4), dtype=np.float32)
        routed = route_prediction_channels(prediction, "simulated_dispatch")
        self.assertEqual(routed["demand"].shape, (3, 4, 3))
        self.assertNotIn("gas", routed["rigid_demand_tasks"])
        self.assertEqual(route_prediction_channels(prediction, "real_replay")["full_loads"].shape, (3, 4, 4))

    def test_prediction_artifact_shape_and_time_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_test.npz"
            times = np.arange("2021-01-01", "2021-01-04", dtype="datetime64[h]")
            prediction = np.ones((len(times), 4, 4), dtype=np.float32)
            np.savez_compressed(path, prediction=prediction, target_times=times)
            loaded = validate_prediction_artifact(path)
            self.assertEqual(loaded["prediction"].shape, (len(times), 4, 4))
            with self.assertRaises(ValueError):
                np.savez_compressed(path, prediction=np.ones((2, 4, 3), dtype=np.float32), target_times=times[:2])
                validate_prediction_artifact(path)

    def test_completed_prediction_and_routing_manifest_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze = _freeze("core_conclusion_stable")
            freeze_path = root / "branch_freeze" / "topology_branch_freeze.json"
            freeze_path.parent.mkdir(parents=True)
            freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
            times = np.arange("2021-01-01", "2021-01-04", dtype="datetime64[h]")
            plan = build_scheduling_run_plan(freeze)
            for model, candidate, seed in sorted({(r["model"], r["candidate_id"], r["seed"]) for r in plan}):
                run_dir = root / "predictions" / model / candidate / f"seed_{seed}"
                run_dir.mkdir(parents=True)
                np.savez_compressed(run_dir / "predictions_test.npz", prediction=np.ones((len(times), 4, 4)), target_times=times)
                (run_dir / "run_manifest.json").write_text(json.dumps({
                    "status": "passed", "branch_frozen_before_test": True,
                    "test_used_for_selection": False,
                }), encoding="utf-8")
            schedule_path = root / "schedule.json"
            schedule_path.write_text(json.dumps({
                "branch_frozen_before_scheduling": True,
                "lp_equations_modified": False,
                "runs": [{} for _ in plan],
            }), encoding="utf-8")
            report = validate_phase_b_result_artifacts(
                freeze_path, root / "predictions", scheduling_manifest_path=schedule_path
            )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["forecast_run_count"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
