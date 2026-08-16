from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frame.src.topology_scheduling_runner import (
    build_scheduling_run_plan,
    route_prediction_channels,
    validate_prediction_artifact,
)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
