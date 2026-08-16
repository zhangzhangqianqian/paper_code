"""Strict tests for cross-topology validation artifact registration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from frame.src.topology_protocol_analysis import (
    align_validation_origins,
    load_validation_artifact,
)


def _write_artifact(root: Path, *, name: str = "predictions_validation.npz", seed: int = 2026,
                    n: int = 8, include_target: bool = True, unsorted: bool = False,
                    test_year: bool = False, target_offset: float = 0.0) -> Path:
    path = root / "full" / "scheme2r" / "H4" / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    times = np.arange(np.datetime64("2020-01-01T00", "h"),
                      np.datetime64("2020-01-01T00", "h") + n, dtype="datetime64[h]")
    if test_year:
        times = times.astype("datetime64[Y]") + 1
    prediction = np.zeros((n, 4, 4), dtype=np.float32)
    target = np.ones((n, 4, 4), dtype=np.float32) + target_offset
    if unsorted:
        order = np.arange(n)[::-1]
        times, prediction, target = times[order], prediction[order], target[order]
    values = {"prediction": prediction, "target_times": times.astype("datetime64[ns]")}
    if include_target:
        values["target"] = target
    output = path / name
    np.savez_compressed(output, **values)
    return output


class TopologyProtocolAnalysisTests(unittest.TestCase):
    def test_loads_exact_validation_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = load_validation_artifact(_write_artifact(Path(directory)))
            self.assertEqual(artifact.prediction.shape, (8, 4, 4))
            self.assertEqual(artifact.sample_count, 8)

    def test_rejects_missing_keys_and_test_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                load_validation_artifact(_write_artifact(root, include_target=False))
            with self.assertRaises(ValueError):
                load_validation_artifact(_write_artifact(root, name="predictions_test.npz"))

    def test_rejects_bad_shape_unsorted_and_wrong_year(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                load_validation_artifact(_write_artifact(root, n=2, unsorted=True))
            with self.assertRaises(ValueError):
                load_validation_artifact(_write_artifact(root, test_year=True))

    def test_rejects_seed_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_artifact(Path(directory), seed=2027)
            with self.assertRaises(ValueError):
                load_validation_artifact(path, expected_seed=2026)

    def test_alignment_rejects_target_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = load_validation_artifact(_write_artifact(root / "left"))
            right = load_validation_artifact(_write_artifact(root / "right", target_offset=2.0))
            with self.assertRaises(ValueError):
                align_validation_origins(left, right)

    def test_alignment_intersects_origins_without_flattening_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_path = _write_artifact(root / "left", n=8)
            right_path = _write_artifact(root / "right", n=6)
            # Shift the right artifact by one origin while retaining the same
            # target values; the common-origin intersection is non-empty.
            with np.load(right_path) as values:
                times = values["target_times"] + np.timedelta64(1, "h")
                np.savez_compressed(
                    right_path,
                    prediction=values["prediction"],
                    target=values["target"],
                    target_times=times,
                )
            left = load_validation_artifact(left_path)
            right = load_validation_artifact(right_path)
            left_aligned, right_aligned = align_validation_origins(left, right)
            self.assertEqual(left_aligned.prediction.shape[1:], (4, 4))
            self.assertEqual(left_aligned.sample_count, 6)
            self.assertTrue(np.array_equal(left_aligned.target_times, right_aligned.target_times))


if __name__ == "__main__":
    unittest.main(verbosity=2)
