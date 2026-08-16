"""Tests for the validation-only topology pilot."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from unittest.mock import patch

import frame.src.topology_validation_runner as runner
from frame.src.topology_validation_runner import (
    PHASE_A_SPLIT,
    PHASE_A_YEARS,
    build_phase_a_run_plan,
    build_train_validation_windows,
    validate_phase_a_years,
)
from frame.src.kitakyushu_pipeline import KITAKYUSHU_EXOG_COLUMNS, KITAKYUSHU_TASKS


def _synthetic_frame() -> pd.DataFrame:
    # Include the preceding 2015--2016 context so the validation protocol can
    # use the 24-hour history at the 2017 boundary, just as the real loader
    # does.  2021 is deliberately absent (the sealed year is never loaded).
    timestamps = pd.date_range("2015-01-01", "2020-12-31 23:00:00", freq="h")
    data: dict[str, object] = {"timestamp": timestamps}
    for index, task in enumerate(KITAKYUSHU_TASKS):
        data[task] = (100.0 + index + np.arange(len(timestamps), dtype=np.float32) % 97)
    for index, column in enumerate(KITAKYUSHU_EXOG_COLUMNS):
        data[column] = np.float32(index) + np.arange(len(timestamps), dtype=np.float32) % 31
    return pd.DataFrame(data)


class TopologyValidationRunnerTests(unittest.TestCase):
    def test_phase_a_years_are_exact_and_sealed(self) -> None:
        self.assertEqual(validate_phase_a_years(PHASE_A_YEARS), (2017, 2018, 2019, 2020))
        with self.assertRaises(ValueError):
            validate_phase_a_years((2017, 2018, 2019, 2020, 2021))
        with self.assertRaises(ValueError):
            validate_phase_a_years((2015, 2016, 2017, 2018, 2019, 2020))

    def test_run_matrix_is_six_runs(self) -> None:
        runs = build_phase_a_run_plan()
        self.assertEqual(len(runs), 6)
        self.assertTrue(all(run["protocol"] == "post_ge_regular_operation" for run in runs))
        self.assertEqual({run["seed"] for run in runs}, {2026, 2027, 2028})

    def test_window_builder_has_only_train_and_validation(self) -> None:
        frame = _synthetic_frame()
        windows, stats = build_train_validation_windows(frame)
        self.assertEqual(set(windows), {"train", "validation"})
        self.assertEqual(windows["train"]["loads"].shape[1:], (24, 4))
        self.assertEqual(windows["validation"]["target"].shape[1:], (4, 4))
        self.assertEqual(len(windows["train"]["target"]), 26277)
        self.assertEqual(len(windows["validation"]["target"]), 8781)
        self.assertEqual(tuple(stats.task_columns), KITAKYUSHU_TASKS)
        self.assertEqual(tuple(stats.exog_columns), KITAKYUSHU_EXOG_COLUMNS)
        self.assertTrue(np.all(np.diff(windows["validation"]["target_times"]) > np.timedelta64(0, "h")))
        self.assertFalse(any("test" in key for key in windows))

    def test_statistics_are_fit_on_2017_to_2019_only(self) -> None:
        frame = _synthetic_frame()
        windows, stats = build_train_validation_windows(frame)
        train = frame[frame["timestamp"].dt.year.between(2017, 2019)]
        expected = train[list(KITAKYUSHU_TASKS)].to_numpy(dtype=np.float32).mean(axis=0)
        np.testing.assert_allclose(stats.load_mean, expected)
        self.assertEqual(windows["validation"]["target"].shape[-1], 4)

    def test_sealed_boundary_is_not_a_window(self) -> None:
        frame = _synthetic_frame()
        windows, _ = build_train_validation_windows(frame)
        for split in windows.values():
            years = pd.to_datetime(split["target_times"]).year
            self.assertTrue(np.all(years == 2020) if split is windows["validation"] else np.all(years <= 2019))
            self.assertFalse(np.any(years == 2021))

    def test_requesting_2021_fails_before_canonical_reader(self) -> None:
        with patch.object(runner, "read_kitakyushu_canonical") as reader:
            with self.assertRaises(ValueError):
                runner.load_phase_a_frame("unused", years=(2017, 2018, 2019, 2020, 2021))
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
