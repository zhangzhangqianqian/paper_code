"""Unit tests for topology-regime audit helpers."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from frame.src.kitakyushu_pipeline import GAS_SOURCE_COLUMNS
from frame.src.topology_regime_audit import (
    summarize_gas_components,
    summarize_task_statistics,
    summarize_yearly_frame,
)


class TopologyRegimeAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.timestamps = pd.date_range("2017-01-01", periods=6, freq="h")
        cls.frame = pd.DataFrame(
            {
                "timestamp": cls.timestamps,
                "electricity": [1, 2, 3, 4, 5, 6],
                "cooling": [0, 0, 1, 1, 0, 0],
                "heating": [4, 4, 3, 3, 2, 2],
                "gas": [10, 11, 12, 13, 14, 15],
                "temperature": [20, 21, 22, 23, 24, 25],
                "humidity": [50, 50, 51, 51, 52, 52],
            }
        )

    def test_yearly_quality_reports_missingness_and_rows(self) -> None:
        report = summarize_yearly_frame(
            self.frame,
            [2017],
            columns=("electricity", "cooling", "heating", "gas", "temperature"),
        )
        self.assertEqual(int(report.loc[0, "rows"]), 6)
        self.assertEqual(int(report.loc[0, "hourly_gaps_gt_1"]), 0)
        self.assertEqual(int(report.loc[0, "gas_missing"]), 0)

    def test_task_statistics_are_numeric_and_deterministic(self) -> None:
        report = summarize_task_statistics(self.frame, [2017])
        electricity = report.loc[report["task"] == "electricity"].iloc[0]
        self.assertAlmostEqual(float(electricity["mean"]), 3.5)
        self.assertAlmostEqual(float(electricity["zero_ratio"]), 0.0)

    def test_gas_component_summary_tracks_nonzero_hours_and_last_timestamp(self) -> None:
        components = pd.DataFrame({"timestamp": self.timestamps})
        for index, column in enumerate(GAS_SOURCE_COLUMNS):
            values = np.zeros(len(self.timestamps), dtype=float)
            values[-1] = index + 1
            components[column] = values
        report = summarize_gas_components(components, [2017])
        self.assertEqual(set(report["component"]), set(GAS_SOURCE_COLUMNS))
        self.assertTrue((report["nonzero_hours"] == 1).all())
        self.assertTrue(report["last_nonzero_timestamp"].notna().all())


if __name__ == "__main__":
    unittest.main()
