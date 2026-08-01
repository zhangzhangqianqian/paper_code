"""Kitakyushu 数据适配器测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_TASKS,
    audit_kitakyushu_dataframe,
    build_kitakyushu_windows,
    clean_kitakyushu_dataframe,
)


def make_frame(hours: int = 40) -> pd.DataFrame:
    timestamps = pd.date_range("2015-01-01", periods=hours, freq="h")
    values = np.arange(hours, dtype=float)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "electricity": values + 100,
            "cooling": values + 200,
            "heating": values + 300,
            "gas": values + 400,
            "temperature": values + 10,
            "humidity": values + 40,
            "solar_irradiance": values,
            "wind_speed": values,
            "wind_direction": values,
        }
    )
    from src.data_pipeline import _add_calendar_features

    return _add_calendar_features(frame)


class KitakyushuPipelineTest(unittest.TestCase):
    def test_audit_reports_four_tasks(self):
        report = audit_kitakyushu_dataframe(make_frame())
        self.assertEqual(report["required_columns_missing"], [])
        self.assertEqual(report["columns"]["gas"]["negative_count"], 0)

    def test_cleaning_does_not_interpolate_targets(self):
        frame = make_frame()
        frame.loc[5, "gas"] = np.nan
        frame.loc[6, "humidity"] = np.nan
        cleaned, report = clean_kitakyushu_dataframe(frame)
        self.assertNotIn(frame.loc[5, "timestamp"], set(cleaned["timestamp"]))
        self.assertTrue(np.isfinite(cleaned["humidity"]).all())
        self.assertFalse(report["targets_were_interpolated"])
        self.assertEqual(report["target_rows_removed"], 1)

    def test_four_task_window_shapes(self):
        windows = build_kitakyushu_windows(make_frame())
        self.assertEqual(windows["loads"].shape, (13, 24, 4))
        self.assertEqual(windows["exog"].shape, (13, 24, len(KITAKYUSHU_EXOG_COLUMNS)))
        self.assertEqual(windows["target"].shape, (13, 4, 4))
        self.assertEqual(len(KITAKYUSHU_TASKS), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)

