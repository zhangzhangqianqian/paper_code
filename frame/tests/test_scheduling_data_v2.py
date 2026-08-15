"""阶段10.3数据计划视图和结算视图测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kitakyushu_pipeline import KITAKYUSHU_CALENDAR_COLUMNS, KITAKYUSHU_WEATHER_COLUMNS  # noqa: E402
from src.scheduling.data import (  # noqa: E402
    ACTUAL_COLUMNS,
    SchedulingFrame,
    make_plan_view,
    make_settlement_view,
)


def _synthetic_frame() -> SchedulingFrame:
    timestamps = pd.date_range("2021-01-01", periods=60, freq="1h")
    frame = pd.DataFrame({"timestamp": timestamps})
    for index, task in enumerate(("electricity", "cooling", "heating", "gas")):
        frame[task] = np.arange(len(frame), dtype=float) + index
    for index, column in enumerate((*KITAKYUSHU_WEATHER_COLUMNS, *KITAKYUSHU_CALENDAR_COLUMNS)):
        frame[column] = float(index)
    frame["actual_grid_import"] = 100.0
    frame["actual_pv"] = 10.0
    return SchedulingFrame(data=frame, metadata={}, years=(2021,))


class SchedulingDataViewTest(unittest.TestCase):
    def test_plan_view_excludes_future_actual_columns(self):
        plan = make_plan_view(_synthetic_frame(), "2021-01-02 00:00:00")
        self.assertEqual(len(plan.history), 24)
        self.assertEqual(len(plan.future_times), 4)
        self.assertFalse(any(column.startswith("actual_") for column in plan.history.columns))
        self.assertNotIn("actual_grid_import", plan.visible_columns)

    def test_settlement_view_contains_only_aligned_future_values(self):
        settlement = make_settlement_view(_synthetic_frame(), "2021-01-02 00:00:00")
        self.assertEqual(len(settlement.realized), 4)
        self.assertEqual(tuple(settlement.protected_columns), ("timestamp", "electricity", "cooling", "heating", "gas", *ACTUAL_COLUMNS))
        self.assertTrue(np.all(settlement.realized["actual_pv"].to_numpy() == 10.0))

    def test_non_contiguous_history_is_rejected(self):
        frame = _synthetic_frame()
        frame = SchedulingFrame(
            data=frame.data.drop(index=10).reset_index(drop=True),
            metadata=frame.metadata,
            years=frame.years,
        )
        with self.assertRaises(ValueError):
            make_plan_view(frame, "2021-01-02 00:00:00")


if __name__ == "__main__":
    unittest.main(verbosity=2)
