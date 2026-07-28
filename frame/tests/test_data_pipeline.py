"""阶段1数据管线测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (  # noqa: E402
    FULL_SPLIT,
    HEEW_EXOG_COLUMNS,
    TASKS,
    audit_dataframe,
    build_windows,
    clean_dataframe,
    read_heew_canonical,
    resolve_columns,
    split_dataframe,
)


def make_synthetic_frame(hours: int = 72) -> pd.DataFrame:
    timestamps = pd.date_range("2015-01-01", periods=hours, freq="h")
    base = np.arange(hours, dtype=float)
    return pd.DataFrame(
        {
            "Date Time": timestamps,
            "Electric Load": base + 100,
            "Cooling Load": base + 200,
            "Heating Load": base + 300,
            "Temperature": base + 10,
        }
    )


def write_heew_pair(directory: Path, hours: int = 40) -> tuple[Path, Path]:
    timestamps = pd.date_range("2014-01-01", periods=hours, freq="h")
    energy = pd.DataFrame(
        {
            "Year": timestamps.year,
            "Month": timestamps.month,
            "Day": timestamps.day,
            "Hour": timestamps.hour,
            "Electricity": np.arange(hours, dtype=float) + 100,
            "Cooling": np.arange(hours, dtype=float) + 200,
            "Heat": np.arange(hours, dtype=float) + 300,
        }
    )
    weather = pd.DataFrame(
        {
            "Year": timestamps.year,
            "Month": timestamps.month,
            "Day": timestamps.day,
            "Hour": timestamps.hour,
            "Temperature": np.arange(hours, dtype=float) + 10,
            "Dew Point": np.arange(hours, dtype=float) + 1,
            "Humidity": np.arange(hours, dtype=float) + 40,
            "Wind Speed": np.arange(hours, dtype=float),
            "Wind Gust": np.arange(hours, dtype=float) + 2,
            "Pressure": np.arange(hours, dtype=float) + 28,
            "Precip": np.zeros(hours),
        }
    )
    energy_path = directory / "Total_energy.csv"
    weather_path = directory / "Total_weather.csv"
    energy.to_csv(energy_path, index=False)
    weather.to_csv(weather_path, index=False)
    return energy_path, weather_path


class DataPipelineTest(unittest.TestCase):
    def test_heew_pair_is_merged_and_calendar_features_are_added(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            energy_path, weather_path = write_heew_pair(Path(temporary_directory))
            canonical, resolved = read_heew_canonical(energy_path, weather_path)

        self.assertEqual(canonical.shape, (40, 1 + len(TASKS) + len(HEEW_EXOG_COLUMNS)))
        self.assertEqual(canonical.columns[0], "timestamp")
        self.assertEqual(canonical.columns[1:4].tolist(), list(TASKS))
        self.assertEqual(resolved["energy.electricity"], "Electricity")
        self.assertEqual(resolved["weather.dew_point"], "Dew Point")
        self.assertTrue(canonical["timestamp"].is_monotonic_increasing)
        self.assertTrue(set(canonical["is_weekend"].unique()).issubset({0.0, 1.0}))

    def test_heew_pair_with_misaligned_timestamps_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            energy_path, weather_path = write_heew_pair(Path(temporary_directory))
            weather = pd.read_csv(weather_path).iloc[:-1]
            weather.to_csv(weather_path, index=False)
            with self.assertRaises(ValueError):
                read_heew_canonical(energy_path, weather_path)

    def test_empty_exogenous_window_dimension_is_supported(self):
        frame = make_synthetic_frame(40).rename(
            columns={
                "Date Time": "timestamp",
                "Electric Load": "electricity",
                "Cooling Load": "cooling",
                "Heating Load": "heating",
                "Temperature": "temperature",
            }
        )
        windows = build_windows(frame, lookback=24, horizon=4, exog_columns=())
        self.assertEqual(windows["loads"].shape, (13, 24, 3))
        self.assertEqual(windows["exog"].shape, (13, 24, 0))

    def test_heew_full_split_matches_downloaded_coverage(self):
        self.assertEqual(FULL_SPLIT.train_start, "2014-01-01 00:00:00")
        self.assertEqual(FULL_SPLIT.train_end, "2019-12-31 23:00:00")
        self.assertEqual(FULL_SPLIT.validation_start, "2020-01-01 00:00:00")
        self.assertEqual(FULL_SPLIT.test_end, "2022-12-31 23:00:00")

    def test_column_alias_resolution(self):
        frame = make_synthetic_frame(10)
        resolved = resolve_columns(frame.columns)
        self.assertEqual(resolved["timestamp"], "Date Time")
        self.assertEqual(resolved["electricity"], "Electric Load")
        self.assertEqual(resolved["cooling"], "Cooling Load")
        self.assertEqual(resolved["heating"], "Heating Load")

    def test_audit_detects_gap_and_negative_value(self):
        frame = make_synthetic_frame(12)
        frame = frame.drop(index=5).reset_index(drop=True)
        frame.loc[3, "Electric Load"] = -1
        canonical = frame.rename(
            columns={
                "Date Time": "timestamp",
                "Electric Load": "electricity",
                "Cooling Load": "cooling",
                "Heating Load": "heating",
                "Temperature": "temperature",
            }
        )
        report = audit_dataframe(canonical)
        self.assertEqual(report["duplicate_timestamp_count"], 0)
        self.assertGreaterEqual(report["abnormal_gap_count"], 1)
        self.assertEqual(report["columns"]["electricity"]["negative_count"], 1)

    def test_cleaning_interpolates_short_internal_gap(self):
        frame = make_synthetic_frame(12).rename(
            columns={
                "Date Time": "timestamp",
                "Electric Load": "electricity",
                "Cooling Load": "cooling",
                "Heating Load": "heating",
                "Temperature": "temperature",
            }
        )
        frame.loc[5, "electricity"] = np.nan
        frame.loc[5, "cooling"] = -10
        cleaned, report = clean_dataframe(frame)
        self.assertTrue(np.isfinite(cleaned.loc[5, "electricity"]))
        self.assertTrue(np.isfinite(cleaned.loc[5, "cooling"]))
        self.assertEqual(report["negative_values_replaced_by_nan"]["cooling"], 1)

    def test_split_has_no_overlap(self):
        frame = make_synthetic_frame(72).rename(
            columns={
                "Date Time": "timestamp",
                "Electric Load": "electricity",
                "Cooling Load": "cooling",
                "Heating Load": "heating",
                "Temperature": "temperature",
            }
        )
        spec_frame = frame.copy()
        spec_frame["timestamp"] = pd.date_range("2015-01-01", periods=72, freq="h")
        # 用自定义时间范围验证切分逻辑，而不是依赖ASU完整年份。
        from src.data_pipeline import SplitSpec

        spec = SplitSpec(
            "2015-01-01 00:00:00",
            "2015-01-01 23:00:00",
            "2015-01-02 00:00:00",
            "2015-01-02 23:00:00",
            "2015-01-03 00:00:00",
            "2015-01-03 23:00:00",
        )
        splits = split_dataframe(spec_frame, spec)
        self.assertEqual(sum(len(part) for part in splits.values()), len(spec_frame))
        self.assertEqual(
            set(splits["train"]["timestamp"]).intersection(
                set(splits["validation"]["timestamp"])
            ),
            set(),
        )

    def test_select_training_frame_respects_both_train_boundaries(self):
        from src.data_pipeline import SplitSpec, select_training_frame

        frame = make_synthetic_frame(72).rename(
            columns={
                "Date Time": "timestamp",
                "Electric Load": "electricity",
                "Cooling Load": "cooling",
                "Heating Load": "heating",
                "Temperature": "temperature",
            }
        )
        frame["timestamp"] = pd.date_range("2015-01-01", periods=72, freq="h")
        spec = SplitSpec(
            "2015-01-02 00:00:00",
            "2015-01-02 23:00:00",
            "2015-01-03 00:00:00",
            "2015-01-03 23:00:00",
            "2015-01-04 00:00:00",
            "2015-01-04 23:00:00",
        )
        selected = select_training_frame(frame, spec)
        self.assertEqual(len(selected), 24)
        self.assertEqual(str(selected["timestamp"].iloc[0]), "2015-01-02 00:00:00")
        self.assertEqual(str(selected["timestamp"].iloc[-1]), "2015-01-02 23:00:00")

    def test_windows_have_expected_shapes_and_no_future_input(self):
        frame = make_synthetic_frame(40).rename(
            columns={
                "Date Time": "timestamp",
                "Electric Load": "electricity",
                "Cooling Load": "cooling",
                "Heating Load": "heating",
                "Temperature": "temperature",
            }
        )
        windows = build_windows(frame, lookback=24, horizon=4, exog_columns=("temperature",))
        self.assertEqual(windows["loads"].shape, (13, 24, 3))
        self.assertEqual(windows["exog"].shape, (13, 24, 1))
        self.assertEqual(windows["target"].shape, (13, 4, 3))
        first_target_index = 24
        self.assertEqual(
            windows["target"][0, 0, 0], frame.loc[first_target_index, "electricity"]
        )
        self.assertEqual(
            windows["loads"][0, -1, 0], frame.loc[first_target_index - 1, "electricity"]
        )

    def test_protocol_windows_keep_boundary_history(self):
        frame = make_synthetic_frame(72).rename(
            columns={
                "Date Time": "timestamp",
                "Electric Load": "electricity",
                "Cooling Load": "cooling",
                "Heating Load": "heating",
                "Temperature": "temperature",
            }
        )
        from src.data_pipeline import SplitSpec, build_protocol_windows

        spec = SplitSpec(
            "2015-01-01 00:00:00",
            "2015-01-01 23:00:00",
            "2015-01-02 00:00:00",
            "2015-01-02 23:00:00",
            "2015-01-03 00:00:00",
            "2015-01-03 23:00:00",
        )
        windows = build_protocol_windows(
            frame,
            spec,
            split_name="test",
            lookback=24,
            horizon=4,
            exog_columns=("temperature",),
        )
        self.assertEqual(windows["loads"].shape, (21, 24, 3))
        self.assertEqual(
            windows["loads"][0, -1, 0], frame.loc[47, "electricity"]
        )
        self.assertEqual(
            windows["target_times"][0], np.datetime64("2015-01-03T00:00:00")
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
