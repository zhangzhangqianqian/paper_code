from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.joint_dispatch.formal_v4_capacity import DEFAULT_QUOTAS, select_capacity_origins
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries


def _base() -> FormalV4BaseSeries:
    timestamps = pd.date_range("2015-01-01", "2018-12-31 23:00", freq="h").to_numpy(dtype="datetime64[ns]")
    n = len(timestamps)
    hours = np.arange(n)
    month = pd.DatetimeIndex(timestamps).month.to_numpy()
    electricity = 100.0 + 20.0 * np.sin(2 * np.pi * hours / 24.0)
    cooling = np.where(np.isin(month, [3, 4, 5, 6, 7, 8, 9, 10, 11]), 80.0 + 10.0 * np.sin(2 * np.pi * hours / 24.0), 0.0)
    heating = np.where(np.isin(month, [12, 1, 2]), 70.0, 20.0)
    gas = 5.0 + 0.1 * np.cos(2 * np.pi * hours / 24.0)
    exog = np.zeros((n, 12), dtype=np.float64)
    loads = np.column_stack((electricity, cooling, heating, gas, exog))
    renewable = np.full((n, 2), 0.5, dtype=np.float64)
    prices = np.ones((n, 3), dtype=np.float64)
    return FormalV4BaseSeries(loads, renewable, renewable, prices, timestamps, "train")


def _config() -> dict[str, object]:
    return {"capacity": {"origin_selection": dict(DEFAULT_QUOTAS)}}


def test_select_capacity_origins_is_deterministic_and_exactly_500() -> None:
    base = _base()
    first = select_capacity_origins(base, _config())
    second = select_capacity_origins(base, _config())
    assert len(first.origin_indices) == 500
    assert np.array_equal(first.origin_indices, second.origin_indices)
    assert np.array_equal(first.strata, second.strata)
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.selection_config == DEFAULT_QUOTAS


def test_origin_manifest_covers_year_season_calendar_and_nonzero_cooling() -> None:
    manifest = select_capacity_origins(_base(), _config())
    times = pd.DatetimeIndex(manifest.origin_timestamps)
    assert set(times.year) == {2015, 2016, 2017, 2018}
    assert set(times.month) >= {1, 3, 6, 9}
    weekdays = manifest.demand_summaries["weekday"]
    assert np.any(weekdays <= 4)
    assert np.any(weekdays >= 5)
    assert set(manifest.demand_summaries["hour_bin"].astype(int)) == {0, 6, 12, 18}
    assert float(manifest.demand_summaries["cooling_sum"].sum()) > 0.0
    assert np.any(manifest.strata == "cooling_top_decile")
    assert np.any(manifest.strata == "heating_top_decile")
    assert np.any(manifest.strata == "electricity_top_decile")


def test_old_chronological_first_500_is_not_the_audit_manifest() -> None:
    base = _base()
    old_origins = np.arange(24, 24 + 500, dtype=np.int64)
    old_cooling = np.asarray(
        [base.load_and_exog[index : index + 4, 1].sum() for index in old_origins]
    )
    assert float(old_cooling.sum()) == 0.0
    manifest = select_capacity_origins(base, _config())
    assert not np.array_equal(manifest.origin_indices, old_origins)
    assert float(manifest.demand_summaries["cooling_sum"].sum()) > 0.0


def test_manifest_save_is_immutable(tmp_path: Path) -> None:
    manifest = select_capacity_origins(_base(), _config())
    path = tmp_path / "capacity_origins.json"
    manifest.save(path)
    assert path.exists()
    try:
        manifest.save(path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("manifest save must refuse overwrite")


def test_capacity_origins_reject_selection_or_evaluation_timestamps() -> None:
    base = _base()
    timestamps = base.timestamps.copy()
    timestamps[-1] = np.datetime64("2019-01-01T00:00:00")
    invalid = FormalV4BaseSeries(
        base.load_and_exog, base.renewable_forecast, base.renewable_realized,
        base.prices_and_weights, timestamps, "train",
    )
    try:
        select_capacity_origins(invalid, _config())
    except PermissionError as exc:
        assert "2015-2018" in str(exc)
    else:
        raise AssertionError("selection-year timestamps must be rejected")
