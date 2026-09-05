from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_4_pilot_data import build_pilot_indices, validate_pilot_split


def _times(start: str, count: int) -> np.ndarray:
    return np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(count, "h"), np.timedelta64(1, "h"))


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_times = np.concatenate((_times("2015-01-01", 8760), _times("2016-01-01", 8760), _times("2017-01-01", 8760), _times("2018-01-01", 8760)))
    selection_times = _times("2019-01-01", 8760)
    train_last = np.zeros(len(train_times), dtype=np.int64)
    train_future = np.zeros((len(train_times), 4), dtype=np.int64)
    train_future[::3, 1] = 1
    train_future[1::3, 2] = 2
    selection_last = np.zeros(len(selection_times), dtype=np.int64)
    selection_future = np.zeros((len(selection_times), 4), dtype=np.int64)
    selection_future[::4, 0] = 1
    selection_future[1::4, 0] = 2
    return train_times, train_last, train_future, selection_times, selection_last, selection_future


def test_pilot_indices_are_deterministic_purged_and_year_safe() -> None:
    values = _inputs()
    a = build_pilot_indices(*values, n_train=4096, seed=2026)
    b = build_pilot_indices(*values, n_train=4096, seed=2026)
    np.testing.assert_array_equal(a.train, b.train)
    assert len(a.train) == 4096
    train_times, _, _, selection_times, _, _ = values
    validate_pilot_split(a, train_times, selection_times)
    assert set((selection_times[a.selection_full].astype("datetime64[Y]").astype(int) + 1970).tolist()) == {2019}


def test_pilot_split_rejects_overlap() -> None:
    values = _inputs()
    receipt = build_pilot_indices(*values, n_train=4096, seed=2026)
    receipt.train[0] = receipt.early_stop[0]
    with pytest.raises(ValueError, match="overlap"):
        validate_pilot_split(receipt, values[0], values[3])
