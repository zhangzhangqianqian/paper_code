from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_4_regime import (
    derive_last_observed_regime,
    derive_thermal_regimes,
    fit_thermal_prior,
)


def _sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = np.zeros((12, 4, 4), dtype=np.float64)
    target[:, :, 0] = 10.0
    target[4:6, :, 1] = 100.0
    target[6:8, :, 2] = 50.0
    target[8:10, :, 1] = 200.0
    target[10:12, :, 2] = 80.0
    history = np.zeros((12, 24, 4), dtype=np.float64)
    history[4:6, -1, 1] = 100.0
    history[6:8, -1, 2] = 50.0
    history[8:10, -1, 1] = 100.0
    history[10:12, -1, 2] = 50.0
    times = np.asarray(["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2015-02-01", "2016-02-01", "2017-02-01", "2018-02-01", "2015-07-01", "2016-07-01", "2017-07-01", "2018-07-01"], dtype="datetime64[ns]")
    return target, history, times


def test_thermal_regimes_encode_off_cooling_and_heating() -> None:
    target = np.zeros((1, 4, 4), dtype=np.float64)
    target[0, 1, 1] = 111.425
    target[0, 2, 2] = 63.7934
    np.testing.assert_array_equal(derive_thermal_regimes(target), [[0, 1, 2, 0]])


def test_thermal_regimes_fail_closed_on_simultaneous_targets() -> None:
    target = np.zeros((1, 4, 4), dtype=np.float64)
    target[0, 0, 1:3] = 1.0
    with pytest.raises(ValueError, match="simultaneous"):
        derive_thermal_regimes(target)


def test_last_regime_uses_last_historical_hour_only() -> None:
    history = np.zeros((2, 24, 4), dtype=np.float64)
    history[0, -1, 1] = 1.0
    history[1, -1, 2] = 1.0
    np.testing.assert_array_equal(derive_last_observed_regime(history), [1, 2])


def test_transition_prior_is_normalized_and_horizon_conditioned() -> None:
    target, history, times = _sample()
    receipt = fit_thermal_prior(target, history, times)
    assert receipt.transition_probability.shape == (4, 3, 3)
    np.testing.assert_allclose(receipt.transition_probability.sum(-1), 1.0)
    assert receipt.years == (2015, 2016, 2017, 2018)
    assert receipt.active_mean.shape == (2,)


def test_prior_rejects_selection_year() -> None:
    target, history, times = _sample()
    times[0] = np.datetime64("2019-01-01")
    with pytest.raises(ValueError, match="training years"):
        fit_thermal_prior(target, history, times)
