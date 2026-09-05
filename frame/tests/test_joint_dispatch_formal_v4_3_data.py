from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_3_data import (
    derive_thermal_regimes,
    fit_thermal_magnitude_statistics,
    thermal_transition_mask,
)


def test_thermal_regimes_encode_off_cooling_and_heating() -> None:
    target = np.zeros((1, 4, 4), dtype=np.float32)
    target[0, 1, 1] = 111.425
    target[0, 2, 2] = 63.7934
    labels = derive_thermal_regimes(target)
    np.testing.assert_array_equal(labels, [[0, 1, 2, 0]])


def test_thermal_regimes_fail_closed_on_simultaneous_targets() -> None:
    target = np.zeros((1, 4, 4), dtype=np.float32)
    target[0, 0, 1:3] = 1.0
    with pytest.raises(ValueError, match="simultaneous cooling and heating"):
        derive_thermal_regimes(target)


def test_active_statistics_ignore_exact_zeros() -> None:
    target = np.zeros((2, 4, 4), dtype=np.float32)
    target[0, 0, 1] = 100.0
    target[1, 0, 1] = 300.0
    target[0, 1, 2] = 50.0
    target[1, 1, 2] = 150.0
    regimes = derive_thermal_regimes(target)
    receipt = fit_thermal_magnitude_statistics(target, regimes)
    np.testing.assert_allclose(receipt.mean, [200.0, 100.0])
    np.testing.assert_array_equal(receipt.active_count, [2, 2])
    np.testing.assert_array_equal(receipt.class_count, [4, 2, 2])


def test_transition_mask_includes_horizon_changes() -> None:
    future = np.asarray([[0, 0, 1, 1], [2, 2, 2, 2]], dtype=np.int64)
    previous = np.asarray([0, 2], dtype=np.int64)
    np.testing.assert_array_equal(thermal_transition_mask(future, previous), [True, False])
