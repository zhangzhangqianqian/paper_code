from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.joint_dispatch.contract import DISPATCH_ORDER
from src.joint_dispatch.formal_v4_data import (
    ACTIVITY_FEATURE_NAMES,
    CONTINUOUS_DEVICE_FEATURE_NAMES,
    FormalV4BaseSeries,
    FormalV4Normalization,
    FormalV4WindowSplit,
    build_same_information_teacher,
    materialize_state_windows,
)


def _base(n: int = 40) -> FormalV4BaseSeries:
    timestamps = np.datetime64("2015-01-01T00:00") + np.arange(n).astype("timedelta64[h]")
    load_exog = np.arange(n * 16, dtype=np.float64).reshape(n, 16) + 1.0
    renewable = np.column_stack((np.linspace(1.0, 2.0, n), np.linspace(2.0, 3.0, n)))
    persistence = np.vstack((renewable[:1], renewable[:-1]))
    prices = np.ones((n, 3), dtype=np.float64)
    return FormalV4BaseSeries(load_exog, persistence, renewable, prices, timestamps, "train")


def _receipt(gate3: bool = False) -> dict[str, object]:
    return {"gate0_authorized": True, "gate3_authorized": gate3, "capacity_scenario_hash": "cap"}


def test_v4_state_projection_excludes_diagnostics():
    base = _base()
    dispatch = np.zeros((len(base.timestamps), len(DISPATCH_ORDER)), dtype=np.float64)
    split = materialize_state_windows(base, dispatch, capacity_receipt=_receipt())
    assert split.device_history.shape[-1] == 17
    assert tuple(split.device_feature_names) == CONTINUOUS_DEVICE_FEATURE_NAMES
    assert not ({"slack_e", "slack_c", "slack_h", "q_dump"} & set(split.device_feature_names))


def test_gas_is_auxiliary_and_keeps_native_target():
    base = _base()
    dispatch = np.zeros((len(base.timestamps), len(DISPATCH_ORDER)), dtype=np.float64)
    split = materialize_state_windows(base, dispatch, capacity_receipt=_receipt())
    assert split.forecast_target.shape[-1] == 4
    assert split.rigid_demand.shape[-1] == 3
    assert split.gas_context_target.shape[-1] == 1


def test_teacher_uses_deployment_information_not_realized_future():
    base = _base()
    dispatch = np.zeros((len(base.timestamps), len(DISPATCH_ORDER)), dtype=np.float64)
    split = materialize_state_windows(base, dispatch, capacity_receipt=_receipt())
    teacher = np.ones((len(split), 4, len(DISPATCH_ORDER)), dtype=np.float64)
    a = build_same_information_teacher(split, teacher, stage_p_checkpoint_sha256="stage-p")
    mutated = replace(base, renewable_realized=base.renewable_realized + 100.0)
    split_mutated = materialize_state_windows(mutated, dispatch, capacity_receipt=_receipt())
    b = build_same_information_teacher(split_mutated, teacher, stage_p_checkpoint_sha256="stage-p")
    np.testing.assert_array_equal(a.dispatch, b.dispatch)


def test_renewable_persistence_repeats_only_last_available_value():
    base = _base()
    dispatch = np.zeros((len(base.timestamps), len(DISPATCH_ORDER)), dtype=np.float64)
    split = materialize_state_windows(base, dispatch, capacity_receipt=_receipt())
    expected = np.repeat(split.renewable_history[:, -1:, :], 4, axis=1)
    np.testing.assert_array_equal(split.renewable_forecast, expected)


def test_evaluation_build_requires_gate3_authorization():
    with pytest.raises(PermissionError, match="Gate 3"):
        materialize_state_windows(_base(), np.zeros((40, 21)), capacity_receipt=_receipt(), split="evaluation")


def test_state_materialization_requires_frozen_capacity_receipt():
    with pytest.raises(PermissionError, match="capacity receipt"):
        materialize_state_windows(_base(), np.zeros((40, 21)), capacity_receipt=None)


def test_state_hash_binds_capacity_receipt_and_trajectory_rule():
    base = _base()
    dispatch = np.zeros((len(base.timestamps), len(DISPATCH_ORDER)), dtype=np.float64)
    first = materialize_state_windows(base, dispatch, capacity_receipt={"gate0_authorized": True, "capacity_scenario_hash": "a"}, trajectory_hash="traj-a")
    second = materialize_state_windows(base, dispatch, capacity_receipt={"gate0_authorized": True, "capacity_scenario_hash": "b"}, trajectory_hash="traj-b")
    assert not np.array_equal(first.state_hashes, second.state_hashes)


def test_normalization_is_train_only():
    base = _base()
    dispatch = np.zeros((len(base.timestamps), len(DISPATCH_ORDER)), dtype=np.float64)
    split = materialize_state_windows(base, dispatch, capacity_receipt=_receipt())
    norm = FormalV4Normalization.fit(split)
    assert norm.fitted_split == "train"
    with pytest.raises(ValueError, match="train"):
        FormalV4Normalization.fit(replace(split, split="selection"))


def test_gap_is_preserved_but_crossing_windows_are_not_materialized():
    base = _base()
    timestamps = base.timestamps.copy()
    timestamps[30:] += np.timedelta64(2, "h")
    gapped = replace(base, timestamps=timestamps)
    split = materialize_state_windows(gapped, np.zeros((40, 21)), capacity_receipt=_receipt())
    assert len(split) < len(base.timestamps) - 27
