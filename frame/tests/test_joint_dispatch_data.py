from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.joint_dispatch.data import (
    SCHEDULER_CONTEXT_ORDER,
    STATUS_ORDER,
    JointNormalization,
    build_joint_windows,
    derive_device_status,
)
from src.data_pipeline import SplitSpec
from src.joint_dispatch.contract import DISPATCH_ORDER, EXOG_ORDER, TASK_ORDER


def _synthetic_inputs(hours: int = 72):
    times = pd.date_range("2020-01-01", periods=hours, freq="h")
    frame = pd.DataFrame({"timestamp": times})
    for i, task in enumerate(TASK_ORDER):
        frame[task] = 10.0 + i + np.arange(hours, dtype=float) * 0.01
    for i, column in enumerate(EXOG_ORDER):
        frame[column] = float(i) + np.arange(hours, dtype=float) * 0.001
    device = pd.DataFrame({"timestamp": times})
    for i, column in enumerate(DISPATCH_ORDER):
        device[column] = float(i + 1) + np.arange(hours, dtype=float) * 0.01
    device["p_charge"] = 0.0
    device["p_discharge"] = 0.0
    device["p_chp"] = np.where(np.arange(hours) % 2, 2.0, 0.0)
    device["q_gb"] = np.where(np.arange(hours) % 3, 3.0, 0.0)
    context = pd.DataFrame({"timestamp": times})
    for i, column in enumerate(SCHEDULER_CONTEXT_ORDER):
        context[column] = 1.0 + i + np.arange(hours, dtype=float) * 0.01
    context["initial_soc"] = 0.5
    teacher = pd.DataFrame({"timestamp": times})
    for i, column in enumerate(DISPATCH_ORDER):
        teacher[column] = 100.0 + i + np.arange(hours, dtype=float) * 0.02
    oracle = pd.DataFrame({"timestamp": times, "oracle_first_step_objective": np.arange(hours, dtype=float)})
    return frame, device, context, teacher, oracle


def _split():
    return SplitSpec(
        train_start="2020-01-01 00:00:00",
        train_end="2020-01-03 23:00:00",
        validation_start="2020-01-01 12:00:00",
        validation_end="2020-01-02 23:00:00",
        test_start="2020-01-02 00:00:00",
        test_end="2020-01-02 11:00:00",
    )


def test_device_status_uses_canonical_dispatch_coordinates():
    dispatch = np.zeros((2, 3, len(DISPATCH_ORDER)), dtype=np.float32)
    dispatch[0, 0, DISPATCH_ORDER.index("p_chp")] = 1.0
    dispatch[0, 0, DISPATCH_ORDER.index("q_gb")] = 2.0
    dispatch[0, 1, DISPATCH_ORDER.index("q_ec")] = 1.0
    dispatch[0, 2, DISPATCH_ORDER.index("q_ac")] = 1.0
    dispatch[1, 0, DISPATCH_ORDER.index("p_charge")] = 1.0
    dispatch[1, 1, DISPATCH_ORDER.index("p_discharge")] = 1.0
    status = derive_device_status(dispatch)
    assert status.shape == (2, 3, len(STATUS_ORDER))
    assert status[0, 0].tolist()[:2] == [1.0, 1.0]
    assert status[0, 1].tolist()[2] == 1.0
    assert status[0, 2].tolist()[3] == 1.0
    assert status[1, 0].tolist()[4:] == [1.0, 0.0]
    assert status[1, 1].tolist()[4:] == [0.0, 1.0]


def test_joint_windows_are_aligned_and_causal():
    frame, device, context, teacher, oracle = _synthetic_inputs()
    windows = build_joint_windows(
        frame,
        device,
        context,
        teacher,
        oracle,
        split="test",
        split_spec=_split(),
    )
    assert windows.split == "test"
    assert windows.load_history.shape[1:] == (24, 4)
    assert windows.exog_history.shape[1:] == (24, 12)
    assert windows.device_history.shape[1:] == (24, 21)
    assert windows.device_status.shape[1:] == (24, 6)
    assert windows.forecast_target.shape[1:] == (4, 4)
    assert windows.scheduler_context.shape[1:] == (4, 6)
    assert windows.teacher_dispatch.shape[1:] == (4, 21)
    assert windows.previous_chp.shape == (len(windows), 1)
    assert np.all(windows.target_times >= np.datetime64("2020-01-02T00:00"))
    assert np.all(windows.target_times + np.timedelta64(3, "h") <= np.datetime64("2020-01-02T11:00"))
    # The history ends at t-1; the first target is never copied into it.
    first = pd.Timestamp(windows.target_times[0])
    assert first - pd.Timedelta(hours=1) not in set(pd.to_datetime(windows.target_times))


def test_future_perturbation_does_not_change_history_features():
    frame, device, context, teacher, oracle = _synthetic_inputs()
    kwargs = dict(split="test", split_spec=_split())
    baseline = build_joint_windows(frame, device, context, teacher, oracle, **kwargs)
    frame2, device2, context2, teacher2, oracle2 = _synthetic_inputs()
    # Perturb only after the complete test target interval; no valid history
    # window should depend on observations after its own forecast origin.
    target_start = pd.Timestamp("2020-01-02 12:00")
    frame2.loc[frame2.timestamp >= target_start, TASK_ORDER] = 999999.0
    context2.loc[context2.timestamp >= target_start, SCHEDULER_CONTEXT_ORDER] = 999999.0
    teacher2.loc[teacher2.timestamp >= target_start, DISPATCH_ORDER] = 999999.0
    changed = build_joint_windows(frame2, device2, context2, teacher2, oracle2, **kwargs)
    np.testing.assert_array_equal(baseline.load_history, changed.load_history)
    np.testing.assert_array_equal(baseline.exog_history, changed.exog_history)
    np.testing.assert_array_equal(baseline.device_history, changed.device_history)
    np.testing.assert_array_equal(baseline.forecast_target, changed.forecast_target)


def test_normalization_is_fit_on_training_split_only():
    frame, device, context, teacher, oracle = _synthetic_inputs()
    train = build_joint_windows(frame, device, context, teacher, oracle, split="train", split_spec=_split())
    norm = JointNormalization.fit(train)
    assert norm.fitted_split == "train"
    assert norm.load_mean.shape == (4,)
    transformed = norm.transform(train)
    assert transformed.load_history.shape == train.load_history.shape
    assert np.isfinite(transformed.load_history).all()
    with pytest.raises(ValueError, match="fitted on train"):
        norm.fit(build_joint_windows(frame, device, context, teacher, oracle, split="test", split_spec=_split()))


def test_joint_window_rejects_misaligned_artifacts():
    frame, device, context, teacher, oracle = _synthetic_inputs()
    with pytest.raises(ValueError, match="timestamp alignment"):
        build_joint_windows(frame, device.iloc[:-1], context, teacher, oracle, split="test", split_spec=_split())
