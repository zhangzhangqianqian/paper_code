from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.joint_dispatch.contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER
from src.joint_dispatch.data import JointWindowSplit, save_joint_split
from src.joint_dispatch.external_baseline_data import (
    ExternalBaselineBatch,
    assert_external_batch_causal,
    build_causal_error_history,
    fit_external_normalization,
    load_external_batches,
)


def _split(split_name: str = "train", n: int = 3) -> JointWindowSplit:
    rng = np.random.default_rng(7)
    return JointWindowSplit(
        load_history=rng.normal(size=(n, 24, len(TASK_ORDER))).astype(np.float32),
        exog_history=rng.normal(size=(n, 24, len(EXOG_ORDER))).astype(np.float32),
        device_history=np.abs(rng.normal(size=(n, 24, len(DISPATCH_ORDER)))).astype(np.float32),
        device_status=np.zeros((n, 24, len(STATUS_ORDER)), dtype=np.float32),
        forecast_target=rng.normal(size=(n, 4, len(TASK_ORDER))).astype(np.float32),
        scheduler_context=np.abs(rng.normal(size=(n, 4, 6))).astype(np.float32),
        previous_chp=np.abs(rng.normal(size=(n, 1))).astype(np.float32),
        teacher_dispatch=np.abs(rng.normal(size=(n, 4, len(DISPATCH_ORDER)))).astype(np.float32),
        oracle_first_step_objective=np.abs(rng.normal(size=(n,))).astype(np.float32),
        target_times=np.datetime64("2025-01-01T00") + np.arange(n).astype("timedelta64[h]"),
        split=split_name,
    )


def test_external_batch_preserves_canonical_shapes_and_labels_are_separate() -> None:
    batch = ExternalBaselineBatch.from_split(_split())
    assert tuple(batch.load_history.shape[1:]) == (24, 4)
    assert tuple(batch.exog_history.shape[1:]) == (24, 12)
    assert tuple(batch.device_history.shape[1:]) == (24, 21)
    assert tuple(batch.device_status.shape[1:]) == (24, 6)
    assert tuple(batch.forecast_target.shape[1:]) == (4, 4)
    assert tuple(batch.teacher_dispatch.shape[1:]) == (4, 21)
    assert_external_batch_causal(batch)


def test_future_target_mutation_does_not_change_causal_error_history() -> None:
    baseline = _split()
    changed = _split()
    changed.forecast_target[...] = 999.0
    np.testing.assert_array_equal(build_causal_error_history(baseline), build_causal_error_history(changed))


def test_normalization_is_train_only_and_transformation_is_finite() -> None:
    train = _split("train")
    normalization = fit_external_normalization(train)
    assert normalization.fitted_split == "train"
    assert normalization.load_mean.shape == (4,)
    transformed = normalization.transform(train)
    assert np.isfinite(transformed.load_history).all()
    with pytest.raises(ValueError, match="test split"):
        normalization.transform(_split("test"))


def test_loader_is_deterministic_and_rejects_test_split(tmp_path: Path) -> None:
    path = tmp_path / "train.npz"
    save_joint_split(_split(), path)
    first = list(load_external_batches(path, batch_size=2, shuffle=True, seed=11))
    second = list(load_external_batches(path, batch_size=2, shuffle=True, seed=11))
    assert len(first) == 2
    assert torch.equal(first[0].load_history, second[0].load_history)
    test_path = tmp_path / "test.npz"
    save_joint_split(_split("test"), test_path)
    with pytest.raises(ValueError, match="test split"):
        list(load_external_batches(test_path, batch_size=2, shuffle=False, seed=11))
