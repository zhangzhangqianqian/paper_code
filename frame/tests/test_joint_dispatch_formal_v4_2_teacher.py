from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_2_teacher import (
    TeacherCacheMismatch,
    TeacherKeyV42,
    build_same_information_teacher_v42,
    load_teacher_overlay,
    save_teacher_overlay,
)
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries, materialize_state_windows
from src.scheduling.dispatch_lp import DispatchResult
from src.scheduling.dispatch_schema import VARIABLES


def _base() -> FormalV4BaseSeries:
    n = 48
    times = np.datetime64("2015-01-01T00:00") + np.arange(n).astype("timedelta64[h]")
    load = np.ones((n, 16), dtype=np.float64)
    load[:, :4] = np.arange(n)[:, None] + np.array([1.0, 2.0, 3.0, 4.0])
    renewable = np.column_stack((np.arange(n) + 1.0, np.arange(n) + 2.0))
    persistence = np.vstack((renewable[:1], renewable[:-1]))
    prices = np.ones((n, 3), dtype=np.float64)
    return FormalV4BaseSeries(load, persistence, renewable, prices, times, "train")


def _window() -> object:
    receipt = {"gate0_authorized": True, "capacity_scenario_hash": "cap"}
    split = materialize_state_windows(_base(), np.zeros((48, len(VARIABLES))), capacity_receipt=receipt)
    return split


def _fake_solver(inputs):
    horizon = inputs.demand.shape[0]
    values = {name: np.zeros(horizon, dtype=np.float64) for name in VARIABLES}
    values["slack_e"] = inputs.demand[:, 0].copy()
    values["slack_c"] = inputs.demand[:, 1].copy()
    values["slack_h"] = inputs.demand[:, 2].copy()
    return DispatchResult("optimal", "fake", 7.0, values, {"electricity": 0.0, "cooling": 0.0, "heating": 0.0}, 0.0)


def _stage(window):
    # The gas component is intentionally different from the realised target.
    return np.full((4, 4), 5.0, dtype=np.float64)


def _key(seed: int = 2026) -> TeacherKeyV42:
    return TeacherKeyV42(
        seed=seed, split="train", timestamp_sha256="1" * 64, state_sha256="2" * 64,
        capacity_sha256="3" * 64, benchmark_sha256="4" * 64, normalization_sha256="5" * 64,
        stage_p_checkpoint_sha256="6" * 64, implementation_sha256="7" * 64,
        source_manifest_sha256="8" * 64,
    )


def test_teacher_uses_predictions_and_persistence_not_realized_future():
    split = _window()
    overlay = build_same_information_teacher_v42(
        _stage, split, seed=2026, solver=_fake_solver,
        stage_p_checkpoint_sha256="6" * 64,
    )
    assert np.array_equal(overlay.rigid_demand, overlay.predicted_forecast[..., :3])
    assert np.array_equal(overlay.renewable_plan[0], np.repeat(split.renewable_history[0, -1:, :], 4, axis=0))
    assert not np.array_equal(overlay.rigid_demand, split.renewable_realized[:, :1, :3] if False else split.forecast_target[:, :3])


def test_teacher_ignores_realized_future_when_prediction_is_fixed():
    split = _window()
    first = build_same_information_teacher_v42(_stage, split, seed=2026, solver=_fake_solver, stage_p_checkpoint_sha256="6" * 64)
    mutated = replace(split, renewable_realized=split.renewable_realized + 1000.0, forecast_target=split.forecast_target + 1000.0)
    second = build_same_information_teacher_v42(_stage, mutated, seed=2026, solver=_fake_solver, stage_p_checkpoint_sha256="6" * 64)
    np.testing.assert_array_equal(first.predicted_forecast, second.predicted_forecast)
    np.testing.assert_array_equal(first.renewable_plan, second.renewable_plan)
    np.testing.assert_array_equal(first.dispatch, second.dispatch)


def test_teacher_cache_rejects_checkpoint_or_state_hash_change(tmp_path: Path):
    split = _window()
    overlay = build_same_information_teacher_v42(_stage, split, seed=2026, solver=_fake_solver, stage_p_checkpoint_sha256="6" * 64)
    save_teacher_overlay(tmp_path, overlay)
    with pytest.raises(TeacherCacheMismatch):
        load_teacher_overlay(tmp_path, replace(overlay.key, checkpoint_sha256="0" * 64))
    with pytest.raises(TeacherCacheMismatch):
        load_teacher_overlay(tmp_path, replace(overlay.key, state_sha256="0" * 64))
    loaded = load_teacher_overlay(tmp_path, overlay.key)
    np.testing.assert_array_equal(loaded.dispatch, overlay.dispatch)


def test_teacher_key_exposes_checkpoint_alias():
    assert _key().checkpoint_sha256 == "6" * 64
