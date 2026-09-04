from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_2_data import (
    apply_normalization,
    fit_train_normalization,
    select_gate1_origins,
)
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries, FormalV4WindowSplit, materialize_state_windows
from src.joint_dispatch.contract import DISPATCH_ORDER


def _split(split: str = "train", n: int = 48) -> FormalV4WindowSplit:
    base = np.arange(n, dtype=np.float64)[:, None]
    load_history = np.broadcast_to(base[:, None, :4], (n, 24, 4)).copy()
    exog_history = np.ones((n, 24, 12), dtype=np.float64)
    renewable_history = np.ones((n, 24, 2), dtype=np.float64)
    device_history = np.ones((n, 24, 17), dtype=np.float64)
    activity_history = np.zeros((n, 24, 6), dtype=np.float64)
    target = np.ones((n, 4, 4), dtype=np.float64)
    target[:, :, 1] = np.where(np.arange(n)[:, None] % 3 == 0, 2.0, 0.0)
    target[:, :, 2] = np.where(np.arange(n)[:, None] % 4 == 0, 3.0, 0.0)
    rigid = target[:, :, :3].copy()
    renew_forecast = np.ones((n, 4, 2), dtype=np.float64)
    renew_realized = np.ones((n, 4, 2), dtype=np.float64)
    prices = np.ones((n, 4, 3), dtype=np.float64)
    initial_soc = np.full((n, 1), 0.5, dtype=np.float64)
    previous_chp = np.ones((n, 1), dtype=np.float64)
    if split == "selection":
        timestamps = np.array([np.datetime64("2019-01-01") + np.timedelta64(i * 30, "D") for i in range(n)])
    else:
        timestamps = np.array([np.datetime64("2018-01-01") + np.timedelta64(i, "D") for i in range(n)])
    return FormalV4WindowSplit(
        load_history,
        exog_history,
        renewable_history,
        device_history,
        activity_history,
        target,
        rigid,
        renew_forecast,
        renew_realized,
        prices,
        initial_soc,
        previous_chp,
        timestamps,
        np.array(["trajectory"] * n),
        np.array([f"{i:064x}" for i in range(n)]),
        split=split,
    )


def test_normalization_uses_train_only_and_handles_zero_scale() -> None:
    train = _split("train")
    receipt = fit_train_normalization(train)
    repeated = fit_train_normalization(train)
    assert repeated.receipt_sha256 == receipt.receipt_sha256
    assert np.all(receipt.field_scale["activity"] == 1.0)
    shifted = replace(train, forecast_target=train.forecast_target + 1.0e9)
    normalized = apply_normalization(shifted, receipt)
    assert np.isfinite(normalized.target).all()
    assert normalized.scheduler_context.shape == (len(train), 4, 6)
    np.testing.assert_array_equal(normalized.scheduler_context[..., :2], shifted.renewable_forecast.astype(np.float32))
    np.testing.assert_array_equal(normalized.scheduler_context[..., 2:5], shifted.prices_and_weights.astype(np.float32))
    np.testing.assert_array_equal(normalized.activity_history, shifted.activity_history.astype(np.float32))
    assert normalized.target_normalized.shape == normalized.target.shape


def test_normalization_rejects_selection_fit() -> None:
    with pytest.raises(ValueError, match="train split"):
        fit_train_normalization(_split("selection"))


def test_v42_capacity_receipt_materializes_state_windows() -> None:
    n = 40
    times = np.datetime64("2015-01-01") + np.arange(n).astype("timedelta64[h]")
    tasks_and_exog = np.ones((n, 16), dtype=np.float64)
    renewable = np.ones((n, 2), dtype=np.float64)
    base = FormalV4BaseSeries(tasks_and_exog, renewable, renewable, np.ones((n, 3)), times, "train")
    receipt = {"schema": "formal-v4.2-capacity-freeze-v1", "status": "pass", "selected": {"multiplier": 1.0}}
    materialized = materialize_state_windows(
        base, np.zeros((n, len(DISPATCH_ORDER))), capacity_receipt=receipt,
        bess_energy_capacity=1.0,
    )
    assert materialized.split == "train"


def test_state_windows_keep_complete_24_hour_device_trajectory() -> None:
    split = _split()
    assert split.device_history.shape[1:] == (24, 17)
    assert split.activity_history.shape[1:] == (24, 6)


def test_gate1_manifest_covers_activity_and_seasons() -> None:
    selection = _split("selection")
    design = {
        "total": 24,
        "cooling_active": 6,
        "heating_active": 6,
        "winter": 4,
        "summer": 4,
        "shoulder": 2,
        "chronological_remaining": 2,
        "minimum_cooling_active_fraction": 0.20,
        "minimum_heating_active_fraction": 0.20,
    }
    manifest = select_gate1_origins(selection, design)
    assert set(manifest.strata) >= {"cooling_active", "heating_active", "winter", "summer", "shoulder"}
    assert manifest.cooling_active_fraction >= 0.20
    assert manifest.heating_active_fraction >= 0.20
    assert len(np.unique(manifest.origin_indices)) == len(manifest.origin_indices)


def test_gate1_manifest_rejects_activity_poor_selection() -> None:
    selection = _split("selection")
    poor_target = np.ones_like(selection.forecast_target)
    poor_target[:, :, 1:3] = 0.0
    selection = replace(selection, forecast_target=poor_target)
    selection = replace(selection, rigid_demand=selection.forecast_target[:, :, :3])
    design = {"total": 12, "cooling_active": 3, "heating_active": 3, "winter": 2, "summer": 2, "shoulder": 1, "chronological_remaining": 1}
    with pytest.raises(ValueError, match="activity coverage"):
        select_gate1_origins(selection, design)
