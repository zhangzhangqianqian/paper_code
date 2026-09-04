from __future__ import annotations

from pathlib import Path

import numpy as np

from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries

from runpy import run_path


PILOT = run_path(str(Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_pilot.py"))


def _receipt():
    return {
        "stage_p_loss_decreased": True, "stage_s_loss_decreased": True, "stage_j_loss_finite": True,
        "persistent_optimizer_steps": True, "stage_s_clone_identical": True,
        "joint_forecast_decision_gradient_positive": True, "decoupled_forecast_decision_gradient_zero": True,
        "first_step_state_carry": True, "shortage_rate": 0.50,
    }


def test_pilot_requires_numerical_and_structural_checks():
    receipt = _receipt()
    assert PILOT["authorize_pilot"](receipt, shortage_rate_max=0.80) is True
    receipt["stage_s_clone_identical"] = False
    assert PILOT["authorize_pilot"](receipt, shortage_rate_max=0.80) is False


def test_pilot_receipt_is_never_paper_eligible(tmp_path: Path):
    receipt = PILOT["run_pilot"]({**_receipt(), "output_root": tmp_path, "run_id": "formal_v4_2_pilot_test"})
    assert receipt["paper_eligible"] is False
    assert receipt["authorized_gate1"] is True
    assert receipt["evaluation_year_accessed"] is False
    assert (tmp_path / "formal_v4_2_pilot_test" / "pilot" / "PILOT_RECEIPT.json").is_file()


def test_pilot_subset_is_train_only_and_seasonally_bounded():
    hours = 365 * 24
    times = np.datetime64("2015-01-01") + np.arange(hours).astype("timedelta64[h]")
    load = np.ones((hours, 16), dtype=np.float64)
    load[:, :4] += np.arange(hours)[:, None] % 24
    renew = np.ones((hours, 2), dtype=np.float64)
    prices = np.ones((hours, 3), dtype=np.float64)
    base = FormalV4BaseSeries(load, renew, renew, prices, times, "train")
    selected = PILOT["_pilot_base"](base, segment_hours=96, segments=4)
    assert selected.split == "train"
    assert len(selected.timestamps) == 384
    assert {str(value)[:4] for value in selected.timestamps} == {"2015"}
