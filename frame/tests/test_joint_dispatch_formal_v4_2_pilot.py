from __future__ import annotations

from pathlib import Path

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
