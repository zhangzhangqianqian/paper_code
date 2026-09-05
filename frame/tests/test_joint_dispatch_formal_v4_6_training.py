from __future__ import annotations

import numpy as np
import torch

from src.joint_dispatch.formal_v4_4_model import ResidualGatedRSCPFModel
from src.joint_dispatch.formal_v4_4_regime import fit_thermal_prior
from src.joint_dispatch.formal_v4_5_training import StageBudgetV45, StageReceiptV45
from src.joint_dispatch.formal_v4_6_contract import load_formal_v4_6_contract
from src.joint_dispatch.formal_v4_6_risk import fit_risk_caps_v46
from src.joint_dispatch.formal_v4_6_training import (
    initialize_v46_parent,
    run_stage_j_pair_v46,
    run_v46_calibration,
    run_zero_risk_control_v46,
)
from src.joint_dispatch.model import JointForecastDispatchModel


CONTRACT = load_formal_v4_6_contract("configs/joint_forecast_dispatch_formal_v4_6.json")


def _prior_and_risk():
    target = np.zeros((12, 4, 4), dtype=np.float64)
    target[:, :, 0] = 10.0
    target[4:8, :, 1] = 2.0
    target[8:12, :, 2] = 2.0
    history = np.zeros((12, 24, 4), dtype=np.float64)
    times = np.asarray([f"{year}-01-01" for year in (2015, 2016, 2017, 2018)] * 3, dtype="datetime64[ns]")
    prior = fit_thermal_prior(target, history, times)
    risk = fit_risk_caps_v46(np.zeros_like(target), target, times, "early_stop", 0.90, CONTRACT.contract_sha256, "1" * 64)
    return prior, risk


def _batch(batch_size: int = 2):
    target = torch.zeros(batch_size, 4, 4)
    target[..., 0] = 10.0
    target[0, 1, 1] = 2.0
    target[1, 2, 2] = 2.0
    return {
        "load_history": torch.ones(batch_size, 24, 4), "exog_history": torch.zeros(batch_size, 24, 12),
        "device_history": torch.zeros(batch_size, 24, 17), "activity_history": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.cat((torch.zeros(batch_size, 4, 5), torch.full((batch_size, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.ones(batch_size, 1), "last_thermal_regime": torch.zeros(batch_size, dtype=torch.long),
        "target_normalized": target, "target_physical": target,
        "realized_renewables": torch.zeros(batch_size, 4, 2), "initial_soc": torch.full((batch_size, 1), 0.5),
        "teacher_dispatch": torch.zeros(batch_size, 4, 21),
    }


def _stage_s():
    model = ResidualGatedRSCPFModel(
        transition_probability=torch.full((4, 3, 3), 1.0 / 3.0),
        decoder_parameters=JointForecastDispatchModel._test_parameters(), task_mean=torch.zeros(4), task_scale=torch.ones(4),
        physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10), dropout=0.0,
    )
    return StageReceiptV45(
        stage="S", mode="scheduler_imitation", parent_sha256="0" * 64, final_sha256="0" * 64,
        best_sha256="0" * 64, terminal_sha256="0" * 64, epochs=1, best_epoch=0, optimizer_steps=1,
        forecast_optimizer_steps=0, loss_history=(0.0,), validation_history=({"metric": 0.0},),
        stopping_reason="budget_exhausted", selection_metric=0.0, model=model,
    )


def _budget():
    return StageBudgetV45(
        max_epochs=1, minimum_epochs=1, p0_lr=1e-3, p1_base_lr=2e-4, p1_head_lr=1e-3,
        s_scheduler_lr=1e-3, j_forecaster_lr=2e-4, j_head_lr=5e-4, j_scheduler_lr=5e-4,
        weight_decay=1e-5, max_grad_norm=1.0, inactive_leakage_target=0.25, ramp_epochs=5, patience=1,
    )


def _parameters():
    result = JointForecastDispatchModel._test_parameters()
    result.update({"grid_energy_price": 1.0, "gas_energy_price": 1.0, "carbon_price_default": 0.1})
    return result


def test_joint_and_decoupled_start_from_byte_identical_parent():
    prior, risk = _prior_and_risk()
    pair = run_stage_j_pair_v46(_stage_s(), {"train": [_batch()], "early_stop": [_batch()]}, _budget(), prior=prior, parameters=_parameters(), contract=CONTRACT, risk_cap=risk, risk_multiplier=1.0, seed=2026)
    assert pair.joint.parent_sha256 == pair.decoupled.parent_sha256
    assert pair.joint.optimizer_groups["risk"]["steps"] == pair.decoupled.optimizer_groups["risk"]["steps"]
    assert pair.joint.optimizer_groups["scheduler"]["steps"] == pair.decoupled.optimizer_groups["scheduler"]["steps"]


def test_decoupled_cuts_only_decision_edges_into_forecast():
    prior, risk = _prior_and_risk()
    pair = run_stage_j_pair_v46(_stage_s(), {"train": [_batch()], "early_stop": [_batch()]}, _budget(), prior=prior, parameters=_parameters(), contract=CONTRACT, risk_cap=risk, risk_multiplier=1.0, seed=2026)
    assert pair.joint.gradient_norms["decision_to_base"] > 0
    assert pair.joint.gradient_norms["decision_to_risk"] > 0
    assert pair.decoupled.gradient_norms["decision_to_base"] <= 1e-12
    assert pair.decoupled.gradient_norms["decision_to_gate"] <= 1e-12
    assert pair.decoupled.gradient_norms["decision_to_magnitude"] <= 1e-12
    assert pair.decoupled.gradient_norms["decision_to_risk"] > 0


def test_calibration_is_train_only_and_zero_risk_is_exact():
    prior, risk = _prior_and_risk()
    loaders = {"train": [_batch()], "early_stop": [_batch()]}
    calibration = run_v46_calibration(_stage_s(), loaders, _budget(), prior=prior, parameters=_parameters(), contract=CONTRACT, risk_cap=risk, seed=2026)
    assert calibration.selected_multiplier in (0.5, 1.0, 2.0)
    assert calibration.selected.validation.eligible
    assert calibration.selection_role == "early_stop"
    assert calibration.selection_year_accessed is False
    zero = run_zero_risk_control_v46(_stage_s(), loaders, _budget(), prior=prior, parameters=_parameters(), contract=CONTRACT, risk_cap=risk, seed=2026)
    assert zero.risk_trainable is False
    assert np.count_nonzero(zero.risk_adjustment) == 0
    assert zero.scheduler_demand_sha256 == zero.forecast_nominal_sha256
