from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from src.joint_dispatch.formal_v4_3_data import derive_thermal_regimes, fit_thermal_magnitude_statistics
from src.joint_dispatch.formal_v4_3_training import forecast_loss_v43
from src.joint_dispatch.formal_v4_3_training import run_stage_j_v43, run_stage_p_v43, run_stage_s_v43
from src.joint_dispatch.formal_v4_2_training import StageBudgetV42
from src.joint_dispatch.formal_v4_models import RegimeAwareRSCPFModel


WEIGHTS = {
    "continuous_weight": 1.0,
    "regime_weight": 1.0,
    "active_magnitude_weight": 1.0,
    "point_weight": 0.5,
    "inactive_leakage_weight": 0.25,
    "gas_task_weight": 0.25,
    "transition_window_weight": 1.5,
}


def _case() -> tuple[SimpleNamespace, dict[str, torch.Tensor], object]:
    physical = torch.tensor(
        [[[10.0, 0.0, 0.0, 2.0], [10.0, 100.0, 0.0, 2.0], [10.0, 0.0, 50.0, 2.0], [10.0, 0.0, 0.0, 2.0]]],
    )
    target = physical.clone()
    labels = derive_thermal_regimes(target.numpy())
    receipt = fit_thermal_magnitude_statistics(target.numpy(), labels)
    task_mean = torch.zeros(4)
    task_scale = torch.ones(4)
    normalized = (physical - task_mean) / task_scale
    logits = torch.full((1, 4, 3), -50.0)
    for horizon, label in enumerate(labels[0]):
        logits[0, horizon, int(label)] = 50.0
    magnitudes = torch.stack((physical[..., 1], physical[..., 2]), dim=-1)
    output = SimpleNamespace(
        forecast_normalized=normalized,
        forecast_physical=physical,
        regime_logits=logits,
        regime_probabilities=torch.softmax(logits, dim=-1),
        thermal_magnitudes=magnitudes,
    )
    batch = {
        "target_normalized": normalized,
        "target_physical": target,
        "thermal_transition_mask": torch.tensor([1.0]),
    }
    return output, batch, receipt


def test_v43_forecast_loss_is_near_zero_for_perfect_regime_and_magnitude() -> None:
    output, batch, receipt = _case()
    loss = forecast_loss_v43(output, batch, receipt, WEIGHTS)
    assert loss.total.item() < 1.0e-5


def test_inactive_leakage_penalizes_false_thermal_output() -> None:
    output, batch, receipt = _case()
    output.forecast_physical = output.forecast_physical.clone()
    output.forecast_physical[0, 0, 1] = 50.0
    loss = forecast_loss_v43(output, batch, receipt, WEIGHTS)
    assert loss.inactive_leakage.item() > 0.0


def test_simultaneous_target_is_rejected_by_training_loss() -> None:
    output, batch, receipt = _case()
    batch["target_physical"] = batch["target_physical"].clone()
    batch["target_physical"][0, 0, 1:3] = 1.0
    try:
        forecast_loss_v43(output, batch, receipt, WEIGHTS)
    except ValueError as exc:
        assert "simultaneous cooling and heating" in str(exc)
    else:
        raise AssertionError("simultaneous thermal target was accepted")


def _model_batch() -> dict[str, torch.Tensor]:
    output, batch, _ = _case()
    size = 1
    context = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.5]).reshape(1, 1, 6).expand(size, 4, 6).clone()
    batch.update({
        "load_history": torch.randn(size, 24, 4),
        "exog_history": torch.randn(size, 24, 12),
        "device_history": torch.randn(size, 24, 17),
        "activity_history": torch.zeros(size, 24, 6),
        "scheduler_context": context,
        "previous_chp": torch.zeros(size, 1),
        "teacher_dispatch": torch.zeros(size, 4, 21),
    })
    return batch


def test_v43_stage_wrappers_keep_joint_and_decoupled_modes_distinct() -> None:
    receipt = _case()[2]
    budget = StageBudgetV42(max_epochs=1, minimum_epochs=1, ramp_epochs=1)
    model = RegimeAwareRSCPFModel.for_test()
    batch = _model_batch()
    loaders = {"train": [batch]}
    stage_p = run_stage_p_v43(model, loaders, receipt, WEIGHTS, budget=budget, seed=2026)
    stage_s = run_stage_s_v43(stage_p.model, loaders, budget=budget, seed=2026)
    joint = run_stage_j_v43(stage_s.model, loaders, mode="joint", receipt=receipt, loss_weights=WEIGHTS, budget=budget, seed=2026)
    decoupled = run_stage_j_v43(stage_s.model, loaders, mode="decoupled", receipt=receipt, loss_weights=WEIGHTS, budget=budget, seed=2026)
    assert joint.mode == "joint_v43"
    assert decoupled.mode == "decoupled_v43"
    assert joint.epochs == decoupled.epochs == 1
