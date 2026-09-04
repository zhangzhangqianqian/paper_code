from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.joint_dispatch.formal_v4_2_training import (
    StageBudgetV42,
    refresh_rollin_sample,
    run_stage_j_pair,
    run_stage_j,
    run_training_seed_v42,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.forecaster = nn.Linear(1, 4)
        self.scheduler = nn.Linear(4, 21)
        self.decoder_parameters = {}

    def forecaster_parameters(self):
        return tuple(self.forecaster.parameters())

    def scheduler_parameters(self):
        return tuple(self.scheduler.parameters())

    def forward(self, *, detach_forecast_for_dispatch=False, **inputs):
        x = inputs["load_history"][:, -1:, :1]
        forecast = self.forecaster(x.squeeze(1)).unsqueeze(1).expand(-1, 4, -1)
        forecast_physical = forecast
        route = forecast.detach() if detach_forecast_for_dispatch else forecast
        dispatch = self.scheduler(route)
        return SimpleNamespace(forecast_normalized=forecast, forecast_physical=forecast_physical, dispatch=dispatch)


def _batch(with_realized=False):
    batch = {
        "load_history": torch.ones(2, 24, 4), "exog_history": torch.zeros(2, 24, 12),
        "device_history": torch.zeros(2, 24, 17), "activity_history": torch.zeros(2, 24, 6),
        "scheduler_context": torch.zeros(2, 4, 6), "previous_chp": torch.zeros(2, 1),
        "target_normalized": torch.ones(2, 4, 4), "target_physical": torch.ones(2, 4, 4),
        "teacher_dispatch": torch.zeros(2, 4, 21),
    }
    if with_realized:
        batch.update({"realized_renewables": torch.zeros(2, 4, 2), "initial_soc": torch.full((2, 1), 0.5)})
    return batch


def test_curriculum_reaches_registered_terminal_weight():
    budget = StageBudgetV42(max_epochs=30, minimum_epochs=18, decision_start=0.05, decision_final=1.0, ramp_epochs=18)
    assert budget.weights(epoch=0).decision == pytest.approx(0.05)
    assert budget.weights(epoch=18).decision == pytest.approx(1.0)


def test_stage_j_pair_has_exact_gradient_boundary():
    joint, decoupled = run_stage_j_pair(ToyModel(), _batch(), budget=StageBudgetV42(max_epochs=1, minimum_epochs=1, ramp_epochs=1))
    assert joint.decision_forecaster_gradient_norm > 0.0
    assert decoupled.decision_forecaster_gradient_norm == pytest.approx(0.0)
    assert joint.scheduler_gradient_norm > 0.0
    assert decoupled.scheduler_gradient_norm > 0.0


def test_refreshed_rollin_detaches_history_and_zeros_stale_imitation():
    value = torch.ones(1, 24, 17, requires_grad=True)
    refreshed = refresh_rollin_sample({"device_history": value, "imitation_weight": 1.0}, recompute_teacher=False)
    assert not refreshed["device_history"].requires_grad
    assert refreshed["imitation_weight"] == 0.0


def test_stage_runner_persists_optimizer_across_batches_and_order():
    model = ToyModel()
    receipt = run_training_seed_v42(
        seed=2026, model=model, loaders={"train": [_batch(), _batch()]}, teacher_overlay=object(),
        budget=StageBudgetV42(max_epochs=2, minimum_epochs=1, ramp_epochs=1),
    )
    assert receipt.stage_events == ("P_complete", "teacher_complete", "S_complete", "clone_verified", "J_complete")
    assert receipt.optimizer_final_steps["P"] > 1
    assert receipt.optimizer_final_steps["S"] > 1
    assert receipt.optimizer_final_steps["J_joint"] > 1


def test_stage_results_record_epoch_loss_history():
    model = ToyModel()
    budget = StageBudgetV42(max_epochs=2, minimum_epochs=1, ramp_epochs=1)
    from src.joint_dispatch.formal_v4_2_training import run_stage_p, run_stage_s
    p = run_stage_p(model, {"train": [_batch(), _batch()]}, budget=budget)
    assert len(p.loss_history) == 2
    s = run_stage_s(p.model, {"train": [_batch(), _batch()]}, budget=budget)
    assert len(s.loss_history) == 2
    j = run_stage_j(s.model, {"train": [_batch()]}, mode="joint", budget=budget)
    assert len(j.loss_history) == 2
    assert j.optimizer_steps > 1
