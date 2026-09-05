from __future__ import annotations

import torch
from torch import nn
from types import SimpleNamespace

from src.joint_dispatch.formal_v4_5_contract import load_formal_v4_5_contract
from src.joint_dispatch.formal_v4_5_training import StageBudgetV45, run_stage_j_pair_v45, run_stage_with_validation_v45


def _model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Linear(1, 1, bias=False)


def _batches(values: list[float]) -> list[dict[str, float]]:
    return [{"value": value} for value in values]


def _train_loss(model: nn.Module, batch: dict[str, float], _epoch: int) -> torch.Tensor:
    prediction = model(torch.ones((1, 1)))
    target = torch.as_tensor([[batch["value"]]], dtype=prediction.dtype)
    return (prediction - target).square().mean()


def test_best_validation_state_is_restored() -> None:
    model = _model()
    receipt = run_stage_with_validation_v45(
        model=model,
        train_batches=_batches([0.8, 0.6, 0.7]),
        validation_batches=_batches([0.0]),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_loss=_train_loss,
        validation_metric=lambda _model, _batch, epoch: [0.4, 0.2, 0.3, 0.4][epoch],
        eligibility=lambda values: values["metric"] < 1.0,
        max_epochs=4,
        minimum_epochs=1,
        patience=2,
        stage="toy",
    )
    assert receipt.best_epoch == 1
    assert receipt.stopping_reason == "early_stopped"
    assert receipt.final_sha256 == receipt.best_sha256


def test_validation_does_not_change_parameters_or_optimizer_steps() -> None:
    model = _model()
    train_batches = _batches([0.8, 0.6])
    receipt = run_stage_with_validation_v45(
        model=model,
        train_batches=train_batches,
        validation_batches=_batches([0.0]),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_loss=_train_loss,
        validation_metric=lambda _model, _batch, epoch: float(epoch),
        eligibility=lambda _values: True,
        max_epochs=2,
        minimum_epochs=1,
        patience=2,
    )
    assert receipt.optimizer_steps == len(train_batches) * receipt.epochs
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())


class _TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.tensor(0.2))
        self.gate = nn.Parameter(torch.tensor(0.1))
        self.magnitude = nn.Parameter(torch.tensor(0.1))
        self.scheduler = nn.Parameter(torch.tensor(0.1))

    def v44_parameter_groups(self):
        return {"base": (self.base,), "gate": (self.gate,), "magnitude": (self.magnitude,), "scheduler": (self.scheduler,)}

    def forward(self, *, detach_forecast_for_dispatch=False, **_inputs):
        value = self.base + self.gate + self.magnitude
        forecast = value.expand(1, 4, 4)
        dispatch_forecast = forecast.detach() if detach_forecast_for_dispatch else forecast
        dispatch = (dispatch_forecast.mean(dim=-1, keepdim=True) + self.scheduler).expand(1, 4, 21)
        return SimpleNamespace(forecast_normalized=forecast, forecast_physical=forecast, dispatch=dispatch)


def _joint_batch() -> dict[str, torch.Tensor]:
    return {
        "load_history": torch.zeros(1, 24, 4),
        "exog_history": torch.zeros(1, 24, 12),
        "device_history": torch.zeros(1, 24, 17),
        "activity_history": torch.zeros(1, 24, 6),
        "scheduler_context": torch.zeros(1, 4, 7),
        "previous_chp": torch.zeros(1, 1),
        "last_thermal_regime": torch.zeros(1, dtype=torch.long),
        "target_normalized": torch.zeros(1, 4, 4),
        "teacher_dispatch": torch.zeros(1, 4, 21),
    }


def test_joint_pair_updates_forecast_path_and_decoupled_branch_is_separated() -> None:
    contract = load_formal_v4_5_contract("configs/joint_forecast_dispatch_formal_v4_5.json")
    budget = StageBudgetV45(
        max_epochs=2, minimum_epochs=1, p0_lr=1e-3, p1_base_lr=1e-3,
        p1_head_lr=1e-3, s_scheduler_lr=1e-3, j_forecaster_lr=2e-4,
        j_head_lr=5e-4, j_scheduler_lr=5e-4, weight_decay=1e-5,
        max_grad_norm=1.0, inactive_leakage_target=0.25, ramp_epochs=5, patience=2,
    )
    parent = _TinyJointModel()
    pair = run_stage_j_pair_v45(
        SimpleNamespace(model=parent), {"train": [_joint_batch()], "early_stop": [_joint_batch()]},
        budget, prior=None, c_ref=1.0, parameters=None, contract=contract,
    )
    assert pair.joint.gradient_norms["decision_to_base"] > 0.0
    assert pair.joint.gradient_norms["decision_to_gate"] > 0.0
    assert pair.joint.gradient_norms["decision_to_magnitude"] > 0.0
    assert pair.decoupled.gradient_norms["decision_to_base"] <= 1.0e-12
    assert pair.decoupled.gradient_norms["decision_to_gate"] <= 1.0e-12
