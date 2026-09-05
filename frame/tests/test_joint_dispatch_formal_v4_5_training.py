from __future__ import annotations

import torch
from torch import nn

from src.joint_dispatch.formal_v4_5_training import run_stage_with_validation_v45


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
