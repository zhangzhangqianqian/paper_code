"""Validation-aware stage primitives for the formal-v4.5 repair."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .formal_v4_4_training import seed_everything, sha256_state_dict


@dataclass(frozen=True)
class StageValidationV45:
    metric: float
    eligible: bool
    details: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StageReceiptV45:
    stage: str
    mode: str
    parent_sha256: str
    final_sha256: str
    best_sha256: str
    terminal_sha256: str
    epochs: int
    best_epoch: int
    optimizer_steps: int
    forecast_optimizer_steps: int
    loss_history: tuple[float, ...]
    validation_history: tuple[Mapping[str, float], ...]
    stopping_reason: str
    selection_metric: float
    model: nn.Module | None = field(default=None, repr=False, compare=False)


def _finite_scalar(value: Any, label: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise FloatingPointError(f"non-finite {label}")
    return scalar


def run_stage_with_validation_v45(
    model: nn.Module,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    optimizer_factory: Callable[[Sequence[nn.Parameter]], torch.optim.Optimizer],
    train_loss: Callable[[nn.Module, Mapping[str, Any], int], Tensor],
    validation_metric: Callable[[nn.Module, Mapping[str, Any], int], float],
    eligibility: Callable[[Mapping[str, float]], bool],
    max_epochs: int,
    minimum_epochs: int,
    patience: int,
    *,
    stage: str = "stage",
    mode: str = "validation",
    seed: int = 2026,
    forecast_steps: bool = False,
) -> StageReceiptV45:
    """Train a stage and restore its best finite validation checkpoint.

    Validation is read-only and never contributes an optimizer step. A stage
    fails closed when every validation epoch is ineligible or non-finite.
    """

    if max_epochs < 1 or minimum_epochs < 1 or minimum_epochs > max_epochs:
        raise ValueError("invalid v4.5 epoch budget")
    if patience < 1:
        raise ValueError("patience must be positive")
    train_batches = list(train_batches)
    validation_batches = list(validation_batches)
    if not train_batches or not validation_batches:
        raise ValueError("v4.5 stages require non-empty train and validation batches")
    params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not params:
        raise ValueError("v4.5 stage has no trainable parameters")
    seed_everything(seed)
    parent_sha256 = sha256_state_dict(model)
    optimizer = optimizer_factory(params)
    loss_history: list[float] = []
    validation_history: list[Mapping[str, float]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_metric = float("inf")
    stale_epochs = 0
    optimizer_steps = 0
    stopping_reason = "budget_exhausted"

    for epoch in range(max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = train_loss(model, batch, epoch)
            if loss.ndim != 0 or not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(f"non-finite {stage} training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer_steps += 1
            epoch_losses.append(float(loss.detach()))
        loss_history.append(float(np.mean(epoch_losses)))

        model.eval()
        with torch.no_grad():
            metrics = [_finite_scalar(validation_metric(model, batch, epoch), "validation metric") for batch in validation_batches]
        metric = float(np.mean(metrics))
        details = {"metric": metric, "validation_batches": float(len(metrics)), "epoch": float(epoch)}
        is_eligible = bool(eligibility(details))
        validation_history.append(dict(details, eligible=float(is_eligible)))
        if is_eligible and metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch + 1 >= minimum_epochs and stale_epochs >= patience:
            stopping_reason = "early_stopped"
            break

    if best_state is None or best_epoch < 0:
        raise RuntimeError(f"{stage} has no eligible finite validation checkpoint")
    terminal_sha256 = sha256_state_dict(model)
    model.load_state_dict(best_state, strict=True)
    final_sha256 = sha256_state_dict(model)
    return StageReceiptV45(
        stage=stage, mode=mode, parent_sha256=parent_sha256,
        final_sha256=final_sha256, best_sha256=final_sha256,
        terminal_sha256=terminal_sha256,
        epochs=len(loss_history), best_epoch=best_epoch,
        optimizer_steps=optimizer_steps,
        forecast_optimizer_steps=optimizer_steps if forecast_steps else 0,
        loss_history=tuple(loss_history),
        validation_history=tuple(validation_history),
        stopping_reason=stopping_reason, selection_metric=best_metric,
        model=model,
    )


__all__ = ["StageReceiptV45", "StageValidationV45", "run_stage_with_validation_v45"]
