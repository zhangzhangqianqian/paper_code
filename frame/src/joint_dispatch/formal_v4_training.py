"""Fair staged training utilities for the formal-v4 model paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .formal_v4_models import RSCPFModel
from .formal_v4_objective import formal_v4_curriculum_weights, formal_v4_joint_loss


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class GradientBoundaryReceipt:
    mode: str
    forecaster_decision_gradient_norm: float
    scheduler_decision_gradient_norm: float


@dataclass(frozen=True)
class StageCheckpointReceipt:
    stage: str
    seed: int
    mode: str
    updated_parameters: tuple[str, ...]
    forecaster_parameters: tuple[str, ...]
    scheduler_parameters: tuple[str, ...]
    forecaster_gradient_norm: float
    scheduler_gradient_norm: float
    decision_forecaster_gradient_norm: float
    loss: float
    model_state_hash: str
    checkpoint_path: str = ""
    test_set_accessed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _groups(model: RSCPFModel) -> tuple[tuple[str, ...], tuple[str, ...]]:
    forecast_ids = {id(parameter) for parameter in model.forecaster_parameters()}
    scheduler_ids = {id(parameter) for parameter in model.scheduler_parameters()}
    names = dict(model.named_parameters())
    forecaster = tuple(name for name, parameter in names.items() if id(parameter) in forecast_ids)
    scheduler = tuple(name for name, parameter in names.items() if id(parameter) in scheduler_ids)
    return forecaster, scheduler


def configure_stage_j(model: RSCPFModel, mode: Literal["joint", "decoupled"]) -> None:
    if mode not in {"joint", "decoupled"}:
        raise ValueError("mode must be joint or decoupled")
    for parameter in model.forecaster_parameters():
        parameter.requires_grad_(mode == "joint")
    for parameter in model.scheduler_parameters():
        parameter.requires_grad_(True)


def _snapshot(model: torch.nn.Module) -> dict[str, Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def _updated(before: Mapping[str, Tensor], model: torch.nn.Module, *, atol: float = 0.0) -> tuple[str, ...]:
    changed = []
    for name, parameter in model.named_parameters():
        if not torch.allclose(before[name], parameter.detach(), rtol=0.0, atol=atol):
            changed.append(name)
    return tuple(changed)


def _grad_norm(parameters: Sequence[Tensor]) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _forward_inputs(batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: batch[name] for name in ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")}


def _receipt(model: RSCPFModel, *, stage: str, seed: int, mode: str, before: Mapping[str, Tensor], loss: Tensor, forecaster_gradient_norm: float, scheduler_gradient_norm: float, decision_forecaster_gradient_norm: float, checkpoint_path: str = "") -> StageCheckpointReceipt:
    forecast_names, scheduler_names = _groups(model)
    return StageCheckpointReceipt(
        stage=stage, seed=int(seed), mode=mode, updated_parameters=_updated(before, model),
        forecaster_parameters=forecast_names, scheduler_parameters=scheduler_names,
        forecaster_gradient_norm=float(forecaster_gradient_norm), scheduler_gradient_norm=float(scheduler_gradient_norm),
        decision_forecaster_gradient_norm=float(decision_forecaster_gradient_norm), loss=float(loss.detach()),
        model_state_hash=_model_hash(model), checkpoint_path=str(checkpoint_path), test_set_accessed=False,
    )


def _step_optimizer(model: RSCPFModel, parameters: Sequence[Tensor], loss: Tensor, *, max_grad_norm: float = 1.0) -> tuple[float, float]:
    optimizer = torch.optim.AdamW(list(parameters), lr=1.0e-3, weight_decay=1.0e-4)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_norm = _grad_norm(parameters)
    torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)
    optimizer.step()
    return total_norm, max_grad_norm


def train_stage_p(model: RSCPFModel, batch: Mapping[str, Tensor], parameters: Mapping[str, Any], *, seed: int, max_grad_norm: float = 1.0, checkpoint_path: str | Path = "") -> StageCheckpointReceipt:
    """Run one deterministic Stage-P forecast update on train data only."""

    configure_stage_j(model, "joint")
    for parameter in model.scheduler_parameters():
        parameter.requires_grad_(False)
    before = _snapshot(model)
    output = model(**_forward_inputs(batch))
    task_weights = output.forecast_normalized.new_tensor((1.0, 1.0, 1.0, 0.25))
    per_task = F.smooth_l1_loss(output.forecast_normalized, batch["target_normalized"], reduction="none").mean(dim=1)
    loss = (per_task * task_weights).sum(dim=-1).mean() / task_weights.sum()
    forecaster_norm, _ = _step_optimizer(model, model.forecaster_parameters(), loss, max_grad_norm=max_grad_norm)
    return _receipt(model, stage="P", seed=seed, mode="forecast_pretrain", before=before, loss=loss, forecaster_gradient_norm=forecaster_norm, scheduler_gradient_norm=0.0, decision_forecaster_gradient_norm=0.0, checkpoint_path=str(checkpoint_path))


def train_stage_s(model: RSCPFModel, batch: Mapping[str, Tensor], parameters: Mapping[str, Any], *, seed: int, max_grad_norm: float = 1.0, checkpoint_path: str | Path = "") -> StageCheckpointReceipt:
    """Run one Stage-S scheduler update with the forecast/state pathway frozen."""

    configure_stage_j(model, "decoupled")
    for parameter in model.scheduler_parameters():
        parameter.requires_grad_(True)
    before = _snapshot(model)
    output = model(**_forward_inputs(batch))
    loss = F.smooth_l1_loss(output.dispatch, batch["teacher_dispatch"].to(output.dispatch))
    scheduler_norm, _ = _step_optimizer(model, model.scheduler_parameters(), loss, max_grad_norm=max_grad_norm)
    return _receipt(model, stage="S", seed=seed, mode="scheduler_pretrain", before=before, loss=loss, forecaster_gradient_norm=0.0, scheduler_gradient_norm=scheduler_norm, decision_forecaster_gradient_norm=0.0, checkpoint_path=str(checkpoint_path))


def train_stage_j(model: RSCPFModel, batch: Mapping[str, Tensor], parameters: Mapping[str, Any], *, c_ref: float, mode: Literal["joint", "decoupled"], seed: int, epoch: int = 0, max_grad_norm: float = 1.0, checkpoint_path: str | Path = "", decision_start: float = 0.05, forecaster_lr: float = 1.0e-5, scheduler_lr: float = 1.0e-3) -> StageCheckpointReceipt:
    """Run one Stage-J update; only ``mode`` changes the decision gradient boundary."""

    configure_stage_j(model, mode)
    before = _snapshot(model)
    output = model(**_forward_inputs(batch))
    # Keep the frozen formal-v4 teacher anchor active throughout Stage-J.
    # Dropping imitation to zero lets the decision term distort the explicit
    # forecast bottleneck before the model has learned a stable dispatch map.
    weights = formal_v4_curriculum_weights(epoch=epoch, ramp_epochs=18, imitation_final=0.0, decision_start=decision_start)
    breakdown = formal_v4_joint_loss(
        output, batch["target_normalized"], batch["target_physical"], batch["realized_renewables"],
        batch["initial_soc"], batch["previous_chp"], batch.get("teacher_dispatch"), parameters,
        c_ref=c_ref, forecast_weight=weights.forecast, imitation_weight=weights.imitation,
        decision_weight=weights.decision,
    )
    decision_term = breakdown.normalized_realized_objective + breakdown.constraint_penalty
    forecast_params = tuple(model.forecaster_parameters())
    scheduler_params = tuple(model.scheduler_parameters())
    if decision_term.requires_grad and any(parameter.requires_grad for parameter in forecast_params):
        decision_grads = torch.autograd.grad(decision_term, forecast_params, allow_unused=True, retain_graph=True)
        decision_forecast_norm = _grad_norm_from_values(decision_grads)
    else:
        decision_forecast_norm = 0.0
    scheduler_decision_grads = torch.autograd.grad(decision_term, scheduler_params, allow_unused=True, retain_graph=True) if decision_term.requires_grad else tuple()
    scheduler_decision_norm = _grad_norm_from_values(scheduler_decision_grads)
    if forecaster_lr <= 0.0 or scheduler_lr <= 0.0:
        raise ValueError("Stage-J learning rates must be positive")
    optimizer = torch.optim.AdamW([
        {"params": [parameter for parameter in forecast_params if parameter.requires_grad], "lr": float(forecaster_lr)},
        {"params": [parameter for parameter in scheduler_params if parameter.requires_grad], "lr": float(scheduler_lr)},
    ], weight_decay=1.0e-4)
    optimizer.zero_grad(set_to_none=True); breakdown.total.backward()
    forecaster_norm = _grad_norm(forecast_params); scheduler_norm = _grad_norm(scheduler_params)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm); optimizer.step()
    return _receipt(model, stage="J", seed=seed, mode=mode, before=before, loss=breakdown.total, forecaster_gradient_norm=forecaster_norm, scheduler_gradient_norm=scheduler_norm, decision_forecaster_gradient_norm=decision_forecast_norm, checkpoint_path=str(checkpoint_path))


def _grad_norm_from_values(values: Sequence[Tensor | None]) -> float:
    tensors = [value.detach().float().norm() for value in values if value is not None]
    return float(torch.stack(tensors).norm()) if tensors else 0.0


def save_stage_checkpoint(path: str | Path, model: torch.nn.Module, receipt: StageCheckpointReceipt) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "receipt": receipt.to_dict(), "test_set_accessed": False}, destination)


def stage_p_checkpoint(run_root: str | Path, seed: int) -> Path:
    if int(seed) not in (2026, 2027, 2028, 2029, 2030):
        raise ValueError("seed is not in the frozen formal-v4 seed list")
    return Path(run_root) / "stage_p" / f"seed_{int(seed)}" / "best.pt"


__all__ = [
    "GradientBoundaryReceipt", "StageCheckpointReceipt", "configure_stage_j", "save_stage_checkpoint",
    "stage_p_checkpoint", "train_stage_j", "train_stage_p", "train_stage_s",
]
