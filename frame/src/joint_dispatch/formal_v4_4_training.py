"""Matched v4.4 training stages and auditable gradient-boundary receipts.

The functions in this module intentionally keep the training protocol small and
explicit.  Every branch consumes the same batches in the same order, and the J
pair starts from a byte-identical S parent.  This makes a comparison between
RSC-PF and Fair Decoupled a comparison of the gradient edge, not of data,
initialization, or optimizer budget.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import io
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .formal_v4_4_loss import CurriculumWeightsV44, forecast_loss_v44
from .formal_v4_4_model import ResidualGatedRSCPFModel
from .formal_v4_objective import formal_v4_joint_loss


@dataclass(frozen=True)
class StageBudgetV44:
    max_epochs: int = 2
    minimum_epochs: int = 1
    p0_lr: float = 1.0e-3
    base_lr: float = 2.0e-4
    head_lr: float = 1.0e-3
    scheduler_lr: float = 5.0e-4
    weight_decay: float = 1.0e-5
    max_grad_norm: float = 1.0
    inactive_leakage_target: float = 0.25
    ramp_epochs: int = 5
    patience: int = 3

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.minimum_epochs < 1 or self.minimum_epochs > self.max_epochs:
            raise ValueError("invalid v4.4 epoch budget")
        for name in ("p0_lr", "base_lr", "head_lr", "scheduler_lr", "weight_decay", "max_grad_norm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.ramp_epochs < 1 or self.patience < 1:
            raise ValueError("ramp_epochs and patience must be positive")


@dataclass(frozen=True)
class StageReceiptV44:
    stage: str
    mode: str
    parent_sha256: str
    final_sha256: str
    epochs: int
    optimizer_steps: int
    forecast_optimizer_steps: int
    loss_history: tuple[float, ...]
    stopping_reason: str
    gradient_norms: Mapping[str, float] = field(default_factory=dict)
    # Kept on the in-process receipt so the next stage can load the exact
    # parent without serializing an ambiguous checkpoint.  Artifact writers
    # should omit this object and persist state_dict/optimizer state instead.
    model: nn.Module | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class JointPairReceiptV44:
    joint: StageReceiptV44
    decoupled: StageReceiptV44


def seed_everything(seed: int) -> None:
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def sha256_state_dict(model: nn.Module) -> str:
    """Hash a state dict in sorted-key order, independent of pickle layout."""

    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8")); digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(repr(tuple(tensor.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(tensor.numpy().tobytes()); digest.update(b"\0")
    return digest.hexdigest()


def named_autograd_norms(loss: Tensor, groups: Mapping[str, Sequence[nn.Parameter]], *, retain_graph: bool = False) -> dict[str, float]:
    """Return separate pre-update gradient norms for named parameter groups."""

    if loss.ndim != 0 or not bool(torch.isfinite(loss).all()):
        raise ValueError("loss must be a finite scalar")
    result: dict[str, float] = {}
    if not loss.requires_grad:
        return {name: 0.0 for name in groups}
    items = list(groups.items())
    for index, (name, parameters) in enumerate(items):
        params = tuple(parameter for parameter in parameters if parameter.requires_grad)
        if not params:
            result[name] = 0.0; continue
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=retain_graph or index < len(items) - 1)
        values = [gradient.detach().float().norm() for gradient in grads if gradient is not None]
        result[name] = float(torch.stack(values).norm()) if values else 0.0
    return result


def _batches(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, Mapping):
        value = data.get("train", data)
        return [value] if isinstance(value, Mapping) else list(value)
    return list(data)


def _inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    required = ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp", "last_thermal_regime")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"training batch is missing {missing[0]}")
    result: dict[str, Tensor] = {}
    for name in required:
        value = batch[name] if isinstance(batch[name], Tensor) else torch.as_tensor(batch[name])
        result[name] = value.float() if name != "last_thermal_regime" else value.long()
    return result


def _target(batch: Mapping[str, Any], name: str, reference: Tensor) -> Tensor:
    value = batch.get(name)
    if value is None:
        raise ValueError(f"training batch requires {name}")
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    if hasattr(model, "v44_parameter_groups"):
        return {name: tuple(params) for name, params in model.v44_parameter_groups().items()}
    if hasattr(model, "forecaster_parameters") and hasattr(model, "scheduler_parameters"):
        return {"base": tuple(model.forecaster_parameters()), "gate": (), "magnitude": (), "scheduler": tuple(model.scheduler_parameters())}
    params = tuple(model.parameters())
    return {"base": params, "gate": (), "magnitude": (), "scheduler": params}


def _all_forecast(groups: Mapping[str, Sequence[nn.Parameter]]) -> tuple[nn.Parameter, ...]:
    return tuple((*groups.get("base", ()), *groups.get("gate", ()), *groups.get("magnitude", ())))


def _forecast_loss(output: Any, batch: Mapping[str, Any], prior: Any | None, budget: StageBudgetV44, epoch: int) -> Tensor:
    if prior is not None and all(hasattr(output, name) for name in ("regime_logits", "thermal_magnitudes", "forecast_physical")):
        weights = CurriculumWeightsV44(1.0, 1.0, 1.0, 0.25 + 0.75 * min(epoch / budget.ramp_epochs, 1.0), 0.10 + (budget.inactive_leakage_target - 0.10) * min(epoch / budget.ramp_epochs, 1.0))
        return forecast_loss_v44(output, batch, prior, weights).total
    target = _target(batch, "target_normalized", output.forecast_normalized)
    per_task = F.smooth_l1_loss(output.forecast_normalized, target, reduction="none").mean(dim=1)
    return (per_task * output.forecast_normalized.new_tensor((1.0, 1.0, 1.0, 0.25))).mean()


def _decision_and_imitation(model: nn.Module, output: Any, batch: Mapping[str, Any], *, c_ref: float, parameters: Mapping[str, Any] | None, detached: bool) -> tuple[Tensor, Tensor, Tensor]:
    teacher = batch.get("teacher_dispatch")
    teacher_t = None if teacher is None else torch.as_tensor(teacher, dtype=output.dispatch.dtype, device=output.dispatch.device)
    if teacher_t is None:
        imitation = output.dispatch.new_zeros(())
    else:
        scale = teacher_t.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
        imitation = F.smooth_l1_loss(output.dispatch / scale, teacher_t / scale)
    required = ("target_normalized", "target_physical", "realized_renewables", "initial_soc", "previous_chp")
    if parameters is None or not all(key in batch for key in required):
        decision = output.dispatch.square().mean()
    else:
        breakdown = formal_v4_joint_loss(
            output,
            _target(batch, "target_normalized", output.forecast_normalized),
            _target(batch, "target_physical", output.forecast_physical),
            torch.as_tensor(batch["realized_renewables"], dtype=output.dispatch.dtype, device=output.dispatch.device),
            torch.as_tensor(batch["initial_soc"], dtype=output.dispatch.dtype, device=output.dispatch.device),
            torch.as_tensor(batch["previous_chp"], dtype=output.dispatch.dtype, device=output.dispatch.device),
            teacher_t, parameters, c_ref=c_ref, forecast_weight=0.0, imitation_weight=0.0, decision_weight=1.0,
        )
        decision = breakdown.normalized_realized_objective + breakdown.constraint_penalty
    return decision, imitation, decision + imitation


def _step(optimizer: torch.optim.Optimizer, params: Sequence[nn.Parameter], loss: Tensor, max_norm: float) -> float:
    optimizer.zero_grad(set_to_none=True); loss.backward()
    values = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    norm = float(torch.stack(values).norm()) if values else 0.0
    torch.nn.utils.clip_grad_norm_(list(params), max_norm); optimizer.step()
    return norm


def _run_epochs(model: nn.Module, batches: list[Mapping[str, Any]], params: Sequence[nn.Parameter], optimizer: torch.optim.Optimizer, loss_fn: Any, budget: StageBudgetV44, *, stage: str, mode: str, parent: str, forecast_steps: bool = False) -> StageReceiptV44:
    history: list[float] = []; steps = 0
    for epoch in range(budget.max_epochs):
        losses: list[float] = []
        for batch in batches:
            loss = loss_fn(batch, epoch)
            if not bool(torch.isfinite(loss).all()):
                raise FloatingPointError(f"non-finite {stage} loss")
            _step(optimizer, params, loss, budget.max_grad_norm); steps += 1; losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return StageReceiptV44(stage, mode, parent, sha256_state_dict(model), budget.max_epochs, steps, steps if forecast_steps else 0, tuple(history), "budget_exhausted", model=model)


def run_stage_p0_v44(model: nn.Module, loaders: Any, budget: StageBudgetV44 | None = None, *, seed: int = 2026) -> StageReceiptV44:
    """Train the base forecast/state pathway on the frozen P0 curriculum."""
    budget = budget or StageBudgetV44(); seed_everything(seed); batches = _batches(loaders)
    if not batches: raise ValueError("P0 requires non-empty training batches")
    groups = _groups(model); forecast_params = _all_forecast(groups)
    for p in forecast_params: p.requires_grad_(True)
    for p in groups.get("scheduler", ()): p.requires_grad_(False)
    parent = sha256_state_dict(model); optimizer = torch.optim.AdamW(forecast_params, lr=budget.p0_lr, weight_decay=budget.weight_decay)
    return _run_epochs(model, batches, forecast_params, optimizer, lambda b, e: _forecast_loss(model(**_inputs(b)), b, None, budget, e), budget, stage="P0", mode="forecast_base", parent=parent, forecast_steps=True)


def run_stage_p1_v44(stage_p0: StageReceiptV44 | Any, loaders: Any, budget: StageBudgetV44 | None = None, *, prior: Any | None = None, seed: int = 2026) -> StageReceiptV44:
    """Load the P0 parent and fit the residual regime/magnitude heads."""
    budget = budget or StageBudgetV44(); seed_everything(seed)
    base_model = stage_p0.model if hasattr(stage_p0, "model") else stage_p0
    model = deepcopy(base_model); batches = _batches(loaders)
    groups = _groups(model); head_params = tuple((*groups.get("gate", ()), *groups.get("magnitude", ())))
    if not head_params: raise ValueError("P1 requires a residual-gated v4.4 model")
    for p in _all_forecast(groups): p.requires_grad_(False)
    for p in head_params: p.requires_grad_(True)
    parent = sha256_state_dict(model); optimizer = torch.optim.AdamW(head_params, lr=budget.head_lr, weight_decay=budget.weight_decay)
    receipt = _run_epochs(model, batches, head_params, optimizer, lambda b, e: _forecast_loss(model(**_inputs(b)), b, prior, budget, e), budget, stage="P1", mode="residual_head", parent=parent, forecast_steps=False)
    return receipt


def run_continuous_control_v44(stage_p0: StageReceiptV44 | Any, loaders: Any, budget: StageBudgetV44 | None = None, *, seed: int = 2026) -> StageReceiptV44:
    """Matched continuous-control run with the same base-update budget as P1."""
    budget = budget or StageBudgetV44(); seed_everything(seed)
    base_model = stage_p0.model if hasattr(stage_p0, "model") else stage_p0; model = deepcopy(base_model); batches = _batches(loaders)
    groups = _groups(model); params = _all_forecast(groups)
    for p in params: p.requires_grad_(True)
    for p in groups.get("scheduler", ()): p.requires_grad_(False)
    parent = sha256_state_dict(model); optimizer = torch.optim.AdamW(params, lr=budget.base_lr, weight_decay=budget.weight_decay)
    return _run_epochs(model, batches, params, optimizer, lambda b, e: _forecast_loss(model(**_inputs(b)), b, None, budget, e), budget, stage="P1-control", mode="continuous_base", parent=parent, forecast_steps=True)


def run_stage_s_v44(stage_p1: StageReceiptV44 | Any, loaders: Any, budget: StageBudgetV44 | None = None, *, seed: int = 2026) -> StageReceiptV44:
    """Fit the scheduler by teacher imitation while the forecast parent is fixed."""
    budget = budget or StageBudgetV44(); seed_everything(seed); model = deepcopy(stage_p1.model if hasattr(stage_p1, "model") else stage_p1); batches = _batches(loaders)
    groups = _groups(model); params = tuple(groups.get("scheduler", ()))
    if not params: raise ValueError("S requires scheduler parameters")
    for p in _all_forecast(groups): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    parent = sha256_state_dict(model); optimizer = torch.optim.AdamW(params, lr=budget.scheduler_lr, weight_decay=budget.weight_decay)
    def loss_fn(batch: Mapping[str, Any], _epoch: int) -> Tensor:
        output = model(**_inputs(batch)); teacher = batch.get("teacher_dispatch")
        if teacher is None: raise ValueError("S requires teacher_dispatch")
        teacher_t = torch.as_tensor(teacher, dtype=output.dispatch.dtype, device=output.dispatch.device); scale = teacher_t.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
        return F.smooth_l1_loss(output.dispatch / scale, teacher_t / scale)
    return _run_epochs(model, batches, params, optimizer, loss_fn, budget, stage="S", mode="scheduler_imitation", parent=parent)


def run_stage_j_pair_v44(stage_s: StageReceiptV44 | Any, loaders: Any, budget: StageBudgetV44 | None = None, *, prior: Any | None = None, c_ref: float = 1.0, parameters: Mapping[str, Any] | None = None, seed: int = 2026) -> JointPairReceiptV44:
    """Run matched joint and decoupled branches from one frozen S parent."""
    budget = budget or StageBudgetV44(); seed_everything(seed); base_model = stage_s.model if hasattr(stage_s, "model") else stage_s; batches = _batches(loaders)
    if not batches: raise ValueError("J requires non-empty training batches")
    parent = sha256_state_dict(base_model)
    receipts: list[StageReceiptV44] = []
    for mode in ("joint", "decoupled"):
        seed_everything(seed); model = deepcopy(base_model); groups = _groups(model)
        forecast_params = _all_forecast(groups); scheduler_params = tuple(groups.get("scheduler", ()))
        # Keep the forecast graph differentiable in both branches so the
        # supervised forecast loss can still update it.  Only the joint branch
        # places those parameters in the optimizer; the decoupled branch uses
        # the detached dispatch edge to guarantee zero decision gradients.
        for p in forecast_params: p.requires_grad_(True)
        for p in scheduler_params: p.requires_grad_(True)
        train_forecast = forecast_params if mode == "joint" else ()
        active = tuple((*train_forecast, *scheduler_params))
        optimizer = torch.optim.AdamW([{"params": list(train_forecast), "lr": budget.base_lr}, {"params": scheduler_params, "lr": budget.scheduler_lr}], weight_decay=budget.weight_decay)
        history: list[float] = []; steps = 0; last_norms: dict[str, float] = {}
        for epoch in range(budget.max_epochs):
            epoch_losses: list[float] = []
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                output = model(detach_forecast_for_dispatch=(mode == "decoupled"), **_inputs(batch))
                forecast = _forecast_loss(output, batch, prior, budget, epoch)
                decision, imitation, _ = _decision_and_imitation(model, output, batch, c_ref=c_ref, parameters=parameters, detached=(mode == "decoupled"))
                total = forecast + imitation + decision
                group_map = {"base": groups.get("base", ()), "gate": groups.get("gate", ()), "magnitude": groups.get("magnitude", ()), "scheduler": scheduler_params}
                decision_norms = named_autograd_norms(decision, group_map, retain_graph=True)
                forecast_norms = named_autograd_norms(forecast, group_map, retain_graph=True)
                total.backward(); torch.nn.utils.clip_grad_norm_(list(active), budget.max_grad_norm); optimizer.step(); steps += 1
                last_norms = {
                    "decision_to_base": decision_norms.get("base", 0.0), "decision_to_gate": decision_norms.get("gate", 0.0),
                    "decision_to_magnitude": decision_norms.get("magnitude", 0.0), "decision_to_scheduler": decision_norms.get("scheduler", 0.0),
                    "forecast_to_base": forecast_norms.get("base", 0.0), "forecast_to_gate": forecast_norms.get("gate", 0.0),
                    "forecast_to_magnitude": forecast_norms.get("magnitude", 0.0),
                }
                epoch_losses.append(float(total.detach()))
            history.append(float(np.mean(epoch_losses)))
        receipts.append(StageReceiptV44("J", mode, parent, sha256_state_dict(model), budget.max_epochs, steps, 0, tuple(history), "budget_exhausted", last_norms, model=model))
    return JointPairReceiptV44(receipts[0], receipts[1])


__all__ = [
    "JointPairReceiptV44", "StageBudgetV44", "StageReceiptV44", "named_autograd_norms", "run_continuous_control_v44",
    "run_stage_j_pair_v44", "run_stage_p0_v44", "run_stage_p1_v44", "run_stage_s_v44", "seed_everything", "sha256_state_dict",
]
