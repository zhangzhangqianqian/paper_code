"""Validation-aware stage primitives for the formal-v4.5 repair."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .formal_v4_4_training import seed_everything, sha256_state_dict
from .formal_v4_4_loss import CurriculumWeightsV44, forecast_loss_v44
from .formal_v4_4_training import _all_forecast, _decision_and_imitation, _groups, _inputs, _target, named_autograd_norms
from .formal_v4_5_loss import JointNormalizationV45, build_j_optimizer_v45, curriculum_weights_v45, joint_loss_v45
from .formal_v4_objective import formal_v4_joint_loss


@dataclass(frozen=True)
class StageBudgetV45:
    max_epochs: int
    minimum_epochs: int
    p0_lr: float
    p1_base_lr: float
    p1_head_lr: float
    s_scheduler_lr: float
    j_forecaster_lr: float
    j_head_lr: float
    j_scheduler_lr: float
    weight_decay: float
    max_grad_norm: float
    inactive_leakage_target: float
    ramp_epochs: int
    patience: int


@dataclass(frozen=True)
class JointPairReceiptV45:
    joint: StageReceiptV45
    decoupled: StageReceiptV45


def _forecast_loss_v45(output: Any, batch: Mapping[str, Any], prior: Any | None, budget: StageBudgetV45, epoch: int) -> Tensor:
    if prior is not None and all(hasattr(output, name) for name in ("regime_logits", "thermal_magnitudes", "forecast_physical")):
        weights = CurriculumWeightsV44(
            1.0, 1.0, 1.0,
            0.25 + 0.75 * min(epoch / max(budget.ramp_epochs, 1), 1.0),
            0.10 + (budget.inactive_leakage_target - 0.10) * min(epoch / max(budget.ramp_epochs, 1), 1.0),
        )
        return forecast_loss_v44(output, batch, prior, weights).total
    target = _target(batch, "target_normalized", output.forecast_normalized)
    per_task = torch.nn.functional.smooth_l1_loss(output.forecast_normalized, target, reduction="none").mean(dim=1)
    return (per_task * output.forecast_normalized.new_tensor((1.0, 1.0, 1.0, 0.25))).mean()


def _stage_loaders(loaders: Any) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not isinstance(loaders, Mapping) or "train" not in loaders or "early_stop" not in loaders:
        raise ValueError("formal-v4.5 stages require train and early_stop loaders")
    train = list(loaders["train"])
    validation = list(loaders["early_stop"])
    if not train or not validation:
        raise ValueError("formal-v4.5 stages require non-empty train and early_stop loaders")
    return train, validation


def _stage_loss_metric(model: nn.Module, batch: Mapping[str, Any], epoch: int, prior: Any | None, budget: StageBudgetV45) -> float:
    with torch.no_grad():
        return float(_forecast_loss_v45(model(**_inputs(batch)), batch, prior, budget, epoch).detach())


def run_stage_p0_v45(model: nn.Module, loaders: Any, budget: StageBudgetV45, *, seed: int = 2026) -> StageReceiptV45:
    train, validation = _stage_loaders(loaders)
    groups = _groups(model)
    params = tuple(groups.get("base", ())) or _all_forecast(groups)
    for parameter in params:
        parameter.requires_grad_(True)
    for parameter in (*groups.get("gate", ()), *groups.get("magnitude", ()), *groups.get("scheduler", ())):
        parameter.requires_grad_(False)
    parent = sha256_state_dict(model)
    return run_stage_with_validation_v45(
        model, train, validation,
        lambda values: torch.optim.AdamW(values, lr=budget.p0_lr, weight_decay=budget.weight_decay),
        lambda current, batch, epoch: _forecast_loss_v45(current(**_inputs(batch)), batch, None, budget, epoch),
        lambda current, batch, epoch: _stage_loss_metric(current, batch, epoch, None, budget),
        lambda details: bool(np.isfinite(details["metric"])),
        budget.max_epochs, budget.minimum_epochs, budget.patience,
        stage="P0", mode="forecast_base", seed=seed, forecast_steps=True,
    )


def run_stage_p1_v45(stage_p0: StageReceiptV45 | Any, loaders: Any, budget: StageBudgetV45, *, prior: Any | None, seed: int = 2026) -> StageReceiptV45:
    train, validation = _stage_loaders(loaders)
    model = deepcopy(stage_p0.model if hasattr(stage_p0, "model") else stage_p0)
    groups = _groups(model)
    params = tuple((*groups.get("gate", ()), *groups.get("magnitude", ())))
    if not params:
        raise ValueError("P1 requires residual gate and magnitude parameters")
    for parameter in _all_forecast(groups):
        parameter.requires_grad_(False)
    for parameter in params:
        parameter.requires_grad_(True)
    return run_stage_with_validation_v45(
        model, train, validation,
        lambda values: torch.optim.AdamW(values, lr=budget.p1_head_lr, weight_decay=budget.weight_decay),
        lambda current, batch, epoch: _forecast_loss_v45(current(**_inputs(batch)), batch, prior, budget, epoch),
        lambda current, batch, epoch: _stage_loss_metric(current, batch, epoch, prior, budget),
        lambda details: bool(np.isfinite(details["metric"])),
        budget.max_epochs, budget.minimum_epochs, budget.patience,
        stage="P1", mode="residual_head", seed=seed,
    )


def run_continuous_control_v45(stage_p0: StageReceiptV45 | Any, model: nn.Module, loaders: Any, budget: StageBudgetV45, *, seed: int = 2026) -> StageReceiptV45:
    train, validation = _stage_loaders(loaders)
    current = deepcopy(model)
    groups = _groups(current)
    params = _all_forecast(groups)
    for parameter in params:
        parameter.requires_grad_(True)
    for parameter in groups.get("scheduler", ()):
        parameter.requires_grad_(False)
    return run_stage_with_validation_v45(
        current, train, validation,
        lambda values: torch.optim.AdamW(values, lr=budget.p1_base_lr, weight_decay=budget.weight_decay),
        lambda active, batch, epoch: _forecast_loss_v45(active(**_inputs(batch)), batch, None, budget, epoch),
        lambda active, batch, epoch: _stage_loss_metric(active, batch, epoch, None, budget),
        lambda details: bool(np.isfinite(details["metric"])),
        budget.max_epochs, budget.minimum_epochs, budget.patience,
        stage="P1-control", mode="continuous_base", seed=seed, forecast_steps=True,
    )


def run_stage_s_v45(stage_p1: StageReceiptV45 | Any, loaders: Any, budget: StageBudgetV45, *, seed: int = 2026) -> StageReceiptV45:
    train, validation = _stage_loaders(loaders)
    model = deepcopy(stage_p1.model if hasattr(stage_p1, "model") else stage_p1)
    groups = _groups(model)
    params = tuple(groups.get("scheduler", ()))
    if not params:
        raise ValueError("S requires scheduler parameters")
    for parameter in _all_forecast(groups):
        parameter.requires_grad_(False)
    for parameter in params:
        parameter.requires_grad_(True)

    def imitation_loss(current: nn.Module, batch: Mapping[str, Any], _epoch: int) -> Tensor:
        output = current(**_inputs(batch))
        teacher = batch.get("teacher_dispatch")
        if teacher is None:
            raise ValueError("S requires teacher_dispatch")
        teacher_tensor = torch.as_tensor(teacher, dtype=output.dispatch.dtype, device=output.dispatch.device)
        scale = teacher_tensor.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
        return torch.nn.functional.smooth_l1_loss(output.dispatch / scale, teacher_tensor / scale)

    return run_stage_with_validation_v45(
        model, train, validation,
        lambda values: torch.optim.AdamW(values, lr=budget.s_scheduler_lr, weight_decay=budget.weight_decay),
        imitation_loss, lambda current, batch, _epoch: float(imitation_loss(current, batch, 0).detach()),
        lambda details: bool(np.isfinite(details["metric"])),
        budget.max_epochs, budget.minimum_epochs, budget.patience,
        stage="S", mode="scheduler_imitation", seed=seed,
    )


def _j_parent_normalization(
    parent_model: nn.Module,
    train_batches: Sequence[Mapping[str, Any]],
    prior: Any | None,
    budget: StageBudgetV45,
    *,
    c_ref: float,
    parameters: Mapping[str, Any] | None,
) -> JointNormalizationV45:
    forecasts: list[Tensor] = []
    imitations: list[Tensor] = []
    decisions: list[Tensor] = []
    parent_model.eval()
    with torch.no_grad():
        for batch in train_batches:
            output = parent_model(**_inputs(batch))
            forecasts.append(_forecast_loss_v45(output, batch, prior, budget, 0))
            decision, imitation, _ = _decision_and_imitation(
                parent_model, output, batch, c_ref=c_ref, parameters=parameters, detached=False,
            )
            decisions.append(decision)
            imitations.append(imitation)
    return JointNormalizationV45.from_values(
        torch.stack(forecasts), torch.stack(imitations), torch.stack(decisions),
    )


def _j_validation_metric(
    model: nn.Module,
    parent_model: nn.Module,
    batch: Mapping[str, Any],
    prior: Any | None,
    budget: StageBudgetV45,
    normalization: JointNormalizationV45,
    *,
    c_ref: float,
    parameters: Mapping[str, Any] | None,
    mode: str,
    epoch: int,
) -> tuple[float, dict[str, float]]:
    with torch.no_grad():
        output = model(detach_forecast_for_dispatch=(mode == "decoupled"), **_inputs(batch))
        parent_output = parent_model(**_inputs(batch))
        decision, imitation, _ = _decision_and_imitation(
            model, output, batch, c_ref=c_ref, parameters=parameters, detached=(mode == "decoupled"),
        )
        forecast = _forecast_loss_v45(output, batch, prior, budget, epoch)
        weights = curriculum_weights_v45(epoch, budget.ramp_epochs)
        terms = joint_loss_v45(
            output, parent_output, batch, normalization, weights,
            decision_loss=decision,
        )
        # Validation guardrails compare against the parent on the same
        # early-stop batch.  Using the train-set parent scale here would make
        # a legitimate seasonal shift look like forecast degradation and can
        # reject every checkpoint before J training starts.
        parent_forecast = _forecast_loss_v45(parent_output, batch, prior, budget, epoch)
        parent_forecast_scale = parent_forecast.detach().abs().clamp_min(1.0e-8)
        details = {
            "metric": float((terms.decision / normalization.decision).detach()),
            "decision": float((decision / normalization.decision).detach()),
            "forecast": float((forecast / parent_forecast_scale).detach()),
            "imitation": float((imitation / normalization.imitation).detach()),
            "anchor": float(terms.anchor.detach()),
        }
        return details["metric"], details


def _run_j_branch_v45(
    base_model: nn.Module,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    budget: StageBudgetV45,
    *,
    prior: Any | None,
    c_ref: float,
    parameters: Mapping[str, Any] | None,
    contract: Any,
    normalization: JointNormalizationV45,
    mode: str,
    seed: int,
    validation_guard: Callable[[Mapping[str, float]], bool] | None,
) -> StageReceiptV45:
    seed_everything(seed)
    model = deepcopy(base_model)
    groups = _groups(model)
    forecast_params = _all_forecast(groups)
    scheduler_params = tuple(groups.get("scheduler", ()))
    for parameter in forecast_params:
        parameter.requires_grad_(mode == "joint")
    for parameter in scheduler_params:
        parameter.requires_grad_(True)
    active = tuple(parameter for parameter in (*forecast_params, *scheduler_params) if parameter.requires_grad)
    if not active:
        raise ValueError("J branch has no trainable parameters")
    optimizer = build_j_optimizer_v45(model, contract, mode=mode)
    parent_model = deepcopy(base_model).eval()
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    parent_sha256 = sha256_state_dict(base_model)
    loss_history: list[float] = []
    validation_history: list[Mapping[str, float]] = []
    last_norms: dict[str, float] = {
        "decision_to_base": 0.0, "decision_to_gate": 0.0,
        "decision_to_magnitude": 0.0, "decision_to_scheduler": 0.0,
        "forecast_to_base": 0.0, "forecast_to_gate": 0.0,
        "forecast_to_magnitude": 0.0,
    }
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_metric = float("inf")
    stale_epochs = 0
    optimizer_steps = 0
    stopping_reason = "budget_exhausted"
    for epoch in range(budget.max_epochs):
        model.train()
        weights = curriculum_weights_v45(epoch, budget.ramp_epochs)
        epoch_losses: list[float] = []
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(detach_forecast_for_dispatch=(mode == "decoupled"), **_inputs(batch))
            with torch.no_grad():
                parent_output = parent_model(**_inputs(batch))
            decision, imitation, _ = _decision_and_imitation(
                model, output, batch, c_ref=c_ref, parameters=parameters, detached=(mode == "decoupled"),
            )
            terms = joint_loss_v45(
                output, parent_output, batch, normalization, weights,
                decision_loss=decision,
            )
            forecast_norms = named_autograd_norms(terms.forecast, {
                "base": groups.get("base", ()), "gate": groups.get("gate", ()),
                "magnitude": groups.get("magnitude", ()), "scheduler": scheduler_params,
            }, retain_graph=True)
            decision_norms = named_autograd_norms(terms.decision, {
                "base": groups.get("base", ()), "gate": groups.get("gate", ()),
                "magnitude": groups.get("magnitude", ()), "scheduler": scheduler_params,
            }, retain_graph=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(active, budget.max_grad_norm)
            optimizer.step()
            optimizer_steps += 1
            epoch_losses.append(float(terms.total.detach()))
            last_norms = {
                "decision_to_base": decision_norms.get("base", 0.0),
                "decision_to_gate": decision_norms.get("gate", 0.0),
                "decision_to_magnitude": decision_norms.get("magnitude", 0.0),
                "decision_to_scheduler": decision_norms.get("scheduler", 0.0),
                "forecast_to_base": forecast_norms.get("base", 0.0),
                "forecast_to_gate": forecast_norms.get("gate", 0.0),
                "forecast_to_magnitude": forecast_norms.get("magnitude", 0.0),
            }
        loss_history.append(float(np.mean(epoch_losses)))
        model.eval()
        details_rows = [
            _j_validation_metric(
                model, parent_model, batch, prior, budget, normalization,
                c_ref=c_ref, parameters=parameters, mode=mode, epoch=epoch,
            )[1]
            for batch in validation_batches
        ]
        details = {name: float(np.mean([row[name] for row in details_rows])) for name in details_rows[0]}
        eligible = bool(validation_guard(details) if validation_guard is not None else np.isfinite(details["metric"]))
        details["eligible"] = float(eligible)
        validation_history.append(details)
        if eligible and details["metric"] < best_metric:
            best_metric = details["metric"]
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch + 1 >= budget.minimum_epochs and stale_epochs >= budget.patience:
            stopping_reason = "early_stopped"
            break
    if best_state is None or best_epoch < 0:
        tail = [dict(row) for row in validation_history[-3:]]
        raise RuntimeError(f"J {mode} has no eligible finite validation checkpoint; validation_tail={tail}")
    terminal_sha256 = sha256_state_dict(model)
    model.load_state_dict(best_state, strict=True)
    final_sha256 = sha256_state_dict(model)
    return StageReceiptV45(
        stage="J", mode=mode, parent_sha256=parent_sha256,
        final_sha256=final_sha256, best_sha256=final_sha256,
        terminal_sha256=terminal_sha256, epochs=len(loss_history),
        best_epoch=best_epoch, optimizer_steps=optimizer_steps,
        forecast_optimizer_steps=optimizer_steps if mode == "joint" else 0,
        loss_history=tuple(loss_history), validation_history=tuple(validation_history),
        stopping_reason=stopping_reason, selection_metric=best_metric,
        gradient_norms=dict(last_norms),
        model=model,
    )


def run_stage_j_pair_v45(
    stage_s: StageReceiptV45 | Any,
    loaders: Any,
    budget: StageBudgetV45,
    *,
    prior: Any | None,
    c_ref: float,
    parameters: Mapping[str, Any] | None,
    contract: Any,
    seed: int = 2026,
    validation_guard: Callable[[Mapping[str, float]], bool] | None = None,
) -> JointPairReceiptV45:
    train, validation = _stage_loaders(loaders)
    base_model = stage_s.model if hasattr(stage_s, "model") else stage_s
    normalization = _j_parent_normalization(
        base_model, train, prior, budget, c_ref=c_ref, parameters=parameters,
    )
    joint = _run_j_branch_v45(
        base_model, train, validation, budget, prior=prior, c_ref=c_ref,
        parameters=parameters, contract=contract, normalization=normalization,
        mode="joint", seed=seed, validation_guard=validation_guard,
    )
    decoupled = _run_j_branch_v45(
        base_model, train, validation, budget, prior=prior, c_ref=c_ref,
        parameters=parameters, contract=contract, normalization=normalization,
        mode="decoupled", seed=seed, validation_guard=validation_guard,
    )
    return JointPairReceiptV45(joint=joint, decoupled=decoupled)


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
    gradient_norms: Mapping[str, float] = field(default_factory=dict)
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


__all__ = [
    "JointPairReceiptV45", "StageBudgetV45", "StageReceiptV45", "StageValidationV45",
    "run_continuous_control_v45", "run_stage_p0_v45", "run_stage_p1_v45",
    "run_stage_j_pair_v45", "run_stage_s_v45", "run_stage_with_validation_v45",
]
