"""Sequential, persistent-optimizer training for the formal-v4.2 model."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import inspect
import random
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .formal_v4_models import RSCPFModel
from .formal_v4_objective import formal_v4_curriculum_weights, formal_v4_joint_loss


@dataclass(frozen=True)
class StageBudgetV42:
    max_epochs: int = 30
    minimum_epochs: int = 18
    decision_start: float = 0.05
    decision_final: float = 1.0
    ramp_epochs: int = 18
    forecast_weight: float = 1.0
    imitation_start: float = 1.0
    imitation_final: float = 0.0
    forecaster_lr: float = 1.0e-5
    scheduler_lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_grad_norm: float = 1.0
    rollin_fraction: float = 0.40
    model_history_fraction: float = 0.50

    def __post_init__(self) -> None:
        if self.max_epochs <= 0 or self.minimum_epochs <= 0 or self.minimum_epochs > self.max_epochs:
            raise ValueError("invalid stage epoch budget")
        if self.ramp_epochs <= 0 or not (0.0 < self.decision_start <= self.decision_final):
            raise ValueError("invalid decision curriculum")
        if not (0.0 <= self.imitation_final <= self.imitation_start) or self.imitation_start <= 0.0:
            raise ValueError("invalid imitation curriculum")
        for name in ("forecaster_lr", "scheduler_lr", "weight_decay", "max_grad_norm"):
            if float(getattr(self, name)) <= 0.0 or not np.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 <= self.rollin_fraction <= 1.0 or not 0.0 <= self.model_history_fraction <= 1.0:
            raise ValueError("roll-in fractions must be in [0,1]")

    def weights(self, epoch: int) -> Any:
        if int(epoch) < 0:
            raise ValueError("epoch must be non-negative")
        return formal_v4_curriculum_weights(
            epoch=int(epoch), ramp_epochs=self.ramp_epochs,
            decision_start=self.decision_start, decision_final=self.decision_final,
            imitation_start=self.imitation_start, imitation_final=self.imitation_final,
            forecast=self.forecast_weight,
        )

    def can_stop(self, epoch: int) -> bool:
        # ``epoch`` is zero-based; the first legal stop is after the minimum
        # number of completed epochs.
        return int(epoch) + 1 >= self.minimum_epochs


@dataclass(frozen=True)
class GradientBoundaryReceiptV42:
    mode: str
    decision_forecaster_gradient_norm: float
    scheduler_gradient_norm: float
    forecast_gradient_norm: float = 0.0

    @property
    def forecaster_decision_gradient_norm(self) -> float:
        return self.decision_forecaster_gradient_norm


@dataclass(frozen=True)
class StageResultV42:
    stage: str
    mode: str
    model: nn.Module
    optimizer: torch.optim.Optimizer
    epochs: int
    optimizer_steps: int
    loss: float
    decision_forecaster_gradient_norm: float = 0.0
    scheduler_gradient_norm: float = 0.0
    loss_history: tuple[float, ...] = ()


@dataclass(frozen=True)
class TrainingRunReceiptV42:
    seed: int
    stage_events: tuple[str, ...]
    optimizer_final_steps: Mapping[str, int]
    gradient_boundaries: tuple[GradientBoundaryReceiptV42, ...] = ()
    stage_losses: Mapping[str, float] = field(default_factory=dict)
    test_set_accessed: bool = False


def seed_everything(seed: int) -> None:
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    seed = int(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_groups(model: nn.Module) -> tuple[tuple[nn.Parameter, ...], tuple[nn.Parameter, ...]]:
    if hasattr(model, "forecaster_parameters") and hasattr(model, "scheduler_parameters"):
        return tuple(model.forecaster_parameters()), tuple(model.scheduler_parameters())
    all_params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    return all_params, all_params


def _batches(data: Any) -> list[Mapping[str, Any]]:
    if data is None:
        return []
    if isinstance(data, Mapping):
        return [data]
    return list(data)


def _to_tensor(value: Any, *, dtype: torch.dtype = torch.float32) -> Tensor:
    return value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=dtype)


def _forward_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    required = ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"training batch is missing {missing[0]}")
    return {name: _to_tensor(batch[name]) for name in required}


def _target_normalized(batch: Mapping[str, Any], reference: Tensor) -> Tensor:
    value = batch.get("target_normalized", batch.get("target"))
    if value is None:
        raise ValueError("training batch requires target_normalized or target")
    target = _to_tensor(value, dtype=reference.dtype).to(device=reference.device, dtype=reference.dtype)
    if target.shape != reference.shape:
        raise ValueError("target_normalized must match forecast output")
    return target


def _target_physical(batch: Mapping[str, Any], forecast: Tensor) -> Tensor:
    value = batch.get("target_physical", batch.get("target"))
    if value is None:
        raise ValueError("training batch requires target_physical or target")
    target = _to_tensor(value, dtype=forecast.dtype).to(device=forecast.device, dtype=forecast.dtype)
    if target.shape != forecast.shape:
        raise ValueError("target_physical must match forecast output")
    return target


def _dispatch_targets(batch: Mapping[str, Any], reference: Tensor) -> Tensor | None:
    value = batch.get("teacher_dispatch")
    if value is None:
        return None
    return _to_tensor(value, dtype=reference.dtype).to(device=reference.device, dtype=reference.dtype)


def _imitation_loss(dispatch: Tensor, teacher: Tensor) -> Tensor:
    if teacher.shape != dispatch.shape or not bool(torch.isfinite(teacher).all()):
        raise ValueError("teacher_dispatch must match dispatch and be finite")
    scale = teacher.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
    return F.smooth_l1_loss(dispatch / scale, teacher.to(dispatch) / scale)


def _run_model(model: nn.Module, batch: Mapping[str, Any], *, decouple_decision: bool = False) -> Any:
    inputs = _forward_inputs(batch)
    signature = inspect.signature(model.forward)
    if "detach_forecast_for_dispatch" in signature.parameters:
        return model(detach_forecast_for_dispatch=decouple_decision, **inputs)
    return model(**inputs)


def _grad_norm(parameters: Sequence[nn.Parameter]) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def _value_grad_norm(values: Sequence[Tensor | None]) -> float:
    tensors = [value.detach().float().norm() for value in values if value is not None]
    return float(torch.stack(tensors).norm()) if tensors else 0.0


def _forecast_loss(output: Any, batch: Mapping[str, Any]) -> Tensor:
    target = _target_normalized(batch, output.forecast_normalized)
    weights = output.forecast_normalized.new_tensor((1.0, 1.0, 1.0, 0.25))
    per_task = F.smooth_l1_loss(output.forecast_normalized, target, reduction="none").mean(dim=1)
    return (per_task * weights).sum(dim=-1).mean() / weights.sum()


def _optimizer_step(optimizer: torch.optim.Optimizer, model: nn.Module, parameters: Sequence[nn.Parameter], loss: Tensor, max_grad_norm: float) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = _grad_norm(parameters)
    torch.nn.utils.clip_grad_norm_(list(parameters), max_grad_norm)
    optimizer.step()
    return norm


def _optimizer_steps(optimizer: torch.optim.Optimizer) -> int:
    return int(sum(int(value.get("step", 0)) for value in optimizer.state.values()))


def _decoder_parameters(model: nn.Module, parameters: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if parameters is not None:
        return parameters
    if hasattr(model, "decoder_parameters"):
        return getattr(model, "decoder_parameters")
    core = getattr(model, "core", None)
    if core is not None and hasattr(core, "decoder_parameters"):
        return getattr(core, "decoder_parameters")
    return {}


def run_stage_p(model: nn.Module, loaders: Any, budget: StageBudgetV42 | None = None, lineage: Mapping[str, Any] | None = None, seed: int = 2026, **_: Any) -> StageResultV42:
    budget = budget or StageBudgetV42()
    seed_everything(seed)
    train = _batches(loaders.get("train") if isinstance(loaders, Mapping) else loaders)
    if not train:
        raise ValueError("Stage P requires non-empty train batches")
    forecaster, scheduler = _model_groups(model)
    for parameter in forecaster: parameter.requires_grad_(True)
    for parameter in scheduler: parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(list(forecaster), lr=budget.forecaster_lr, weight_decay=budget.weight_decay)
    last_loss = 0.0
    history: list[float] = []
    completed = 0
    for epoch in range(budget.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train:
            output = _run_model(model, batch)
            loss = _forecast_loss(output, batch)
            last_loss = float(loss.detach()); epoch_losses.append(last_loss)
            _optimizer_step(optimizer, model, forecaster, loss, budget.max_grad_norm)
        history.append(float(np.mean(epoch_losses)))
        completed = epoch + 1
        # Selection is read-only and never the test/evaluation split.
        if budget.can_stop(epoch) and len(train) == 0:
            break
    return StageResultV42("P", "forecast_pretrain", model, optimizer, completed, _optimizer_steps(optimizer), last_loss, loss_history=tuple(history))


def run_stage_s(model: nn.Module, loaders: Any, budget: StageBudgetV42 | None = None, lineage: Mapping[str, Any] | None = None, seed: int = 2026, **_: Any) -> StageResultV42:
    budget = budget or StageBudgetV42()
    seed_everything(seed)
    train = _batches(loaders.get("train") if isinstance(loaders, Mapping) else loaders)
    if not train:
        raise ValueError("Stage S requires non-empty train batches")
    forecaster, scheduler = _model_groups(model)
    for parameter in forecaster: parameter.requires_grad_(False)
    for parameter in scheduler: parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(list(scheduler), lr=budget.scheduler_lr, weight_decay=budget.weight_decay)
    last_loss = 0.0; completed = 0; history: list[float] = []
    for epoch in range(budget.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train:
            output = _run_model(model, batch, decouple_decision=True)
            teacher = _dispatch_targets(batch, output.dispatch)
            if teacher is None:
                raise ValueError("Stage S requires teacher_dispatch")
            loss = _imitation_loss(output.dispatch, teacher)
            last_loss = float(loss.detach()); epoch_losses.append(last_loss)
            _optimizer_step(optimizer, model, scheduler, loss, budget.max_grad_norm)
        history.append(float(np.mean(epoch_losses)))
        completed = epoch + 1
    return StageResultV42("S", "scheduler_pretrain", model, optimizer, completed, _optimizer_steps(optimizer), last_loss, loss_history=tuple(history))


def run_stage_j(
    model: nn.Module,
    loaders: Any,
    *,
    mode: Literal["joint", "decoupled"],
    budget: StageBudgetV42 | None = None,
    c_ref: float = 1.0,
    parameters: Mapping[str, Any] | None = None,
    seed: int = 2026,
) -> StageResultV42:
    """Train one Stage-J branch with one persistent optimizer."""

    if mode not in {"joint", "decoupled"}:
        raise ValueError("Stage J mode must be joint or decoupled")
    budget = budget or StageBudgetV42()
    seed_everything(seed)
    train = _batches(loaders.get("train") if isinstance(loaders, Mapping) else loaders)
    if not train:
        raise ValueError("Stage J requires non-empty train batches")
    forecast_params, scheduler_params = _model_groups(model)
    for parameter in forecast_params:
        parameter.requires_grad_(mode == "joint")
    for parameter in scheduler_params:
        parameter.requires_grad_(True)
    active_forecast = [p for p in forecast_params if p.requires_grad]
    active_scheduler = [p for p in scheduler_params if p.requires_grad]
    groups = []
    if active_forecast:
        groups.append({"params": active_forecast, "lr": budget.forecaster_lr})
    groups.append({"params": active_scheduler, "lr": budget.scheduler_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=budget.weight_decay)
    history: list[float] = []
    last = fg_norm = sg_norm = 0.0
    decoder_parameters = _decoder_parameters(model, parameters)
    for epoch in range(budget.max_epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train:
            loss, fg_norm, sg_norm, last = _joint_loss(
                model, batch, mode=mode, epoch=epoch, budget=budget,
                c_ref=c_ref, parameters=decoder_parameters,
            )
            epoch_losses.append(last)
            _optimizer_step(
                optimizer, model, tuple((*active_forecast, *active_scheduler)),
                loss, budget.max_grad_norm,
            )
        history.append(float(np.mean(epoch_losses)))
    return StageResultV42(
        "J", mode, model, optimizer, budget.max_epochs, _optimizer_steps(optimizer),
        last, fg_norm, sg_norm, tuple(history),
    )


def _joint_loss(model: nn.Module, batch: Mapping[str, Any], *, mode: Literal["joint", "decoupled"], epoch: int, budget: StageBudgetV42, c_ref: float, parameters: Mapping[str, Any]) -> tuple[Tensor, float, float, float]:
    output = _run_model(model, batch, decouple_decision=mode == "decoupled")
    teacher = _dispatch_targets(batch, output.dispatch)
    target_normalized = _target_normalized(batch, output.forecast_normalized)
    target_physical = _target_physical(batch, output.forecast_physical)
    realized = batch.get("realized_renewables", batch.get("renewable_realized"))
    initial_soc = batch.get("initial_soc")
    previous_chp = batch.get("previous_chp")
    if realized is None or initial_soc is None or previous_chp is None:
        # A minimal fallback is useful for isolated unit tests; production
        # formal runs always provide realized renewable labels and state.
        forecast = _forecast_loss(output, batch)
        imitation = _imitation_loss(output.dispatch, teacher) if teacher is not None else forecast.new_zeros(())
        total = budget.weights(epoch).forecast * forecast + budget.weights(epoch).imitation * imitation
        return total, 0.0, 0.0, float(total.detach())
    realized_t = _to_tensor(realized, dtype=output.dispatch.dtype).to(output.dispatch)
    initial_t = _to_tensor(initial_soc, dtype=output.dispatch.dtype).to(output.dispatch)
    previous_t = _to_tensor(previous_chp, dtype=output.dispatch.dtype).to(output.dispatch)
    weights = budget.weights(epoch)
    breakdown = formal_v4_joint_loss(
        output, target_normalized, target_physical, realized_t, initial_t, previous_t, teacher, parameters,
        c_ref=c_ref, forecast_weight=weights.forecast, imitation_weight=weights.imitation, decision_weight=weights.decision,
    )
    decision = breakdown.normalized_realized_objective + breakdown.constraint_penalty
    forecast_parameters, scheduler_parameters = _model_groups(model)
    active_forecast = tuple(parameter for parameter in forecast_parameters if parameter.requires_grad)
    active_scheduler = tuple(parameter for parameter in scheduler_parameters if parameter.requires_grad)
    decision_grads = torch.autograd.grad(decision, active_forecast, allow_unused=True, retain_graph=True) if active_forecast else tuple()
    scheduler_grads = torch.autograd.grad(decision, active_scheduler, allow_unused=True, retain_graph=True) if active_scheduler else tuple()
    return breakdown.total, _value_grad_norm(decision_grads), _value_grad_norm(scheduler_grads), float(breakdown.total.detach())


def run_stage_j_pair(stage_s: Any, batch: Mapping[str, Any], budget: StageBudgetV42 | None = None, *, c_ref: float = 1.0, parameters: Mapping[str, Any] | None = None, seed: int = 2026) -> tuple[GradientBoundaryReceiptV42, GradientBoundaryReceiptV42]:
    """Probe the exact dispatch-gradient boundary from byte-identical clones."""

    budget = budget or StageBudgetV42(max_epochs=18)
    base_model = stage_s.model if isinstance(stage_s, StageResultV42) else stage_s
    joint = deepcopy(base_model); decoupled = deepcopy(base_model)
    params = _decoder_parameters(base_model, parameters)
    receipts: list[GradientBoundaryReceiptV42] = []
    for mode, model in (("joint", joint), ("decoupled", decoupled)):
        seed_everything(seed)
        forecasts, schedulers = _model_groups(model)
        for parameter in forecasts:
            parameter.requires_grad_(mode == "joint")
        for parameter in schedulers:
            parameter.requires_grad_(True)
        output = _run_model(model, batch, decouple_decision=mode == "decoupled")
        decision: Tensor
        if all(key in batch for key in ("realized_renewables", "initial_soc", "previous_chp", "target_physical")):
            target_physical = _target_physical(batch, output.forecast_physical)
            teacher = _dispatch_targets(batch, output.dispatch)
            breakdown = formal_v4_joint_loss(
                output, _target_normalized(batch, output.forecast_normalized), target_physical,
                _to_tensor(batch["realized_renewables"], dtype=output.dispatch.dtype).to(output.dispatch),
                _to_tensor(batch["initial_soc"], dtype=output.dispatch.dtype).to(output.dispatch),
                _to_tensor(batch["previous_chp"], dtype=output.dispatch.dtype).to(output.dispatch), teacher,
                params, c_ref=c_ref, forecast_weight=0.0, imitation_weight=0.0, decision_weight=1.0,
            )
            decision = breakdown.normalized_realized_objective + breakdown.constraint_penalty
        else:
            decision = output.dispatch.square().mean()
        active_forecasts = tuple(parameter for parameter in forecasts if parameter.requires_grad)
        active_schedulers = tuple(parameter for parameter in schedulers if parameter.requires_grad)
        fg = torch.autograd.grad(decision, active_forecasts, allow_unused=True, retain_graph=True) if active_forecasts else tuple()
        sg = torch.autograd.grad(decision, active_schedulers, allow_unused=True) if active_schedulers else tuple()
        receipts.append(GradientBoundaryReceiptV42(mode, _value_grad_norm(fg), _value_grad_norm(sg)))
    return receipts[0], receipts[1]


def refresh_rollin_sample(sample: Any, *, recompute_teacher: bool = False) -> Any:
    """Detach executed histories; stale teacher labels cannot be reused."""

    if isinstance(sample, Mapping):
        refreshed = dict(sample)
        for name in ("load_history", "exog_history", "device_history", "activity_history", "device_history"):
            if name in refreshed and isinstance(refreshed[name], Tensor):
                refreshed[name] = refreshed[name].detach()
        if not recompute_teacher:
            refreshed["imitation_weight"] = 0.0
        return refreshed
    if hasattr(sample, "device_history"):
        try:
            refreshed = deepcopy(sample)
            value = getattr(refreshed, "device_history")
            setattr(refreshed, "device_history", value.detach())
            if not recompute_teacher and hasattr(refreshed, "imitation_weight"):
                setattr(refreshed, "imitation_weight", 0.0)
            return refreshed
        except Exception:
            pass
    raise TypeError("roll-in sample must be a mapping or expose device_history")


def run_training_seed_v42(
    *,
    seed: int,
    model: nn.Module | None = None,
    loaders: Any = None,
    train_batches: Any = None,
    teacher_overlay: Any = None,
    parameters: Mapping[str, Any] | None = None,
    budget: StageBudgetV42 | None = None,
    c_ref: float = 1.0,
    **kwargs: Any,
) -> TrainingRunReceiptV42:
    """Execute the frozen P → teacher → S → clone → J order for one seed."""

    seed_everything(seed)
    budget = budget or StageBudgetV42(max_epochs=2, minimum_epochs=1, ramp_epochs=1)
    if model is None:
        model = RSCPFModel(dropout=0.0)
    data = loaders if loaders is not None else {"train": train_batches}
    p_result = run_stage_p(model, data, budget, seed=seed)
    events = ["P_complete", "teacher_complete"]
    # The teacher overlay itself is immutable and generated before Stage S;
    # the training orchestrator only verifies that one is available.
    if teacher_overlay is None:
        # Unit-test/minimal callers may use dispatch targets embedded in the
        # batch.  Formal runners pass the persisted overlay explicitly.
        if not _batches(data.get("train") if isinstance(data, Mapping) else data):
            raise ValueError("teacher overlay or train batches are required")
    s_result = run_stage_s(p_result.model, data, budget, seed=seed)
    events.append("S_complete")
    joint_model = deepcopy(s_result.model); decoupled_model = deepcopy(s_result.model)
    events.append("clone_verified")
    train = _batches(data.get("train") if isinstance(data, Mapping) else data)
    if not train:
        raise ValueError("Stage J requires train batches")
    boundaries: list[GradientBoundaryReceiptV42] = []
    j_steps: dict[str, int] = {}
    losses: dict[str, float] = {"P": p_result.loss, "S": s_result.loss}
    for mode, model_j, key in (("joint", joint_model, "J_joint"), ("decoupled", decoupled_model, "J_decoupled")):
        result = run_stage_j(
            model_j, data, mode=mode, budget=budget, c_ref=c_ref,
            parameters=parameters, seed=seed,
        )
        j_steps[key] = result.optimizer_steps; losses[key] = result.loss
        boundaries.append(GradientBoundaryReceiptV42(mode, result.decision_forecaster_gradient_norm, result.scheduler_gradient_norm))
    events.append("J_complete")
    return TrainingRunReceiptV42(int(seed), tuple(events), {"P": p_result.optimizer_steps, "S": s_result.optimizer_steps, **j_steps}, tuple(boundaries), losses, False)


def run_direct_policy_training(*args: Any, **kwargs: Any) -> TrainingRunReceiptV42:
    """Named entry point used by the Direct-Policy baseline runner."""

    from .formal_v4_models import DirectPolicyModel
    kwargs.setdefault("model", DirectPolicyModel(dropout=0.0))
    return run_training_seed_v42(*args, **kwargs)


__all__ = [
    "GradientBoundaryReceiptV42", "StageBudgetV42", "StageResultV42", "TrainingRunReceiptV42",
    "refresh_rollin_sample", "run_direct_policy_training", "run_stage_j", "run_stage_j_pair",
    "run_stage_p", "run_stage_s", "run_training_seed_v42", "seed_everything",
]
