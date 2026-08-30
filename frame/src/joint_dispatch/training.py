"""One-optimizer joint training and provenance-aware checkpoint helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .contract import JointTrainingContract, JointVariant
from .losses import CurriculumWeights, JointLossBreakdown, joint_forecast_dispatch_loss, weights_for_epoch
from .model import JointForecastDispatchModel
from .rollout import ClosedLoopRollout, rollout_joint_policy


@dataclass(frozen=True)
class GradientCouplingAudit:
    forecaster_norm_from_total: float
    scheduler_norm_from_total: float
    forecaster_norm_from_decision_only: float
    forecaster_nonzero: bool
    scheduler_nonzero: bool


@dataclass(frozen=True)
class JointStepResult:
    loss: JointLossBreakdown
    gradient_audit: GradientCouplingAudit
    weights: CurriculumWeights


@dataclass(frozen=True)
class WarmStartReceipt:
    loaded: tuple[str, ...]
    skipped: tuple[str, ...]
    rejected: tuple[str, ...]
    source_sha256: str


def _contract_value(contract: JointTrainingContract | None, name: str, default: float) -> float:
    if contract is None:
        return float(default)
    # The frozen v1 contract intentionally keeps optimizer details outside the
    # scientific selection fields; use conservative fixed defaults here.
    return float(getattr(contract, name, default))


def build_joint_optimizer(
    model: JointForecastDispatchModel,
    contract: JointTrainingContract | None = None,
    *,
    variant: JointVariant | str = "joint_from_scratch",
) -> torch.optim.Optimizer | None:
    """Create one AdamW optimizer covering every trainable graph parameter."""

    variant_name = variant.name if isinstance(variant, JointVariant) else str(variant)
    if variant_name not in {"joint_from_scratch", "warm_started_joint", "frozen_pto"}:
        raise ValueError("unknown joint variant")
    if variant_name == "frozen_pto":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return None
    groups = [
        {"params": list(model.forecaster.parameters()), "lr": _contract_value(contract, "forecaster_lr", 1.0e-3)},
        {"params": list(model.scheduler.parameters()), "lr": _contract_value(contract, "scheduler_lr", 1.0e-3)},
    ]
    seen: set[int] = set()
    deduplicated: list[dict[str, object]] = []
    for group in groups:
        parameters = []
        for parameter in group["params"]:  # type: ignore[union-attr]
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameter.requires_grad_(True)
            parameters.append(parameter)
        if parameters:
            deduplicated.append({"params": parameters, "lr": group["lr"]})
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    included = {id(parameter) for group in deduplicated for parameter in group["params"]}  # type: ignore[union-attr]
    if expected != included:
        raise ValueError("joint optimizer omitted trainable parameters")
    return torch.optim.AdamW(deduplicated, weight_decay=_contract_value(contract, "weight_decay", 1.0e-4))


def _norm(parameters: Iterable[Tensor]) -> float:
    values = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm().item()) if values else 0.0


def _batch_value(batch: Mapping[str, Tensor], name: str) -> Tensor:
    if name not in batch:
        raise KeyError(f"joint batch is missing {name}")
    return batch[name]


def joint_train_step(
    model: JointForecastDispatchModel,
    batch: Mapping[str, Tensor],
    parameters: Mapping[str, Any],
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    ramp_epochs: int = 20,
    grad_clip_norm: float = 1.0,
) -> JointStepResult:
    """Run exactly one complete forward/backward/optimizer step."""

    weights = weights_for_epoch(epoch=epoch, ramp_epochs=ramp_epochs)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = model(
        load_history=_batch_value(batch, "load_history"),
        exog_history=_batch_value(batch, "exog_history"),
        device_history=_batch_value(batch, "device_history"),
        device_status=_batch_value(batch, "device_status"),
        scheduler_context=_batch_value(batch, "scheduler_context"),
        previous_chp=_batch_value(batch, "previous_chp"),
    )
    breakdown = joint_forecast_dispatch_loss(
        output,
        _batch_value(batch, "target_normalized"),
        _batch_value(batch, "target_physical"),
        _batch_value(batch, "teacher_dispatch"),
        _batch_value(batch, "oracle_first_step_objective"),
        parameters,
        weights,
    )
    decision_term = breakdown.regret + breakdown.shortage
    decision_grads = torch.autograd.grad(
        decision_term,
        tuple(model.forecaster.parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    decision_norms = [value.detach().float().norm() for value in decision_grads if value is not None]
    forecaster_decision_norm = float(torch.stack(decision_norms).norm().item()) if decision_norms else 0.0
    if optimizer is None:
        return JointStepResult(
            breakdown,
            GradientCouplingAudit(0.0, 0.0, forecaster_decision_norm, forecaster_decision_norm > 0.0, False),
            weights,
        )
    breakdown.total.backward()
    forecaster_norm = _norm(model.forecaster.parameters())
    scheduler_norm = _norm(model.scheduler.parameters())
    if grad_clip_norm <= 0.0:
        raise ValueError("grad_clip_norm must be positive")
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    audit = GradientCouplingAudit(
        forecaster_norm, scheduler_norm, forecaster_decision_norm,
        forecaster_norm > 0.0, scheduler_norm > 0.0,
    )
    return JointStepResult(breakdown, audit, weights)


def collect_training_rollin(*args: Any, **kwargs: Any) -> ClosedLoopRollout:
    """Collect one training-only policy roll-in through the shared rollout."""

    return rollout_joint_policy(*args, **kwargs)


def _atomic_torch_save(payload: Mapping[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
    temporary.replace(destination)


def save_joint_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    epoch: int,
    variant: str,
    seed: int,
    contract_hash: str = "",
    normalization_hash: str = "",
    data_hash: str = "",
    gradient_audit: GradientCouplingAudit | None = None,
    best_validation_score: float | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch), "variant": str(variant), "seed": int(seed),
        "contract_hash": str(contract_hash), "normalization_hash": str(normalization_hash),
        "data_hash": str(data_hash), "best_validation_score": best_validation_score,
        "gradient_audit": None if gradient_audit is None else asdict(gradient_audit),
        "test_set_accessed": False,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    _atomic_torch_save(payload, destination)


def load_allowed_warm_start(
    model: JointForecastDispatchModel,
    checkpoint_path: str | Path,
    *,
    variant: str,
    forecast_prefixes: tuple[str, ...],
    scheduler_prefixes: tuple[str, ...],
) -> WarmStartReceipt:
    """Load only exact-name/exact-shape tensors admitted by the contract."""

    if variant == "joint_from_scratch":
        raise ValueError("joint_from_scratch cannot load a checkpoint")
    source = Path(checkpoint_path)
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model state must be a mapping")
    current = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    rejected: list[str] = []
    for raw_name, value in state.items():
        name = str(raw_name)
        normalized = name.removeprefix("forecaster.")
        admitted = any(normalized.startswith(prefix) for prefix in forecast_prefixes) or any(name.startswith(prefix) for prefix in scheduler_prefixes)
        if not admitted:
            rejected.append(name)
            continue
        if name not in current or tuple(current[name].shape) != tuple(value.shape):
            skipped.append(name)
            continue
        current[name].copy_(value)
        loaded.append(name)
    model.load_state_dict(current)
    return WarmStartReceipt(tuple(sorted(loaded)), tuple(sorted(skipped)), tuple(sorted(rejected)), source_hash)


def run_joint_training(
    model: JointForecastDispatchModel,
    batches: Iterable[Mapping[str, Tensor]],
    parameters: Mapping[str, Any],
    *,
    epochs: int,
    optimizer: torch.optim.Optimizer | None = None,
    variant: str = "joint_from_scratch",
) -> tuple[JointStepResult, ...]:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if optimizer is None and variant != "frozen_pto":
        optimizer = build_joint_optimizer(model, variant=variant)
    history: list[JointStepResult] = []
    for epoch in range(epochs):
        for batch in batches:
            history.append(joint_train_step(model, batch, parameters, optimizer, epoch=epoch))
    return tuple(history)


__all__ = [
    "GradientCouplingAudit", "JointStepResult", "WarmStartReceipt",
    "build_joint_optimizer", "collect_training_rollin", "joint_train_step",
    "load_allowed_warm_start", "run_joint_training", "save_joint_checkpoint",
]
