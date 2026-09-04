"""One-optimizer joint training and provenance-aware checkpoint helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .contract import JointTrainingContract, JointVariant
from .data import JointWindowSplit
from .losses import CurriculumWeights, JointLossBreakdown, JointV3LossBreakdown, joint_forecast_dispatch_loss, joint_forecast_dispatch_loss_v3, weights_for_epoch
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
    normalize_decision: bool = False,
    loss_weight_multipliers: Mapping[str, float] | None = None,
    forecast_task_weights: Sequence[float] | None = None,
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
    if all(name in batch for name in ("realized_renewables", "initial_soc", "oracle_four_step_objective")):
        breakdown = joint_forecast_dispatch_loss_v3(
            output,
            _batch_value(batch, "target_normalized"),
            _batch_value(batch, "target_physical"),
            _batch_value(batch, "realized_renewables"),
            _batch_value(batch, "initial_soc"),
            _batch_value(batch, "previous_chp"),
            _batch_value(batch, "teacher_dispatch"),
            _batch_value(batch, "oracle_four_step_objective"),
            parameters,
            weights,
            forecast_task_weights=forecast_task_weights,
        )
        decision_shortage = breakdown.normalized_shortage
        decision_term = breakdown.four_hour_optimality_gap + decision_shortage + breakdown.constraint_penalty
    else:
        breakdown = joint_forecast_dispatch_loss(
            output,
            _batch_value(batch, "target_normalized"),
            _batch_value(batch, "target_physical"),
            _batch_value(batch, "teacher_dispatch"),
            _batch_value(batch, "oracle_first_step_objective"),
            parameters,
            weights,
            normalize_decision=normalize_decision,
        )
        decision_shortage = breakdown.normalized_shortage if normalize_decision else breakdown.shortage
        decision_term = breakdown.regret + decision_shortage
    if decision_shortage is None:  # pragma: no cover - defensive for custom breakdowns
        decision_shortage = breakdown.shortage
    multipliers = {"forecast": 1.0, "imitation": 1.0, "decision": 1.0}
    if loss_weight_multipliers is not None:
        for name, value in loss_weight_multipliers.items():
            if name not in multipliers or not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("loss_weight_multipliers must contain finite non-negative forecast/imitation/decision values")
            multipliers[name] = float(value)
    if isinstance(breakdown, JointV3LossBreakdown):
        total = (
            multipliers["forecast"] * weights.forecast * breakdown.forecast
            + multipliers["imitation"] * weights.imitation * breakdown.imitation
            + multipliers["decision"] * weights.decision * (breakdown.four_hour_optimality_gap + breakdown.normalized_shortage + breakdown.constraint_penalty)
        )
        breakdown = JointV3LossBreakdown(
            total=total, forecast=breakdown.forecast, imitation=breakdown.imitation,
            four_hour_optimality_gap=breakdown.four_hour_optimality_gap, shortage=breakdown.shortage,
            constraint_penalty=breakdown.constraint_penalty, normalized_shortage=breakdown.normalized_shortage,
        )
    else:
        total = (
            multipliers["forecast"] * weights.forecast * breakdown.forecast
            + multipliers["imitation"] * weights.imitation * breakdown.imitation
            + multipliers["decision"] * weights.decision * (breakdown.regret + decision_shortage)
        )
        breakdown = JointLossBreakdown(
            total=total, forecast=breakdown.forecast, imitation=breakdown.imitation,
            regret=breakdown.regret, shortage=breakdown.shortage, carbon_metric=breakdown.carbon_metric,
            planned_feasibility=breakdown.planned_feasibility, normalized_shortage=breakdown.normalized_shortage,
        )
    trainable_forecaster = tuple(parameter for parameter in model.forecaster.parameters() if parameter.requires_grad)
    if decision_term.requires_grad and trainable_forecaster:
        decision_grads = torch.autograd.grad(
            decision_term,
            trainable_forecaster,
            allow_unused=True,
            retain_graph=True,
        )
    else:
        decision_grads = tuple(None for _ in model.forecaster.parameters())
    decision_norms = [value.detach().float().norm() for value in decision_grads if value is not None]
    forecaster_decision_norm = float(torch.stack(decision_norms).norm().item()) if decision_norms else 0.0
    if optimizer is None:
        return JointStepResult(
            breakdown,
            GradientCouplingAudit(0.0, 0.0, forecaster_decision_norm, forecaster_decision_norm > 0.0, False),
            weights,
        )
    breakdown.total.backward()
    forecaster_norm = _norm(trainable_forecaster)
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


def rollin_refresh_epoch(max_epochs: int, start_fraction: float = 0.4) -> int:
    if max_epochs <= 0 or not 0.0 < start_fraction < 1.0:
        raise ValueError("max_epochs must be positive and start_fraction must be in (0,1)")
    # Refresh only after at least one initial LP-history epoch has completed;
    # this keeps short smoke runs from collecting a roll-in before the policy
    # has taken a gradient step.  For the formal 3+ epoch protocol this is
    # exactly floor(0.4 * epochs).
    return max(1, int(np.floor(float(max_epochs) * float(start_fraction))))


def mix_history_windows(
    lp_history: JointWindowSplit,
    model_history: JointWindowSplit,
    *,
    model_history_fraction: float = 0.5,
    seed: int = 2026,
) -> JointWindowSplit:
    """Sample a reproducible LP/model-history mixture for train only."""

    if lp_history.split != "train" or model_history.split != "train":
        raise ValueError("history mixing is legal only for train splits")
    if lp_history.history_source != "causal_lp" or model_history.history_source != "joint_policy_rollin":
        raise ValueError("history sources must be causal_lp and joint_policy_rollin")
    if not 0.0 <= model_history_fraction <= 1.0:
        raise ValueError("model_history_fraction must be in [0,1]")
    total = min(len(lp_history), len(model_history))
    if total <= 0:
        raise ValueError("cannot mix empty histories")
    model_count = int(round(total * model_history_fraction))
    lp_count = total - model_count
    rng = np.random.default_rng(seed)
    # LP and roll-in artifacts are generated for the same target origins.  A
    # mixture therefore selects the history source *per origin*; concatenating
    # both copies would create duplicate target timestamps and violate the
    # chronological-window contract.  If origins differ, fail closed rather
    # than silently pairing the wrong future teacher with a history window.
    lp_times = np.asarray(lp_history.target_times, dtype="datetime64[ns]")
    model_times = np.asarray(model_history.target_times, dtype="datetime64[ns]")
    if not np.array_equal(np.sort(lp_times), np.sort(model_times)):
        raise ValueError("LP and roll-in histories must share identical target origins")
    lp_by_time = {time: idx for idx, time in enumerate(lp_times)}
    model_by_time = {time: idx for idx, time in enumerate(model_times)}
    origins = np.sort(lp_times)
    model_positions = set(rng.choice(len(origins), size=model_count, replace=False).tolist()) if model_count else set()
    selected = []
    for position, time in enumerate(origins):
        source = model_history if position in model_positions else lp_history
        index = (model_by_time if source is model_history else lp_by_time)[time]
        selected.append((source, index))
    fields = {}
    names = (
        "load_history", "exog_history", "device_history", "device_status", "forecast_target",
        "scheduler_context", "previous_chp", "teacher_dispatch", "oracle_first_step_objective", "target_times",
    )
    for name in names:
        fields[name] = np.stack([getattr(source, name)[index] for source, index in selected], axis=0)
    return JointWindowSplit(
        **fields,
        split="train",
        history_source="joint_policy_rollin",
    )


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
    loss_weights: Mapping[str, float] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "epoch": int(epoch), "variant": str(variant), "seed": int(seed),
        "contract_hash": str(contract_hash), "normalization_hash": str(normalization_hash),
        "data_hash": str(data_hash), "best_validation_score": best_validation_score,
        "loss_weights": None if loss_weights is None else dict(loss_weights),
        "lr_scheduler": None,
        "gradient_audit": None if gradient_audit is None else asdict(gradient_audit),
        "test_set_accessed": False,
        "rng_state": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "metadata": None if metadata is None else dict(metadata),
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
    state = payload.get("model", payload.get("model_state_dict", payload))
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint model state must be a mapping")
    current = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    rejected: list[str] = []
    for raw_name, value in state.items():
        name = str(raw_name)
        # Legacy Scheme2R checkpoints predate the connected wrapper and use
        # ``encoders.*``/``router.*`` at the root.  Admit them only through the
        # exact forecast whitelist after mapping into ``forecaster.base.*``;
        # newly added device/state-fusion/scheduler-input parameters remain
        # random by construction.
        candidate_names = [name]
        if not name.startswith("forecaster.") and not name.startswith("scheduler."):
            candidate_names.append(f"forecaster.base.{name}")
            if name.startswith(("residual.", "output_projection.")):
                candidate_names.append(f"scheduler.{name}")
        selected_name = None
        for candidate in candidate_names:
            normalized = candidate.removeprefix("forecaster.")
            scheduler_local = candidate.removeprefix("scheduler.")
            admitted = any(normalized.startswith(prefix) for prefix in forecast_prefixes) or any(scheduler_local.startswith(prefix) for prefix in scheduler_prefixes)
            if admitted and candidate in current:
                selected_name = candidate
                break
        if selected_name is None:
            rejected.append(name)
            continue
        if tuple(current[selected_name].shape) != tuple(value.shape):
            skipped.append(name)
            continue
        current[selected_name].copy_(value)
        loaded.append(selected_name)
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
    "load_allowed_warm_start", "mix_history_windows", "rollin_refresh_epoch",
    "run_joint_training", "save_joint_checkpoint",
]
