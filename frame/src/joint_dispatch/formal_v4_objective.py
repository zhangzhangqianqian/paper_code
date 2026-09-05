"""Formal-v4 decision objective and differentiable four-hour settlement.

The v4 objective deliberately does not divide a decision loss by a per-window
oracle.  A single ``C_ref`` is fitted on the training split and then frozen for
all methods, seeds, and evaluation windows.  This keeps the objective
comparable and prevents a near-zero oracle from amplifying gradients.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..scheduling.dispatch_schema import VARIABLES
from .contract import FORECAST_TASK_WEIGHTS
from .formal_v4_recourse import settle_first_step_v4


STEP_WEIGHTS: tuple[float, float, float, float] = (0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
_I = {name: index for index, name in enumerate(VARIABLES)}


def _finite_scalar(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class TrainingObjectiveScale:
    """One immutable decision scale fitted from training-only PI-LP windows."""

    c_ref: float
    source_split: str = "train"
    raw_mean: float = 0.0
    raw_median: float = 0.0
    raw_iqr: float = 0.0
    sample_count: int = 0
    capacity_scenario_hash: str = ""
    source_hash: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.c_ref)) or float(self.c_ref) <= 0.0:
            raise ValueError("c_ref must be finite and positive")
        if self.source_split != "train":
            raise ValueError("formal-v4 C_ref must come from the train split")
        if int(self.sample_count) < 0:
            raise ValueError("sample_count must be non-negative")
        for name in ("raw_mean", "raw_median", "raw_iqr"):
            if not np.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_c_ref_receipt(
    receipt: Mapping[str, Any],
    *,
    train_archive_sha256: str | None = None,
    capacity_receipt_sha256: str | None = None,
) -> TrainingObjectiveScale:
    """Validate a persisted train-only ``C_ref`` receipt before reuse."""

    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != "formal-v4.1-c-ref-receipt-v1":
        raise ValueError("unsupported C_ref receipt schema")
    if receipt.get("source_split") != "train":
        raise ValueError("C_ref receipt must be fitted on train")
    if train_archive_sha256 is not None and receipt.get("train_archive_sha256") != train_archive_sha256:
        raise ValueError("C_ref receipt train archive hash mismatch")
    if capacity_receipt_sha256 is not None and receipt.get("capacity_receipt_sha256") != capacity_receipt_sha256:
        raise ValueError("C_ref receipt capacity hash mismatch")
    weights = tuple(float(value) for value in receipt.get("step_weights", ()))
    if weights != STEP_WEIGHTS:
        raise ValueError("C_ref receipt step weights are not frozen")
    unsigned = {str(key): value for key, value in receipt.items() if key != "c_ref_sha256"}
    expected_hash = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    if receipt.get("c_ref_sha256") != expected_hash:
        raise ValueError("C_ref receipt hash mismatch")
    return TrainingObjectiveScale(
        c_ref=float(receipt["c_ref"]), source_split="train",
        raw_mean=float(receipt["raw_mean"]), raw_median=float(receipt["raw_median"]),
        raw_iqr=float(receipt["raw_iqr"]), sample_count=int(receipt["sample_count"]),
        capacity_scenario_hash=str(receipt.get("capacity_scenario_hash", "")),
        source_hash=str(receipt["train_archive_sha256"]),
    )


@dataclass(frozen=True)
class FormalV4FourHourSettlement:
    """Differentiable realized settlement used by formal-v4 training."""

    per_step_penalized_objective: Tensor  # [B,4], shortage/dump penalties included
    constraint_penalty: Tensor  # [B]
    normalized_shortage: Tensor  # [B], reporting-only; never added twice

    def __post_init__(self) -> None:
        if self.per_step_penalized_objective.ndim != 2 or tuple(self.per_step_penalized_objective.shape[1:]) != (4,):
            raise ValueError("per_step_penalized_objective must have shape [B,4]")
        batch = self.per_step_penalized_objective.shape[0]
        for name, value in (("constraint_penalty", self.constraint_penalty), ("normalized_shortage", self.normalized_shortage)):
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
        if not bool(torch.isfinite(self.per_step_penalized_objective).all()):
            raise ValueError("per_step_penalized_objective must be finite")


@dataclass(frozen=True)
class FormalV4LossBreakdown:
    total: Tensor
    forecast: Tensor
    imitation: Tensor
    normalized_realized_objective: Tensor
    normalized_shortage: Tensor
    constraint_penalty: Tensor


def fit_training_objective_scale(
    train_objectives: np.ndarray,
    *,
    capacity_scenario_hash: str = "",
    source_hash: str = "",
) -> TrainingObjectiveScale:
    """Fit the frozen ``C_ref=max(median,1)`` rule on train-only objectives.

    A one-dimensional input is interpreted as already aggregated window
    objectives.  A ``[N,4]`` matrix is aggregated with the fixed execution-step
    weights before fitting, so the same rule applies to per-hour PI-LP output.
    """

    values = np.asarray(train_objectives, dtype=np.float64)
    if values.ndim == 2:
        if values.shape[1] != 4:
            raise ValueError("train_objectives matrix must have four horizon steps")
        values = values @ np.asarray(STEP_WEIGHTS, dtype=np.float64)
    elif values.ndim != 1:
        raise ValueError("train_objectives must be one-dimensional or [N,4]")
    if values.size == 0 or not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("train_objectives must be non-empty, finite and non-negative")
    mean = float(np.mean(values))
    median = float(np.median(values))
    q25, q75 = np.percentile(values, (25.0, 75.0))
    return TrainingObjectiveScale(
        c_ref=max(median, 1.0), source_split="train", raw_mean=mean,
        raw_median=median, raw_iqr=float(q75 - q25), sample_count=int(values.size),
        capacity_scenario_hash=str(capacity_scenario_hash), source_hash=str(source_hash),
    )


def formal_v4_curriculum_weights(
    *,
    epoch: int,
    ramp_epochs: int,
    decision_start: float = 0.05,
    decision_final: float = 1.0,
    imitation_start: float = 1.0,
    imitation_final: float = 0.0,
    forecast: float = 1.0,
) -> Any:
    """Return finite curriculum weights with decision learning active at epoch 0."""

    if int(epoch) < 0 or int(ramp_epochs) <= 0:
        raise ValueError("epoch must be non-negative and ramp_epochs must be positive")
    endpoints = (decision_start, decision_final, imitation_start, imitation_final, forecast)
    if not all(np.isfinite(float(value)) and float(value) >= 0.0 for value in endpoints):
        raise ValueError("curriculum weights must be finite and non-negative")
    if decision_start <= 0.0 or decision_final < decision_start or imitation_final > imitation_start:
        raise ValueError("invalid curriculum endpoints")
    alpha = min(max(float(epoch) / float(ramp_epochs), 0.0), 1.0)
    from .losses import CurriculumWeights

    return CurriculumWeights(
        forecast=float(forecast),
        imitation=float(imitation_start + alpha * (imitation_final - imitation_start)),
        decision=float(decision_start + alpha * (decision_final - decision_start)),
    )


def clip_formal_v4_gradients(parameters: Sequence[torch.nn.Parameter], *, max_norm: float) -> float:
    """Clip a batch's gradients and return the pre-clipping total norm."""

    limit = _finite_scalar(max_norm, "max_norm")
    if limit <= 0.0:
        raise ValueError("max_norm must be positive")
    return float(torch.nn.utils.clip_grad_norm_(list(parameters), limit))


def _as_column(value: Tensor, batch: int, name: str) -> Tensor:
    if value.ndim == 1 and value.shape[0] == batch:
        value = value.unsqueeze(-1)
    if value.shape != (batch, 1) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must have shape [B] or [B,1] and be finite")
    return value


def _parameter_at_step(parameters: Mapping[str, Any], name: str, reference: Tensor, step: int, default: float = 0.0) -> Tensor:
    raw = parameters.get(name, default)
    tensor = torch.as_tensor(raw, dtype=reference.dtype, device=reference.device)
    if tensor.ndim == 0:
        return tensor.expand(reference.shape)
    if tensor.ndim == 1 and tensor.shape[0] == 4:
        return tensor[step].expand(reference.shape)
    if tensor.shape == reference.shape:
        return tensor
    raise ValueError(f"{name} must be scalar, [4], or [B]")


def settle_formal_v4_four_hour(
    planned_dispatch: Tensor,
    actual_demand: Tensor,
    actual_renewables: Tensor,
    initial_soc: Tensor,
    previous_chp: Tensor,
    parameters: Mapping[str, Any],
) -> FormalV4FourHourSettlement:
    """Settle four planned actions against realized labels without an LP call."""

    if planned_dispatch.ndim != 3 or tuple(planned_dispatch.shape[1:]) != (4, len(VARIABLES)):
        raise ValueError(f"planned_dispatch must have shape [B,4,{len(VARIABLES)}]")
    batch = planned_dispatch.shape[0]
    if actual_demand.shape != (batch, 4, 3) or actual_renewables.shape != (batch, 4, 2):
        raise ValueError("actual demand and renewables must have shapes [B,4,3] and [B,4,2]")
    if not bool(torch.isfinite(planned_dispatch).all() and torch.isfinite(actual_demand).all() and torch.isfinite(actual_renewables).all()):
        raise ValueError("settlement inputs must be finite")
    initial_soc = _as_column(initial_soc, batch, "initial_soc")
    previous_chp = _as_column(previous_chp, batch, "previous_chp")

    outcomes = []
    current_soc = initial_soc
    current_previous_chp = previous_chp
    for step in range(4):
        outcome = settle_first_step_v4(
            planned_dispatch[:, step, :], actual_demand[:, step, :],
            actual_renewables[:, step, :], parameters,
            initial_soc=current_soc, previous_chp=current_previous_chp,
            grid_price=_parameter_at_step(parameters, "grid_energy_price", planned_dispatch[:, step, 0], step, 0.0),
            gas_price=_parameter_at_step(parameters, "gas_energy_price", planned_dispatch[:, step, 0], step, 0.0),
            carbon_price=_parameter_at_step(parameters, "carbon_price", planned_dispatch[:, step, 0], step, float(parameters.get("carbon_price_default", 0.0))),
        )
        outcomes.append(outcome)
        current_soc = outcome.next_soc
        current_previous_chp = outcome.next_previous_chp

    settled_dispatch = torch.stack([item.realized_dispatch for item in outcomes], dim=1)
    per_step = torch.stack([item.penalized_objective for item in outcomes], dim=1)
    shortage = torch.stack([item.shortage for item in outcomes], dim=1).sum(dim=(1, 2))
    normalized_shortage = shortage / actual_demand.abs().sum(dim=(1, 2)).clamp_min(1.0)

    balance = torch.stack([item.balance_residuals for item in outcomes], dim=1)
    conversion = torch.stack([item.conversion_residuals for item in outcomes], dim=1)
    capacity = _finite_scalar(parameters.get("bess_energy_capacity", 1.0), "bess_energy_capacity")
    eta = _finite_scalar(parameters.get("bess_roundtrip_efficiency", 1.0), "bess_roundtrip_efficiency") ** 0.5
    previous_energy = initial_soc[:, 0] * capacity
    soc_residual = []
    for step in range(4):
        soc = settled_dispatch[:, step, _I["soc"]]
        soc_residual.append(soc - previous_energy - eta * settled_dispatch[:, step, _I["p_charge"]] + settled_dispatch[:, step, _I["p_discharge"]] / max(eta, 1.0e-8))
        previous_energy = soc
    soc_residual_tensor = torch.stack(soc_residual, dim=1)
    constraint_penalty = balance.abs().mean(dim=(1, 2)) + conversion.abs().mean(dim=(1, 2)) + soc_residual_tensor.abs().mean(dim=1)
    return FormalV4FourHourSettlement(per_step, constraint_penalty, normalized_shortage)


def _scalar_c_ref(c_ref: float | TrainingObjectiveScale) -> float:
    if isinstance(c_ref, TrainingObjectiveScale):
        return float(c_ref.c_ref)
    return _finite_scalar(c_ref, "c_ref")


def formal_v4_joint_loss(
    output: Any,
    target_normalized: Tensor,
    target_physical: Tensor,
    realized_renewables: Tensor,
    initial_soc: Tensor,
    previous_chp: Tensor,
    teacher_dispatch: Tensor | None,
    parameters: Mapping[str, Any],
    *,
    c_ref: float | TrainingObjectiveScale,
    forecast_weight: float = 1.0,
    imitation_weight: float = 0.25,
    decision_weight: float = 1.0,
    physics_weight: float = 1.0,
    forecast_task_weights: Sequence[float] = FORECAST_TASK_WEIGHTS,
    step_weights: Sequence[float] = STEP_WEIGHTS,
    oracle_diagnostic: Tensor | None = None,
    settled: FormalV4FourHourSettlement | None = None,
    supervised_forecast_loss: Tensor | None = None,
) -> FormalV4LossBreakdown:
    """Compute the formal-v4 loss while preserving forecast-to-dispatch gradients."""

    forecast = getattr(output, "forecast_normalized", None)
    dispatch = getattr(output, "dispatch", None)
    if not isinstance(forecast, Tensor) or not isinstance(dispatch, Tensor):
        raise ValueError("output must expose forecast_normalized and dispatch tensors")
    if forecast.shape != target_normalized.shape or forecast.ndim != 3 or tuple(forecast.shape[1:]) != (4, 4):
        raise ValueError("forecast and target_normalized must have shape [B,4,4]")
    if dispatch.ndim != 3 or tuple(dispatch.shape[1:]) != (4, len(VARIABLES)):
        raise ValueError(f"dispatch must have shape [B,4,{len(VARIABLES)}]")
    if target_physical.shape != (dispatch.shape[0], 4, 4):
        raise ValueError("target_physical must have shape [B,4,4]")
    if realized_renewables.shape != (dispatch.shape[0], 4, 2):
        raise ValueError("realized_renewables must have shape [B,4,2]")
    if not bool(torch.isfinite(target_normalized).all() and torch.isfinite(target_physical).all() and torch.isfinite(realized_renewables).all()):
        raise ValueError("loss targets must be finite")
    task_weights = forecast.new_tensor(tuple(float(value) for value in forecast_task_weights))
    if task_weights.shape != (4,) or bool((task_weights < 0.0).any()) or float(task_weights.sum()) <= 0.0:
        raise ValueError("forecast_task_weights must contain four non-negative values with a positive sum")
    horizons = forecast.new_tensor(tuple(float(value) for value in step_weights))
    if horizons.shape != (4,) or bool((horizons < 0.0).any()) or not torch.isclose(horizons.sum(), forecast.new_tensor(1.0), atol=1.0e-8):
        raise ValueError("step_weights must be non-negative and sum to one")

    if supervised_forecast_loss is None:
        per_task = F.smooth_l1_loss(forecast, target_normalized, reduction="none").mean(dim=1)
        forecast_loss = (per_task * task_weights).sum(dim=-1).mean() / task_weights.sum()
    else:
        if supervised_forecast_loss.ndim != 0 or not bool(torch.isfinite(supervised_forecast_loss).all()):
            raise ValueError("supervised_forecast_loss must be a finite scalar")
        forecast_loss = supervised_forecast_loss.to(dtype=forecast.dtype, device=forecast.device)
    if teacher_dispatch is None:
        imitation_loss = forecast_loss.new_zeros(())
    else:
        if teacher_dispatch.shape != dispatch.shape or not bool(torch.isfinite(teacher_dispatch).all()):
            raise ValueError("teacher_dispatch must match dispatch and be finite")
        denominator = teacher_dispatch.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
        imitation_loss = F.smooth_l1_loss(dispatch / denominator, teacher_dispatch.to(dispatch) / denominator)

    settlement = settled if settled is not None else settle_formal_v4_four_hour(
        dispatch, target_physical[..., :3], realized_renewables, initial_soc, previous_chp, parameters,
    )
    reference = dispatch.new_tensor(_scalar_c_ref(c_ref)).clamp_min(1.0e-12)
    normalized_objective = (settlement.per_step_penalized_objective * horizons).sum(dim=1).mean() / reference
    penalty = settlement.constraint_penalty.mean()
    decision = normalized_objective + float(physics_weight) * penalty
    total = float(forecast_weight) * forecast_loss + float(imitation_weight) * imitation_loss + float(decision_weight) * decision
    # The oracle is intentionally diagnostic-only.  Keeping this explicit
    # prevents accidental reintroduction of per-window oracle normalization.
    _ = oracle_diagnostic
    return FormalV4LossBreakdown(
        total=total, forecast=forecast_loss, imitation=imitation_loss,
        normalized_realized_objective=normalized_objective,
        normalized_shortage=settlement.normalized_shortage.mean(),
        constraint_penalty=penalty,
    )


__all__ = [
    "STEP_WEIGHTS", "FormalV4FourHourSettlement", "FormalV4LossBreakdown",
    "TrainingObjectiveScale", "clip_formal_v4_gradients", "fit_training_objective_scale", "validate_c_ref_receipt",
    "formal_v4_curriculum_weights", "formal_v4_joint_loss", "settle_formal_v4_four_hour",
]
