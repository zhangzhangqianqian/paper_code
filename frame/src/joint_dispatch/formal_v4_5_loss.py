"""Normalized Joint curriculum and optimizer-group helpers for formal-v4.5."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JointNormalizationV45:
    forecast: float
    imitation: float
    decision: float

    def __post_init__(self) -> None:
        for name in ("forecast", "imitation", "decision"):
            value = float(getattr(self, name))
            if not torch.isfinite(torch.as_tensor(value)) or value <= 0.0:
                raise ValueError(f"{name} normalization must be finite and positive")

    @classmethod
    def from_values(cls, forecast: Tensor | float, imitation: Tensor | float, decision: Tensor | float) -> "JointNormalizationV45":
        epsilon = 1.0e-8
        return cls(
            max(float(torch.as_tensor(forecast).detach().abs().mean()), epsilon),
            max(float(torch.as_tensor(imitation).detach().abs().mean()), epsilon),
            max(float(torch.as_tensor(decision).detach().abs().mean()), epsilon),
        )


@dataclass(frozen=True)
class JointLossV45:
    total: Tensor
    forecast: Tensor
    imitation: Tensor
    decision: Tensor
    anchor: Tensor
    weights: Mapping[str, float]


def curriculum_weights_v45(epoch: int, ramp_epochs: int = 5) -> dict[str, float]:
    if int(epoch) < 0 or int(ramp_epochs) < 1:
        raise ValueError("epoch must be non-negative and ramp_epochs must be positive")
    fraction = min(float(epoch) / float(ramp_epochs - 1 if ramp_epochs > 1 else 1), 1.0)
    return {
        "forecast": 1.0,
        "anchor": 0.5,
        "decision": 0.05 + (0.50 - 0.05) * fraction,
        "imitation": 1.00 + (0.25 - 1.00) * fraction,
    }


def _teacher_imitation(dispatch: Tensor, teacher: Any) -> Tensor:
    if teacher is None:
        return dispatch.new_zeros(())
    teacher_tensor = torch.as_tensor(teacher, dtype=dispatch.dtype, device=dispatch.device)
    scale = teacher_tensor.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
    return F.smooth_l1_loss(dispatch / scale, teacher_tensor / scale)


def joint_loss_v45(
    output: Any,
    parent_output: Any,
    batch: Mapping[str, Any],
    normalization: JointNormalizationV45,
    weights: Mapping[str, float],
    *,
    anchor_weight: float | None = None,
    decision_loss: Tensor | None = None,
) -> JointLossV45:
    """Compute normalized forecast, imitation, decision, and P1-anchor terms."""

    target = torch.as_tensor(batch["target_normalized"], dtype=output.forecast_normalized.dtype, device=output.forecast_normalized.device)
    forecast = F.smooth_l1_loss(output.forecast_normalized, target)
    imitation = _teacher_imitation(output.dispatch, batch.get("teacher_dispatch"))
    decision = output.dispatch.square().mean() if decision_loss is None else decision_loss
    if decision.ndim != 0:
        decision = decision.mean()
    anchor = F.smooth_l1_loss(output.forecast_normalized, parent_output.forecast_normalized.detach())
    anchor_weight = float(weights.get("anchor", 0.5) if anchor_weight is None else anchor_weight)
    total = (
        float(weights["forecast"]) * forecast / float(normalization.forecast)
        + float(weights["imitation"]) * imitation / float(normalization.imitation)
        + float(weights["decision"]) * decision / float(normalization.decision)
        + anchor_weight * anchor
    )
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("non-finite formal-v4.5 Joint loss")
    return JointLossV45(total, forecast, imitation, decision, anchor, dict(weights))


def _parameter_groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    if not hasattr(model, "v44_parameter_groups"):
        raise TypeError("formal-v4.5 optimizer requires v44_parameter_groups")
    groups = {name: tuple(params) for name, params in model.v44_parameter_groups().items()}
    return groups


def build_j_optimizer_v45(model: nn.Module, contract: Any, *, mode: str = "joint") -> torch.optim.Optimizer:
    """Build the audited J optimizer with separate base/head/scheduler rates."""

    if mode not in {"joint", "decoupled"}:
        raise ValueError("mode must be joint or decoupled")
    groups = _parameter_groups(model)
    budget = contract.payload["pilot_budget"]
    rates = {
        "base": float(budget["j_forecaster_lr"]),
        "head": float(budget["j_head_lr"]),
        "scheduler": float(budget["j_scheduler_lr"]),
    }
    parameters: list[dict[str, Any]] = []
    base = tuple(groups.get("base", ()))
    head = tuple((*groups.get("gate", ()), *groups.get("magnitude", ())))
    scheduler = tuple(groups.get("scheduler", ()))
    if mode == "joint":
        named = (("base", base), ("head", head), ("scheduler", scheduler))
    else:
        named = (("scheduler", scheduler),)
    seen: set[int] = set()
    for name, values in named:
        if not values:
            continue
        if any(id(parameter) in seen for parameter in values):
            raise ValueError("J optimizer parameter groups overlap")
        seen.update(id(parameter) for parameter in values)
        parameters.append({"name": name, "params": list(values), "lr": rates[name]})
    if not parameters:
        raise ValueError("J optimizer has no trainable parameters")
    return torch.optim.AdamW(parameters, weight_decay=float(budget["weight_decay"]))


__all__ = [
    "JointLossV45", "JointNormalizationV45", "build_j_optimizer_v45",
    "curriculum_weights_v45", "joint_loss_v45",
]
