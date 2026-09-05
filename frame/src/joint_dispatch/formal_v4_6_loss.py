"""Joint forecast, dispatch, and bounded-risk losses for formal-v4.6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .formal_v4_4_loss import CurriculumWeightsV44, ForecastLossV44, forecast_loss_v44
from .formal_v4_5_loss import JointNormalizationV45, _teacher_imitation


@dataclass(frozen=True)
class JointLossV46:
    total: Tensor
    forecast: ForecastLossV44
    imitation: Tensor
    decision: Tensor
    anchor: Tensor
    risk_size: Tensor
    off_risk: Tensor
    weights: Mapping[str, float]


def _thermal_inactive_mask_v46(target_physical: Tensor) -> Tensor:
    if target_physical.ndim != 3 or tuple(target_physical.shape[1:]) != (4, 4):
        raise ValueError("target_physical must have shape [B,4,4]")
    if not bool(torch.isfinite(target_physical).all()):
        raise ValueError("target_physical must be finite")
    return (target_physical[..., 1:3] <= 1.0e-9).to(dtype=target_physical.dtype)


def risk_penalties_v46(output: Any, target_physical: Tensor) -> tuple[Tensor, Tensor]:
    """Return mean normalized risk size and thermal risk in inactive windows."""

    adjustment = getattr(output, "risk_adjustment", None)
    cap = getattr(output, "risk_cap", None)
    if not isinstance(adjustment, Tensor) or not isinstance(cap, Tensor):
        raise ValueError("output must expose risk_adjustment and risk_cap")
    if adjustment.ndim != 3 or tuple(adjustment.shape[1:]) != (4, 3) or cap.shape != adjustment.shape:
        raise ValueError("risk tensors must have shape [B,4,3]")
    if not bool(torch.isfinite(adjustment).all() and torch.isfinite(cap).all()):
        raise ValueError("risk tensors must be finite")
    if bool((adjustment < -1.0e-8).any()) or bool((cap <= 0.0).any()):
        raise ValueError("risk adjustment must be non-negative and caps positive")
    fraction = adjustment / cap.clamp_min(1.0e-8)
    risk_size = fraction.mean()
    inactive = _thermal_inactive_mask_v46(target_physical).to(device=fraction.device, dtype=fraction.dtype)
    if inactive.shape[0] != fraction.shape[0]:
        raise ValueError("target and risk batch dimensions must align")
    off_risk = (fraction[..., 1:3] * inactive).sum() / inactive.sum().clamp_min(1.0)
    return risk_size, off_risk


def _forecast_weights(curriculum: Mapping[str, float]) -> CurriculumWeightsV44:
    point = float(curriculum.get("point", curriculum.get("forecast_point", 1.0)))
    leakage = float(curriculum.get("inactive_leakage", curriculum.get("forecast_inactive_leakage", 0.25)))
    return CurriculumWeightsV44(
        electricity_gas=1.0,
        regime=1.0,
        active_magnitude=1.0,
        point=point,
        inactive_leakage=leakage,
    )


def joint_loss_v46(
    output: Any,
    parent_output: Any,
    batch: Mapping[str, Any],
    prior: Any,
    normalization: JointNormalizationV45,
    curriculum: Mapping[str, float],
    risk_multiplier: float,
    decision_loss: Tensor,
) -> JointLossV46:
    """Combine the full v4.4 forecast objective with decision and risk terms."""

    if not isinstance(decision_loss, Tensor) or decision_loss.ndim != 0 or not bool(torch.isfinite(decision_loss).all()):
        raise ValueError("decision_loss must be a finite scalar tensor")
    multiplier = float(risk_multiplier)
    if not torch.isfinite(torch.as_tensor(multiplier)) or multiplier < 0.0:
        raise ValueError("risk_multiplier must be finite and non-negative")
    if not isinstance(normalization, JointNormalizationV45):
        raise TypeError("normalization must be JointNormalizationV45")
    if not isinstance(curriculum, Mapping):
        raise TypeError("curriculum must be a mapping")
    v44_output = output.as_v44_forecast_output() if hasattr(output, "as_v44_forecast_output") else output
    forecast = forecast_loss_v44(v44_output, batch, prior, _forecast_weights(curriculum))
    imitation = _teacher_imitation(output.dispatch, batch.get("teacher_dispatch"))
    parent_forecast = getattr(parent_output, "forecast_normalized", None)
    current_forecast = getattr(output, "forecast_normalized", None)
    if not isinstance(parent_forecast, Tensor) or not isinstance(current_forecast, Tensor) or parent_forecast.shape != current_forecast.shape:
        raise ValueError("parent and current outputs must expose aligned forecast_normalized tensors")
    anchor = torch.nn.functional.smooth_l1_loss(current_forecast, parent_forecast.detach())
    risk_size, off_risk = risk_penalties_v46(output, torch.as_tensor(batch["target_physical"], dtype=current_forecast.dtype, device=current_forecast.device))
    weights = {
        "forecast": float(curriculum.get("forecast", 1.0)),
        "imitation": float(curriculum.get("imitation", 0.0)),
        "decision": float(curriculum.get("decision", 0.0)),
        "anchor": float(curriculum.get("anchor", 0.5)),
    }
    if any(value < 0.0 or not bool(torch.isfinite(torch.as_tensor(value))) for value in weights.values()):
        raise ValueError("curriculum weights must be finite and non-negative")
    total = (
        weights["forecast"] * forecast.total / float(normalization.forecast)
        + weights["imitation"] * imitation / float(normalization.imitation)
        + weights["decision"] * decision_loss / float(normalization.decision)
        + weights["anchor"] * anchor
        + 0.10 * multiplier * risk_size
        + 0.50 * multiplier * off_risk
    )
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("non-finite formal-v4.6 Joint loss")
    return JointLossV46(total, forecast, imitation, decision_loss, anchor, risk_size, off_risk, weights)


def _groups(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    if not hasattr(model, "v46_parameter_groups"):
        raise TypeError("formal-v4.6 optimizer requires v46_parameter_groups")
    return {name: tuple(values) for name, values in model.v46_parameter_groups().items()}


def build_j_optimizer_v46(model: nn.Module, contract: Any, mode: str = "joint") -> torch.optim.Optimizer:
    """Build the frozen J optimizer, with a separate bounded-risk learning rate."""

    if mode not in {"joint", "decoupled"}:
        raise ValueError("mode must be joint or decoupled")
    groups = _groups(model)
    payload = contract.payload if hasattr(contract, "payload") else contract
    budget = payload["pilot_budget"]
    risk_config = payload["risk_adjustment"]
    rates = {
        "base": float(budget["j_forecaster_lr"]),
        "head": float(budget["j_head_lr"]),
        "risk": float(risk_config["j_risk_lr"]),
        "scheduler": float(budget["j_scheduler_lr"]),
    }
    weight_decay = float(budget["weight_decay"])
    selected = (
        (("base", groups.get("base", ())), ("head", (*groups.get("gate", ()), *groups.get("magnitude", ()))), ("risk", groups.get("risk", ())), ("scheduler", groups.get("scheduler", ())))
        if mode == "joint"
        else (("risk", groups.get("risk", ())), ("scheduler", groups.get("scheduler", ())))
    )
    parameter_groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for name, values in selected:
        values = tuple(values)
        if any(id(parameter) in seen for parameter in values):
            raise ValueError("v4.6 J optimizer groups overlap")
        seen.update(id(parameter) for parameter in values)
        if values:
            parameter_groups.append({"name": name, "params": list(values), "lr": rates[name]})
    if not parameter_groups:
        raise ValueError("v4.6 J optimizer has no trainable parameters")
    return torch.optim.AdamW(parameter_groups, weight_decay=weight_decay)


__all__ = [
    "JointLossV46",
    "JointNormalizationV45",
    "build_j_optimizer_v46",
    "joint_loss_v46",
    "risk_penalties_v46",
    "_thermal_inactive_mask_v46",
]
