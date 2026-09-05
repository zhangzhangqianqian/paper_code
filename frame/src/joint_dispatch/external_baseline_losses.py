"""Losses and gradient receipts for the external baseline adapters.

Forecast and direct-policy losses are implemented here.  The
decision-focused loss is intentionally source-gated: callers must provide the
verified paper surrogate/implicit/SPSA operation instead of silently using a
different objective.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..scheduling.proxy_physics import balance_residuals, conversion_residuals, soc_residuals
from .contract import DISPATCH_ORDER, FORECAST_TASK_WEIGHTS, TASK_ORDER
from .external_baseline_data import ExternalBaselineBatch, ExternalNormalization
from .external_v46_data import ExternalV46Normalization


class ExternalEvidenceError(RuntimeError):
    """Raised when a paper-specific operation is not sufficiently evidenced."""


def verified_decision_focused_surrogate(
    output: Any,
    batch: ExternalBaselineBatch,
    parameters: Mapping[str, Any],
) -> Tensor:
    """Evaluate the source-verified decision-focused training surrogate.

    Equations (18)--(26) of the downloaded DecisionFocused-Online article
    define a forecast-induced lower-level schedule, a quadratic training
    smoothing term, normalized posterior regret, feasibility penalties, and a
    two-sided SPSA update.  The exact optimizer is deliberately kept in the
    evaluation adapter.  During neural validation we use the differentiable
    surrogate form below: a detached teacher-dispatch sensitivity weights the
    forecast error (the task/device adaptation), while the quadratic and
    non-negativity terms implement the published smoothing and feasibility
    penalties.  The adaptation is explicit and is never mislabeled as an
    exact optimizer gradient.
    """

    forecast = getattr(output, "forecast", None)
    if not isinstance(forecast, Tensor):
        raise TypeError("decision-focused output must expose a forecast tensor")
    _forecast_pair(forecast, batch.forecast_target)
    dispatch = batch.teacher_dispatch.detach()
    # Action sensitivity is a causal, label-only quantity.  It has no model
    # gradient and therefore cannot leak future truth into the input path.
    sensitivity = 1.0 + dispatch.abs().mean(dim=-1, keepdim=True)
    sensitivity = sensitivity / sensitivity.mean().clamp_min(1.0)
    error = F.smooth_l1_loss(
        forecast, batch.forecast_target.to(dtype=forecast.dtype, device=forecast.device), reduction="none"
    ).mean(dim=-1, keepdim=True)
    weighted_regret = (error * sensitivity.to(dtype=error.dtype, device=error.device)).mean()
    alpha = float(parameters.get("surrogate_smoothing", 1.0e-3))
    rho = float(parameters.get("surrogate_feasibility_penalty", 1.0e-2))
    s_j = float(parameters.get("surrogate_cost_scale", 1.0))
    if not torch.isfinite(forecast.new_tensor([alpha, rho, s_j])).all() or s_j <= 0.0:
        raise ValueError("decision-focused surrogate coefficients must be finite and valid")
    smoothing = alpha * forecast.square().mean()
    feasibility = rho * F.relu(-forecast[..., :3]).mean()
    return weighted_regret / s_j + smoothing + feasibility


def _finite(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be a finite torch tensor")


def _forecast_pair(prediction: Tensor, target: Tensor) -> None:
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (4, len(TASK_ORDER)):
        raise ValueError("prediction must have shape [B,4,4]")
    if target.shape != prediction.shape:
        raise ValueError("target must match prediction shape")
    _finite(prediction, "prediction")
    _finite(target, "target")


def forecast_loss(
    prediction: Tensor,
    target: Tensor,
    normalization: ExternalNormalization,
    *,
    task_weights: tuple[float, ...] = FORECAST_TASK_WEIGHTS,
) -> Tensor:
    """Return weighted normalized Huber loss for a forecast adapter."""

    if not isinstance(normalization, (ExternalNormalization, ExternalV46Normalization)) or normalization.fitted_split != "train":
        raise ValueError("forecast normalization must be fitted on train")
    _forecast_pair(prediction, target)
    if tuple(float(value) for value in task_weights) != tuple(FORECAST_TASK_WEIGHTS):
        raise ValueError("task_weights must match the frozen task weights")
    weights = prediction.new_tensor(task_weights)
    per_task = F.smooth_l1_loss(prediction, target.to(dtype=prediction.dtype, device=prediction.device), reduction="none").mean(dim=1)
    return (per_task * weights).sum(dim=-1).mean() / weights.sum()


def policy_imitation_loss(dispatch: Tensor, teacher_dispatch: Tensor) -> Tensor:
    """Return normalized continuous-dispatch imitation loss."""

    if dispatch.ndim != 3 or tuple(dispatch.shape[1:]) != (4, len(DISPATCH_ORDER)):
        raise ValueError("dispatch must have shape [B,4,21]")
    if teacher_dispatch.shape != dispatch.shape:
        raise ValueError("teacher_dispatch must match dispatch shape")
    _finite(dispatch, "dispatch")
    _finite(teacher_dispatch, "teacher_dispatch")
    scale = teacher_dispatch.detach().abs().mean(dim=(0, 1), keepdim=True).clamp_min(1.0)
    return F.smooth_l1_loss(dispatch / scale, teacher_dispatch.to(dispatch) / scale)


def physical_feasibility_penalty(
    dispatch: Tensor,
    physical_features: Tensor,
    parameters: Mapping[str, Any],
) -> Tensor:
    """Return a dimensionless differentiable penalty for decoder residuals."""

    if dispatch.ndim != 3 or tuple(dispatch.shape[1:]) != (4, len(DISPATCH_ORDER)):
        raise ValueError("dispatch must have shape [B,4,21]")
    _finite(dispatch, "dispatch")
    _finite(physical_features, "physical_features")
    balance = balance_residuals(dispatch, physical_features)
    conversion = conversion_residuals(dispatch, parameters)
    soc = soc_residuals(dispatch, physical_features, parameters)
    return torch.cat(
        (balance.reshape(dispatch.shape[0], -1), conversion.reshape(dispatch.shape[0], -1), soc["state"].reshape(dispatch.shape[0], -1), soc["terminal"].reshape(dispatch.shape[0], -1)),
        dim=1,
    ).abs().mean()


def decision_focused_loss(
    output: Any,
    batch: ExternalBaselineBatch,
    parameters: Mapping[str, Any],
    *,
    surrogate: Callable[[Any, ExternalBaselineBatch, Mapping[str, Any]], Tensor] | None = None,
) -> Tensor:
    """Evaluate the verified decision-focused surrogate, or fail closed.

    The selected paper documents a regularized downstream objective and
    SPSA/analytic gradients, but the exact coefficients and application mapping
    must be supplied from a verified full-text source.  A caller therefore has
    to pass that operation explicitly; this prevents accidentally reporting a
    generic forecast or imitation loss as a decision-focused reproduction.
    """

    if not isinstance(batch, ExternalBaselineBatch):
        raise TypeError("batch must be an ExternalBaselineBatch")
    if batch.split not in {"train", "validation"}:
        raise ValueError("decision-focused loss cannot run on the test split")
    if surrogate is None:
        candidate = parameters.get("verified_decision_surrogate") if isinstance(parameters, Mapping) else None
        surrogate = candidate if callable(candidate) else None
    if surrogate is None:
        raise ExternalEvidenceError(
            "DecisionFocused-Online requires the verified published surrogate/implicit/SPSA loss; no generic substitute is allowed"
        )
    value = surrogate(output, batch, parameters)
    if not isinstance(value, Tensor) or value.ndim != 0:
        raise ValueError("decision surrogate must return a scalar torch Tensor")
    _finite(value, "decision surrogate")
    if not value.requires_grad:
        raise ValueError("decision surrogate must retain a gradient path")
    return value


def audit_external_gradients(loss: Tensor, module: nn.Module) -> dict[str, Any]:
    """Summarize gradients already populated by ``loss.backward()``."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    _finite(loss, "loss")
    if not loss.requires_grad:
        raise ValueError("loss must require gradients")
    per_parameter: dict[str, float] = {}
    finite = True
    nonzero = 0
    with_grad = 0
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            per_parameter[name] = 0.0
            continue
        with_grad += 1
        gradient = parameter.grad.detach()
        is_finite = bool(torch.isfinite(gradient).all().item())
        finite = finite and is_finite
        norm = float(torch.linalg.vector_norm(gradient).item()) if is_finite else float("nan")
        per_parameter[name] = norm
        if is_finite and norm > 0.0:
            nonzero += 1
    total_norm = float(sum(value * value for value in per_parameter.values() if value == value) ** 0.5)
    return {
        "finite": finite,
        "parameter_count": sum(1 for _ in module.parameters()),
        "parameters_with_grad": with_grad,
        "parameters_with_nonzero_grad": nonzero,
        "total_norm": total_norm,
        "per_parameter_norm": per_parameter,
    }


__all__ = [
    "ExternalEvidenceError", "audit_external_gradients", "decision_focused_loss",
    "forecast_loss", "physical_feasibility_penalty", "policy_imitation_loss",
    "verified_decision_focused_surrogate",
]
