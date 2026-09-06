"""Complete-formal Differentiable-LP integration.

The legacy formal-v4 DiffLP implementation remains untouched.  This module
adds the complete-formal adapter, an inference provider with explicit solve
accounting, and a one-step training evidence helper.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from importlib import metadata
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .complete_formal_providers import CompletePlannedStep
from .formal_v4_diffopt import DifferentiableIESLayer, RIGID_TASKS
from .formal_v4_2_gate2_training import train_differentiable_lp


@dataclass(frozen=True)
class CompleteDiffLPProblemSpec:
    rigid_demand_tasks: tuple[str, ...] = RIGID_TASKS
    horizon: int = 4
    output_dimension: int = 21
    inference_lp_calls_per_origin: int = 1


def _normalization_vector(normalization: Any, field: str, name: str, width: int, default: float) -> np.ndarray:
    if normalization is None:
        return np.full(width, default, dtype=np.float32)
    means = getattr(normalization, "field_mean", {})
    scales = getattr(normalization, "field_scale", {})
    if isinstance(normalization, Mapping):
        means = normalization.get("field_mean", means)
        scales = normalization.get("field_scale", scales)
    source = means if name == "mean" else scales
    value = source.get(field, np.full(width, default, dtype=np.float32)) if isinstance(source, Mapping) else np.full(width, default, dtype=np.float32)
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (width,) or not np.isfinite(result).all() or (name == "scale" and np.any(result <= 0.0)):
        raise ValueError(f"invalid {field} normalization {name}")
    return result


def _as_price_parameters(scheduler_context: np.ndarray) -> np.ndarray:
    context = np.asarray(scheduler_context[0], dtype=np.float64)
    if context.shape != (4, 6):
        raise ValueError("scheduler_context must have shape [1,4,6]")
    prices = context[:, 2:5]
    return np.column_stack((prices[:, 0], prices[:, 1], np.zeros(4), prices[:, 2]))


class DifferentiableLPProvider:
    method_id = "Differentiable-LP"
    optimizer_role = "exact optimizer at inference"
    forecast_metrics_applicable = True

    def __init__(
        self,
        forecaster: nn.Module,
        layer: nn.Module,
        parameters: Mapping[str, Any],
        *,
        normalization: Any = None,
        task_mean: Optional[Sequence[float]] = None,
        task_scale: Optional[Sequence[float]] = None,
    ) -> None:
        self.forecaster = forecaster.eval()
        self.layer = layer.eval()
        self.parameters = dict(parameters)
        self.problem_spec = CompleteDiffLPProblemSpec()
        self.load_mean = np.asarray(
            task_mean if task_mean is not None else _normalization_vector(normalization, "load", "mean", 4, 0.0),
            dtype=np.float32,
        )
        self.load_scale = np.asarray(
            task_scale if task_scale is not None else _normalization_vector(normalization, "load", "scale", 4, 1.0),
            dtype=np.float32,
        )
        if self.load_mean.shape != (4,) or self.load_scale.shape != (4,) or np.any(self.load_scale <= 0.0):
            raise ValueError("DiffLP task mean/scale must have shape [4] and positive scale")
        self.exog_mean = _normalization_vector(normalization, "exog", "mean", 12, 0.0)
        self.exog_scale = _normalization_vector(normalization, "exog", "scale", 12, 1.0)

    def plan(self, origin: Any) -> CompletePlannedStep:
        load = (np.asarray(origin.load_history, dtype=np.float32) - self.load_mean.reshape(1, 1, 4)) / self.load_scale.reshape(1, 1, 4)
        exog = (np.asarray(origin.exog_history, dtype=np.float32) - self.exog_mean.reshape(1, 1, 12)) / self.exog_scale.reshape(1, 1, 12)
        with torch.inference_mode():
            output = self.forecaster(
                torch.as_tensor(load, dtype=torch.float32),
                torch.as_tensor(exog, dtype=torch.float32),
            )
            if isinstance(output, (tuple, list)):
                output = output[0]
            forecast = torch.clamp(
                output * torch.as_tensor(self.load_scale, dtype=output.dtype, device=output.device)
                + torch.as_tensor(self.load_mean, dtype=output.dtype, device=output.device),
                min=0.0,
            )
            dispatch = self.layer(
                forecast[..., :3],
                torch.as_tensor(np.asarray(origin.renewable_forecast)[None], dtype=torch.float64),
                torch.as_tensor(_as_price_parameters(np.asarray(origin.scheduler_context))[None], dtype=torch.float64),
                torch.as_tensor(np.asarray(origin.scheduler_context)[:, :1, 5], dtype=torch.float64),
                torch.as_tensor(np.asarray(origin.previous_chp), dtype=torch.float64),
            )
        forecast_np = forecast.detach().cpu().numpy()[0].astype(np.float64)
        dispatch_np = dispatch.detach().cpu().numpy()[0].astype(np.float64)
        return CompletePlannedStep(
            forecast=forecast_np,
            scheduler_demand=forecast_np,
            renewable_forecast=np.asarray(origin.renewable_forecast, dtype=np.float64).copy(),
            dispatch=dispatch_np,
            inference_lp_calls=1,
        )


def build_difflp_provider(
    parameters: Mapping[str, Any],
    *,
    forecaster: nn.Module,
    layer: nn.Module,
    normalization: Any = None,
    task_mean: Optional[Sequence[float]] = None,
    task_scale: Optional[Sequence[float]] = None,
) -> DifferentiableLPProvider:
    """Build a causal DiffLP provider from an explicit forecaster and layer."""

    if not isinstance(forecaster, nn.Module) or not isinstance(layer, nn.Module):
        raise TypeError("DiffLP requires a torch forecaster and differentiable layer")
    return DifferentiableLPProvider(
        forecaster=forecaster,
        layer=layer,
        parameters=parameters,
        normalization=normalization,
        task_mean=task_mean,
        task_scale=task_scale,
    )


@dataclass(frozen=True)
class DiffLPStepArtifact:
    lp_calls: int
    forecaster_gradient_norm: float
    decision_forecaster_gradient_norm: float
    loss: float


def _gradient_norm(values: Sequence[Optional[Tensor]]) -> float:
    tensors = [value.detach().float().norm() for value in values if value is not None]
    return float(torch.stack(tensors).norm()) if tensors else 0.0


def one_difflp_training_step(
    model: nn.Module,
    native_layer: nn.Module,
    batch: Mapping[str, Tensor],
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
    task_mean: Optional[Tensor] = None,
    task_scale: Optional[Tensor] = None,
) -> DiffLPStepArtifact:
    """Perform one differentiable forecast-to-LP update and record gradients."""

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    forecast_normalized = model(batch["load_history"], batch["exog_history"])
    if isinstance(forecast_normalized, (tuple, list)):
        forecast_normalized = forecast_normalized[0]
    if task_mean is None:
        task_mean = torch.zeros(4, dtype=forecast_normalized.dtype, device=forecast_normalized.device)
    if task_scale is None:
        task_scale = torch.ones(4, dtype=forecast_normalized.dtype, device=forecast_normalized.device)
    forecast_physical = torch.clamp(forecast_normalized * task_scale + task_mean, min=0.0)
    prices = batch["prices_and_weights"]
    if prices.shape[-1] == 3:
        prices = torch.cat((prices[..., :2], torch.zeros_like(prices[..., :1]), prices[..., 2:3]), dim=-1)
    dispatch = native_layer(
        forecast_physical[..., :3],
        batch["renewable_forecast"],
        prices,
        batch["initial_soc"],
        batch["previous_chp"],
    )
    forecast_loss = F.smooth_l1_loss(forecast_normalized, batch["target_normalized"].to(forecast_normalized))
    decision_loss = dispatch[..., 0].mean() + 1.0e-3 * dispatch[..., 18:20].mean()
    loss = forecast_loss + decision_loss
    parameters = tuple(model.parameters())
    decision_gradients = torch.autograd.grad(decision_loss, parameters, allow_unused=True, retain_graph=True)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    forecaster_gradient_norm = _gradient_norm(tuple(parameter.grad for parameter in parameters))
    decision_forecaster_gradient_norm = _gradient_norm(decision_gradients)
    optimizer.step()
    if not np.isfinite(forecaster_gradient_norm) or forecaster_gradient_norm <= 0.0:
        raise RuntimeError("DiffLP one-step forecaster gradient is non-finite or zero")
    return DiffLPStepArtifact(
        lp_calls=1,
        forecaster_gradient_norm=forecaster_gradient_norm,
        decision_forecaster_gradient_norm=decision_forecaster_gradient_norm,
        loss=float(loss.detach()),
    )


def _package_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for name in ("cvxpy", "cvxpylayers", "diffcp", "torch"):
        try:
            versions[name] = torch.__version__ if name == "torch" else metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _dpp_status(layer: nn.Module) -> Optional[bool]:
    problem = getattr(layer, "problem", None)
    checker = getattr(problem, "is_dpp", None)
    return bool(checker()) if callable(checker) else None


def train_complete_difflp(
    seed: int,
    data: Any,
    freeze: Mapping[str, Any],
    diffopt_receipt: Mapping[str, Any],
    parameters: Mapping[str, Any],
    output_dir: str | Path,
    *,
    budget: Any = None,
    model: Optional[nn.Module] = None,
    layer: Optional[nn.Module] = None,
    micro_batch_size: int = 1,
) -> Any:
    """Run the existing DiffLP trainer and add complete-formal provenance."""

    active_layer = layer if layer is not None else DifferentiableIESLayer(parameters)
    artifact = train_differentiable_lp(
        seed,
        data,
        freeze,
        diffopt_receipt,
        parameters,
        output_dir,
        budget=budget,
        model=model,
        layer=active_layer,
        micro_batch_size=micro_batch_size,
    )
    receipt = dict(artifact.training_receipt)
    receipt.update({
        "schema_version": "rsc-pf-complete-formal-difflp-training-v1",
        "method_id": "Differentiable-LP",
        "seed": int(seed),
        "reproduction_level": "cvxpylayers_methodology_adaptation",
        "rigid_demand_tasks": list(RIGID_TASKS),
        "dpp_status": _dpp_status(active_layer),
        "parity_residual": diffopt_receipt.get("parity_residual"),
        "gradient_norm": float(artifact.decision_forecaster_gradient_norm),
        "inference_lp_calls_per_origin": 1,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "package_versions": _package_versions(),
        "python_executable": sys.executable,
    })
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "COMPLETE_DIFFLP_RECEIPT.json").write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    return artifact


__all__ = [
    "CompleteDiffLPProblemSpec",
    "DiffLPStepArtifact",
    "DifferentiableLPProvider",
    "build_difflp_provider",
    "one_difflp_training_step",
    "train_complete_difflp",
]
