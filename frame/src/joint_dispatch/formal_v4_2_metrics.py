"""Metric aggregation and residual auditing for formal-v4.2 rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .formal_v4_2_rollout import ChronologicalRolloutV42, PhysicalResidualsV42, SettledStepV42


def _safe_wape(prediction: np.ndarray, target: np.ndarray, active: np.ndarray | None = None) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes must match")
    if active is not None:
        mask = np.asarray(active, dtype=bool)
        pred = pred[mask]; truth = truth[mask]
    denominator = float(np.sum(np.abs(truth)))
    return float(np.sum(np.abs(pred - truth)) / denominator * 100.0) if denominator > 1.0e-12 else float("nan")


@dataclass(frozen=True)
class MetricsV42:
    forecast_mae: np.ndarray
    forecast_rmse: np.ndarray
    forecast_wape: np.ndarray
    rigid_macro_wape: float
    gas_wape: float
    shortage_energy: np.ndarray
    shortage_rate: np.ndarray
    operating_cost: float
    physical_carbon: float
    penalized_objective: float
    curtailment: np.ndarray
    p_dump: np.ndarray
    q_dump: np.ndarray
    balance_residual_max: float
    capacity_violation_max: float
    conversion_residual_max: float
    soc_residual_max: float
    ramp_violation_max: float
    exclusivity_max: float
    renewable_residual_max: float
    finite_violation_max: float
    constraint_violation_rate: float
    optimizer_calls: int = 0
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_mae": self.forecast_mae.tolist(), "forecast_rmse": self.forecast_rmse.tolist(),
            "forecast_wape": self.forecast_wape.tolist(), "rigid_macro_wape": self.rigid_macro_wape,
            "gas_wape": self.gas_wape, "shortage_energy": self.shortage_energy.tolist(),
            "shortage_rate": self.shortage_rate.tolist(), "operating_cost": self.operating_cost,
            "physical_carbon": self.physical_carbon, "penalized_objective": self.penalized_objective,
            "curtailment": self.curtailment.tolist(), "p_dump": self.p_dump.tolist(), "q_dump": self.q_dump.tolist(),
            "balance_residual_max": self.balance_residual_max, "capacity_violation_max": self.capacity_violation_max,
            "conversion_residual_max": self.conversion_residual_max, "soc_residual_max": self.soc_residual_max,
            "ramp_violation_max": self.ramp_violation_max, "exclusivity_max": self.exclusivity_max,
            "renewable_residual_max": self.renewable_residual_max, "finite_violation_max": self.finite_violation_max,
            "constraint_violation_rate": self.constraint_violation_rate,
            "optimizer_calls": self.optimizer_calls, "latency_seconds": self.latency_seconds,
        }

    # Common spelling used by result tables.
    @property
    def shortage(self) -> np.ndarray:
        return self.shortage_energy


def _residual_max(residuals: tuple[PhysicalResidualsV42, ...], name: str) -> float:
    values = [np.max(np.abs(getattr(item, name))) for item in residuals if getattr(item, name).size]
    return float(max(values, default=0.0))


def compute_v42_metrics(outcome: ChronologicalRolloutV42 | SettledStepV42) -> MetricsV42:
    """Compute task-level forecast and dispatch metrics from settled outcomes."""

    if isinstance(outcome, SettledStepV42):
        result = None
        shortage = np.asarray(outcome.shortage, dtype=np.float64).reshape(1, 3)
        p_dump = np.asarray([outcome.p_dump], dtype=np.float64)
        q_dump = np.asarray([outcome.q_dump], dtype=np.float64)
        residuals = (outcome.residuals,)
        operating = outcome.operating_cost; carbon = outcome.physical_carbon; objective = outcome.penalized_objective
        curtailment = np.asarray([outcome.settled[[1, 3]] .sum()], dtype=np.float64)
        # A single settled step has no independent forecast label.
        forecast_mae = np.full(4, np.nan); forecast_rmse = np.full(4, np.nan); forecast_wape = np.full(4, np.nan)
        rigid_wape = gas_wape = float("nan")
    elif isinstance(outcome, ChronologicalRolloutV42):
        result = outcome
        target = np.asarray(outcome.forecast_target, dtype=np.float64)
        prediction = np.asarray(outcome.forecast_prediction, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError("rollout forecast prediction/target shapes differ")
        error = prediction - target
        forecast_mae = np.mean(np.abs(error), axis=(0, 1)); forecast_rmse = np.sqrt(np.mean(error ** 2, axis=(0, 1)))
        forecast_wape = np.asarray([_safe_wape(prediction[..., i], target[..., i]) for i in range(4)])
        rigid_wape = _safe_wape(prediction[..., :3], target[..., :3])
        gas_wape = _safe_wape(prediction[..., 3], target[..., 3])
        shortage = np.asarray(outcome.shortage_energy, dtype=np.float64)
        p_dump = np.asarray(outcome.p_dump, dtype=np.float64); q_dump = np.asarray(outcome.q_dump, dtype=np.float64)
        residuals = tuple(outcome.residuals)
        operating = float(np.sum(outcome.operating_cost)); carbon = float(np.sum(outcome.physical_carbon)); objective = float(np.sum(outcome.penalized_objective))
        curtailment = np.asarray([np.sum(item) for item in p_dump.reshape(-1, 1)], dtype=np.float64)
    else:
        raise TypeError("outcome must be ChronologicalRolloutV42 or SettledStepV42")
    if shortage.ndim != 2 or shortage.shape[1] != 3:
        raise ValueError("shortage must have shape [N,3]")
    shortage_energy = np.sum(shortage, axis=0)
    # Rate is reported against the evaluated realized target only when it is
    # available; otherwise it remains NaN rather than inventing a denominator.
    if result is None:
        shortage_rate = np.full(3, np.nan)
    else:
        target = np.asarray(result.forecast_target[..., :3], dtype=np.float64)
        shortage_rate = shortage_energy / np.maximum(np.sum(np.abs(target), axis=(0, 1)), 1.0e-12)
    threshold = 1.0e-7
    violation_flags = []
    for item in residuals:
        violation_flags.append(any(float(np.max(np.abs(getattr(item, name)))) > threshold for name in (
            "balance", "capacity", "conversion", "soc", "ramp", "exclusivity", "renewable_accounting", "finite")))
    return MetricsV42(
        forecast_mae=np.asarray(forecast_mae), forecast_rmse=np.asarray(forecast_rmse), forecast_wape=np.asarray(forecast_wape),
        rigid_macro_wape=float(rigid_wape), gas_wape=float(gas_wape), shortage_energy=shortage_energy,
        shortage_rate=shortage_rate, operating_cost=float(operating), physical_carbon=float(carbon),
        penalized_objective=float(objective), curtailment=np.asarray(curtailment), p_dump=p_dump, q_dump=q_dump,
        balance_residual_max=_residual_max(residuals, "balance"), capacity_violation_max=_residual_max(residuals, "capacity"),
        conversion_residual_max=_residual_max(residuals, "conversion"), soc_residual_max=_residual_max(residuals, "soc"),
        ramp_violation_max=_residual_max(residuals, "ramp"), exclusivity_max=_residual_max(residuals, "exclusivity"),
        renewable_residual_max=_residual_max(residuals, "renewable_accounting"), finite_violation_max=_residual_max(residuals, "finite"),
        constraint_violation_rate=float(np.mean(violation_flags)) if violation_flags else 0.0,
    )


__all__ = ["MetricsV42", "compute_v42_metrics"]
