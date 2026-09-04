"""Checkpoint-backed rolling execution for formal-v4.2 Gate 2."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import torch

from .formal_v4_2_artifacts import sha256_file, write_once_json
from .formal_v4_2_gate2_artifacts import write_gate2_rollout, write_gate2_row_receipt
from .formal_v4_2_gate2_training import Gate2DataBundle, TrainedMethodArtifact
from .formal_v4_2_metrics import compute_v42_metrics_from_arrays
from .formal_v4_2_rollout import evaluate_chronological_v42
from .formal_v4_state import FormalV4ClosedLoopState


@dataclass(frozen=True)
class Gate2RowKey:
    method_id: str
    seed: int | None

    @property
    def label(self) -> str:
        return self.method_id if self.seed is None else f"{self.method_id}__seed_{self.seed}"


def evaluation_windows(data: Gate2DataBundle) -> list[dict[str, Any]]:
    split = data.evaluation
    rows: list[dict[str, Any]] = []
    for index in range(len(split)):
        realized_exog = split.exog_history[index + 1, -1] if index + 1 < len(split) else split.exog_history[index, -1]
        rows.append({
            "forecast_target": split.forecast_target[index],
            "renewable_forecast": split.renewable_forecast[index],
            "renewable_realized": split.renewable_realized[index],
            "prices_and_weights": split.prices_and_weights[index],
            "target_time": split.target_times[index],
            "realized_exog": realized_exog,
        })
    return rows


def initial_evaluation_state(data: Gate2DataBundle) -> FormalV4ClosedLoopState:
    split = data.evaluation
    if len(split) == 0:
        raise ValueError("Gate 2 evaluation is empty")
    return FormalV4ClosedLoopState(
        torch.as_tensor(split.load_history[:1], dtype=torch.float64),
        torch.as_tensor(split.exog_history[:1], dtype=torch.float64),
        torch.as_tensor(split.device_history[:1], dtype=torch.float64),
        torch.as_tensor(split.activity_history[:1], dtype=torch.float64),
        torch.as_tensor(split.initial_soc[:1], dtype=torch.float64),
        torch.as_tensor(split.previous_chp[:1], dtype=torch.float64),
        split.target_times[0], str(split.trajectory_ids[0]),
    )


def _rollout_arrays(outcome: Any) -> dict[str, np.ndarray]:
    residuals = tuple(outcome.residuals)
    return {
        "forecast_target": np.asarray(outcome.forecast_target),
        "forecast_prediction": np.asarray(outcome.forecast_prediction),
        "planned_dispatch": np.asarray(outcome.planned_dispatch),
        "settled_dispatch": np.asarray(outcome.settled_dispatch),
        "shortage_energy": np.asarray(outcome.shortage_energy),
        "p_dump": np.asarray(outcome.p_dump),
        "q_dump": np.asarray(outcome.q_dump),
        "operating_cost": np.asarray(outcome.operating_cost),
        "physical_carbon": np.asarray(outcome.physical_carbon),
        "penalized_objective": np.asarray(outcome.penalized_objective),
        "target_times": np.asarray(outcome.target_times).astype("datetime64[ns]").astype(np.int64),
        "state_hashes": np.asarray([state.state_hash for state in outcome.next_states]),
        "residual_balance": np.stack([item.balance for item in residuals]),
        "residual_capacity": np.stack([item.capacity for item in residuals]),
        "residual_conversion": np.stack([item.conversion for item in residuals]),
        "residual_soc": np.stack([item.soc for item in residuals]),
        "residual_ramp": np.stack([item.ramp for item in residuals]),
        "residual_exclusivity": np.stack([item.exclusivity for item in residuals]),
        "residual_renewable_accounting": np.stack([item.renewable_accounting for item in residuals]),
        "residual_finite": np.stack([item.finite for item in residuals]),
    }


def execute_gate2_row(
    key: Gate2RowKey,
    trained: Any,
    data: Gate2DataBundle,
    parameters: Mapping[str, Any],
    row_dir: str | Path,
    lineage: Mapping[str, str],
) -> Mapping[str, Any]:
    """Execute, reopen, recompute, and commit one immutable method row."""

    directory = Path(row_dir)
    directory.mkdir(parents=True, exist_ok=True)
    method = getattr(trained, "method", trained)
    call_count = 0
    latency = 0.0

    def counted(window: Any, state: FormalV4ClosedLoopState) -> Any:
        nonlocal call_count, latency
        result = method.plan(window, state) if hasattr(method, "plan") else method(window, state)
        call_count += int(getattr(result, "optimizer_calls", 0))
        latency += float(getattr(result, "latency_seconds", 0.0))
        return result

    counted.method_id = key.method_id  # type: ignore[attr-defined]
    started = time.perf_counter()
    outcome = evaluate_chronological_v42(
        counted, evaluation_windows(data), initial_state=initial_evaluation_state(data),
        parameters=parameters,
    )
    runtime = time.perf_counter() - started
    arrays = _rollout_arrays(outcome)
    rollout_hash = write_gate2_rollout(directory / "ROLLOUT.npz", arrays, lineage)
    with np.load(directory / "ROLLOUT.npz", allow_pickle=False) as reopened:
        reopened_arrays = {name: reopened[name] for name in reopened.files if name not in {"schema", "lineage_json"}}
    metrics = compute_v42_metrics_from_arrays(reopened_arrays, method_id=key.method_id)
    metrics_payload = metrics.to_dict()
    forecast_applicable = bool(getattr(method, "forecast_metrics_applicable", key.method_id not in {"Direct-Policy", "Perfect-Information-MPC"}))
    metrics_payload.update({
        "schema": "formal-v4.2-gate2-metrics-v1",
        "method_id": key.method_id,
        "seed": key.seed,
        "settled_hours": len(reopened_arrays["settled_dispatch"]),
        "optimizer_calls": int(call_count),
        "latency_seconds": float(latency),
        "runtime_seconds": float(runtime),
        "forecast_metrics_applicable": forecast_applicable,
        "forecast_metrics": None if not forecast_applicable else {
            "mae": metrics.forecast_mae.tolist(),
            "rmse": metrics.forecast_rmse.tolist(),
            "wape": metrics.forecast_wape.tolist(),
            "rigid_macro_wape": metrics.rigid_macro_wape,
            "gas_wape": metrics.gas_wape,
        },
    })
    write_once_json(directory / "METRICS.json", metrics_payload)
    metrics_hash = sha256_file(directory / "METRICS.json")
    if key.seed is None:
        checkpoint_hash = training_hash = "not_applicable"
    else:
        checkpoint_hash = sha256_file(directory / "CHECKPOINT.pt")
        training_hash = sha256_file(directory / "TRAINING_RECEIPT.json")
    row_payload = {
        **dict(lineage),
        "method_id": key.method_id,
        "seed": key.seed,
        "checkpoint_sha256": checkpoint_hash,
        "training_receipt_sha256": training_hash,
        "rollout_sha256": rollout_hash,
        "metrics_sha256": metrics_hash,
        "settled_hours": len(reopened_arrays["settled_dispatch"]),
        "optimizer_calls": int(call_count),
        "latency_seconds": float(latency),
        "runtime_seconds": float(runtime),
        "forecast_metrics_applicable": forecast_applicable,
        "forecast_metrics": metrics_payload["forecast_metrics"],
        "shortage_energy": metrics.shortage_energy.tolist(),
        "shortage_rate": metrics.shortage_rate.tolist(),
        "penalized_objective": metrics.penalized_objective,
        "operating_cost": metrics.operating_cost,
        "physical_carbon": metrics.physical_carbon,
        "constraint_violation_rate": metrics.constraint_violation_rate,
        "physical_residual_max": max(
            metrics.balance_residual_max, metrics.capacity_violation_max,
            metrics.conversion_residual_max, metrics.soc_residual_max,
            metrics.ramp_violation_max, metrics.exclusivity_max,
            metrics.renewable_residual_max, metrics.finite_violation_max,
        ),
    }
    write_gate2_row_receipt(directory, row_payload)
    return json.loads((directory / "ROW_RECEIPT.json").read_text(encoding="utf-8"))


__all__ = [
    "Gate2RowKey", "evaluation_windows", "execute_gate2_row",
    "initial_evaluation_state",
]
