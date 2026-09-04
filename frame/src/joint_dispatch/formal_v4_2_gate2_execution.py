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
        training_payload: Mapping[str, Any] = {}
    else:
        checkpoint_hash = sha256_file(directory / "CHECKPOINT.pt")
        training_hash = sha256_file(directory / "TRAINING_RECEIPT.json")
        training_payload = json.loads((directory / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
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
        "stage_s_parent_sha256": training_payload.get("stage_s_parent_sha256", "not_applicable"),
        "decision_forecaster_gradient_norm": training_payload.get("decision_forecaster_gradient_norm", "not_applicable"),
        "test_set_accessed": False,
        "shortage_energy": float(np.sum(metrics.shortage_energy)),
        "shortage_by_carrier": metrics.shortage_energy.tolist(),
        "shortage_rate": metrics.shortage_rate.tolist(),
        "penalized_objective": metrics.penalized_objective,
        "operating_cost": metrics.operating_cost,
        "physical_carbon": metrics.physical_carbon,
        "constraint_violation_rate": metrics.constraint_violation_rate,
        "balance_residual_max": metrics.balance_residual_max,
        "capacity_violation_max": metrics.capacity_violation_max,
        "physical_residual_max": max(
            metrics.balance_residual_max, metrics.capacity_violation_max,
            metrics.conversion_residual_max, metrics.soc_residual_max,
            metrics.ramp_violation_max, metrics.exclusivity_max,
            metrics.renewable_residual_max, metrics.finite_violation_max,
        ),
    }
    write_gate2_row_receipt(directory, row_payload)
    return json.loads((directory / "ROW_RECEIPT.json").read_text(encoding="utf-8"))


def _registered_from_artifact(
    artifact: TrainedMethodArtifact,
    data: Gate2DataBundle,
    parameters: Mapping[str, Any],
    *,
    diff_layer: Any = None,
) -> Any:
    from .formal_v4_2_methods import build_v42_method
    from .formal_v4_method_adapter import build_formal_v4_method_adapter

    kwargs = {
        "task_mean": data.normalization.field_mean["load"],
        "task_scale": data.normalization.field_scale["load"],
        "normalization": data.normalization,
    }
    if artifact.method_id == "Official iTransformer-PTO":
        adapter = build_formal_v4_method_adapter(
            artifact.method_id, parameters, forecaster=artifact.model, **kwargs,
        )
    elif artifact.method_id == "Differentiable-LP":
        if diff_layer is None:
            raise ValueError("Differentiable-LP evaluation requires its frozen layer")
        adapter = build_formal_v4_method_adapter(
            artifact.method_id, parameters, forecaster=artifact.model,
            layer=diff_layer, **kwargs,
        )
    else:
        adapter = build_formal_v4_method_adapter(artifact.method_id, parameters, **kwargs)
        adapter.model = artifact.model
    checkpoint = {
        "path": artifact.checkpoint_path,
        "model_sha256": artifact.checkpoint_sha256,
        "optimizer_steps": int(artifact.training_receipt["optimizer_steps"]),
    }
    return build_v42_method(
        artifact.method_id, seed=artifact.seed, checkpoint=checkpoint,
        parameters=parameters, adapter=adapter, normalization=data.normalization,
    )


def _perfect_information_method(parameters: Mapping[str, Any]) -> Any:
    from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
    from ..scheduling.dispatch_schema import VARIABLES
    from .formal_v4_2_methods import build_v42_method

    def planner(window: Mapping[str, Any], state: FormalV4ClosedLoopState) -> Mapping[str, Any]:
        target = np.asarray(window["forecast_target"], dtype=np.float64)
        renew = np.asarray(window["renewable_realized"], dtype=np.float64)
        prices = np.asarray(window["prices_and_weights"], dtype=np.float64)
        context = dict(parameters)
        context["grid_energy_price"] = prices[:, 0]
        context["gas_energy_price"] = prices[:, 1]
        context["carbon_price"] = prices[:, 2]
        solved = solve_dispatch_lp(DispatchInputs(
            target[:, :3], renew[:, 0], renew[:, 1], context,
            float(state.initial_soc[0, 0]), float(state.previous_chp[0, 0]),
        ))
        if not solved.success:
            raise RuntimeError(f"Perfect-Information-MPC failed: {solved.message}")
        return {"dispatch": np.column_stack([solved.values[name] for name in VARIABLES])}

    return build_v42_method("Perfect-Information-MPC", parameters=parameters, planner=planner)


def execute_gate2_matrix(
    *,
    contract: Any,
    run_root: str | Path,
    parameters: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Train and evaluate the exact 23-row registered Gate 2 matrix."""

    from .formal_v4_2_artifacts import sha256_file
    from .formal_v4_2_gate2_training import (
        load_gate2_data, train_differentiable_lp, train_direct_policy,
        train_official_itransformer_pto, train_rsc_family, train_scheme2r_pto,
    )
    from .formal_v4_2_methods import build_v42_method
    from .formal_v4_diffopt import DifferentiableIESLayer
    from .formal_v4_method_adapter import build_formal_v4_method_adapter

    root = Path(run_root).resolve()
    gate2 = root / "gate2"
    if gate2.exists():
        raise FileExistsError(f"Gate 2 directory already exists: {gate2}")
    rows_root = gate2 / "rows"
    rows_root.mkdir(parents=True)
    data = load_gate2_data(root, contract)
    freeze = json.loads((root / "gate1" / "GATE1_FREEZE.json").read_text(encoding="utf-8"))
    evidence = json.loads((root / "gate1" / "GATE1_EVIDENCE.json").read_text(encoding="utf-8"))
    freeze = {**freeze, **{
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": evidence["source_manifest_sha256"],
        "c_ref": evidence["c_ref"],
    }}
    itransformer_receipt = json.loads((root / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json").read_text(encoding="utf-8"))
    diffopt_receipt = json.loads((root / "protocol" / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json").read_text(encoding="utf-8"))
    legacy_itransformer = {
        "schema_version": "formal-v4.1-itransformer-source-v1",
        **{name: itransformer_receipt[name] for name in (
            "source_root", "repository", "commit", "backbone_class", "imported_file_hashes",
            "license_file", "license_sha256", "reproduction_level", "verified",
        )},
    }
    legacy_path = gate2 / "ITRANSFORMER_ADAPTER_RECEIPT.json"
    write_once_json(legacy_path, legacy_itransformer)
    lineage = {
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": evidence["source_manifest_sha256"],
        "train_manifest_sha256": data.train_manifest_sha256,
        "calibration_manifest_sha256": data.calibration_manifest_sha256,
        "evaluation_manifest_sha256": data.evaluation_manifest_sha256,
        "normalization_sha256": data.normalization_sha256,
    }
    completed: list[Mapping[str, Any]] = []
    for seed in contract.gate2_seeds:
        family = train_rsc_family(seed, data, freeze, parameters, rows_root)
        teacher_path = rows_root / "_shared" / str(seed) / "TEACHER.npz"
        with np.load(teacher_path, allow_pickle=False) as teacher_payload:
            teacher_dispatch = teacher_payload["dispatch"]
        direct = train_direct_policy(
            seed, data, freeze, parameters, rows_root / "Direct-Policy" / str(seed),
            teacher_dispatch=teacher_dispatch,
        )
        scheme = train_scheme2r_pto(
            seed, data, freeze, rows_root / "Scheme2R-PTO" / str(seed),
        )
        official = train_official_itransformer_pto(
            seed, data, freeze, itransformer_receipt,
            rows_root / "Official iTransformer-PTO" / str(seed),
            source_root=Path(__file__).resolve().parents[2] / "third_party" / "iTransformer_source",
            receipt_path=legacy_path,
        )
        diff_layer = DifferentiableIESLayer(parameters)
        diff = train_differentiable_lp(
            seed, data, freeze, diffopt_receipt, parameters,
            rows_root / "Differentiable-LP" / str(seed), layer=diff_layer,
            micro_batch_size=8,
        )
        artifacts = {
            **family, "Direct-Policy": direct, "Scheme2R-PTO": scheme,
            "Official iTransformer-PTO": official, "Differentiable-LP": diff,
        }
        for method_id in (
            "RSC-PF", "Decoupled-RSC-PF", "Direct-Policy", "Scheme2R-PTO",
            "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP",
        ):
            artifact = artifacts[method_id]
            layer = diff_layer if method_id == "Differentiable-LP" else None
            method = _registered_from_artifact(artifact, data, parameters, diff_layer=layer)
            completed.append(execute_gate2_row(
                Gate2RowKey(method_id, seed), method, data, parameters,
                artifact.checkpoint_path.parent, lineage,
            ))
    seasonal_adapter = build_formal_v4_method_adapter(
        "Seasonal-Naive-PTO", parameters,
        task_mean=data.normalization.field_mean["load"],
        task_scale=data.normalization.field_scale["load"],
        normalization=data.normalization,
    )
    seasonal = build_v42_method("Seasonal-Naive-PTO", parameters=parameters, adapter=seasonal_adapter)
    completed.append(execute_gate2_row(
        Gate2RowKey("Perfect-Information-MPC", None), _perfect_information_method(parameters),
        data, parameters, rows_root / "Perfect-Information-MPC" / "deterministic", lineage,
    ))
    completed.append(execute_gate2_row(
        Gate2RowKey("Seasonal-Naive-PTO", None), seasonal, data, parameters,
        rows_root / "Seasonal-Naive-PTO" / "deterministic", lineage,
    ))
    write_once_json(gate2 / "rows.json", {Gate2RowKey(row["method_id"], row["seed"]).label: row for row in completed})
    return completed


__all__ = [
    "Gate2RowKey", "evaluation_windows", "execute_gate2_row",
    "execute_gate2_matrix", "initial_evaluation_state",
]
