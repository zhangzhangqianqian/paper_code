"""Production P0/P1/S/J executor for the formal-v4.4 Pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml
from torch import nn

from .formal_v4_2_data import fit_train_normalization
from .formal_v4_4_contract import FormalV44Contract
from .formal_v4_4_model import ContinuousControlRSCPFModel, ResidualGatedRSCPFModel
from .formal_v4_4_pilot_data import build_v44_batches
from .formal_v4_4_pilot_materializer import MaterializedV44PilotData
from .formal_v4_4_regime import ThermalPriorReceiptV44, fit_thermal_prior
from .formal_v4_4_rollout import RolloutResultV44, rollout_v44_2019
from .formal_v4_4_teacher import TeacherReceiptV44
from .formal_v4_4_training import (
    JointPairReceiptV44, StageBudgetV44, StageReceiptV44, run_continuous_control_v44,
    run_stage_j_pair_v44, run_stage_p0_v44, run_stage_p1_v44, run_stage_s_v44,
    sha256_state_dict,
)


@dataclass(frozen=True)
class TrainingBundleV44:
    p0: StageReceiptV44
    p1: StageReceiptV44
    continuous_control: StageReceiptV44
    s: StageReceiptV44
    j: JointPairReceiptV44
    teacher: TeacherReceiptV44
    prior: ThermalPriorReceiptV44


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    mean = array.mean(axis=(0, 1)).astype(np.float32)
    scale = array.std(axis=(0, 1)).astype(np.float32)
    return mean, np.where(scale < 1.0e-6, 1.0, scale).astype(np.float32)


def build_v44_model(
    materialized: MaterializedV44PilotData, contract: FormalV44Contract,
    prior: ThermalPriorReceiptV44, parameters: Mapping[str, Any],
) -> ResidualGatedRSCPFModel:
    """Construct RSC-PF using train-only normalization and prior statistics."""

    contract.validate()
    source = materialized.normalization_source.split
    normalization = fit_train_normalization(source)
    renew_mean, renew_scale = _stats(source.renewable_forecast)
    soc_values = np.repeat(source.initial_soc[:, None, :], 4, axis=1)
    soc_mean, soc_scale = _stats(soc_values)
    physical_mean = np.concatenate((normalization.field_mean["load"], renew_mean, normalization.field_mean["scheduler"], soc_mean))
    physical_scale = np.concatenate((normalization.field_scale["load"], renew_scale, normalization.field_scale["scheduler"], soc_scale))
    previous_mean = float(np.mean(source.previous_chp))
    previous_scale = max(float(np.std(source.previous_chp)), 1.0)
    temperature = float(contract.payload.get("pilot_candidate", {}).get("temperature", 1.0))
    return ResidualGatedRSCPFModel(
        transition_probability=torch.as_tensor(prior.transition_probability, dtype=torch.float32),
        regime_temperature=temperature, decoder_parameters=dict(parameters), dropout=0.0,
        task_mean=torch.as_tensor(normalization.field_mean["load"]),
        task_scale=torch.as_tensor(normalization.field_scale["load"]),
        physical_feature_mean=torch.as_tensor(physical_mean), physical_feature_scale=torch.as_tensor(physical_scale),
        previous_chp_mean=previous_mean, previous_chp_scale=previous_scale,
    )


def build_continuous_control_v44(p0_model: ResidualGatedRSCPFModel) -> ContinuousControlRSCPFModel:
    """Clone the P0 parent while removing regime-gate/residual parameters."""

    if not isinstance(p0_model, ResidualGatedRSCPFModel):
        raise TypeError("continuous control must be cloned from a v4.4 residual P0 model")
    model = ContinuousControlRSCPFModel(
        decoder_parameters=dict(p0_model.core.decoder_parameters),
        task_mean=p0_model.core.task_mean.detach().clone(), task_scale=p0_model.core.task_scale.detach().clone(),
        physical_feature_mean=p0_model.core.physical_feature_mean.detach().clone(),
        physical_feature_scale=p0_model.core.physical_feature_scale.detach().clone(),
        previous_chp_mean=p0_model.previous_chp_mean.detach().clone(), previous_chp_scale=p0_model.previous_chp_scale.detach().clone(),
        dropout=0.0,
    )
    destination = model.state_dict()
    source = p0_model.state_dict()
    for name in destination:
        if name not in source:
            raise ValueError(f"P0 parent is missing continuous-control parameter {name}")
        destination[name].copy_(source[name])
    model.load_state_dict(destination, strict=True)
    if model.v44_parameter_groups().get("gate") or model.v44_parameter_groups().get("magnitude"):
        raise AssertionError("continuous-control model unexpectedly contains residual heads")
    return model


def _budget(contract: FormalV44Contract, stage: str) -> StageBudgetV44:
    payload = contract.payload["pilot_budget"]
    epochs = int(payload[f"{stage}_max_epochs"])
    return StageBudgetV44(
        max_epochs=epochs, minimum_epochs=min(int(payload["minimum_epochs"]), epochs),
        p0_lr=float(payload["p0_forecaster_lr"] if stage == "p0" else payload["p1_base_lr"]),
        base_lr=float(payload["p1_base_lr"]), head_lr=float(payload["p1_head_lr"]),
        scheduler_lr=float(payload["s_scheduler_lr"]), weight_decay=float(payload["weight_decay"]),
        max_grad_norm=float(payload["max_grad_norm"]), inactive_leakage_target=float(contract.payload["pilot_candidate"]["inactive_leakage_weight"]),
        patience=int(payload["early_stopping_patience"]),
    )


def _parameters(benchmark: Mapping[str, Any], capacity: Mapping[str, Any]) -> dict[str, Any]:
    values = benchmark.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("benchmark values are missing")
    result = dict(values)
    selected = capacity.get("selected")
    if not isinstance(selected, Mapping):
        audit = capacity.get("capacity_audit")
        selected = audit.get("selected") if isinstance(audit, Mapping) else None
    multiplier = float(selected.get("multiplier", capacity.get("capacity_multiplier", 1.0))) if isinstance(selected, Mapping) else float(capacity.get("capacity_multiplier", 1.0))
    for name in ("electric_chiller_capacity", "absorption_chiller_capacity"):
        if name in result:
            result[name] = float(result[name]) * multiplier
    return result


def _loaders(materialized: MaterializedV44PilotData, *, teacher: TeacherReceiptV44 | None, batch_size: int) -> tuple[list[Mapping[str, Any]], Any]:
    normalization = fit_train_normalization(materialized.normalization_source.split)
    indices = np.arange(len(materialized.train), dtype=np.int64)
    dispatch = None if teacher is None else teacher.dispatch
    return build_v44_batches(materialized.train, normalization, indices, batch_size=batch_size, teacher_dispatch=dispatch), normalization


def execute_training_stages_v44(
    *, materialized: MaterializedV44PilotData, contract: FormalV44Contract, artifact_root: str | Path,
    seed: int, parameters: Mapping[str, Any], prior: ThermalPriorReceiptV44 | None = None,
    teacher: TeacherReceiptV44 | None = None, model: ResidualGatedRSCPFModel | None = None,
    teacher_factory: Callable[[nn.Module, Path], TeacherReceiptV44] | None = None,
) -> TrainingBundleV44:
    """Run the frozen matched P0/P1/continuous/S/J stage sequence."""

    contract.validate()
    if prior is None:
        source = materialized.normalization_source
        prior = fit_thermal_prior(source.forecast_target, source.load_history, source.target_times)
    model = build_v44_model(materialized, contract, prior, parameters) if model is None else model
    batch_size = int(contract.payload["pilot_budget"]["batch_size"])
    p0_loaders, _normalization = _loaders(materialized, teacher=None, batch_size=batch_size)
    p0 = run_stage_p0_v44(model, p0_loaders, _budget(contract, "p0"), seed=seed)
    p1 = run_stage_p1_v44(p0, p0_loaders, _budget(contract, "p1"), prior=prior, seed=seed)
    continuous = run_continuous_control_v44(p0, p0_loaders, _budget(contract, "p1"), seed=seed)
    if teacher is None:
        if teacher_factory is None:
            raise ValueError("formal-v4.4 S/J stages require a same-information teacher or teacher_factory")
        teacher = teacher_factory(p1.model, Path(artifact_root))
    teacher_loaders, _ = _loaders(materialized, teacher=teacher, batch_size=batch_size)
    s = run_stage_s_v44(p1, teacher_loaders, _budget(contract, "s"), seed=seed)
    pair = run_stage_j_pair_v44(
        s, teacher_loaders, _budget(contract, "j"), prior=prior,
        c_ref=float(parameters.get("unserved_penalty", 1.0)), parameters=parameters, seed=seed,
    )
    return TrainingBundleV44(p0=p0, p1=p1, continuous_control=continuous, s=s, j=pair, teacher=teacher, prior=prior)


def _load_runtime_parameters(benchmark: str | Path, capacity_receipt: str | Path) -> dict[str, Any]:
    benchmark_payload = yaml.safe_load(Path(benchmark).read_text(encoding="utf-8"))
    capacity_payload = __import__("json").loads(Path(capacity_receipt).read_text(encoding="utf-8"))
    return _parameters(benchmark_payload, capacity_payload)


def _persist_rollout_alias(result: RolloutResultV44, artifact_root: Path, method_id: str) -> None:
    from .formal_v4_4_artifacts import write_json_once, write_npz_once
    root = artifact_root / "rollout"; root.mkdir(parents=True, exist_ok=True)
    write_npz_once(root / f"{method_id}.npz", {
        "prediction": result.prediction, "target": result.target, "probability": result.probability,
        "prior_probability": result.prior_probability, "regimes": result.regimes, "planned_dispatch": result.planned_dispatch,
        "settled_dispatch": result.settled_dispatch, "shortage": result.shortage, "physical_residual": result.physical_residual,
        "operating_cost": result.operating_cost, "physical_carbon": result.physical_carbon, "penalized_objective": result.penalized_objective,
        "initial_soc": result.initial_soc, "previous_chp": result.previous_chp, "realized_demand": result.realized_demand,
        "realized_renewables": result.realized_renewables, "realized_prices": result.realized_prices,
        "times": result.times.astype("datetime64[ns]").astype(np.int64), "state_hashes": result.state_hashes,
    })
    write_json_once(root / f"{method_id}.json", {"schema": "formal-v4.4-rollout-v1", "method_id": str(method_id), "rows": int(len(result.times)), "times": [str(value) for value in result.times]})


def _save_stage_checkpoint(root: Path, name: str, receipt: StageReceiptV44) -> str:
    """Persist a CPU-only, immutable checkpoint for independent audit.

    The checkpoint intentionally stores only the model state and the receipt
    metadata.  Optimizer internals are not evidence for the Pilot claim and
    would make the artifact unnecessarily non-portable.
    """

    if receipt.model is None:
        raise ValueError(f"stage {name} has no in-memory model")
    from .formal_v4_4_artifacts import sha256_file
    target = root / "stages" / f"{name}.pt"
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "formal-v4.4-stage-checkpoint-v1",
        "stage": str(receipt.stage), "mode": str(receipt.mode),
        "parent_sha256": str(receipt.parent_sha256), "final_sha256": str(receipt.final_sha256),
        "epochs": int(receipt.epochs), "optimizer_steps": int(receipt.optimizer_steps),
        "forecast_optimizer_steps": int(receipt.forecast_optimizer_steps),
        "loss_history": [float(value) for value in receipt.loss_history],
        "stopping_reason": str(receipt.stopping_reason),
        "gradient_norms": {str(key): float(value) for key, value in receipt.gradient_norms.items()},
        "state_dict": {str(key): value.detach().cpu().clone() for key, value in receipt.model.state_dict().items()},
    }
    temporary = target.with_name(f".{target.name}.writing")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(payload, temporary)
    temporary.replace(target)
    return sha256_file(target)


def _jsonable_runtime(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_runtime(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_runtime(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _model_runtime_config(
    model: ResidualGatedRSCPFModel, parameters: Mapping[str, Any], prior: ThermalPriorReceiptV44,
    materialized: MaterializedV44PilotData, contract: FormalV44Contract, batch_size: int,
) -> dict[str, Any]:
    """Return every non-learned value needed to replay a saved J batch."""

    return _jsonable_runtime({
        "schema": "formal-v4.4-stage-runtime-v1",
        "contract_sha256": contract.contract_sha256,
        "batch_size": int(batch_size),
        "transition_probability": model.thermal_head.transition_log_prior.detach().exp().cpu().tolist(),
        "regime_temperature": float(model.thermal_head.temperature),
        "decoder_parameters": dict(model.core.decoder_parameters),
        "task_mean": model.core.task_mean.detach().cpu().tolist(),
        "task_scale": model.core.task_scale.detach().cpu().tolist(),
        "physical_feature_mean": model.core.physical_feature_mean.detach().cpu().tolist(),
        "physical_feature_scale": model.core.physical_feature_scale.detach().cpu().tolist(),
        "previous_chp_mean": model.previous_chp_mean.detach().cpu().tolist(),
        "previous_chp_scale": model.previous_chp_scale.detach().cpu().tolist(),
        "parameters": dict(parameters),
        "prior": prior.to_dict(),
        "materialized_lineage": dict(materialized.lineage),
    })


def _metric_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("inf")
    return float(numerator / denominator)


def execute_real_pilot_v44(
    *, train_data: str | Path, selection_data: str | Path, benchmark: str | Path,
    capacity_receipt: str | Path, split: Mapping[str, np.ndarray], contract: FormalV44Contract,
    artifact_root: str | Path,
) -> Mapping[str, Any]:
    """Run the real v4.4 Pilot and return only measured rollout evidence."""

    from .formal_v4_2_data import fit_train_normalization
    from .formal_v4_4_artifacts import write_json_once, write_npz_once
    from .formal_v4_4_metrics import compute_forecast_metrics_v44
    from .formal_v4_4_pilot import EXPECTED_ROWS
    from .formal_v4_4_pilot_materializer import materialize_v44_pilot_data

    artifact_root = Path(artifact_root)
    materialized = materialize_v44_pilot_data(
        train_data=train_data, selection_data=selection_data, benchmark=benchmark,
        capacity_receipt=capacity_receipt, split=split, artifact_root=artifact_root, contract=contract,
    )
    runtime_parameters = _load_runtime_parameters(benchmark, capacity_receipt)
    source = materialized.normalization_source
    prior = fit_thermal_prior(source.forecast_target, source.load_history, source.target_times)
    seed = int(contract.payload["pilot_budget"]["seed"])
    teacher_cache = artifact_root

    def teacher_factory(p1_model: nn.Module, root: Path) -> TeacherReceiptV44:
        from .formal_v4_4_teacher import build_same_information_teacher_v44
        return build_same_information_teacher_v44(
            model=p1_model, materialized=materialized, indices=np.arange(len(materialized.train), dtype=np.int64),
            benchmark=benchmark, capacity_receipt=capacity_receipt, artifact_root=teacher_cache,
            contract=contract, seed=seed,
        )

    bundle = execute_training_stages_v44(
        materialized=materialized, contract=contract, artifact_root=artifact_root, seed=seed,
        parameters=runtime_parameters, prior=prior, teacher_factory=teacher_factory,
    )
    # Save immutable stage evidence before any full-2019 rollout.  These
    # checkpoints are the source used by the independent gradient audit; the
    # Pilot receipt never relies solely on in-memory norms.
    checkpoint_hashes = {
        "P0": _save_stage_checkpoint(artifact_root, "P0", bundle.p0),
        "P1": _save_stage_checkpoint(artifact_root, "P1", bundle.p1),
        "continuous_control": _save_stage_checkpoint(artifact_root, "continuous_control", bundle.continuous_control),
        "S": _save_stage_checkpoint(artifact_root, "S", bundle.s),
        "J_joint": _save_stage_checkpoint(artifact_root, "J_joint", bundle.j.joint),
        "J_decoupled": _save_stage_checkpoint(artifact_root, "J_decoupled", bundle.j.decoupled),
    }
    normalization = fit_train_normalization(source.split)
    teacher_loaders, _ = _loaders(materialized, teacher=bundle.teacher, batch_size=int(contract.payload["pilot_budget"]["batch_size"]))
    if not teacher_loaders:
        raise ValueError("formal-v4.4 Pilot produced no replay batch")
    replay_batch = teacher_loaders[0]
    write_npz_once(artifact_root / "PILOT_BATCH.npz", {
        key: (value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value))
        for key, value in replay_batch.items()
    })
    runtime_payload = _model_runtime_config(
        bundle.j.joint.model, runtime_parameters, prior, materialized, contract,
        int(contract.payload["pilot_budget"]["batch_size"]),
    )
    runtime_payload["checkpoint_hashes"] = checkpoint_hashes
    write_json_once(artifact_root / "STAGE_RUNTIME.json", runtime_payload)
    indices = np.arange(len(materialized.selection_full), dtype=np.int64)
    rollout_models = {
        "continuous_control": (bundle.continuous_control.model, "continuous_control"),
        "residual_stage_p1": (bundle.p1.model, "residual_stage_p1"),
        "rsc_pf_joint": (bundle.j.joint.model, "rsc_pf_joint"),
        "fair_decoupled": (bundle.j.decoupled.model, "fair_decoupled"),
    }
    rollouts: dict[str, RolloutResultV44] = {}
    for row, (model, method_id) in rollout_models.items():
        rollouts[row] = rollout_v44_2019(
            model=model, materialized=materialized, indices=indices, normalization=normalization,
            parameters=runtime_parameters, artifact_root=artifact_root, method_id=method_id,
        )
    # The transition-prior row keeps the fixed P1 magnitude backbone but
    # replaces learned probabilities by the train-only causal transition prior.
    p1_rollout = rollouts["residual_stage_p1"]
    transition = RolloutResultV44(
        method_id="transition_prior", prediction=p1_rollout.prediction, target=p1_rollout.target,
        probability=p1_rollout.prior_probability, prior_probability=p1_rollout.prior_probability,
        regimes=p1_rollout.regimes, planned_dispatch=p1_rollout.planned_dispatch,
        settled_dispatch=p1_rollout.settled_dispatch, shortage=p1_rollout.shortage,
        physical_residual=p1_rollout.physical_residual, operating_cost=p1_rollout.operating_cost,
        physical_carbon=p1_rollout.physical_carbon, penalized_objective=p1_rollout.penalized_objective,
        initial_soc=p1_rollout.initial_soc, previous_chp=p1_rollout.previous_chp,
        realized_demand=p1_rollout.realized_demand, realized_renewables=p1_rollout.realized_renewables,
        realized_prices=p1_rollout.realized_prices,
        times=p1_rollout.times, state_hashes=p1_rollout.state_hashes,
    )
    _persist_rollout_alias(transition, artifact_root, "transition_prior")
    rollouts["transition_prior"] = transition
    metrics = {name: compute_forecast_metrics_v44(value.prediction, value.target, value.probability, value.prior_probability, value.regimes, value.times) for name, value in rollouts.items()}
    joint = rollouts["rsc_pf_joint"]; decoupled = rollouts["fair_decoupled"]; baseline = rollouts["residual_stage_p1"]
    joint_metrics = metrics["rsc_pf_joint"]; base_metrics = metrics["residual_stage_p1"]
    comparisons = {
        "leakage_ratio": {name: _metric_ratio(joint_metrics.inactive_leakage[name], base_metrics.inactive_leakage[name]) for name in ("cooling", "heating")},
        "active_wape_ratio": {name: _metric_ratio(joint_metrics.active_only[name]["wape"], base_metrics.active_only[name]["wape"]) for name in ("cooling", "heating")},
        "electricity_gas_wape_ratio": {name: _metric_ratio(joint_metrics.task[name]["wape"], base_metrics.task[name]["wape"]) for name in ("electricity", "gas")},
        "four_task_score_ratio": _metric_ratio(joint_metrics.four_task_score, base_metrics.four_task_score),
    }
    joint_objective = float(np.mean(joint.penalized_objective)); dec_objective = float(np.mean(decoupled.penalized_objective))
    joint_shortage = float(np.mean(joint.shortage.sum(axis=1))); dec_shortage = float(np.mean(decoupled.shortage.sum(axis=1)))
    joint_grad = dict(bundle.j.joint.gradient_norms); dec_grad = dict(bundle.j.decoupled.gradient_norms)
    result = {
        "prediction": joint.prediction, "target": joint.target, "probability": joint.probability,
        "prior_probability": joint.prior_probability, "regimes": joint.regimes, "times": joint.times,
        "comparisons": comparisons,
        "joint": {"penalized_objective": joint_objective, "shortage": joint_shortage, "gradient_norms": joint_grad},
        "decoupled": {"penalized_objective": dec_objective, "shortage": dec_shortage, "gradient_norms": dec_grad},
        "physics": {"max_residual": float(np.max(joint.physical_residual))},
        "rows": list(EXPECTED_ROWS),
    }
    # Keep a compact stage lineage receipt beside the row rollouts so the
    # independent audit can identify the exact parent hashes used by J.
    write_json_once(artifact_root / "STAGE_LINEAGE.json", {
        "schema": "formal-v4.4-stage-lineage-v1", "p0": sha256_state_dict(bundle.p0.model),
        "p1": sha256_state_dict(bundle.p1.model), "s": sha256_state_dict(bundle.s.model),
        "joint": sha256_state_dict(bundle.j.joint.model), "decoupled": sha256_state_dict(bundle.j.decoupled.model),
        "teacher": dict(bundle.teacher.lineage), "evaluation_year_accessed": False,
    })
    return result


__all__ = [
    "TrainingBundleV44", "build_continuous_control_v44", "build_v44_model", "execute_real_pilot_v44", "execute_training_stages_v44",
]
