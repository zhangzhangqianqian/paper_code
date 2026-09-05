"""Production P0/P1/S/J executor for the formal-v4.4 Pilot."""

from __future__ import annotations

from dataclasses import dataclass
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


__all__ = [
    "TrainingBundleV44", "build_continuous_control_v44", "build_v44_model", "execute_training_stages_v44",
]
