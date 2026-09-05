"""Training-stage executor for the formal-v4.5 diagnostic and Pilot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .formal_v4_2_data import fit_train_normalization
from .formal_v4_4_pilot_executor import build_continuous_control_v44, build_v44_model, _parameters
from .formal_v4_4_pilot_materializer import MaterializedV44PilotData
from .formal_v4_4_regime import ThermalPriorReceiptV44, fit_thermal_prior
from .formal_v4_4_teacher import TeacherReceiptV44, build_same_information_teacher_v44
from .formal_v4_5_contract import FormalV45Contract
from .formal_v4_5_pilot_data import build_v45_loaders
from .formal_v4_5_training import (
    JointPairReceiptV45, StageBudgetV45, StageReceiptV45,
    run_continuous_control_v45, run_stage_j_pair_v45, run_stage_p0_v45,
    run_stage_p1_v45, run_stage_s_v45,
)


@dataclass(frozen=True)
class TeacherBundleV45:
    train: TeacherReceiptV44
    early_stop: TeacherReceiptV44

    @property
    def dispatch(self) -> dict[str, np.ndarray]:
        return {"train": self.train.dispatch, "early_stop": self.early_stop.dispatch}


@dataclass(frozen=True)
class TrainingBundleV45:
    p0: StageReceiptV45
    p1: StageReceiptV45
    continuous_control: StageReceiptV45
    s: StageReceiptV45
    j: JointPairReceiptV45
    teacher: TeacherBundleV45
    prior: ThermalPriorReceiptV44
    normalization: Any


def _budget(contract: FormalV45Contract, stage: str) -> StageBudgetV45:
    payload = contract.payload["pilot_budget"]
    if stage not in {"p0", "p1", "s", "j"}:
        raise ValueError(f"unsupported formal-v4.5 stage: {stage}")
    return StageBudgetV45(
        max_epochs=int(payload[f"{stage}_max_epochs"]),
        minimum_epochs=min(int(payload["minimum_epochs"]), int(payload[f"{stage}_max_epochs"])),
        p0_lr=float(payload["p0_forecaster_lr"]),
        p1_base_lr=float(payload["p1_base_lr"]),
        p1_head_lr=float(payload["p1_head_lr"]),
        s_scheduler_lr=float(payload["s_scheduler_lr"]),
        j_forecaster_lr=float(payload["j_forecaster_lr"]),
        j_head_lr=float(payload["j_head_lr"]),
        j_scheduler_lr=float(payload["j_scheduler_lr"]),
        weight_decay=float(payload["weight_decay"]),
        max_grad_norm=float(payload["max_grad_norm"]),
        inactive_leakage_target=float(contract.payload["pilot_candidate"]["inactive_leakage_weight"]),
        ramp_epochs=int(contract.payload["joint_curriculum"]["ramp_epochs"]),
        patience=int(payload["early_stopping_patience"]),
    )


def build_v45_teachers(
    *,
    model: Any,
    materialized: MaterializedV44PilotData,
    benchmark: str | Path,
    capacity_receipt: str | Path,
    artifact_root: str | Path,
    contract: FormalV45Contract,
    seed: int,
) -> TeacherBundleV45:
    """Generate separate same-information teachers for train and early-stop."""

    root = Path(artifact_root)
    train = build_same_information_teacher_v44(
        model=model, materialized=materialized.train,
        indices=np.arange(len(materialized.train), dtype=np.int64),
        benchmark=benchmark, capacity_receipt=capacity_receipt,
        artifact_root=root / "teacher_train", contract=contract, seed=seed,
    )
    early_stop = build_same_information_teacher_v44(
        model=model, materialized=materialized.early_stop,
        indices=np.arange(len(materialized.early_stop), dtype=np.int64),
        benchmark=benchmark, capacity_receipt=capacity_receipt,
        artifact_root=root / "teacher_early_stop", contract=contract, seed=seed,
    )
    return TeacherBundleV45(train=train, early_stop=early_stop)


def execute_training_stages_v45(
    *,
    materialized: MaterializedV44PilotData,
    contract: FormalV45Contract,
    artifact_root: str | Path,
    benchmark: str | Path,
    capacity_receipt: str | Path,
    seed: int,
    parameters: Mapping[str, Any],
    prior: ThermalPriorReceiptV44 | None = None,
    teacher: TeacherBundleV45 | None = None,
) -> TrainingBundleV45:
    """Run P0/P1/continuous/S/J using explicit train and early-stop loaders."""

    contract.validate()
    if prior is None:
        source = materialized.normalization_source
        prior = fit_thermal_prior(source.forecast_target, source.load_history, source.target_times)
    model = build_v44_model(materialized, contract, prior, parameters)
    normalization = fit_train_normalization(materialized.normalization_source.split)
    batch_size = int(contract.payload["pilot_budget"]["batch_size"])
    loaders = build_v45_loaders(materialized, normalization, batch_size=batch_size)
    p0 = run_stage_p0_v45(model, loaders, _budget(contract, "p0"), seed=seed)
    p1 = run_stage_p1_v45(p0, loaders, _budget(contract, "p1"), prior=prior, seed=seed)
    continuous_model = build_continuous_control_v44(p0.model)
    continuous = run_continuous_control_v45(p0, continuous_model, loaders, _budget(contract, "p1"), seed=seed)
    if teacher is None:
        teacher = build_v45_teachers(
            model=p1.model, materialized=materialized, benchmark=benchmark,
            capacity_receipt=capacity_receipt,
            artifact_root=artifact_root, contract=contract, seed=seed,
        )
    teacher_loaders = build_v45_loaders(
        materialized, normalization, batch_size=batch_size, teacher=teacher.dispatch,
    )
    s = run_stage_s_v45(p1, teacher_loaders, _budget(contract, "s"), seed=seed)

    def validation_guard(details: Mapping[str, float]) -> bool:
        # The parent-normalized validation loss is a train-only forecast
        # guardrail. Detailed WAPE/F1/leakage guardrails are recomputed from the
        # persisted early-stop arrays by the independent v4.5 audit.
        return bool(
            np.isfinite(details["metric"])
            and details["forecast"] <= 1.02
            and details["anchor"] <= 0.25
        )

    pair = run_stage_j_pair_v45(
        s, teacher_loaders, _budget(contract, "j"), prior=prior,
        c_ref=float(parameters.get("unserved_penalty", 1.0)), parameters=parameters,
        contract=contract, seed=seed, validation_guard=validation_guard,
    )
    return TrainingBundleV45(
        p0=p0, p1=p1, continuous_control=continuous, s=s, j=pair,
        teacher=teacher, prior=prior, normalization=normalization,
    )


__all__ = ["TeacherBundleV45", "TrainingBundleV45", "build_v45_teachers", "execute_training_stages_v45"]
