"""Order-safe Pilot executor for formal-v4.6."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml

from .formal_v4_2_data import fit_train_normalization
from .formal_v4_4_pilot_executor import _parameters, build_v44_model
from .formal_v4_4_regime import fit_thermal_prior
from .formal_v4_4_teacher import build_same_information_teacher_v44
from .formal_v4_5_pilot_data import MaterializedPilotV45, MaterializedTrainingV45, build_v45_loaders
from .formal_v4_5_training import StageBudgetV45, StageReceiptV45, run_stage_p0_v45, run_stage_p1_v45, run_stage_s_v45
from .formal_v4_6_contract import FormalV46Contract
from .formal_v4_6_risk import fit_risk_caps_v46
from .formal_v4_6_rollout import FrozenCandidateV46, collect_p1_early_stop_evidence_v46, freeze_v46_candidate, run_matched_v46_rollouts
from .formal_v4_6_training import CalibrationReceiptV46, JointPairReceiptV46, run_v46_calibration


@dataclass(frozen=True)
class TrainingBundleV46:
    p0: StageReceiptV45
    p1: StageReceiptV45
    s: StageReceiptV45
    prior: Any
    normalization: Any
    risk_caps: Any
    calibration: CalibrationReceiptV46
    frozen_candidate: FrozenCandidateV46


def _budget(contract: FormalV46Contract, stage: str) -> StageBudgetV45:
    payload = contract.payload["pilot_budget"]
    if stage not in {"p0", "p1", "s", "j"}:
        raise ValueError(f"unsupported v4.6 stage: {stage}")
    return StageBudgetV45(
        max_epochs=int(payload[f"{stage}_max_epochs"]), minimum_epochs=min(int(payload["minimum_epochs"]), int(payload[f"{stage}_max_epochs"])),
        p0_lr=float(payload["p0_forecaster_lr"]), p1_base_lr=float(payload["p1_base_lr"]), p1_head_lr=float(payload["p1_head_lr"]), s_scheduler_lr=float(payload["s_scheduler_lr"]),
        j_forecaster_lr=float(payload["j_forecaster_lr"]), j_head_lr=float(payload["j_head_lr"]), j_scheduler_lr=float(payload["j_scheduler_lr"]), weight_decay=float(payload["weight_decay"]), max_grad_norm=float(payload["max_grad_norm"]),
        inactive_leakage_target=float(contract.payload["pilot_candidate"]["inactive_leakage_weight"]), ramp_epochs=int(contract.payload["joint_curriculum"]["ramp_epochs"]), patience=int(payload["early_stopping_patience"]),
    )


def _teacher_bundle(model: Any, materialized: MaterializedTrainingV45, benchmark: str | Path, capacity_receipt: str | Path, artifact_root: Path, contract: FormalV46Contract, seed: int) -> dict[str, np.ndarray]:
    train = build_same_information_teacher_v44(model=model, materialized=materialized.train, indices=np.arange(len(materialized.train), dtype=np.int64), benchmark=benchmark, capacity_receipt=capacity_receipt, artifact_root=artifact_root / "teacher_train", contract=contract, seed=seed)
    early = build_same_information_teacher_v44(model=model, materialized=materialized.early_stop, indices=np.arange(len(materialized.early_stop), dtype=np.int64), benchmark=benchmark, capacity_receipt=capacity_receipt, artifact_root=artifact_root / "teacher_early_stop", contract=contract, seed=seed)
    return {"train": train.dispatch, "early_stop": early.dispatch}


def execute_real_pilot_v46(
    *,
    materialized_training: MaterializedTrainingV45,
    selection_loader: Callable[[], MaterializedPilotV45],
    contract: FormalV46Contract,
    artifact_root: str | Path,
    benchmark: str | Path,
    capacity_receipt: str | Path,
    seed: int,
) -> Mapping[str, Any]:
    """Train only on 2015--2018, calibrate, then open the 2019 view once."""

    contract.validate()
    root = Path(artifact_root)
    source = materialized_training.normalization_source
    prior = fit_thermal_prior(source.forecast_target, source.load_history, source.target_times)
    benchmark_payload = yaml.safe_load(Path(benchmark).read_text(encoding="utf-8"))
    capacity_payload = json.loads(Path(capacity_receipt).read_text(encoding="utf-8"))
    parameters = _parameters(benchmark_payload, capacity_payload)
    normalization = fit_train_normalization(source.split)
    model = build_v44_model(materialized_training, contract, prior, parameters)
    batch_size = int(contract.payload["pilot_budget"]["batch_size"])
    loaders = build_v45_loaders(materialized_training, normalization, batch_size=batch_size)
    p0 = run_stage_p0_v45(model, loaders, _budget(contract, "p0"), seed=seed)
    p1 = run_stage_p1_v45(p0, loaders, _budget(contract, "p1"), prior=prior, seed=seed)
    early_batches = loaders["early_stop"]
    evidence = collect_p1_early_stop_evidence_v46(p1.model, early_batches, materialized_training.early_stop.timestamps)
    risk_caps = fit_risk_caps_v46(evidence.prediction, evidence.target, evidence.timestamps, "early_stop", 0.90, contract.contract_sha256, p1.final_sha256)
    teacher = _teacher_bundle(p1.model, materialized_training, benchmark, capacity_receipt, root, contract, seed)
    teacher_loaders = build_v45_loaders(materialized_training, normalization, batch_size=batch_size, teacher=teacher)
    s = run_stage_s_v45(p1, teacher_loaders, _budget(contract, "s"), seed=seed)
    # Gate 1 calibration is deliberately completed before the 2019 selection
    # loader is invoked.  This preserves the contract's no-selection-read
    # calibration boundary even when the caller's loader opens disk artifacts.
    calibration = run_v46_calibration(s, teacher_loaders, _budget(contract, "j"), prior=prior, parameters=parameters, contract=contract, risk_cap=risk_caps, seed=seed)
    frozen = freeze_v46_candidate(calibration.selected)
    selection = selection_loader()
    rollouts = run_matched_v46_rollouts(frozen, selection, normalization, parameters, root)
    return {
        "bundle": TrainingBundleV46(p0, p1, s, prior, normalization, risk_caps, calibration, frozen),
        "calibration": calibration,
        "frozen_candidate": frozen,
        "rollouts": rollouts,
        "selection_lineage": dict(selection.lineage),
        "evaluation_year_accessed": False,
    }


__all__ = ["TrainingBundleV46", "execute_real_pilot_v46"]
