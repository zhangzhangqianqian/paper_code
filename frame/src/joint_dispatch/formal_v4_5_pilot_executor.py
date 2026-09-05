"""Training-stage executor for the formal-v4.5 diagnostic and Pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def _budget(contract: FormalV45Contract, stage: str, *, epoch_cap: int | None = None) -> StageBudgetV45:
    payload = contract.payload["pilot_budget"]
    if stage not in {"p0", "p1", "s", "j"}:
        raise ValueError(f"unsupported formal-v4.5 stage: {stage}")
    configured_epochs = int(payload[f"{stage}_max_epochs"])
    max_epochs = configured_epochs if epoch_cap is None else min(configured_epochs, int(epoch_cap))
    if max_epochs < 1:
        raise ValueError("epoch_cap must be positive")
    return StageBudgetV45(
        max_epochs=max_epochs,
        minimum_epochs=min(int(payload["minimum_epochs"]), max_epochs),
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
    epoch_cap: int | None = None,
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
    p0 = run_stage_p0_v45(model, loaders, _budget(contract, "p0", epoch_cap=epoch_cap), seed=seed)
    p1 = run_stage_p1_v45(p0, loaders, _budget(contract, "p1", epoch_cap=epoch_cap), prior=prior, seed=seed)
    continuous_model = build_continuous_control_v44(p0.model)
    continuous = run_continuous_control_v45(p0, continuous_model, loaders, _budget(contract, "p1", epoch_cap=epoch_cap), seed=seed)
    if teacher is None:
        teacher = build_v45_teachers(
            model=p1.model, materialized=materialized, benchmark=benchmark,
            capacity_receipt=capacity_receipt,
            artifact_root=artifact_root, contract=contract, seed=seed,
        )
    teacher_loaders = build_v45_loaders(
        materialized, normalization, batch_size=batch_size, teacher=teacher.dispatch,
    )
    s = run_stage_s_v45(p1, teacher_loaders, _budget(contract, "s", epoch_cap=epoch_cap), seed=seed)

    def validation_guard(details: Mapping[str, float]) -> bool:
        # This parent-relative loss is only a conservative checkpoint guard.
        # The final 2019 Pilot still applies the frozen WAPE/F1/leakage
        # thresholds; a 2% final semantic threshold is not a valid proxy for
        # seasonal variation in normalized early-stop loss.
        return bool(
            np.isfinite(details["metric"])
            and details["forecast"] <= 1.10
            and details["anchor"] <= 0.25
        )

    pair = run_stage_j_pair_v45(
        s, teacher_loaders, _budget(contract, "j", epoch_cap=epoch_cap), prior=prior,
        c_ref=float(parameters.get("unserved_penalty", 1.0)), parameters=parameters,
        contract=contract, seed=seed, validation_guard=validation_guard,
    )
    return TrainingBundleV45(
        p0=p0, p1=p1, continuous_control=continuous, s=s, j=pair,
        teacher=teacher, prior=prior, normalization=normalization,
    )


def execute_real_pilot_v45(
    *,
    materialized: Any,
    contract: FormalV45Contract,
    artifact_root: str | Path,
    benchmark: str | Path,
    capacity_receipt: str | Path,
    seed: int,
    epoch_cap: int | None = None,
) -> Mapping[str, Any]:
    """Train the frozen v4.5 stages and evaluate the permitted 2019 view."""

    import json
    from .formal_v4_2_data import fit_train_normalization
    from .formal_v4_4_metrics import compute_forecast_metrics_v44
    from .formal_v4_4_rollout import rollout_v44_2019

    artifact_root = Path(artifact_root)
    benchmark_payload = __import__("yaml").safe_load(Path(benchmark).read_text(encoding="utf-8"))
    capacity_payload = json.loads(Path(capacity_receipt).read_text(encoding="utf-8"))
    parameters = _parameters(benchmark_payload, capacity_payload)
    prior = fit_thermal_prior(
        materialized.normalization_source.forecast_target,
        materialized.normalization_source.load_history,
        materialized.normalization_source.target_times,
    )
    bundle = execute_training_stages_v45(
        materialized=materialized, contract=contract, artifact_root=artifact_root,
        benchmark=benchmark, capacity_receipt=capacity_receipt, seed=int(seed),
        parameters=parameters, prior=prior, epoch_cap=epoch_cap,
    )
    normalization = fit_train_normalization(materialized.normalization_source.split)
    indices = np.arange(len(materialized.selection_full), dtype=np.int64)
    models = {
        "continuous_control": bundle.continuous_control.model,
        "residual_stage_p1": bundle.p1.model,
        "rsc_pf_joint": bundle.j.joint.model,
        "fair_decoupled": bundle.j.decoupled.model,
    }
    rollouts = {
        name: rollout_v44_2019(
            model=model, materialized=materialized, indices=indices,
            normalization=normalization, parameters=parameters,
            artifact_root=artifact_root, method_id=name,
        ) for name, model in models.items()
    }
    p1 = rollouts["residual_stage_p1"]
    from .formal_v4_4_rollout import RolloutResultV44
    transition = RolloutResultV44(
        method_id="transition_prior", prediction=p1.prediction, target=p1.target,
        probability=p1.prior_probability, prior_probability=p1.prior_probability,
        regimes=p1.regimes, planned_dispatch=p1.planned_dispatch,
        settled_dispatch=p1.settled_dispatch, shortage=p1.shortage,
        physical_residual=p1.physical_residual, operating_cost=p1.operating_cost,
        physical_carbon=p1.physical_carbon, penalized_objective=p1.penalized_objective,
        initial_soc=p1.initial_soc, previous_chp=p1.previous_chp,
        realized_demand=p1.realized_demand, realized_renewables=p1.realized_renewables,
        realized_prices=p1.realized_prices, times=p1.times, state_hashes=p1.state_hashes,
    )
    rollouts["transition_prior"] = transition
    metrics = {
        name: compute_forecast_metrics_v44(
            value.prediction, value.target, value.probability,
            value.prior_probability, value.regimes, value.times,
        ) for name, value in rollouts.items()
    }
    joint = rollouts["rsc_pf_joint"]; decoupled = rollouts["fair_decoupled"]
    baseline = rollouts["residual_stage_p1"]

    def ratio(numerator: float, denominator: float) -> float:
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
            return float("inf")
        return float(numerator / denominator)

    joint_metrics = metrics["rsc_pf_joint"]; base_metrics = metrics["residual_stage_p1"]
    comparisons = {
        "leakage_ratio": {name: ratio(joint_metrics.inactive_leakage[name], base_metrics.inactive_leakage[name]) for name in ("cooling", "heating")},
        "active_wape_ratio": {name: ratio(joint_metrics.active_only[name]["wape"], base_metrics.active_only[name]["wape"]) for name in ("cooling", "heating")},
        "electricity_gas_wape_ratio": {name: ratio(joint_metrics.task[name]["wape"], base_metrics.task[name]["wape"]) for name in ("electricity", "gas")},
        "four_task_score_ratio": ratio(joint_metrics.four_task_score, base_metrics.four_task_score),
    }
    return {
        "prediction": joint.prediction, "target": joint.target,
        "probability": joint.probability, "prior_probability": joint.prior_probability,
        "regimes": joint.regimes, "times": joint.times,
        "comparisons": comparisons,
        "joint": {
            "penalized_objective": float(np.mean(joint.penalized_objective)),
            "shortage": float(np.mean(joint.shortage.sum(axis=1))),
            "gradient_norms": dict(bundle.j.joint.gradient_norms),
        },
        "decoupled": {
            "penalized_objective": float(np.mean(decoupled.penalized_objective)),
            "shortage": float(np.mean(decoupled.shortage.sum(axis=1))),
            "gradient_norms": dict(bundle.j.decoupled.gradient_norms),
        },
        "physics": {"max_residual": float(np.max(joint.physical_residual))},
        "metrics": {name: asdict(metric) for name, metric in metrics.items()},
        "rows": ["continuous_control", "residual_stage_p1", "rsc_pf_joint", "fair_decoupled", "transition_prior"],
        "bundle": bundle,
    }


__all__ = [
    "TeacherBundleV45", "TrainingBundleV45", "build_v45_teachers",
    "execute_real_pilot_v45", "execute_training_stages_v45",
]
