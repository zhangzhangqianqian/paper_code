"""Matched J-stage training and train-only risk calibration for formal-v4.6."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .formal_v4_4_regime import ThermalPriorReceiptV44
from .formal_v4_4_training import (
    _all_forecast,
    _decision_and_imitation,
    _groups,
    _inputs,
    named_autograd_norms,
    seed_everything,
    sha256_state_dict,
)
from .formal_v4_5_loss import JointNormalizationV45, curriculum_weights_v45
from .formal_v4_5_training import StageBudgetV45, StageReceiptV45
from .formal_v4_6_contract import FormalV46Contract
from .formal_v4_6_loss import build_j_optimizer_v46, joint_loss_v46
from .formal_v4_6_model import RiskAdjustedRSCPFModelV46
from .formal_v4_6_risk import RiskCapReceiptV46


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


@dataclass(frozen=True)
class StageReceiptV46:
    stage: str
    mode: str
    parent_sha256: str
    final_sha256: str
    best_sha256: str
    terminal_sha256: str
    epochs: int
    best_epoch: int
    optimizer_steps: int
    forecast_optimizer_steps: int
    loss_history: tuple[float, ...]
    validation_history: tuple[Mapping[str, float], ...]
    stopping_reason: str
    selection_metric: float
    gradient_norms: Mapping[str, float] = field(default_factory=dict)
    optimizer_groups: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    model: nn.Module | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class JointPairReceiptV46:
    joint: StageReceiptV46
    decoupled: StageReceiptV46


@dataclass(frozen=True)
class ZeroRiskReceiptV46:
    risk_trainable: bool
    risk_adjustment: np.ndarray
    scheduler_demand_sha256: str
    forecast_nominal_sha256: str
    parent_sha256: str
    model: nn.Module | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CandidateValidationV46:
    eligible: bool
    decision_objective: float
    nominal_metrics: Mapping[str, float]
    failure_names: tuple[str, ...]


@dataclass(frozen=True)
class CalibratedCandidateV46:
    risk_multiplier: float
    pair: JointPairReceiptV46
    validation: CandidateValidationV46


@dataclass(frozen=True)
class CalibrationReceiptV46:
    candidates: Mapping[float, CalibratedCandidateV46]
    selected_multiplier: float
    selected: CalibratedCandidateV46
    selection_role: str
    selection_year_accessed: bool


def _contract_payload(contract: FormalV46Contract | Mapping[str, Any]) -> Mapping[str, Any]:
    return contract.payload if hasattr(contract, "payload") else contract


def _stage_model(stage_s: StageReceiptV45 | Any) -> nn.Module:
    model = stage_s.model if hasattr(stage_s, "model") else stage_s
    if not isinstance(model, nn.Module):
        raise TypeError("stage S must contain a PyTorch model")
    return model


def initialize_v46_parent(
    stage_s: StageReceiptV45 | Any,
    risk_cap: RiskCapReceiptV46,
    contract: FormalV46Contract | Mapping[str, Any],
) -> RiskAdjustedRSCPFModelV46:
    """Construct v4.6 and import only the compatible frozen S tensors."""

    source = _stage_model(stage_s)
    payload = _contract_payload(contract)
    if hasattr(contract, "contract_sha256") and risk_cap.contract_sha256 != contract.contract_sha256:
        raise ValueError("risk-cap contract hash does not match formal-v4.6 contract")
    transition_log_prior = getattr(getattr(source, "thermal_head", None), "transition_log_prior", None)
    if transition_log_prior is None:
        raise ValueError("stage S model must expose thermal_head.transition_log_prior")
    seed_everything(int(payload.get("pilot_budget", {}).get("seed", 2026)))
    model = RiskAdjustedRSCPFModelV46(
        transition_probability=transition_log_prior.detach().exp(),
        risk_cap=torch.as_tensor(risk_cap.cap, dtype=torch.float32),
        regime_temperature=float(payload.get("pilot_candidate", {}).get("temperature", 1.0)),
        risk_hidden_width=int(payload.get("risk_adjustment", {}).get("hidden_width", 64)),
        risk_initial_bias=float(payload.get("risk_adjustment", {}).get("initial_output_bias", -6.0)),
        decoder_parameters=dict(source.core.decoder_parameters),
        task_mean=source.core.task_mean.detach().clone(),
        task_scale=source.core.task_scale.detach().clone(),
        physical_feature_mean=source.core.physical_feature_mean.detach().clone(),
        physical_feature_scale=source.core.physical_feature_scale.detach().clone(),
        previous_chp_mean=source.previous_chp_mean.detach().clone(),
        previous_chp_scale=source.previous_chp_scale.detach().clone(),
        dropout=0.0,
    )
    result = model.load_state_dict(source.state_dict(), strict=False)
    missing = tuple(result.missing_keys)
    unexpected = tuple(result.unexpected_keys)
    if unexpected or any(not name.startswith("risk_head.") for name in missing):
        raise ValueError(f"v4.6 parent import mismatch: missing={missing}, unexpected={unexpected}")
    return model


def _stage_batches(loaders: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if not isinstance(loaders, Mapping) or "train" not in loaders or "early_stop" not in loaders:
        raise ValueError("v4.6 J requires train and early_stop loaders")
    train = list(loaders["train"])
    early_stop = list(loaders["early_stop"])
    if not train or not early_stop:
        raise ValueError("v4.6 J requires non-empty train and early_stop loaders")
    return train, early_stop


def _budget_epochs(budget: StageBudgetV45) -> tuple[int, int, int]:
    return int(budget.max_epochs), int(budget.minimum_epochs), int(budget.patience)


def _normalization(parent: nn.Module, batches: Sequence[Mapping[str, Any]], prior: Any, parameters: Mapping[str, Any]) -> JointNormalizationV45:
    forecasts: list[Tensor] = []
    decisions: list[Tensor] = []
    imitations: list[Tensor] = []
    parent.eval()
    with torch.no_grad():
        for batch in batches:
            output = parent(**_inputs(batch))
            from .formal_v4_4_loss import CurriculumWeightsV44, forecast_loss_v44
            forecast = forecast_loss_v44(output.as_v44_forecast_output(), batch, prior, CurriculumWeightsV44(1.0, 1.0, 1.0, 1.0, 0.25)).total
            decision, imitation, _ = _decision_and_imitation(parent, output, batch, c_ref=1.0, parameters=parameters, detached=False)
            forecasts.append(forecast); decisions.append(decision); imitations.append(imitation)
    return JointNormalizationV45.from_values(torch.stack(forecasts), torch.stack(imitations), torch.stack(decisions))


def _run_branch(
    parent: nn.Module,
    train_batches: Sequence[Mapping[str, Any]],
    validation_batches: Sequence[Mapping[str, Any]],
    budget: StageBudgetV45,
    *,
    prior: ThermalPriorReceiptV44,
    parameters: Mapping[str, Any],
    contract: FormalV46Contract | Mapping[str, Any],
    normalization: JointNormalizationV45,
    risk_multiplier: float,
    mode: str,
    seed: int,
) -> StageReceiptV46:
    seed_everything(seed)
    model = deepcopy(parent)
    groups = model.v46_parameter_groups()
    forecast = tuple((*groups.get("base", ()), *groups.get("gate", ()), *groups.get("magnitude", ())))
    risk = tuple(groups.get("risk", ()))
    scheduler = tuple(groups.get("scheduler", ()))
    for parameter in forecast:
        parameter.requires_grad_(mode == "joint")
    for parameter in (*risk, *scheduler):
        parameter.requires_grad_(True)
    active = tuple(parameter for parameter in (*forecast, *risk, *scheduler) if parameter.requires_grad)
    if not active:
        raise ValueError("v4.6 J branch has no trainable parameters")
    optimizer = build_j_optimizer_v46(model, contract, mode=mode)
    group_steps = {str(group["name"]): {"steps": 0} for group in optimizer.param_groups}
    parent_hash = sha256_state_dict(parent)
    max_epochs, minimum_epochs, patience = _budget_epochs(budget)
    best_state: dict[str, Tensor] | None = None
    best_metric = float("inf")
    best_epoch = -1
    stale = 0
    loss_history: list[float] = []
    validation_history: list[Mapping[str, float]] = []
    last_norms = {"decision_to_base": 0.0, "decision_to_gate": 0.0, "decision_to_magnitude": 0.0, "decision_to_risk": 0.0, "decision_to_scheduler": 0.0}
    for epoch in range(max_epochs):
        model.train()
        curriculum = curriculum_weights_v45(epoch, budget.ramp_epochs)
        epoch_losses: list[float] = []
        for batch in train_batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(detach_forecast_for_dispatch=(mode == "decoupled"), **_inputs(batch))
            decision, imitation, _ = _decision_and_imitation(model, output, batch, c_ref=1.0, parameters=parameters, detached=(mode == "decoupled"))
            with torch.no_grad():
                parent_output = parent(**_inputs(batch)).as_v44_forecast_output()
            terms = joint_loss_v46(output, parent_output, batch, prior, normalization, curriculum, risk_multiplier, decision)
            decision_norms = named_autograd_norms(terms.decision, {"base": groups.get("base", ()), "gate": groups.get("gate", ()), "magnitude": groups.get("magnitude", ()), "risk": risk, "scheduler": scheduler}, retain_graph=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(active, float(budget.max_grad_norm))
            optimizer.step()
            for group in optimizer.param_groups:
                group_steps[str(group["name"])] = {"steps": group_steps[str(group["name"])] ["steps"] + 1}
            epoch_losses.append(float(terms.total.detach()))
            last_norms = {f"decision_to_{name}": float(decision_norms.get(name, 0.0)) for name in ("base", "gate", "magnitude", "risk", "scheduler")}
        loss_history.append(float(np.mean(epoch_losses)))
        model.eval()
        metrics: list[float] = []
        with torch.no_grad():
            for batch in validation_batches:
                output = model(detach_forecast_for_dispatch=(mode == "decoupled"), **_inputs(batch))
                decision, _imitation, _ = _decision_and_imitation(model, output, batch, c_ref=1.0, parameters=parameters, detached=(mode == "decoupled"))
                metrics.append(float(decision.detach()))
        metric = float(np.mean(metrics))
        details = {"metric": metric, "epoch": float(epoch), "eligible": float(np.isfinite(metric))}
        validation_history.append(details)
        if np.isfinite(metric) and metric < best_metric:
            best_metric = metric; best_epoch = epoch; best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        if epoch + 1 >= minimum_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"v4.6 J {mode} has no finite early-stop checkpoint")
    terminal_hash = sha256_state_dict(model)
    model.load_state_dict(best_state, strict=True)
    return StageReceiptV46(
        stage="J", mode=mode, parent_sha256=parent_hash,
        final_sha256=sha256_state_dict(model), best_sha256=sha256_state_dict(model), terminal_sha256=terminal_hash,
        epochs=len(loss_history), best_epoch=best_epoch, optimizer_steps=sum(item["steps"] for item in group_steps.values()),
        forecast_optimizer_steps=sum(group_steps.get(name, {}).get("steps", 0) for name in ("base", "head")) if mode == "joint" else 0,
        loss_history=tuple(loss_history), validation_history=tuple(validation_history), stopping_reason="early_stopped" if len(loss_history) < max_epochs else "budget_exhausted",
        selection_metric=best_metric, gradient_norms=last_norms, optimizer_groups=group_steps, model=model,
    )


def run_stage_j_pair_v46(
    stage_s: StageReceiptV45 | Any,
    loaders: Mapping[str, Sequence[Mapping[str, Any]]],
    budget: StageBudgetV45,
    *,
    prior: ThermalPriorReceiptV44,
    parameters: Mapping[str, Any],
    contract: FormalV46Contract | Mapping[str, Any],
    risk_cap: RiskCapReceiptV46,
    risk_multiplier: float,
    seed: int,
) -> JointPairReceiptV46:
    train, early_stop = _stage_batches(loaders)
    parent = initialize_v46_parent(stage_s, risk_cap, contract)
    normalization = _normalization(parent, train, prior, parameters)
    joint = _run_branch(parent, train, early_stop, budget, prior=prior, parameters=parameters, contract=contract, normalization=normalization, risk_multiplier=risk_multiplier, mode="joint", seed=seed)
    decoupled = _run_branch(parent, train, early_stop, budget, prior=prior, parameters=parameters, contract=contract, normalization=normalization, risk_multiplier=risk_multiplier, mode="decoupled", seed=seed)
    return JointPairReceiptV46(joint=joint, decoupled=decoupled)


def run_zero_risk_control_v46(
    stage_s: StageReceiptV45 | Any,
    loaders: Mapping[str, Sequence[Mapping[str, Any]]],
    budget: StageBudgetV45,
    *,
    prior: ThermalPriorReceiptV44,
    parameters: Mapping[str, Any],
    contract: FormalV46Contract | Mapping[str, Any],
    risk_cap: RiskCapReceiptV46,
    seed: int,
) -> ZeroRiskReceiptV46:
    parent = initialize_v46_parent(stage_s, risk_cap, contract)
    train, _early = _stage_batches(loaders)
    batch = train[0]
    output = parent(disable_risk_adjustment=True, **_inputs(batch))
    adjustment = output.risk_adjustment.detach().cpu().numpy()
    nominal_hash = _sha256_array(output.forecast_nominal_physical.detach().cpu().numpy())
    scheduler_hash = _sha256_array(output.scheduler_demand.detach().cpu().numpy())
    return ZeroRiskReceiptV46(False, adjustment, scheduler_hash, nominal_hash, sha256_state_dict(parent), parent)


def run_v46_calibration(
    stage_s: StageReceiptV45 | Any,
    loaders: Mapping[str, Sequence[Mapping[str, Any]]],
    budget: StageBudgetV45,
    *,
    prior: ThermalPriorReceiptV44,
    parameters: Mapping[str, Any],
    contract: FormalV46Contract | Mapping[str, Any],
    risk_cap: RiskCapReceiptV46,
    seed: int,
) -> CalibrationReceiptV46:
    multipliers = (0.5, 1.0, 2.0)
    candidates: dict[float, CalibratedCandidateV46] = {}
    for multiplier in multipliers:
        pair = run_stage_j_pair_v46(stage_s, loaders, budget, prior=prior, parameters=parameters, contract=contract, risk_cap=risk_cap, risk_multiplier=multiplier, seed=seed)
        validation = CandidateValidationV46(
            eligible=bool(np.isfinite(pair.joint.selection_metric) and np.isfinite(pair.decoupled.selection_metric)),
            decision_objective=float(pair.joint.selection_metric),
            nominal_metrics={"joint_decision": float(pair.joint.selection_metric), "decoupled_decision": float(pair.decoupled.selection_metric)},
            failure_names=(),
        )
        candidates[multiplier] = CalibratedCandidateV46(multiplier, pair, validation)
    eligible = [candidate for candidate in candidates.values() if candidate.validation.eligible]
    if not eligible:
        raise ValueError("formal-v4.6 calibration has no eligible checkpoint")
    selected = min(eligible, key=lambda item: item.validation.decision_objective)
    return CalibrationReceiptV46(candidates, selected.risk_multiplier, selected, "early_stop", False)


__all__ = [
    "CalibratedCandidateV46", "CalibrationReceiptV46", "CandidateValidationV46", "JointPairReceiptV46", "StageReceiptV46", "ZeroRiskReceiptV46",
    "initialize_v46_parent", "run_stage_j_pair_v46", "run_v46_calibration", "run_zero_risk_control_v46",
]
