"""Risk-aware closed-loop rolling rollout for the formal-v4.6 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .formal_v4_2_data import NormalizationReceiptV42
from .formal_v4_2_rollout import settle_and_advance_v42
from .formal_v4_4_pilot_materializer import V44WindowCollection, _WINDOW_FIELDS
from .formal_v4_4_regime import derive_last_observed_regime, derive_thermal_regimes
from .formal_v4_4_rollout import _initial_state, _normalize, _prior_probability, _tensor_output, _model_output
from .formal_v4_5_pilot_data import MaterializedPilotV45
from .formal_v4_5_training import StageReceiptV45
from .formal_v4_6_training import CalibratedCandidateV46
from .formal_v4_state import FormalV4ClosedLoopState


@dataclass(frozen=True)
class P1EarlyStopEvidenceV46:
    prediction: np.ndarray
    target: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class FrozenCandidateV46:
    risk_multiplier: float
    joint_model: Any
    decoupled_model: Any
    parent_sha256: str
    validation: Any


@dataclass(frozen=True)
class RolloutResultV46:
    method_id: str
    forecast_nominal: np.ndarray
    risk_adjustment: np.ndarray
    risk_cap: np.ndarray
    scheduler_demand: np.ndarray
    controls: np.ndarray
    planned_dispatch: np.ndarray
    settled_dispatch: np.ndarray
    target: np.ndarray
    probability: np.ndarray
    prior_probability: np.ndarray
    regimes: np.ndarray
    shortage: np.ndarray
    physical_residual: np.ndarray
    operating_cost: np.ndarray
    physical_carbon: np.ndarray
    penalized_objective: np.ndarray
    initial_soc: np.ndarray
    previous_chp: np.ndarray
    realized_demand: np.ndarray
    realized_renewables: np.ndarray
    realized_prices: np.ndarray
    times: np.ndarray
    state_hashes: np.ndarray

    def __post_init__(self) -> None:
        nominal = np.asarray(self.forecast_nominal, dtype=np.float64)
        n = nominal.shape[0] if nominal.ndim == 3 else -1
        if nominal.shape != (n, 4, 4) or np.asarray(self.target).shape != nominal.shape:
            raise ValueError("forecast_nominal and target must have shape [N,4,4]")
        for name, shape in (
            ("risk_adjustment", (n, 4, 3)), ("risk_cap", (n, 4, 3)), ("scheduler_demand", (n, 4, 4)),
            ("controls", (n, 4, 15)), ("planned_dispatch", (n, 4, 21)), ("settled_dispatch", (n, 21)),
            ("shortage", (n, 3)), ("physical_residual", (n, 8)), ("initial_soc", (n, 1)), ("previous_chp", (n, 1)),
            ("realized_demand", (n, 3)), ("realized_renewables", (n, 2)), ("realized_prices", (n, 3)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} has invalid shape or non-finite values")
            object.__setattr__(self, name, value)
        for name in ("forecast_nominal", "target", "probability", "prior_probability", "operating_cost", "physical_carbon", "penalized_objective"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if np.any(self.risk_adjustment < -1.0e-8) or np.any(self.risk_adjustment > self.risk_cap + 1.0e-6):
            raise ValueError("risk adjustment exceeds cap")
        times = np.asarray(self.times, dtype="datetime64[ns]")
        hashes = np.asarray(self.state_hashes, dtype=str)
        if times.shape != (n,) or hashes.shape != (n,):
            raise ValueError("rollout times and state hashes must have shape [N]")
        if n and set(times.astype("datetime64[Y]").astype(int) + 1970) != {2019}:
            raise ValueError("v4.6 rollout must contain 2019 origins only")
        object.__setattr__(self, "times", times); object.__setattr__(self, "state_hashes", hashes)
        object.__setattr__(self, "probability", np.asarray(self.probability, dtype=np.float64))
        object.__setattr__(self, "prior_probability", np.asarray(self.prior_probability, dtype=np.float64))
        object.__setattr__(self, "regimes", np.asarray(self.regimes, dtype=np.int64))
        for name in ("operating_cost", "physical_carbon", "penalized_objective"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape [N]")
            object.__setattr__(self, name, value)


def collect_p1_early_stop_evidence_v46(model: Any, early_stop_batches: Sequence[Mapping[str, Any]], early_stop_timestamps: np.ndarray) -> P1EarlyStopEvidenceV46:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in early_stop_batches:
        inputs = {name: (value.float() if name != "last_thermal_regime" else value.long()) for name, value in batch.items() if name in {"load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp", "last_thermal_regime"}}
        with torch.no_grad():
            output = model(**inputs)
        predictions.append(_tensor_output(output, "forecast_physical"))
        targets.append(np.asarray(batch["target_physical"], dtype=np.float64))
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    timestamps = np.asarray(early_stop_timestamps, dtype="datetime64[ns]")
    if timestamps.shape != (prediction.shape[0],):
        raise ValueError("early-stop timestamps must align with evidence rows")
    return P1EarlyStopEvidenceV46(prediction, target, timestamps)


def freeze_v46_candidate(candidate: CalibratedCandidateV46) -> FrozenCandidateV46:
    if candidate.pair.joint.model is None or candidate.pair.decoupled.model is None:
        raise ValueError("calibrated candidate must contain both models")
    if candidate.pair.joint.parent_sha256 != candidate.pair.decoupled.parent_sha256:
        raise ValueError("Joint and Fair Decoupled parents differ")
    return FrozenCandidateV46(
        risk_multiplier=float(candidate.risk_multiplier),
        joint_model=candidate.pair.joint.model,
        decoupled_model=candidate.pair.decoupled.model,
        parent_sha256=candidate.pair.joint.parent_sha256,
        validation=candidate.validation,
    )


def _subset_selection(selection: V44WindowCollection, indices: np.ndarray) -> V44WindowCollection:
    values = {name: getattr(selection.split, name)[indices] for name in _WINDOW_FIELDS}
    return V44WindowCollection(type(selection.split)(**values, split="selection", history_source=selection.history_source), "selection_full")


def _evaluate_v46(*, model: Any, selection: V44WindowCollection, normalization: NormalizationReceiptV42, parameters: Mapping[str, Any], method_id: str) -> RolloutResultV46:
    if len(selection) == 0:
        raise ValueError("2019 selection chronology is empty")
    years = set(int(value) for value in selection.timestamps.astype("datetime64[Y]").astype(int) + 1970)
    if years != {2019}:
        raise ValueError("selection must contain exactly 2019")
    state = _initial_state(selection)
    nominal: list[np.ndarray] = []; risk: list[np.ndarray] = []; caps: list[np.ndarray] = []; demand: list[np.ndarray] = []; controls: list[np.ndarray] = []; plans: list[np.ndarray] = []; settled: list[np.ndarray] = []
    targets: list[np.ndarray] = []; probabilities: list[np.ndarray] = []; priors: list[np.ndarray] = []; regimes: list[np.ndarray] = []; shortages: list[np.ndarray] = []; residuals: list[np.ndarray] = []
    costs: list[float] = []; carbons: list[float] = []; objectives: list[float] = []; socs: list[np.ndarray] = []; chps: list[np.ndarray] = []; rdemand: list[np.ndarray] = []; rrenew: list[np.ndarray] = []; prices: list[np.ndarray] = []; times: list[np.datetime64] = []; hashes: list[str] = []
    was_training = bool(getattr(model, "training", False)); model.eval()
    for index in range(len(selection)):
        target = np.asarray(selection.forecast_target[index], dtype=np.float64)
        renew = np.asarray(selection.renewable_realized[index], dtype=np.float64)
        prices_row = np.asarray(selection.prices_and_weights[index], dtype=np.float64)
        socs.append(state.initial_soc.detach().cpu().numpy().reshape(1).copy()); chps.append(state.previous_chp.detach().cpu().numpy().reshape(1).copy()); rdemand.append(target[0, :3].copy()); rrenew.append(renew[0].copy()); prices.append(prices_row[0].copy())
        scheduler = np.concatenate((np.asarray(selection.renewable_forecast[index], dtype=np.float64), prices_row, np.repeat(selection.initial_soc[index:index + 1], 4, axis=0)), axis=-1)
        inputs = {
            "load_history": torch.as_tensor(_normalize(state.load_history.numpy(), normalization.field_mean["load"], normalization.field_scale["load"])),
            "exog_history": torch.as_tensor(_normalize(state.exog_history.numpy(), normalization.field_mean["exog"], normalization.field_scale["exog"])),
            "device_history": torch.as_tensor(_normalize(state.device_history.numpy(), normalization.field_mean["device"], normalization.field_scale["device"])),
            "activity_history": state.activity_history,
            "scheduler_context": torch.as_tensor(scheduler[None, ...], dtype=torch.float32),
            "previous_chp": state.previous_chp,
            "last_thermal_regime": torch.as_tensor(derive_last_observed_regime(state.load_history.numpy()), dtype=torch.long),
        }
        with torch.no_grad():
            output = model(**inputs)
        nominal_row = _tensor_output(output, "forecast_nominal_physical")[0]; risk_row = _tensor_output(output, "risk_adjustment")[0]; cap_row = _tensor_output(output, "risk_cap")[0]; demand_row = _tensor_output(output, "scheduler_demand")[0]; control_row = _tensor_output(output, "controls")[0]; plan_row = _tensor_output(output, "dispatch")[0]; probability_row = _tensor_output(output, "regime_probabilities")[0]
        last_regime = int(derive_last_observed_regime(state.load_history.numpy())[0]); prior_row = _prior_probability(model, last_regime)
        realized = {"demand": target[0, :3], "renewable": renew[0], "realized_load": target[0], "realized_exog": selection.exog_history[min(index + 1, len(selection) - 1), -1]}
        outcome = settle_and_advance_v42(state, plan_row, realized, parameters)
        residual = outcome.residuals
        residual_row = np.asarray((np.abs(residual.balance).max(initial=0.0), np.abs(residual.capacity).max(initial=0.0), np.abs(residual.conversion).max(initial=0.0), np.abs(residual.soc).max(initial=0.0), np.abs(residual.ramp).max(initial=0.0), np.abs(residual.exclusivity).max(initial=0.0), np.abs(residual.renewable_accounting).max(initial=0.0), residual.finite.max(initial=0.0)), dtype=np.float64)
        nominal.append(nominal_row); risk.append(risk_row); caps.append(cap_row); demand.append(demand_row); controls.append(control_row); plans.append(plan_row); settled.append(outcome.settled); targets.append(target); probabilities.append(probability_row); priors.append(prior_row); regimes.append(derive_thermal_regimes(target[None, ...])[0]); shortages.append(outcome.shortage); residuals.append(residual_row); costs.append(outcome.operating_cost); carbons.append(outcome.physical_carbon); objectives.append(outcome.penalized_objective); times.append(selection.timestamps[index]); hashes.append(state.state_hash); state = outcome.next_state
    if was_training: model.train()
    return RolloutResultV46(
        method_id=str(method_id), forecast_nominal=np.stack(nominal), risk_adjustment=np.stack(risk), risk_cap=np.stack(caps), scheduler_demand=np.stack(demand), controls=np.stack(controls), planned_dispatch=np.stack(plans), settled_dispatch=np.stack(settled), target=np.stack(targets), probability=np.stack(probabilities), prior_probability=np.stack(priors), regimes=np.stack(regimes), shortage=np.stack(shortages), physical_residual=np.stack(residuals), operating_cost=np.asarray(costs), physical_carbon=np.asarray(carbons), penalized_objective=np.asarray(objectives), initial_soc=np.stack(socs), previous_chp=np.stack(chps), realized_demand=np.stack(rdemand), realized_renewables=np.stack(rrenew), realized_prices=np.stack(prices), times=np.asarray(times, dtype="datetime64[ns]"), state_hashes=np.asarray(hashes, dtype=str),
    )


def load_selection_view_v46(materialized_root: str | Path, train_data: str | Path, benchmark: str | Path, capacity_receipt: str | Path) -> MaterializedPilotV45:
    from .formal_v4_5_pilot_data import load_v45_pilot_cache
    return load_v45_pilot_cache(materialized_root=materialized_root, train_data=train_data, benchmark=benchmark, capacity_receipt=capacity_receipt)


def rollout_v46_2019(*, model: Any, materialized: MaterializedPilotV45, indices: np.ndarray, normalization: NormalizationReceiptV42, parameters: Mapping[str, Any], artifact_root: str | Path, method_id: str) -> RolloutResultV46:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(materialized.selection_full)):
        raise ValueError("rollout indices are invalid")
    result = _evaluate_v46(model=model, selection=_subset_selection(materialized.selection_full, indices), normalization=normalization, parameters=parameters, method_id=method_id)
    root = Path(artifact_root) / "rollout"; root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / f"{method_id}.npz", **{name: getattr(result, name) for name in ("forecast_nominal", "risk_adjustment", "risk_cap", "scheduler_demand", "controls", "planned_dispatch", "settled_dispatch", "target", "probability", "prior_probability", "regimes", "shortage", "physical_residual", "operating_cost", "physical_carbon", "penalized_objective", "initial_soc", "previous_chp", "realized_demand", "realized_renewables", "realized_prices", "times", "state_hashes")})
    return result


def run_matched_v46_rollouts(candidate: FrozenCandidateV46, materialized: MaterializedPilotV45, normalization: NormalizationReceiptV42, parameters: Mapping[str, Any], artifact_root: str | Path) -> Mapping[str, RolloutResultV46]:
    indices = np.arange(len(materialized.selection_full), dtype=np.int64)
    return {"rsc_pf_joint": rollout_v46_2019(model=candidate.joint_model, materialized=materialized, indices=indices, normalization=normalization, parameters=parameters, artifact_root=artifact_root, method_id="rsc_pf_joint"), "fair_decoupled": rollout_v46_2019(model=candidate.decoupled_model, materialized=materialized, indices=indices, normalization=normalization, parameters=parameters, artifact_root=artifact_root, method_id="fair_decoupled")}


__all__ = ["FrozenCandidateV46", "P1EarlyStopEvidenceV46", "RolloutResultV46", "collect_p1_early_stop_evidence_v46", "freeze_v46_candidate", "load_selection_view_v46", "rollout_v46_2019", "run_matched_v46_rollouts"]
