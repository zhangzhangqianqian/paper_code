"""Full-chronology 2019 rolling evaluation for formal-v4.4 Pilot models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .formal_v4_2_data import NormalizationReceiptV42, fit_train_normalization
from .formal_v4_2_rollout import calculate_all_residual_families, settle_and_advance_v42
from .formal_v4_4_pilot_materializer import MaterializedV44PilotData, V44WindowCollection
from .formal_v4_4_regime import derive_last_observed_regime, derive_thermal_regimes
from .formal_v4_state import FormalV4ClosedLoopState


@dataclass(frozen=True)
class RolloutResultV44:
    method_id: str
    prediction: np.ndarray
    target: np.ndarray
    probability: np.ndarray
    prior_probability: np.ndarray
    regimes: np.ndarray
    planned_dispatch: np.ndarray
    settled_dispatch: np.ndarray
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
        prediction = np.asarray(self.prediction, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        probability = np.asarray(self.probability, dtype=np.float64)
        prior = np.asarray(self.prior_probability, dtype=np.float64)
        regimes = np.asarray(self.regimes, dtype=np.int64)
        n = prediction.shape[0] if prediction.ndim == 3 else -1
        if prediction.shape != (n, 4, 4) or target.shape != prediction.shape:
            raise ValueError("rollout prediction/target must have shape [N,4,4]")
        if probability.shape != (n, 4, 3) or prior.shape != probability.shape or regimes.shape != (n, 4):
            raise ValueError("rollout regime arrays have invalid shapes")
        for name, value, shape in (
            ("planned_dispatch", self.planned_dispatch, (n, 4, 21)),
            ("settled_dispatch", self.settled_dispatch, (n, 21)),
            ("shortage", self.shortage, (n, 3)),
            ("physical_residual", self.physical_residual, (n, 8)),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} has invalid rollout shape or non-finite values")
            object.__setattr__(self, name, array)
        for name, value in (("prediction", prediction), ("target", target), ("probability", probability), ("prior_probability", prior)):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        times = np.asarray(self.times, dtype="datetime64[ns]")
        hashes = np.asarray(self.state_hashes, dtype=str)
        if times.shape != (n,) or hashes.shape != (n,):
            raise ValueError("rollout times and state hashes must have shape [N]")
        if n > 1 and not np.all(np.diff(times) > np.timedelta64(0, "s")):
            raise ValueError("2019 rollout must be chronological")
        object.__setattr__(self, "times", times); object.__setattr__(self, "state_hashes", hashes)
        for name in ("operating_cost", "physical_carbon", "penalized_objective"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != (n,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must have shape [N] and be finite")
            object.__setattr__(self, name, array)
        for name, shape in (
            ("initial_soc", (n, 1)), ("previous_chp", (n, 1)),
            ("realized_demand", (n, 3)), ("realized_renewables", (n, 2)),
            ("realized_prices", (n, 3)),
        ):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must have shape {shape} and be finite")
            object.__setattr__(self, name, array)
        object.__setattr__(self, "regimes", regimes)

    @property
    def next_initial_soc(self) -> np.ndarray:
        """Raw carried SOC column for the next chronological origin."""

        values = np.full((len(self.settled_dispatch), 1), np.nan, dtype=np.float64)
        if len(values) > 1:
            values[1:, 0] = self.settled_dispatch[:-1, 16]
        return values


def _model_output(model: Any, inputs: Mapping[str, torch.Tensor]) -> Any:
    with torch.no_grad():
        return model(**inputs)


def _tensor_output(output: Any, name: str) -> np.ndarray:
    value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
    if value is None:
        raise ValueError(f"rollout model output has no {name}")
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def _normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(np.float32)


def _prior_probability(model: Any, last_regime: int) -> np.ndarray:
    head = getattr(model, "thermal_head", None)
    if head is None or not hasattr(head, "transition_log_prior"):
        return np.full((4, 3), 1.0 / 3.0, dtype=np.float64)
    logits = head.transition_log_prior.detach().cpu().numpy()[:, int(last_regime), :]
    logits = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(logits); probability /= probability.sum(axis=-1, keepdims=True)
    return probability


def _initial_state(selection: V44WindowCollection) -> FormalV4ClosedLoopState:
    return FormalV4ClosedLoopState(
        torch.as_tensor(selection.load_history[0:1, -24:], dtype=torch.float32),
        torch.as_tensor(selection.exog_history[0:1, -24:], dtype=torch.float32),
        torch.as_tensor(selection.device_history[0:1, -24:], dtype=torch.float32),
        torch.as_tensor(selection.activity_history[0:1, -24:], dtype=torch.float32),
        torch.as_tensor(selection.initial_soc[0:1], dtype=torch.float32),
        torch.as_tensor(selection.previous_chp[0:1], dtype=torch.float32),
        selection.timestamps[0], str(selection.trajectory_ids[0]),
    )


def evaluate_full_2019_rollout_v44(
    *, model: Any, selection: V44WindowCollection, normalization: NormalizationReceiptV42 | None,
    parameters: Mapping[str, Any], method_id: str = "RSC-PF",
) -> RolloutResultV44:
    """Run every valid 2019 origin, settling only the first planned hour."""

    if len(selection) == 0:
        raise ValueError("2019 selection chronology is empty")
    years = set(int(value) for value in selection.timestamps.astype("datetime64[Y]").astype(int) + 1970)
    if years != {2019}:
        raise ValueError("rollout selection must contain exactly 2019")
    if normalization is None:
        raise ValueError("rollout requires train-only normalization")
    state = _initial_state(selection)
    predictions: list[np.ndarray] = []; targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []; priors: list[np.ndarray] = []; regimes: list[np.ndarray] = []
    plans: list[np.ndarray] = []; settled: list[np.ndarray] = []; shortages: list[np.ndarray] = []
    residuals: list[np.ndarray] = []; operating: list[float] = []; carbon: list[float] = []; objective: list[float] = []
    initial_socs: list[np.ndarray] = []; previous_chps: list[np.ndarray] = []
    realized_demands: list[np.ndarray] = []; realized_renewables: list[np.ndarray] = []; realized_prices: list[np.ndarray] = []
    times: list[np.datetime64] = []; state_hashes: list[str] = []
    previous_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    for index in range(len(selection)):
        target = np.asarray(selection.forecast_target[index], dtype=np.float64)
        renew = np.asarray(selection.renewable_realized[index], dtype=np.float64)
        raw_renew_forecast = np.asarray(selection.renewable_forecast[index], dtype=np.float64)
        prices = np.asarray(selection.prices_and_weights[index], dtype=np.float64)
        initial_socs.append(state.initial_soc.detach().cpu().numpy().reshape(1).copy())
        previous_chps.append(state.previous_chp.detach().cpu().numpy().reshape(1).copy())
        realized_demands.append(target[0, :3].copy()); realized_renewables.append(renew[0].copy()); realized_prices.append(prices[0].copy())
        scheduler = np.concatenate((raw_renew_forecast, prices, np.repeat(selection.initial_soc[index:index + 1], 4, axis=0)), axis=-1)
        inputs = {
            "load_history": torch.as_tensor(_normalize(state.load_history.numpy(), normalization.field_mean["load"], normalization.field_scale["load"])),
            "exog_history": torch.as_tensor(_normalize(state.exog_history.numpy(), normalization.field_mean["exog"], normalization.field_scale["exog"])),
            "device_history": torch.as_tensor(_normalize(state.device_history.numpy(), normalization.field_mean["device"], normalization.field_scale["device"])),
            "activity_history": state.activity_history,
            "scheduler_context": torch.as_tensor(scheduler[None, ...], dtype=torch.float32),
            "previous_chp": state.previous_chp,
            "last_thermal_regime": torch.as_tensor(derive_last_observed_regime(state.load_history.numpy()), dtype=torch.long),
        }
        output = _model_output(model, inputs)
        prediction = _tensor_output(output, "forecast_physical")[0]
        probability = _tensor_output(output, "regime_probabilities")[0]
        plan = _tensor_output(output, "dispatch")[0]
        if prediction.shape != (4, 4) or probability.shape != (4, 3) or plan.shape != (4, 21):
            raise ValueError("rollout model output shapes are invalid")
        last_regime = int(derive_last_observed_regime(state.load_history.numpy())[0])
        prior = _prior_probability(model, last_regime)
        realized = {"demand": target[0, :3], "renewable": renew[0], "realized_load": target[0], "realized_exog": selection.exog_history[min(index + 1, len(selection) - 1), -1]}
        result = settle_and_advance_v42(state, plan, realized, parameters)
        residual = result.residuals
        residual_vector = np.asarray((
            np.abs(residual.balance).max(initial=0.0), np.abs(residual.capacity).max(initial=0.0), np.abs(residual.conversion).max(initial=0.0),
            np.abs(residual.soc).max(initial=0.0), np.abs(residual.ramp).max(initial=0.0), np.abs(residual.exclusivity).max(initial=0.0),
            np.abs(residual.renewable_accounting).max(initial=0.0), residual.finite.max(initial=0.0),
        ), dtype=np.float64)
        predictions.append(prediction); targets.append(target); probabilities.append(probability); priors.append(prior)
        regimes.append(derive_thermal_regimes(target[None, ...])[0]); plans.append(plan); settled.append(result.settled)
        shortages.append(result.shortage); residuals.append(residual_vector); operating.append(result.operating_cost); carbon.append(result.physical_carbon); objective.append(result.penalized_objective)
        times.append(selection.timestamps[index]); state_hashes.append(state.state_hash)
        state = result.next_state
    if previous_training and hasattr(model, "train"):
        model.train()
    return RolloutResultV44(
        method_id=str(method_id), prediction=np.stack(predictions), target=np.stack(targets), probability=np.stack(probabilities), prior_probability=np.stack(priors), regimes=np.stack(regimes),
        planned_dispatch=np.stack(plans), settled_dispatch=np.stack(settled), shortage=np.stack(shortages), physical_residual=np.stack(residuals),
        operating_cost=np.asarray(operating), physical_carbon=np.asarray(carbon), penalized_objective=np.asarray(objective), times=np.asarray(times, dtype="datetime64[ns]"), state_hashes=np.asarray(state_hashes, dtype=str),
        initial_soc=np.stack(initial_socs), previous_chp=np.stack(previous_chps), realized_demand=np.stack(realized_demands), realized_renewables=np.stack(realized_renewables), realized_prices=np.stack(realized_prices),
    )


def rollout_v44_2019(
    *, model: Any, materialized: MaterializedV44PilotData, indices: np.ndarray,
    normalization: NormalizationReceiptV42, parameters: Mapping[str, Any],
    artifact_root: str | Path, method_id: str,
) -> RolloutResultV44:
    """Evaluate a selected chronological 2019 index set and persist its arrays."""

    selected = materialized.selection_full
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(selected)):
        raise ValueError("rollout indices are invalid")
    times = selected.timestamps[indices]
    if len(times) > 1 and not np.all(np.diff(times) > np.timedelta64(0, "s")):
        raise ValueError("rollout indices must be chronological")
    fields = (
        "load_history", "exog_history", "renewable_history", "device_history", "activity_history",
        "forecast_target", "rigid_demand", "renewable_forecast", "renewable_realized", "prices_and_weights",
        "initial_soc", "previous_chp", "target_times", "trajectory_ids", "state_hashes",
    )
    values = {name: getattr(selected.split, name)[indices] for name in fields}
    subset = V44WindowCollection(type(selected.split)(**values, split="selection", history_source=selected.history_source), "selection_full")
    result = evaluate_full_2019_rollout_v44(model=model, selection=subset, normalization=normalization, parameters=parameters, method_id=method_id)
    from .formal_v4_4_artifacts import write_json_once, write_npz_once
    root = Path(artifact_root) / "rollout"; root.mkdir(parents=True, exist_ok=True)
    arrays = {
        "prediction": result.prediction, "target": result.target, "probability": result.probability,
        "prior_probability": result.prior_probability, "regimes": result.regimes, "planned_dispatch": result.planned_dispatch,
        "settled_dispatch": result.settled_dispatch, "shortage": result.shortage, "physical_residual": result.physical_residual,
        "operating_cost": result.operating_cost, "physical_carbon": result.physical_carbon, "penalized_objective": result.penalized_objective,
        "initial_soc": result.initial_soc, "previous_chp": result.previous_chp, "realized_demand": result.realized_demand,
        "realized_renewables": result.realized_renewables, "realized_prices": result.realized_prices,
        "times": result.times.astype("datetime64[ns]").astype(np.int64), "state_hashes": result.state_hashes,
    }
    write_npz_once(root / f"{method_id}.npz", arrays)
    write_json_once(root / f"{method_id}.json", {"schema": "formal-v4.4-rollout-v1", "method_id": str(method_id), "rows": int(len(result.times)), "times": [str(value) for value in result.times]})
    return result


__all__ = ["RolloutResultV44", "evaluate_full_2019_rollout_v44", "rollout_v44_2019"]
