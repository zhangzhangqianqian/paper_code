"""External-baseline adapters for the formal v4.6 materialized windows.

The formal joint-data container stores 17 continuous device channels and six
binary activity channels.  The external baselines need the same causal
information, but their historical-device contract is intentionally kept here
so the core ``JointWindowSplit`` contract is not changed accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from ..scheduling.dispatch_schema import VARIABLES


TASK_COUNT = 4
EXOG_COUNT = 12
DEVICE_HISTORY_COUNT = 17
STATUS_COUNT = 6
HORIZON = 4
LOOKBACK = 24


def _array(value: Any, *, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    result = np.asarray(value)
    if shape is not None and result.shape[1:] != shape:
        raise ValueError(f"{name} must have trailing shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.astype(np.float32, copy=False)


@dataclass(frozen=True)
class ExternalV46Split:
    """Causal windows used only by the external comparison methods."""

    load_history: np.ndarray
    exog_history: np.ndarray
    device_history: np.ndarray
    device_status: np.ndarray
    scheduler_context: np.ndarray
    previous_chp: np.ndarray
    forecast_target: np.ndarray
    renewable_realized: np.ndarray
    target_times: np.ndarray
    split: Literal["train", "validation", "pilot"]
    teacher_dispatch: np.ndarray | None = None
    oracle_first_step_objective: np.ndarray | None = None
    source_path: str = ""
    trajectory_ids: np.ndarray | None = None
    source_state_hashes: np.ndarray | None = None

    def __post_init__(self) -> None:
        fields = {
            "load_history": (LOOKBACK, TASK_COUNT),
            "exog_history": (LOOKBACK, EXOG_COUNT),
            "device_history": (LOOKBACK, DEVICE_HISTORY_COUNT),
            "device_status": (LOOKBACK, STATUS_COUNT),
            "scheduler_context": (HORIZON, 6),
            "forecast_target": (HORIZON, TASK_COUNT),
            "renewable_realized": (HORIZON, 2),
        }
        converted: dict[str, np.ndarray] = {}
        for name, shape in fields.items():
            converted[name] = _array(getattr(self, name), name=name, shape=shape)
        n = int(converted["load_history"].shape[0])
        for name, value in converted.items():
            if int(value.shape[0]) != n:
                raise ValueError(f"{name} has a different sample count")
            object.__setattr__(self, name, value)
        previous = _array(self.previous_chp, name="previous_chp")
        if previous.shape != (n, 1):
            raise ValueError("previous_chp must have shape [N,1]")
        object.__setattr__(self, "previous_chp", previous)
        times = np.asarray(self.target_times, dtype="datetime64[ns]")
        if times.shape != (n,):
            raise ValueError("target_times must have shape [N]")
        object.__setattr__(self, "target_times", times)
        for name in ("trajectory_ids", "source_state_hashes"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=str)
                if array.shape != (n,) or not array.astype(str).tolist():
                    raise ValueError(f"{name} must have shape [N]")
                if any(not item for item in array.tolist()):
                    raise ValueError(f"{name} must contain non-empty strings")
                object.__setattr__(self, name, array)
        if self.split not in {"train", "validation", "pilot"}:
            raise ValueError("split must be train, validation, or pilot")
        if not np.isin(converted["device_status"], (0.0, 1.0)).all():
            raise ValueError("device_status must be binary")
        if not np.allclose(converted["scheduler_context"][:, :, 5], converted["scheduler_context"][:, :1, 5], atol=1e-6):
            raise ValueError("initial_soc must be repeated across the horizon")
        if self.teacher_dispatch is not None:
            teacher = _array(self.teacher_dispatch, name="teacher_dispatch", shape=(HORIZON, len(VARIABLES)))
            if teacher.shape[0] != n:
                raise ValueError("teacher_dispatch has a different sample count")
            object.__setattr__(self, "teacher_dispatch", teacher)
        if self.oracle_first_step_objective is not None:
            oracle = _array(self.oracle_first_step_objective, name="oracle_first_step_objective")
            if oracle.shape != (n,):
                raise ValueError("oracle_first_step_objective must have shape [N]")
            object.__setattr__(self, "oracle_first_step_objective", oracle)

    def __len__(self) -> int:
        return int(self.load_history.shape[0])

    def take(self, indices: np.ndarray) -> "ExternalV46Split":
        index = np.asarray(indices, dtype=np.int64)
        fields = {
            name: getattr(self, name)[index]
            for name in (
                "load_history", "exog_history", "device_history", "device_status",
                "scheduler_context", "previous_chp", "forecast_target",
                "renewable_realized", "target_times",
            )
        }
        if self.teacher_dispatch is not None:
            fields["teacher_dispatch"] = self.teacher_dispatch[index]
        if self.oracle_first_step_objective is not None:
            fields["oracle_first_step_objective"] = self.oracle_first_step_objective[index]
        if self.trajectory_ids is not None:
            fields["trajectory_ids"] = self.trajectory_ids[index]
        if self.source_state_hashes is not None:
            fields["source_state_hashes"] = self.source_state_hashes[index]
        return replace(self, **fields)


@dataclass(frozen=True)
class ExternalV46Normalization:
    """Affine statistics fitted on the training split only."""

    load_mean: np.ndarray
    load_scale: np.ndarray
    exog_mean: np.ndarray
    exog_scale: np.ndarray
    device_mean: np.ndarray
    device_scale: np.ndarray
    scheduler_mean: np.ndarray
    scheduler_scale: np.ndarray
    fitted_split: str = "train"

    @staticmethod
    def _fit(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(value.mean(axis=(0, 1)), dtype=np.float32)
        scale = np.asarray(value.std(axis=(0, 1)), dtype=np.float32)
        return mean, np.where(scale < 1e-6, 1.0, scale).astype(np.float32)

    @classmethod
    def fit(cls, split: ExternalV46Split) -> "ExternalV46Normalization":
        if split.split != "train" or len(split) == 0:
            raise ValueError("external normalization must be fitted on non-empty train data")
        return cls(
            *cls._fit(split.load_history), *cls._fit(split.exog_history),
            *cls._fit(split.device_history), *cls._fit(split.scheduler_context),
        )

    @staticmethod
    def _apply(value: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return ((value - mean.reshape((1, 1, -1))) / scale.reshape((1, 1, -1))).astype(np.float32)

    def transform(self, split: ExternalV46Split) -> ExternalV46Split:
        return replace(
            split,
            load_history=self._apply(split.load_history, self.load_mean, self.load_scale),
            exog_history=self._apply(split.exog_history, self.exog_mean, self.exog_scale),
            device_history=self._apply(split.device_history, self.device_mean, self.device_scale),
            forecast_target=self._apply(split.forecast_target, self.load_mean, self.load_scale),
            scheduler_context=self._apply(split.scheduler_context, self.scheduler_mean, self.scheduler_scale),
        )


def load_external_v46_split(path: str | Path, split_name: Literal["train", "validation", "pilot"]) -> ExternalV46Split:
    """Load a materialized formal-v4.6 NPZ without opening a test artifact."""

    source = Path(path)
    normalized = str(source).replace("\\", "/").lower()
    if any(token in normalized for token in ("/test/", "sealed_test", "2021_test", "test-set")):
        raise ValueError("external v4.6 adapter refuses sealed test paths")
    required = {
        "load_history", "exog_history", "device_history", "activity_history",
        "forecast_target", "renewable_forecast", "renewable_realized",
        "prices_and_weights", "initial_soc", "previous_chp", "target_times",
    }
    with np.load(source, allow_pickle=False) as payload:
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"v4.6 split is missing fields: {missing}")
        initial_soc = np.asarray(payload["initial_soc"], dtype=np.float32)
        renewable_forecast = np.asarray(payload["renewable_forecast"], dtype=np.float32)
        prices = np.asarray(payload["prices_and_weights"], dtype=np.float32)
        if initial_soc.shape != (len(payload["load_history"]), 1):
            raise ValueError("initial_soc must have shape [N,1]")
        context = np.concatenate((renewable_forecast, prices, np.repeat(initial_soc[:, None, :], HORIZON, axis=1)), axis=-1)
        return ExternalV46Split(
            load_history=payload["load_history"], exog_history=payload["exog_history"],
            device_history=payload["device_history"], device_status=payload["activity_history"],
            scheduler_context=context, previous_chp=payload["previous_chp"],
            forecast_target=payload["forecast_target"], renewable_realized=payload["renewable_realized"],
            target_times=payload["target_times"], split=split_name, source_path=str(source.resolve()),
            trajectory_ids=payload["trajectory_ids"] if "trajectory_ids" in payload.files else None,
            source_state_hashes=payload["state_hashes"] if "state_hashes" in payload.files else None,
        )


def _first_step_objective(dispatch: Mapping[str, np.ndarray], context: np.ndarray, parameters: Mapping[str, Any]) -> float:
    grid = float(dispatch["grid"][0])
    gas = float(dispatch["g_chp"][0] + dispatch["g_gb"][0])
    slack = float(dispatch["slack_e"][0] + dispatch["slack_c"][0] + dispatch["slack_h"][0])
    grid_factor = float(parameters.get("grid_emission_factor", 0.5))
    gas_factor = float(parameters.get("gas_emission_factor", 0.25))
    carbon = grid * grid_factor + gas * gas_factor
    return grid * float(context[0, 2]) + gas * float(context[0, 3]) + carbon * float(context[0, 4]) + 100.0 * slack


def bounded_previous_chp(value: float, parameters: Mapping[str, Any]) -> float:
    """Apply the formal decoder's tolerance-aware CHP state bound."""

    pmax = float(parameters["chp_electric_capacity"])
    value = float(value)
    if value < -1.0e-4 or value > pmax + 1.0e-3:
        raise ValueError("previous_chp is outside the formal CHP capacity")
    return float(np.clip(value, 0.0, pmax))


def build_external_v46_teacher(split: ExternalV46Split, parameters: Mapping[str, Any]) -> np.ndarray:
    """Create label-only same-information LP dispatch targets."""

    if split.split == "pilot":
        raise ValueError("pilot labels are not used for external training")
    teacher = np.empty((len(split), HORIZON, len(VARIABLES)), dtype=np.float32)
    for index in range(len(split)):
        context = np.asarray(split.scheduler_context[index], dtype=np.float64)
        lp_parameters = dict(parameters)
        lp_parameters["grid_energy_price"] = context[:, 2]
        lp_parameters["gas_energy_price"] = context[:, 3]
        lp_parameters["carbon_price"] = context[:, 4]
        result = solve_dispatch_lp(DispatchInputs(
            demand=np.asarray(split.forecast_target[index, :, :3], dtype=np.float64),
            pv_available=np.maximum(context[:, 0], 0.0), wt_available=np.maximum(context[:, 1], 0.0),
            parameters=lp_parameters, initial_soc=float(split.scheduler_context[index, 0, 5]),
            previous_chp=bounded_previous_chp(float(split.previous_chp[index, 0]), lp_parameters),
        ))
        if not result.success:
            raise RuntimeError(f"teacher LP failed at window {index}: {result.message}")
        teacher[index] = np.column_stack([result.values[name] for name in VARIABLES]).astype(np.float32)
    return teacher


def build_external_v46_oracle(split: ExternalV46Split, parameters: Mapping[str, Any]) -> np.ndarray:
    """Create first-step perfect-information objectives for evaluation labels."""

    oracle = np.empty((len(split),), dtype=np.float32)
    for index in range(len(split)):
        context = np.asarray(split.scheduler_context[index], dtype=np.float64)
        lp_parameters = dict(parameters)
        lp_parameters["grid_energy_price"] = context[:, 2]
        lp_parameters["gas_energy_price"] = context[:, 3]
        lp_parameters["carbon_price"] = context[:, 4]
        result = solve_dispatch_lp(DispatchInputs(
            demand=np.asarray(split.forecast_target[index, :, :3], dtype=np.float64),
            pv_available=np.maximum(split.renewable_realized[index, :, 0], 0.0),
            wt_available=np.maximum(split.renewable_realized[index, :, 1], 0.0),
            parameters=lp_parameters, initial_soc=float(split.scheduler_context[index, 0, 5]),
            previous_chp=bounded_previous_chp(float(split.previous_chp[index, 0]), lp_parameters),
        ))
        if not result.success:
            raise RuntimeError(f"oracle LP failed at window {index}: {result.message}")
        oracle[index] = _first_step_objective(result.values, context, parameters)
    return oracle


__all__ = [
    "ExternalV46Normalization", "ExternalV46Split", "build_external_v46_oracle",
    "build_external_v46_teacher", "bounded_previous_chp", "load_external_v46_split",
]
