"""Shared forecast-then-optimize helpers for the external comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from ..scheduling.dispatch_schema import VARIABLES
from .external_v46_data import ExternalV46Split, bounded_previous_chp


@dataclass(frozen=True)
class PTOForecasts:
    method_id: str
    prediction: np.ndarray
    target: np.ndarray

    def __post_init__(self) -> None:
        prediction = np.asarray(self.prediction, dtype=np.float64)
        target = np.asarray(self.target, dtype=np.float64)
        if prediction.ndim != 3 or prediction.shape[1:] != (4, 4):
            raise ValueError("prediction must have shape [N,4,4]")
        if target.shape != prediction.shape:
            raise ValueError("target must have the same shape as prediction")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError("PTO forecasts and targets must be finite")
        object.__setattr__(self, "prediction", prediction.astype(np.float32))
        object.__setattr__(self, "target", target.astype(np.float32))
        if not self.method_id:
            raise ValueError("method_id must be non-empty")


@dataclass(frozen=True)
class PTODispatchCache:
    dispatch: np.ndarray
    success: np.ndarray
    messages: tuple[str, ...]
    offline_exact_lp_calls: int

    def __post_init__(self) -> None:
        dispatch = np.asarray(self.dispatch, dtype=np.float32)
        success = np.asarray(self.success, dtype=bool)
        if dispatch.ndim != 3 or dispatch.shape[1:] != (4, len(VARIABLES)):
            raise ValueError("dispatch must have shape [N,4,21]")
        if success.shape != (dispatch.shape[0],):
            raise ValueError("success must have shape [N]")
        if not np.isfinite(dispatch).all():
            raise ValueError("dispatch cache must be finite")
        if len(self.messages) != dispatch.shape[0]:
            raise ValueError("messages must have one entry per window")
        if int(self.offline_exact_lp_calls) != dispatch.shape[0]:
            raise ValueError("PTO must account for one exact LP call per window")
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "success", success)
        object.__setattr__(self, "offline_exact_lp_calls", int(self.offline_exact_lp_calls))


def solve_pto_windows(
    forecasts: PTOForecasts,
    split: ExternalV46Split,
    parameters: Mapping[str, Any],
) -> PTODispatchCache:
    """Run one exact LP per window using forecasted rigid demands."""

    if len(forecasts.prediction) != len(split):
        raise ValueError("forecast and split lengths differ")
    if split.split not in {"validation", "pilot"}:
        raise ValueError("PTO evaluation requires validation or pilot windows")
    dispatch = np.zeros((len(split), 4, len(VARIABLES)), dtype=np.float32)
    success = np.zeros((len(split),), dtype=bool)
    messages: list[str] = []
    for index in range(len(split)):
        context = np.asarray(split.scheduler_context[index], dtype=np.float64)
        lp_parameters = dict(parameters)
        lp_parameters["grid_energy_price"] = context[:, 2]
        lp_parameters["gas_energy_price"] = context[:, 3]
        lp_parameters["carbon_price"] = context[:, 4]
        result = solve_dispatch_lp(DispatchInputs(
            demand=np.maximum(np.asarray(forecasts.prediction[index, :, :3], dtype=np.float64), 0.0),
            pv_available=np.maximum(context[:, 0], 0.0), wt_available=np.maximum(context[:, 1], 0.0),
            parameters=lp_parameters, initial_soc=float(context[0, 5]),
            previous_chp=bounded_previous_chp(float(split.previous_chp[index, 0]), lp_parameters),
        ))
        messages.append(str(result.message))
        if result.success:
            dispatch[index] = np.column_stack([result.values[name] for name in VARIABLES]).astype(np.float32)
            success[index] = True
    return PTODispatchCache(dispatch, success, tuple(messages), len(split))


def seasonal_naive_forecasts(load_history: np.ndarray, season_length: int = 24) -> np.ndarray:
    """Return a causal daily seasonal-naive four-step forecast."""

    history = np.asarray(load_history, dtype=np.float32)
    if history.ndim != 3 or history.shape[1:] != (24, 4):
        raise ValueError("load_history must have shape [N,24,4]")
    if int(season_length) != 24:
        raise ValueError("the v4.6 external contract supports season_length=24 only")
    prediction = history[:, :4, :].copy()
    if not np.isfinite(prediction).all():
        raise ValueError("load_history must be finite")
    return prediction


def save_pto_cache(path: str | Path, cache: PTODispatchCache) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, dispatch=cache.dispatch, success=cache.success,
        messages=np.asarray(cache.messages, dtype="U"),
        offline_exact_lp_calls=np.asarray(cache.offline_exact_lp_calls, dtype=np.int64),
    )
    return destination


def load_pto_cache(path: str | Path) -> PTODispatchCache:
    with np.load(Path(path), allow_pickle=False) as payload:
        messages = tuple(str(value) for value in payload["messages"].tolist())
        return PTODispatchCache(
            payload["dispatch"], payload["success"], messages,
            int(payload["offline_exact_lp_calls"]),
        )


__all__ = [
    "PTOForecasts", "PTODispatchCache", "load_pto_cache", "save_pto_cache",
    "seasonal_naive_forecasts", "solve_pto_windows",
]
