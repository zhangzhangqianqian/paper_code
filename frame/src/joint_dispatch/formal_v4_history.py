"""Causal, realized-settled device histories for formal-v4.1.

The historical trajectory is generated with the same one-hour rolling
transition used at deployment: a four-hour causal plan is solved, only its
first action is executed, and that action is settled against the current
realized demand and renewable availability.  Planned LP values are therefore
never written directly into the historical device features.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..scheduling.dispatch_lp import DispatchInputs, DispatchResult, solve_dispatch_lp
from .contract import DISPATCH_ORDER, STATUS_ORDER
from .data import derive_device_status
from .formal_v4_data import FormalV4BaseSeries
from .formal_v4_recourse import settle_first_step_v4


LOOKBACK = 24
HORIZON = 4
FIRST_SETTLED_INDEX = LOOKBACK
FIRST_MODEL_ORIGIN_INDEX = 2 * LOOKBACK
TRAJECTORY_RULE_VERSION = "formal-v4.1-causal-realized-settlement-v1"
_I = {name: index for index, name in enumerate(DISPATCH_ORDER)}


def _sha256_arrays(*arrays: np.ndarray, extra: str = "") -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(extra.encode("utf-8"))
    return digest.hexdigest()


def _receipt_payload(value: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            raise PermissionError("causal trajectory requires an existing capacity receipt")
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("gate0_authorized") is not True:
        raise PermissionError("causal trajectory requires a Gate 0 authorized capacity receipt")
    return value


def _capacity_multiplier(receipt: Mapping[str, Any]) -> float:
    audit = receipt.get("capacity_audit", receipt)
    selected = audit.get("selected") if isinstance(audit, Mapping) else None
    value = selected.get("multiplier") if isinstance(selected, Mapping) else receipt.get("capacity_multiplier", 1.0)
    multiplier = float(value)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("capacity receipt contains an invalid selected multiplier")
    return multiplier


@dataclass(frozen=True)
class TrajectoryAudit:
    solved_hours: int
    settled_hours: int
    warmup_hours: int
    future_label_reads: int
    max_balance_residual: float
    max_conversion_residual: float
    max_soc_recursion_residual: float
    max_chp_ramp_violation: float
    max_renewable_availability_violation: float
    max_soc_bound_violation: float
    reset_indices: tuple[int, ...]
    first_model_origin_index: int | None
    segment_first_model_origins: tuple[int, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "solved_hours": self.solved_hours,
            "settled_hours": self.settled_hours,
            "warmup_hours": self.warmup_hours,
            "future_label_reads": self.future_label_reads,
            "max_balance_residual": self.max_balance_residual,
            "max_conversion_residual": self.max_conversion_residual,
            "max_soc_recursion_residual": self.max_soc_recursion_residual,
            "max_chp_ramp_violation": self.max_chp_ramp_violation,
            "max_renewable_availability_violation": self.max_renewable_availability_violation,
            "max_soc_bound_violation": self.max_soc_bound_violation,
            "reset_indices": list(self.reset_indices),
            "first_model_origin_index": self.first_model_origin_index,
            "segment_first_model_origins": list(self.segment_first_model_origins),
        }


@dataclass(frozen=True)
class SettledTrajectory:
    """Raw hourly settlement and state metadata for one chronological stream."""

    settled_dispatch: np.ndarray
    activity_indicators: np.ndarray
    initial_soc: np.ndarray
    previous_chp: np.ndarray
    next_soc: np.ndarray
    next_previous_chp: np.ndarray
    settled_mask: np.ndarray
    target_times: np.ndarray
    trajectory_id: str
    audit: TrajectoryAudit

    def __post_init__(self) -> None:
        n = len(self.target_times)
        dispatch = np.asarray(self.settled_dispatch, dtype=np.float64)
        activity = np.asarray(self.activity_indicators, dtype=np.float64)
        settled_mask = np.asarray(self.settled_mask, dtype=bool)
        target_times = np.asarray(self.target_times, dtype="datetime64[ns]")
        if dispatch.shape != (n, len(DISPATCH_ORDER)):
            raise ValueError("settled_dispatch must have shape [T,21]")
        if activity.shape != (n, len(STATUS_ORDER)):
            raise ValueError("activity_indicators must have shape [T,6]")
        for name, value in {
            "initial_soc": self.initial_soc,
            "previous_chp": self.previous_chp,
            "next_soc": self.next_soc,
            "next_previous_chp": self.next_previous_chp,
        }.items():
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (n, 1):
                raise ValueError(f"{name} must have shape [T,1]")
            object.__setattr__(self, name, array)
        if settled_mask.shape != (n,) or target_times.shape != (n,):
            raise ValueError("settled_mask and target_times must have shape [T]")
        if not np.isfinite(dispatch).all() or not np.isfinite(activity).all():
            raise ValueError("trajectory values must be finite")
        if not np.isin(activity, (0.0, 1.0)).all():
            raise ValueError("activity_indicators must be binary")
        object.__setattr__(self, "settled_dispatch", dispatch)
        object.__setattr__(self, "activity_indicators", activity)
        object.__setattr__(self, "settled_mask", settled_mask)
        object.__setattr__(self, "target_times", target_times)

    @property
    def device_history(self) -> np.ndarray:
        """The frozen 17 continuous fields exposed to the forecaster."""

        return self.settled_dispatch[:, :17]

    @property
    def trajectory_sha256(self) -> str:
        return _sha256_arrays(
            self.settled_dispatch,
            self.activity_indicators,
            self.initial_soc,
            self.previous_chp,
            self.next_soc,
            self.next_previous_chp,
            self.settled_mask.astype(np.uint8),
            extra=f"{TRAJECTORY_RULE_VERSION}|{self.trajectory_id}",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "settled_dispatch": self.settled_dispatch,
            "activity_indicators": self.activity_indicators,
            "initial_soc": self.initial_soc,
            "previous_chp": self.previous_chp,
            "next_soc": self.next_soc,
            "next_previous_chp": self.next_previous_chp,
            "settled_mask": self.settled_mask,
            "target_times": self.target_times,
            "trajectory_id": self.trajectory_id,
            "trajectory_sha256": self.trajectory_sha256,
            "rule_version": TRAJECTORY_RULE_VERSION,
            "audit": self.audit.to_payload(),
        }


def _segment_starts(timestamps: np.ndarray) -> tuple[int, ...]:
    starts = [0]
    for index in range(1, len(timestamps)):
        if timestamps[index] - timestamps[index - 1] != np.timedelta64(1, "h"):
            starts.append(index)
    return tuple(starts)


def _settle_numpy(
    planned: np.ndarray,
    demand: np.ndarray,
    renewable: np.ndarray,
    parameters: Mapping[str, Any],
    initial_soc: float,
    previous_chp: float,
) -> tuple[np.ndarray, np.ndarray, float, float, dict[str, float]]:
    """Call the canonical differentiable settlement once and detach to NumPy."""

    dtype = torch.float64
    with torch.no_grad():
        outcome = settle_first_step_v4(
            torch.as_tensor(planned, dtype=dtype).reshape(1, -1),
            torch.as_tensor(demand, dtype=dtype).reshape(1, 3),
            torch.as_tensor(renewable, dtype=dtype).reshape(1, 2),
            parameters,
            initial_soc=torch.tensor([[initial_soc]], dtype=dtype),
            previous_chp=torch.tensor([[previous_chp]], dtype=dtype),
        )
    dispatch = outcome.realized_dispatch[0].cpu().numpy()
    activity = outcome.activity_indicators[0].cpu().numpy()
    bess_energy = float(parameters["bess_energy_capacity"])
    eta_bess = float(parameters["bess_roundtrip_efficiency"]) ** 0.5
    expected_soc = np.clip(
        (float(initial_soc) * bess_energy + eta_bess * dispatch[_I["p_charge"]] - dispatch[_I["p_discharge"]] / max(eta_bess, 1.0e-12)) / bess_energy,
        0.0,
        1.0,
    )
    metrics = {
        "balance": float(torch.max(torch.abs(outcome.balance_residuals)).item()),
        "conversion": float(torch.max(torch.abs(outcome.conversion_residuals)).item()),
        "soc": float(outcome.next_soc[0, 0].item()),
        "next_chp": float(outcome.next_previous_chp[0, 0].item()),
        "soc_recursion": float(abs(outcome.next_soc[0, 0].item() - expected_soc)),
        "renewable_violation": float(max(
            0.0,
            abs(dispatch[_I["pv_use"]] + dispatch[_I["pv_curt"]] - float(renewable[0])),
            abs(dispatch[_I["wt_use"]] + dispatch[_I["wt_curt"]] - float(renewable[1])),
            dispatch[_I["pv_use"]] - float(renewable[0]),
            dispatch[_I["wt_use"]] - float(renewable[1]),
        )),
    }
    return dispatch, activity, metrics["soc"], metrics["next_chp"], metrics


def generate_settled_device_trajectory(
    base: FormalV4BaseSeries,
    parameters: Mapping[str, Any],
    *,
    capacity_receipt: Mapping[str, Any] | str | Path,
    solver: Any = solve_dispatch_lp,
    trajectory_id: str = "formal_v4_1_causal_settled",
) -> SettledTrajectory:
    """Generate a causal realized-settled trajectory over one chronological base.

    The first 24 continuous hours of each segment are warm-up only.  The first
    settled transition is index 24 and the first eligible 24-hour device
    history is index 48.  No current/future target is passed to the planner.
    """

    if not isinstance(base, FormalV4BaseSeries):
        raise TypeError("base must be FormalV4BaseSeries")
    receipt = _receipt_payload(capacity_receipt)
    multiplier = _capacity_multiplier(receipt)
    context_parameters = dict(parameters)
    for key in ("electric_chiller_capacity", "absorption_chiller_capacity"):
        if key not in context_parameters:
            raise ValueError(f"trajectory parameters missing {key}")
        context_parameters[key] = float(context_parameters[key]) * multiplier
    required = ("bess_energy_capacity", "chp_electric_capacity")
    if any(key not in context_parameters for key in required):
        raise ValueError("trajectory parameters are missing state capacities")

    loads = np.asarray(base.load_and_exog[:, :3], dtype=np.float64)
    renewable = np.asarray(base.renewable_realized, dtype=np.float64)
    timestamps = np.asarray(base.timestamps, dtype="datetime64[ns]")
    n = len(timestamps)
    dispatch = np.zeros((n, len(DISPATCH_ORDER)), dtype=np.float64)
    activity = np.zeros((n, len(STATUS_ORDER)), dtype=np.float64)
    initial_soc = np.full((n, 1), np.nan, dtype=np.float64)
    previous_chp = np.full((n, 1), np.nan, dtype=np.float64)
    next_soc = np.full((n, 1), np.nan, dtype=np.float64)
    next_previous_chp = np.full((n, 1), np.nan, dtype=np.float64)
    settled_mask = np.zeros(n, dtype=bool)
    reset_indices: list[int] = []
    first_model_origins: list[int] = []
    solved_hours = 0
    max_balance = max_conversion = max_soc_recursion = 0.0
    max_ramp = max_renewable = max_soc_bound = 0.0
    soc, previous = 0.5, 0.0
    segment_start_set = set(_segment_starts(timestamps))
    for index in range(n):
        segment_start = index in segment_start_set
        relative = index - max(start for start in segment_start_set if start <= index)
        if segment_start:
            soc, previous = 0.5, 0.0
            reset_indices.append(index)
        # A 24-hour causal load forecast is unavailable during warm-up.  No LP
        # call and no realized settlement occurs in those rows.
        if relative < FIRST_SETTLED_INDEX:
            continue
        if relative >= FIRST_MODEL_ORIGIN_INDEX:
            first_model_origins.append(index) if relative == FIRST_MODEL_ORIGIN_INDEX else None
        if index < LOOKBACK or index + HORIZON > n:
            # The final three rows cannot support a complete four-hour plan;
            # leave them unsettled rather than truncating the horizon.
            continue
        demand_plan = loads[index - LOOKBACK:index - LOOKBACK + HORIZON]
        renewable_plan = np.repeat(renewable[index - 1:index], HORIZON, axis=0)
        if demand_plan.shape != (HORIZON, 3) or renewable_plan.shape != (HORIZON, 2):
            raise ValueError("causal planner received an invalid four-hour horizon")
        planner_context = dict(context_parameters)
        planner_context["grid_energy_price"] = np.repeat(base.prices_and_weights[index, 0], HORIZON)
        planner_context["gas_energy_price"] = np.repeat(base.prices_and_weights[index, 1], HORIZON)
        planner_context["carbon_price"] = np.repeat(base.prices_and_weights[index, 2], HORIZON)
        result: DispatchResult = solver(DispatchInputs(
            demand=demand_plan,
            pv_available=renewable_plan[:, 0],
            wt_available=renewable_plan[:, 1],
            parameters=planner_context,
            initial_soc=soc,
            previous_chp=previous,
        ))
        solved_hours += 1
        if not result.success:
            raise RuntimeError(f"causal trajectory LP failed at row {index}: {result.message}")
        planned = np.asarray([float(result.values[name][0]) for name in DISPATCH_ORDER], dtype=np.float64)
        initial_soc[index, 0] = soc
        previous_chp[index, 0] = previous
        settled, settled_activity, soc, previous, metrics = _settle_numpy(
            planned, loads[index], renewable[index], context_parameters, soc, previous,
        )
        dispatch[index] = settled
        activity[index] = settled_activity
        settled_mask[index] = True
        next_soc[index, 0] = soc
        next_previous_chp[index, 0] = previous
        max_balance = max(max_balance, metrics["balance"])
        max_conversion = max(max_conversion, metrics["conversion"])
        max_renewable = max(max_renewable, metrics["renewable_violation"])
        max_soc_bound = max(max_soc_bound, max(0.0, -soc, soc - 1.0))
        max_ramp = max(max_ramp, max(0.0, abs(previous - float(previous_chp[index, 0])) - float(context_parameters["chp_ramp_fraction"]) * float(context_parameters["chp_electric_capacity"])))
        # SOC recursion is represented by the canonical settlement state; this
        # audit field remains explicit for downstream receipts.
        max_soc_recursion = max(max_soc_recursion, metrics["soc_recursion"])

    # The first eligible origin is the first row with 24 settled predecessors;
    # rows at the end without a complete plan remain unavailable.
    eligible = np.zeros(n, dtype=bool)
    for index in range(n):
        if index < LOOKBACK or not settled_mask[index]:
            continue
        history = settled_mask[index - LOOKBACK:index]
        eligible[index] = bool(history.all())
    actual_origins = np.flatnonzero(eligible)
    first_origin = int(actual_origins[0]) if actual_origins.size else None
    if first_origin is not None:
        first_model_origins = [first_origin] + [int(value) for value in first_model_origins if value != first_origin]
    audit = TrajectoryAudit(
        solved_hours=solved_hours,
        settled_hours=int(settled_mask.sum()),
        warmup_hours=int(n - settled_mask.sum()),
        future_label_reads=0,
        max_balance_residual=max_balance,
        max_conversion_residual=max_conversion,
        max_soc_recursion_residual=max_soc_recursion,
        max_chp_ramp_violation=max_ramp,
        max_renewable_availability_violation=max_renewable,
        max_soc_bound_violation=max_soc_bound,
        reset_indices=tuple(reset_indices),
        first_model_origin_index=first_origin,
        segment_first_model_origins=tuple(first_model_origins),
    )
    return SettledTrajectory(
        dispatch, activity, initial_soc, previous_chp, next_soc, next_previous_chp,
        settled_mask, timestamps, trajectory_id, audit,
    )


__all__ = [
    "FIRST_MODEL_ORIGIN_INDEX", "FIRST_SETTLED_INDEX", "LOOKBACK", "HORIZON",
    "SettledTrajectory", "TrajectoryAudit", "TRAJECTORY_RULE_VERSION",
    "generate_settled_device_trajectory",
]
