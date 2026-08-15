"""Stage 10.9: four-hour rolling planning with first-step realization."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .dispatch_lp import DispatchInputs, solve_dispatch_lp
from .recourse import RealizedStep, settle_first_step


@dataclass(frozen=True)
class RollingForecastSet:
    origin_times: np.ndarray
    demand: np.ndarray  # [N,H,3]
    pv_available: np.ndarray  # [N,H]
    wt_available: np.ndarray  # [N,H]


@dataclass(frozen=True)
class ActualStream:
    origin_times: np.ndarray
    electricity: np.ndarray
    cooling: np.ndarray
    heating: np.ndarray
    pv_available: np.ndarray
    wt_available: np.ndarray


@dataclass(frozen=True)
class RollingResult:
    rows: tuple[Mapping[str, object], ...]
    plans: tuple[object, ...]
    realized_steps: tuple[RealizedStep, ...]
    final_soc: float


def _validate_inputs(forecast_set: RollingForecastSet, actual: ActualStream) -> int:
    origins = np.asarray(forecast_set.origin_times, dtype="datetime64[ns]")
    demand = np.asarray(forecast_set.demand, dtype=np.float64)
    pv = np.asarray(forecast_set.pv_available, dtype=np.float64)
    wt = np.asarray(forecast_set.wt_available, dtype=np.float64)
    if demand.ndim != 3 or demand.shape[1:] != (4, 3):
        raise ValueError("forecast demand must have shape [N,4,3]")
    n = demand.shape[0]
    if origins.shape != (n,) or pv.shape != (n, 4) or wt.shape != (n, 4):
        raise ValueError("forecast arrays are not aligned")
    actual_origins = np.asarray(actual.origin_times, dtype="datetime64[ns]")
    if actual_origins.shape != (n,) or not np.array_equal(actual_origins, origins):
        raise ValueError("planning and actual origin_times must match exactly")
    for array in (actual.electricity, actual.cooling, actual.heating, actual.pv_available, actual.wt_available):
        if np.asarray(array).shape != (n,):
            raise ValueError("actual stream must contain one first-step value per origin")
    if len(np.unique(origins)) != n:
        raise ValueError("each origin may be settled only once")
    return n


def run_rolling_dispatch(
    forecast_set: RollingForecastSet,
    actual_stream: ActualStream,
    parameters: Mapping[str, float],
    initial_soc: float = 0.5,
    checkpoint_path: str | Path | None = None,
) -> RollingResult:
    """Solve every four-hour plan and settle only its first hour."""

    n = _validate_inputs(forecast_set, actual_stream)
    soc = float(initial_soc)
    rows: list[Mapping[str, object]] = []
    plans: list[object] = []
    realized_steps: list[RealizedStep] = []
    start_index = 0
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    if checkpoint is not None and checkpoint.exists():
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        completed = np.asarray(payload.get("completed_origins", []), dtype="datetime64[ns]")
        expected_prefix = np.asarray(forecast_set.origin_times[: len(completed)], dtype="datetime64[ns]")
        if not np.array_equal(completed, expected_prefix):
            raise ValueError("Checkpoint origins are not a prefix of the current rolling protocol")
        start_index = len(completed)
        rows.extend(payload.get("rows", []))
        soc = float(payload.get("next_initial_soc", initial_soc))
        if start_index > n:
            raise ValueError("Checkpoint has more origins than the current protocol")
    for i in range(start_index, n):
        plan = solve_dispatch_lp(
            DispatchInputs(
                demand=np.asarray(forecast_set.demand[i], dtype=np.float64),
                pv_available=np.asarray(forecast_set.pv_available[i], dtype=np.float64),
                wt_available=np.asarray(forecast_set.wt_available[i], dtype=np.float64),
                parameters=parameters,
                initial_soc=soc,
            )
        )
        if not plan.success:
            raise RuntimeError(f"LP failed at origin {forecast_set.origin_times[i]}: {plan.message}")
        realized = settle_first_step(
            plan,
            {
                "electricity": float(actual_stream.electricity[i]),
                "cooling": float(actual_stream.cooling[i]),
                "heating": float(actual_stream.heating[i]),
                "pv_available": float(actual_stream.pv_available[i]),
                "wt_available": float(actual_stream.wt_available[i]),
            },
            parameters,
        )
        soc = float(plan.values["soc"][0] / parameters["bess_energy_capacity"])
        plans.append(plan)
        realized_steps.append(realized)
        rows.append(
            {
                "origin": str(forecast_set.origin_times[i]),
                "planned_cost": float(plan.objective),
                "realized_cost": float(realized.realized_cost),
                "regret": float(realized.realized_cost - plan.objective),
                "solver_status": plan.status,
                "max_balance_residual": float(max(plan.balance_residuals.values())),
                "simultaneous_charge_discharge": float(plan.simultaneous_charge_discharge),
                "grid_upward": float(realized.grid_upward),
                "grid_downward": float(realized.grid_downward),
                "gas_upward": float(realized.gas_upward),
                "gas_downward": float(realized.gas_downward),
                "unserved_electricity": float(realized.unserved_electricity),
                "unserved_cooling": float(realized.unserved_cooling),
                "unserved_heating": float(realized.unserved_heating),
            }
        )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(
                json.dumps(
                    {
                        "schema_version": "rolling-checkpoint-v1",
                        "completed_origins": [str(value) for value in forecast_set.origin_times[: i + 1]],
                        "next_initial_soc": soc,
                        "rows": rows,
                        "origin_completed": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    return RollingResult(tuple(rows), tuple(plans), tuple(realized_steps), soc)
