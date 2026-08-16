"""Stage 10.9: four-hour rolling planning with first-step realization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from .dispatch_lp import DispatchInputs, DispatchResult, solve_dispatch_lp
from .recourse import RealizedStep, evaluate_planned_first_step, settle_first_step


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
    solver_times_seconds: tuple[float, ...] = ()


def _dispatch_to_json(plan: DispatchResult) -> dict[str, object]:
    return {
        "status": plan.status,
        "message": plan.message,
        "objective": float(plan.objective),
        "values": {name: np.asarray(value, dtype=float).tolist() for name, value in plan.values.items()},
        "balance_residuals": {name: float(value) for name, value in plan.balance_residuals.items()},
        "simultaneous_charge_discharge": float(plan.simultaneous_charge_discharge),
    }


def _dispatch_from_json(payload: Mapping[str, object]) -> DispatchResult:
    return DispatchResult(
        status=str(payload["status"]),
        message=str(payload["message"]),
        objective=float(payload["objective"]),
        values={str(name): np.asarray(value, dtype=float) for name, value in dict(payload["values"]).items()},
        balance_residuals={str(name): float(value) for name, value in dict(payload["balance_residuals"]).items()},
        simultaneous_charge_discharge=float(payload["simultaneous_charge_discharge"]),
    )


def _realized_to_json(realized: RealizedStep) -> dict[str, float]:
    return {name: float(value) for name, value in realized.__dict__.items()}


def _realized_from_json(payload: Mapping[str, object]) -> RealizedStep:
    return RealizedStep(**{name: float(value) for name, value in payload.items()})


def _validate_inputs(forecast_set: RollingForecastSet, actual: ActualStream) -> int:
    origins = np.asarray(forecast_set.origin_times, dtype="datetime64[ns]")
    demand = np.asarray(forecast_set.demand, dtype=np.float64)
    pv = np.asarray(forecast_set.pv_available, dtype=np.float64)
    wt = np.asarray(forecast_set.wt_available, dtype=np.float64)
    if demand.ndim != 3 or demand.shape[2] != 3 or demand.shape[1] <= 0:
        raise ValueError("forecast demand must have shape [N,H,3]")
    n = demand.shape[0]
    horizon = demand.shape[1]
    if origins.shape != (n,) or pv.shape != (n, horizon) or wt.shape != (n, horizon):
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
    solver_times: list[float] = []
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
        if start_index and ("plans" not in payload or "realized_steps" not in payload):
            raise ValueError("Checkpoint lacks serialized plans/realized_steps; start a new checkpoint")
        plans.extend(_dispatch_from_json(item) for item in payload.get("plans", []))
        realized_steps.extend(_realized_from_json(item) for item in payload.get("realized_steps", []))
        soc = float(payload.get("next_initial_soc", initial_soc))
        if start_index > n:
            raise ValueError("Checkpoint has more origins than the current protocol")
    for i in range(start_index, n):
        solve_start = time.perf_counter()
        plan = solve_dispatch_lp(
            DispatchInputs(
                demand=np.asarray(forecast_set.demand[i], dtype=np.float64),
                pv_available=np.asarray(forecast_set.pv_available[i], dtype=np.float64),
                wt_available=np.asarray(forecast_set.wt_available[i], dtype=np.float64),
                parameters=parameters,
                initial_soc=soc,
            )
        )
        solver_times.append(float(time.perf_counter() - solve_start))
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
        planned_first = evaluate_planned_first_step(plan, parameters)
        energy_capacity = float(parameters["bess_energy_capacity"])
        if energy_capacity > 1e-12:
            # The LP bound is [0, capacity]; clip only floating-point roundoff
            # at the boundary before carrying SOC into the next window.
            raw_soc = float(plan.values["soc"][0] / energy_capacity)
            soc = float(np.clip(raw_soc, 0.0, 1.0))
        else:
            soc = float(initial_soc)
        plans.append(plan)
        realized_steps.append(realized)
        rows.append(
            {
                "origin": str(forecast_set.origin_times[i]),
                # Legacy field: objective over the complete planning horizon.
                "planned_cost": float(plan.objective),
                "horizon_objective": float(plan.objective),
                "planned_cost_first_step": float(planned_first["planned_cost_first_step"]),
                "realized_cost": float(realized.realized_cost),
                "planning_deviation_cost": float(realized.realized_cost - planned_first["planned_cost_first_step"]),
                "planned_carbon_first_step": float(planned_first["planned_carbon_first_step"]),
                "realized_carbon_first_step": float(
                    realized.realized_grid * float(parameters.get("grid_emission_factor", 0.0))
                    + realized.realized_gas * float(parameters.get("gas_emission_factor", 0.0))
                ),
                "planned_curtailment_first_step": float(planned_first["planned_curtailment_first_step"]),
                "realized_curtailment_first_step": float(realized.curtailment),
                "planned_grid_first_step": float(planned_first["planned_grid_first_step"]),
                "planned_gas_first_step": float(planned_first["planned_gas_first_step"]),
                "planned_chp_electricity": float(plan.values["p_chp"][0]),
                "planned_chp_heat": float(plan.values["q_chp"][0]),
                "planned_gas_boiler_heat": float(plan.values["q_gb"][0]),
                "planned_electric_chiller_electricity": float(plan.values["p_ec"][0]),
                "planned_electric_chiller_cooling": float(plan.values["q_ec"][0]),
                "planned_absorption_chiller_cooling": float(plan.values["q_ac"][0]),
                "planned_bess_charge": float(plan.values["p_charge"][0]),
                "planned_bess_discharge": float(plan.values["p_discharge"][0]),
                "electric_chiller_electricity": float(realized.electric_chiller_electricity),
                "electric_chiller_cooling": float(realized.electric_chiller_cooling),
                "gas_boiler_heat": float(realized.gas_boiler_heat),
                "absorption_chiller_cooling": float(realized.absorption_chiller_cooling),
                "chp_electricity": float(realized.chp_electricity),
                "chp_heat": float(realized.chp_heat),
                "bess_charge": float(realized.bess_charge),
                "bess_discharge": float(realized.bess_discharge),
                "soc_after_execution": float(realized.soc_after_execution),
                "actual_electricity": float(realized.actual_electricity),
                "actual_cooling": float(realized.actual_cooling),
                "actual_heating": float(realized.actual_heating),
                "actual_pv": float(realized.actual_pv),
                "actual_wt": float(realized.actual_wt),
                "solver_status": plan.status,
                "max_balance_residual": float(max(plan.balance_residuals.values())),
                "simultaneous_charge_discharge": float(plan.simultaneous_charge_discharge),
                "solver_time_seconds": solver_times[-1],
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
                        "plans": [_dispatch_to_json(item) for item in plans],
                        "realized_steps": [_realized_to_json(item) for item in realized_steps],
                        "origin_completed": True,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    return RollingResult(tuple(rows), tuple(plans), tuple(realized_steps), soc, tuple(solver_times))
