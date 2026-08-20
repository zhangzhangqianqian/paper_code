"""Validation-only economic/carbon frontier for the heat-pump S track."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..synthetic_scenarios import SyntheticScenarioBatch
from .dispatch_lp import (
    HeatPumpDispatchInputs,
    HeatPumpDispatchResult,
    HeatPumpDispatchSolveOptions,
    solve_heat_pump_dispatch_lp,
)
from .parameters import HeatPumpParameters


SCHEMA_VERSION = "heat-pump-frontier-runner-v1"
EXPECTED_CONTRACT = "heat-pump-frontier-contract-v1"


@dataclass(frozen=True)
class HeatPumpFrontierContract:
    schema_version: str
    source_type: str
    split: str
    cop_values: tuple[float, ...]
    capacity_multipliers: tuple[float, ...]
    gas_price_multipliers: tuple[float, ...]
    epsilon_cost_tolerances: tuple[float, ...]
    legacy_carbon_weights: tuple[float, ...]
    scenario_generation: Mapping[str, Any]
    acceptance: Mapping[str, float]
    test_set_accessed: bool


def _tuple_numbers(payload: Mapping[str, Any], key: str, *, minimum: float = 0.0) -> tuple[float, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{key} must be a non-empty array")
    values = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) and item >= minimum for item in values):
        raise ValueError(f"{key} must contain finite values >= {minimum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{key} must not contain duplicate values")
    return values


def load_frontier_contract(path: str | Path) -> HeatPumpFrontierContract:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != EXPECTED_CONTRACT:
        raise ValueError(f"frontier contract must use {EXPECTED_CONTRACT}")
    if payload.get("source_type") != "pure_simulation_heat_pump" or payload.get("split") != "validation":
        raise ValueError("heat-pump frontier is validation-only pure simulation")
    if payload.get("test_set_accessed") is not False:
        raise ValueError("frontier contract must seal the test split")
    acceptance = payload.get("acceptance")
    generation = payload.get("scenario_generation")
    if not isinstance(acceptance, Mapping) or not isinstance(generation, Mapping):
        raise ValueError("frontier contract requires acceptance and scenario_generation objects")
    required_acceptance = {
        "lp_success_rate", "physical_feasible_rate", "balance_residual_tolerance",
        "distinct_dispatch_rate", "emission_reduction_sample_rate", "median_emission_reduction",
    }
    if set(acceptance) != required_acceptance:
        raise ValueError("frontier acceptance fields are incomplete")
    acceptance_values = {key: float(value) for key, value in acceptance.items()}
    if not all(math.isfinite(value) and value >= 0.0 for value in acceptance_values.values()):
        raise ValueError("frontier acceptance values must be finite and non-negative")
    if acceptance_values["lp_success_rate"] != 1.0 or acceptance_values["physical_feasible_rate"] != 1.0:
        raise ValueError("frontier requires 100% LP success and physical feasibility")
    return HeatPumpFrontierContract(
        schema_version=str(payload["schema_version"]),
        source_type=str(payload["source_type"]),
        split=str(payload["split"]),
        cop_values=_tuple_numbers(payload, "cop_values", minimum=1e-12),
        capacity_multipliers=_tuple_numbers(payload, "capacity_multipliers", minimum=0.0),
        gas_price_multipliers=_tuple_numbers(payload, "gas_price_multipliers", minimum=1e-12),
        epsilon_cost_tolerances=_tuple_numbers(payload, "epsilon_cost_tolerances", minimum=0.0),
        legacy_carbon_weights=_tuple_numbers(payload, "legacy_carbon_weights", minimum=0.0),
        scenario_generation=dict(generation),
        acceptance=acceptance_values,
        test_set_accessed=False,
    )


def _scenario_inputs(batch: SyntheticScenarioBatch, index: int, benchmark_values: Mapping[str, Any], heat_pump: HeatPumpParameters, gas_price_multiplier: float) -> HeatPumpDispatchInputs:
    parameters = {str(key): value for key, value in benchmark_values.items()}
    parameters["grid_energy_price"] = np.asarray(batch.prices[index, :, 0], dtype=np.float64)
    parameters["gas_energy_price"] = np.asarray(batch.prices[index, :, 1], dtype=np.float64) * gas_price_multiplier
    parameters["carbon_price"] = np.zeros(batch.horizon, dtype=np.float64)
    return HeatPumpDispatchInputs(
        demand=np.asarray(batch.demand[index], dtype=np.float64),
        pv_available=np.asarray(batch.pv_available[index], dtype=np.float64),
        wt_available=np.asarray(batch.wt_available[index], dtype=np.float64),
        parameters=parameters,
        heat_pump=heat_pump,
        initial_soc=float(batch.initial_soc[index]),
    )


def _slack_vector(result: HeatPumpDispatchResult) -> np.ndarray:
    return np.asarray([np.sum(result.values[name]) for name in ("slack_e", "slack_c", "slack_h")], dtype=np.float64)


def _dispatch_distance(left: HeatPumpDispatchResult, right: HeatPumpDispatchResult) -> float:
    names = ("grid", "g_chp", "g_gb", "p_ec", "q_gb", "p_hp", "q_hp", "p_charge", "p_discharge")
    numerator = sum(float(np.sum(np.abs(left.values[name] - right.values[name]))) for name in names)
    denominator = max(sum(float(np.sum(np.abs(left.values[name]))) for name in names), 1e-9)
    return numerator / denominator


def solve_scenario_frontier(batch: SyntheticScenarioBatch, index: int, benchmark_values: Mapping[str, Any], heat_pump: HeatPumpParameters, gas_price_multiplier: float, epsilon_cost_tolerances: Sequence[float]) -> list[dict[str, Any]]:
    inputs = _scenario_inputs(batch, index, benchmark_values, heat_pump, gas_price_multiplier)
    economic = solve_heat_pump_dispatch_lp(inputs, HeatPumpDispatchSolveOptions(objective_mode="operating_cost"))
    if not economic.success:
        return [
            {
                "scenario_index": index,
                "scenario_id": int(batch.scenario_ids[index]),
                "epsilon": float(epsilon),
                "status": "economic_failed",
            }
            for epsilon in epsilon_cost_tolerances
        ]
    slack_caps = tuple(float(value) + 1e-8 for value in _slack_vector(economic))
    records: list[dict[str, Any]] = []
    for epsilon in epsilon_cost_tolerances:
        low_carbon = solve_heat_pump_dispatch_lp(
            inputs,
            HeatPumpDispatchSolveOptions(
                objective_mode="physical_carbon",
                # Keep epsilon=0 as the same economic frontier while leaving
                # a tiny numerical margin for HiGHS feasibility tolerances.
                operating_cost_cap=max(
                    (1.0 + float(epsilon)) * economic.operating_cost,
                    economic.operating_cost + 1e-7 * max(1.0, abs(economic.operating_cost)),
                ),
                slack_caps=slack_caps,
            ),
        )
        if not low_carbon.success:
            records.append({"scenario_index": index, "scenario_id": int(batch.scenario_ids[index]), "epsilon": float(epsilon), "status": "frontier_failed"})
            continue
        reduction = (economic.physical_carbon - low_carbon.physical_carbon) / max(abs(economic.physical_carbon), 1e-9)
        records.append({
            "scenario_index": index, "scenario_id": int(batch.scenario_ids[index]), "epsilon": float(epsilon), "status": "optimal",
            "economic_cost": float(economic.operating_cost), "low_carbon_cost": float(low_carbon.operating_cost),
            "economic_carbon": float(economic.physical_carbon), "low_carbon_carbon": float(low_carbon.physical_carbon),
            "carbon_reduction_rate": float(reduction), "dispatch_distance": float(_dispatch_distance(economic, low_carbon)),
            "economic_slack": float(np.sum(_slack_vector(economic))), "low_carbon_slack": float(np.sum(_slack_vector(low_carbon))),
        })
    return records


def summarize_frontier(records: Sequence[Mapping[str, Any]], contract: HeatPumpFrontierContract) -> dict[str, Any]:
    rows = [row for row in records if row.get("status") == "optimal"]
    successful = len(rows)
    total = len(records)
    nonzero_rows = [row for row in rows if float(row["epsilon"]) > 0.0]
    reductions = np.asarray([float(row["carbon_reduction_rate"]) for row in nonzero_rows], dtype=np.float64)
    distances = np.asarray([float(row["dispatch_distance"]) for row in nonzero_rows], dtype=np.float64)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "total_records": total,
        "successful_frontier_records": successful,
        "lp_success_rate": float(successful / total) if total else 0.0,
        "physical_feasible_rate": float(successful / total) if total else 0.0,
        "nonzero_epsilon_record_count": len(nonzero_rows),
        "distinct_dispatch_rate": float(np.mean(distances > 1e-8)) if len(nonzero_rows) else 0.0,
        "emission_reduction_sample_rate": float(np.mean(reductions > 1e-9)) if len(nonzero_rows) else 0.0,
        "median_emission_reduction": float(np.median(reductions)) if len(nonzero_rows) else 0.0,
        "test_set_accessed": False,
    }
    checks = {
        "lp_success_rate": summary["lp_success_rate"] >= contract.acceptance["lp_success_rate"],
        "physical_feasible_rate": summary["physical_feasible_rate"] >= contract.acceptance["physical_feasible_rate"],
    }
    by_epsilon: dict[str, dict[str, float | bool]] = {}
    for epsilon in sorted({float(row["epsilon"]) for row in nonzero_rows}):
        level = [row for row in nonzero_rows if float(row["epsilon"]) == epsilon]
        level_reduction = np.asarray([float(row["carbon_reduction_rate"]) for row in level], dtype=np.float64)
        level_distance = np.asarray([float(row["dispatch_distance"]) for row in level], dtype=np.float64)
        level_metrics = {
            "distinct_dispatch_rate": float(np.mean(level_distance > 1e-8)) if len(level) else 0.0,
            "emission_reduction_sample_rate": float(np.mean(level_reduction > 1e-9)) if len(level) else 0.0,
            "median_emission_reduction": float(np.median(level_reduction)) if len(level) else 0.0,
        }
        level_metrics["passes"] = bool(
            level_metrics["distinct_dispatch_rate"] >= contract.acceptance["distinct_dispatch_rate"]
            and level_metrics["emission_reduction_sample_rate"] >= contract.acceptance["emission_reduction_sample_rate"]
            and level_metrics["median_emission_reduction"] >= contract.acceptance["median_emission_reduction"]
        )
        by_epsilon[str(epsilon)] = level_metrics
    checks["nonzero_epsilon_frontier"] = any(bool(level["passes"]) for level in by_epsilon.values())
    summary["gate_checks"] = checks
    summary["gate_by_epsilon"] = by_epsilon
    summary["decision_status"] = "passed" if all(checks.values()) else "failed_no_identifiable_heat_pump_frontier"
    return summary


def frontier_contract_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["HeatPumpFrontierContract", "frontier_contract_hash", "load_frontier_contract", "solve_scenario_frontier", "summarize_frontier"]
