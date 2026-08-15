"""First-step realization and asymmetric recourse for the S-track plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .dispatch_lp import DispatchResult


@dataclass(frozen=True)
class RealizedStep:
    planned_grid: float
    realized_grid: float
    planned_gas: float
    realized_gas: float
    grid_upward: float
    grid_downward: float
    gas_upward: float
    gas_downward: float
    unserved_electricity: float
    unserved_cooling: float
    unserved_heating: float
    curtailment: float
    realized_cost: float


def settle_first_step(
    plan: DispatchResult,
    actual: Mapping[str, float],
    parameters: Mapping[str, float],
) -> RealizedStep:
    """Settle one real first hour while keeping slow plan variables fixed.

    Grid import, the electric chiller and the gas boiler are treated as fast
    recourse variables. CHP output, BESS charge/discharge and absorption
    chiller input remain fixed at their planned first-hour values.
    """

    if not plan.success:
        raise ValueError("Cannot settle an unsuccessful dispatch plan")
    required = ("electricity", "cooling", "heating", "pv_available", "wt_available")
    missing = [key for key in required if key not in actual]
    if missing:
        raise ValueError(f"Missing actual first-step fields: {missing}")
    v = plan.values
    p = {key: float(value) for key, value in parameters.items()}
    planned_grid = float(v["grid"][0])
    planned_gas = float(v["g_chp"][0] + v["g_gb"][0])
    p_chp = float(v["p_chp"][0])
    p_discharge = float(v["p_discharge"][0])
    p_charge = float(v["p_charge"][0])
    q_ac = float(v["q_ac"][0])
    q_ac_in = float(v["q_ac_in"][0])
    q_ec = min(max(float(actual["cooling"]) - q_ac, 0.0), p["electric_chiller_capacity"])
    p_ec = q_ec / p["electric_chiller_cop"]
    unserved_cooling = max(float(actual["cooling"]) - q_ac - q_ec, 0.0)
    q_gb = max(float(actual["heating"]) + q_ac_in - float(v["q_chp"][0]), 0.0)
    available_gb = p["gas_boiler_capacity"]
    unserved_heating = max(q_gb - available_gb, 0.0)
    q_gb = min(q_gb, available_gb)
    realized_gas = float(v["g_chp"][0]) + q_gb / p["gas_boiler_efficiency"]
    electric_balance = float(actual["electricity"]) + p_ec + p_charge - float(actual["pv_available"]) - float(actual["wt_available"]) - p_chp - p_discharge
    realized_grid = min(max(electric_balance, 0.0), p["grid_import_capacity"])
    unserved_electricity = max(electric_balance - p["grid_import_capacity"], 0.0)
    renewable_surplus = max(-electric_balance, 0.0)
    curtailment = renewable_surplus
    grid_error = realized_grid - planned_grid
    gas_error = realized_gas - planned_gas
    grid_upward = max(grid_error, 0.0)
    grid_downward = max(-grid_error, 0.0)
    gas_upward = max(gas_error, 0.0)
    gas_downward = max(-gas_error, 0.0)
    realized_cost = (
        realized_grid * p["grid_energy_price"]
        + realized_gas * p["gas_energy_price"]
        + p["unserved_penalty"] * (unserved_electricity + unserved_cooling + unserved_heating)
        + p.get("bess_throughput_cost", 0.0) * (p_charge + p_discharge)
    )
    return RealizedStep(
        planned_grid=planned_grid,
        realized_grid=realized_grid,
        planned_gas=planned_gas,
        realized_gas=realized_gas,
        grid_upward=grid_upward,
        grid_downward=grid_downward,
        gas_upward=gas_upward,
        gas_downward=gas_downward,
        unserved_electricity=unserved_electricity,
        unserved_cooling=unserved_cooling,
        unserved_heating=unserved_heating,
        curtailment=curtailment,
        realized_cost=float(realized_cost),
    )
