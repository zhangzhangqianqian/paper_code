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
    electric_chiller_electricity: float = 0.0
    electric_chiller_cooling: float = 0.0
    gas_boiler_heat: float = 0.0
    absorption_chiller_cooling: float = 0.0
    chp_electricity: float = 0.0
    chp_heat: float = 0.0
    bess_charge: float = 0.0
    bess_discharge: float = 0.0
    soc_after_execution: float = 0.0
    actual_electricity: float = 0.0
    actual_cooling: float = 0.0
    actual_heating: float = 0.0
    actual_pv: float = 0.0
    actual_wt: float = 0.0


def evaluate_planned_first_step(
    plan: DispatchResult,
    parameters: Mapping[str, float],
) -> Mapping[str, float]:
    """Evaluate only the first hour of a multi-hour planning result."""

    if not plan.success:
        raise ValueError("Cannot evaluate an unsuccessful dispatch plan")
    values = plan.values
    p = {key: float(value) for key, value in parameters.items()}
    carbon_price = p.get("carbon_price", p.get("carbon_price_default", 0.0))
    grid = float(values["grid"][0])
    gas = float(values["g_chp"][0] + values["g_gb"][0])
    slack = float(values["slack_e"][0] + values["slack_c"][0] + values["slack_h"][0])
    charge = float(values["p_charge"][0])
    discharge = float(values["p_discharge"][0])
    grid_factor = p.get("grid_emission_factor", 0.0)
    gas_factor = p.get("gas_emission_factor", 0.0)
    return {
        "planned_cost_first_step": float(
            grid * (p["grid_energy_price"] + carbon_price * grid_factor)
            + gas * (p["gas_energy_price"] + carbon_price * gas_factor)
            + p["unserved_penalty"] * slack
            + p.get("bess_throughput_cost", 0.0) * (charge + discharge)
        ),
        "planned_carbon_first_step": float(grid * grid_factor + gas * gas_factor),
        "planned_curtailment_first_step": float(values["pv_curt"][0] + values["wt_curt"][0]),
        "planned_grid_first_step": grid,
        "planned_gas_first_step": gas,
    }


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
    carbon_price = p.get("carbon_price", p.get("carbon_price_default", 0.0))
    realized_cost = (
        realized_grid * (p["grid_energy_price"] + carbon_price * p.get("grid_emission_factor", 0.0))
        + realized_gas * (p["gas_energy_price"] + carbon_price * p.get("gas_emission_factor", 0.0))
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
        electric_chiller_electricity=float(p_ec),
        electric_chiller_cooling=float(q_ec),
        gas_boiler_heat=float(q_gb),
        absorption_chiller_cooling=float(q_ac),
        chp_electricity=float(p_chp),
        chp_heat=float(v["q_chp"][0]),
        bess_charge=float(p_charge),
        bess_discharge=float(p_discharge),
        soc_after_execution=float(v["soc"][0]),
        actual_electricity=float(actual["electricity"]),
        actual_cooling=float(actual["cooling"]),
        actual_heating=float(actual["heating"]),
        actual_pv=float(actual["pv_available"]),
        actual_wt=float(actual["wt_available"]),
    )
