"""Stage 10.8: deterministic continuous LP for the simulated IES track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class DispatchInputs:
    """One four-hour planning window; all arrays use horizon-first order."""

    demand: np.ndarray  # [H, 3] electricity, cooling, heating
    pv_available: np.ndarray  # [H]
    wt_available: np.ndarray  # [H]
    parameters: Mapping[str, float]
    initial_soc: float = 0.5


@dataclass(frozen=True)
class DispatchResult:
    status: str
    message: str
    objective: float
    values: Mapping[str, np.ndarray]
    balance_residuals: Mapping[str, float]
    simultaneous_charge_discharge: float

    @property
    def success(self) -> bool:
        return self.status == "optimal"


VARIABLES = (
    "grid",
    "pv_use",
    "pv_curt",
    "wt_use",
    "wt_curt",
    "g_chp",
    "g_gb",
    "p_chp",
    "q_chp",
    "q_gb",
    "p_ec",
    "q_ec",
    "q_ac_in",
    "q_ac",
    "p_charge",
    "p_discharge",
    "soc",
    "slack_e",
    "slack_c",
    "slack_h",
    "q_dump",
)


def _validate_inputs(inputs: DispatchInputs) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    demand = np.asarray(inputs.demand, dtype=np.float64)
    pv = np.asarray(inputs.pv_available, dtype=np.float64)
    wt = np.asarray(inputs.wt_available, dtype=np.float64)
    if demand.ndim != 2 or demand.shape[1] != 3:
        raise ValueError("demand must have shape [H,3]")
    horizon = demand.shape[0]
    if horizon <= 0 or pv.shape != (horizon,) or wt.shape != (horizon,):
        raise ValueError("renewable arrays must have shape [H]")
    if not np.isfinite(demand).all() or not np.isfinite(pv).all() or not np.isfinite(wt).all():
        raise ValueError("dispatch inputs must be finite")
    if (demand < 0).any() or (pv < 0).any() or (wt < 0).any():
        raise ValueError("dispatch inputs must be non-negative")
    if not 0.0 <= float(inputs.initial_soc) <= 1.0:
        raise ValueError("initial_soc must be in [0,1]")
    required = (
        "grid_import_capacity", "chp_electric_capacity", "chp_heat_capacity",
        "gas_boiler_capacity", "electric_chiller_capacity", "absorption_chiller_capacity",
        "bess_power_capacity", "bess_energy_capacity", "chp_electric_efficiency",
        "chp_heat_efficiency", "gas_boiler_efficiency", "electric_chiller_cop",
        "absorption_chiller_cop", "bess_roundtrip_efficiency", "bess_throughput_cost", "grid_energy_price",
        "gas_energy_price", "unserved_penalty", "chp_ramp_fraction",
    )
    missing = [name for name in required if name not in inputs.parameters]
    if missing:
        raise ValueError(f"Missing dispatch parameters: {missing}")
    return horizon, demand, pv, wt


def solve_dispatch_lp(inputs: DispatchInputs) -> DispatchResult:
    """Solve one deterministic energy-hub window with HiGHS."""

    horizon, demand, pv, wt = _validate_inputs(inputs)
    p = {key: float(value) for key, value in inputs.parameters.items()}
    eta_bess = p["bess_roundtrip_efficiency"] ** 0.5
    n_per_step = len(VARIABLES)
    total = horizon * n_per_step

    def idx(name: str, t: int) -> int:
        return t * n_per_step + VARIABLES.index(name)

    c = np.zeros(total, dtype=np.float64)
    for t in range(horizon):
        carbon_price = p.get("carbon_price", p.get("carbon_price_default", 0.0))
        grid_carbon = carbon_price * p.get("grid_emission_factor", 0.0)
        gas_carbon = carbon_price * p.get("gas_emission_factor", 0.0)
        c[idx("grid", t)] = p["grid_energy_price"] + grid_carbon
        c[idx("g_chp", t)] = p["gas_energy_price"] + gas_carbon
        c[idx("g_gb", t)] = p["gas_energy_price"] + gas_carbon
        c[idx("slack_e", t)] = p["unserved_penalty"]
        c[idx("slack_c", t)] = p["unserved_penalty"]
        c[idx("slack_h", t)] = p["unserved_penalty"]
        c[idx("p_charge", t)] = p["bess_throughput_cost"]
        c[idx("p_discharge", t)] = p["bess_throughput_cost"]

    equalities: list[tuple[dict[int, float], float]] = []
    inequalities: list[tuple[dict[int, float], float]] = []

    def eq(coefficients: dict[int, float], rhs: float) -> None:
        equalities.append((coefficients, rhs))

    def le(coefficients: dict[int, float], rhs: float) -> None:
        inequalities.append((coefficients, rhs))

    for t in range(horizon):
        # Electricity, cooling and heat balances.
        eq({idx("grid", t): 1, idx("pv_use", t): 1, idx("wt_use", t): 1, idx("p_chp", t): 1, idx("p_discharge", t): 1, idx("slack_e", t): 1, idx("p_ec", t): -1, idx("p_charge", t): -1}, demand[t, 0])
        eq({idx("q_ec", t): 1, idx("q_ac", t): 1, idx("slack_c", t): 1}, demand[t, 1])
        eq({idx("q_chp", t): 1, idx("q_gb", t): 1, idx("slack_h", t): 1, idx("q_ac_in", t): -1, idx("q_dump", t): -1}, demand[t, 2])
        # Renewable availability is split into use and curtailment.
        eq({idx("pv_use", t): 1, idx("pv_curt", t): 1}, pv[t])
        eq({idx("wt_use", t): 1, idx("wt_curt", t): 1}, wt[t])
        # Conversion equations.
        eq({idx("p_chp", t): 1, idx("g_chp", t): -p["chp_electric_efficiency"]}, 0.0)
        eq({idx("q_chp", t): 1, idx("g_chp", t): -p["chp_heat_efficiency"]}, 0.0)
        eq({idx("q_gb", t): 1, idx("g_gb", t): -p["gas_boiler_efficiency"]}, 0.0)
        eq({idx("q_ec", t): 1, idx("p_ec", t): -p["electric_chiller_cop"]}, 0.0)
        eq({idx("q_ac", t): 1, idx("q_ac_in", t): -p["absorption_chiller_cop"]}, 0.0)
        # Battery state equation; terminal SOC equals the initial SOC.
        previous_soc = float(inputs.initial_soc) * p["bess_energy_capacity"] if t == 0 else None
        if t == 0:
            eq({idx("soc", t): 1, idx("p_charge", t): -eta_bess, idx("p_discharge", t): 1.0 / eta_bess}, previous_soc)
        else:
            eq({idx("soc", t): 1, idx("soc", t - 1): -1, idx("p_charge", t): -eta_bess, idx("p_discharge", t): 1.0 / eta_bess}, 0.0)
        if t == horizon - 1:
            eq({idx("soc", t): 1}, previous_soc if horizon == 1 else float(inputs.initial_soc) * p["bess_energy_capacity"])
        # CHP ramping, measured on electric output, with zero prior output.
        ramp = p["chp_ramp_fraction"] * p["chp_electric_capacity"]
        if t == 0:
            le({idx("p_chp", t): 1}, ramp)
            le({idx("p_chp", t): -1}, 0.0)
        else:
            le({idx("p_chp", t): 1, idx("p_chp", t - 1): -1}, ramp)
            le({idx("p_chp", t): -1, idx("p_chp", t - 1): 1}, ramp)

    # Bounds are explicit and make nonnegative flows auditable.
    bounds: list[tuple[float, float | None]] = []
    for t in range(horizon):
        bounds.extend([
            (0.0, p["grid_import_capacity"]),
            (0.0, None), (0.0, None), (0.0, None), (0.0, None),
            (0.0, p["chp_electric_capacity"] / p["chp_electric_efficiency"]),
            (0.0, p["gas_boiler_capacity"] / p["gas_boiler_efficiency"]),
            (0.0, p["chp_electric_capacity"]), (0.0, p["chp_heat_capacity"]),
            (0.0, p["gas_boiler_capacity"]),
            (0.0, p["electric_chiller_capacity"] / p["electric_chiller_cop"]),
            (0.0, p["electric_chiller_capacity"]),
            (0.0, p["absorption_chiller_capacity"] / p["absorption_chiller_cop"]),
            (0.0, p["absorption_chiller_capacity"]),
            (0.0, p["bess_power_capacity"]), (0.0, p["bess_power_capacity"]),
            (0.0, p["bess_energy_capacity"]),
            (0.0, None), (0.0, None), (0.0, None), (0.0, None),
        ])
    if len(bounds) != total:
        raise AssertionError("Dispatch variable bound count mismatch")

    a_eq = np.zeros((len(equalities), total), dtype=np.float64)
    b_eq = np.zeros(len(equalities), dtype=np.float64)
    for row, (coefficients, rhs) in enumerate(equalities):
        for column, value in coefficients.items():
            a_eq[row, column] = value
        b_eq[row] = rhs
    a_ub = np.zeros((len(inequalities), total), dtype=np.float64)
    b_ub = np.zeros(len(inequalities), dtype=np.float64)
    for row, (coefficients, rhs) in enumerate(inequalities):
        for column, value in coefficients.items():
            a_ub[row, column] = value
        b_ub[row] = rhs

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        return DispatchResult(result.status.__str__(), result.message, float("nan"), {}, {}, float("nan"))
    values = {
        name: result.x[[idx(name, t) for t in range(horizon)]].copy()
        for name in VARIABLES
    }
    residuals = {
        "electricity": float(np.max(np.abs(values["grid"] + values["pv_use"] + values["wt_use"] + values["p_chp"] + values["p_discharge"] + values["slack_e"] - values["p_ec"] - values["p_charge"] - demand[:, 0]))),
        "cooling": float(np.max(np.abs(values["q_ec"] + values["q_ac"] + values["slack_c"] - demand[:, 1]))),
        "heating": float(np.max(np.abs(values["q_chp"] + values["q_gb"] + values["slack_h"] - values["q_ac_in"] - values["q_dump"] - demand[:, 2]))),
    }
    simultaneous = float(np.max(np.minimum(values["p_charge"], values["p_discharge"])))
    return DispatchResult("optimal", result.message, float(result.fun), values, residuals, simultaneous)
