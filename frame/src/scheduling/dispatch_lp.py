"""Stage 10.8: deterministic continuous LP for the simulated IES track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .dispatch_schema import VARIABLES


@dataclass(frozen=True)
class DispatchInputs:
    """One four-hour planning window; all arrays use horizon-first order."""

    demand: np.ndarray  # [H, 3] electricity, cooling, heating
    pv_available: np.ndarray  # [H]
    wt_available: np.ndarray  # [H]
    parameters: Mapping[str, object]
    initial_soc: float = 0.5
    previous_chp: float = 0.0


@dataclass(frozen=True)
class DispatchSolveOptions:
    """Optional LP objective and frontier constraints.

    ``None`` is intentionally the legacy weighted-sum call.  The three slack
    caps follow the fixed task order electricity, cooling, heating and are
    cumulative over the planning window.
    """

    objective_mode: str = "weighted_sum"
    carbon_cap: float | None = None
    slack_caps: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.objective_mode not in {"weighted_sum", "operating_cost", "physical_carbon"}:
            raise ValueError("unsupported objective_mode")
        if self.carbon_cap is not None:
            if not np.isfinite(self.carbon_cap) or self.carbon_cap < 0.0:
                raise ValueError("carbon_cap must be finite and non-negative")
        if self.slack_caps is not None:
            if len(self.slack_caps) != 3:
                raise ValueError("slack_caps must follow electricity, cooling, heating")
            if any(not np.isfinite(value) or value < 0.0 for value in self.slack_caps):
                raise ValueError("slack_caps must be finite and non-negative")
        if self.objective_mode == "physical_carbon" and self.slack_caps is None:
            raise ValueError("physical_carbon objective requires per-task slack_caps")


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


_TIME_VARYING_PARAMETERS = {
    "grid_energy_price",
    "gas_energy_price",
    "grid_emission_factor",
    "gas_emission_factor",
    "carbon_price",
    "carbon_price_default",
}


def _horizon_parameter(
    parameters: Mapping[str, object], name: str, horizon: int, default: float = 0.0
) -> np.ndarray:
    """Broadcast a scalar parameter or validate an explicit horizon vector."""

    value = parameters.get(name, default)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(horizon, float(array), dtype=np.float64)
    elif array.shape != (horizon,):
        raise ValueError(f"{name} must be scalar or have shape [H]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _scalar_parameter(parameters: Mapping[str, object], name: str) -> float:
    value = np.asarray(parameters[name], dtype=np.float64)
    if value.ndim != 0 or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


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
    if not np.isfinite(float(inputs.previous_chp)) or float(inputs.previous_chp) < 0.0:
        raise ValueError("previous_chp must be finite and non-negative")
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
    # Physical capacities, efficiencies and penalties are scalar.  Prices and
    # emission factors may additionally be supplied as exact [H] profiles.
    for name in required:
        if name not in _TIME_VARYING_PARAMETERS:
            _scalar_parameter(inputs.parameters, name)
    for name in ("grid_energy_price", "gas_energy_price", "grid_emission_factor", "gas_emission_factor"):
        if name in inputs.parameters:
            _horizon_parameter(inputs.parameters, name, horizon)
    if "carbon_price" in inputs.parameters:
        _horizon_parameter(inputs.parameters, "carbon_price", horizon)
    if "carbon_price_default" in inputs.parameters:
        _horizon_parameter(inputs.parameters, "carbon_price_default", horizon)
    chp_capacity = _scalar_parameter(inputs.parameters, "chp_electric_capacity")
    if float(inputs.previous_chp) > chp_capacity:
        raise ValueError("previous_chp cannot exceed chp_electric_capacity")
    return horizon, demand, pv, wt


def solve_dispatch_lp(inputs: DispatchInputs, options: DispatchSolveOptions | None = None) -> DispatchResult:
    """Solve one deterministic energy-hub window with HiGHS."""

    horizon, demand, pv, wt = _validate_inputs(inputs)
    options = options or DispatchSolveOptions()
    options.validate()
    p = dict(inputs.parameters)
    scalar = lambda name: _scalar_parameter(p, name)
    eta_bess = scalar("bess_roundtrip_efficiency") ** 0.5
    n_per_step = len(VARIABLES)
    total = horizon * n_per_step

    def idx(name: str, t: int) -> int:
        return t * n_per_step + VARIABLES.index(name)

    grid_prices = _horizon_parameter(p, "grid_energy_price", horizon)
    gas_prices = _horizon_parameter(p, "gas_energy_price", horizon)
    grid_emissions = _horizon_parameter(p, "grid_emission_factor", horizon)
    gas_emissions = _horizon_parameter(p, "gas_emission_factor", horizon)
    carbon_prices = _horizon_parameter(
        p, "carbon_price", horizon, default=_scalar_parameter(p, "carbon_price_default") if "carbon_price_default" in p else 0.0
    )
    if options.objective_mode == "weighted_sum":
        grid_unit_cost = grid_prices + carbon_prices * grid_emissions
        gas_unit_cost = gas_prices + carbon_prices * gas_emissions
        storage_unit_cost = scalar("bess_throughput_cost")
        slack_unit_cost = scalar("unserved_penalty")
    elif options.objective_mode == "operating_cost":
        grid_unit_cost = grid_prices
        gas_unit_cost = gas_prices
        storage_unit_cost = scalar("bess_throughput_cost")
        slack_unit_cost = scalar("unserved_penalty")
    else:
        grid_unit_cost = grid_emissions
        gas_unit_cost = gas_emissions
        storage_unit_cost = 0.0
        slack_unit_cost = 0.0

    c = np.zeros(total, dtype=np.float64)
    for t in range(horizon):
        c[idx("grid", t)] = grid_unit_cost[t]
        c[idx("g_chp", t)] = gas_unit_cost[t]
        c[idx("g_gb", t)] = gas_unit_cost[t]
        c[idx("slack_e", t)] = slack_unit_cost
        c[idx("slack_c", t)] = slack_unit_cost
        c[idx("slack_h", t)] = slack_unit_cost
        c[idx("p_charge", t)] = storage_unit_cost
        c[idx("p_discharge", t)] = storage_unit_cost

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
        eq({idx("p_chp", t): 1, idx("g_chp", t): -scalar("chp_electric_efficiency")}, 0.0)
        eq({idx("q_chp", t): 1, idx("g_chp", t): -scalar("chp_heat_efficiency")}, 0.0)
        eq({idx("q_gb", t): 1, idx("g_gb", t): -scalar("gas_boiler_efficiency")}, 0.0)
        eq({idx("q_ec", t): 1, idx("p_ec", t): -scalar("electric_chiller_cop")}, 0.0)
        eq({idx("q_ac", t): 1, idx("q_ac_in", t): -scalar("absorption_chiller_cop")}, 0.0)
        # Battery state equation; terminal SOC equals the initial SOC.
        previous_soc = float(inputs.initial_soc) * scalar("bess_energy_capacity") if t == 0 else None
        if t == 0:
            eq({idx("soc", t): 1, idx("p_charge", t): -eta_bess, idx("p_discharge", t): 1.0 / eta_bess}, previous_soc)
        else:
            eq({idx("soc", t): 1, idx("soc", t - 1): -1, idx("p_charge", t): -eta_bess, idx("p_discharge", t): 1.0 / eta_bess}, 0.0)
        if t == horizon - 1:
            eq({idx("soc", t): 1}, previous_soc if horizon == 1 else float(inputs.initial_soc) * p["bess_energy_capacity"])
        # CHP ramping, measured on electric output, from the supplied prior
        # state for the first step and from the previous planned step after it.
        ramp = p["chp_ramp_fraction"] * p["chp_electric_capacity"]
        if t == 0:
            previous_chp = float(inputs.previous_chp)
            le({idx("p_chp", t): 1}, previous_chp + ramp)
            le({idx("p_chp", t): -1}, ramp - previous_chp)
        else:
            le({idx("p_chp", t): 1, idx("p_chp", t - 1): -1}, ramp)
            le({idx("p_chp", t): -1, idx("p_chp", t - 1): 1}, ramp)

    if options.carbon_cap is not None:
        coefficients: dict[int, float] = {}
        for t in range(horizon):
            coefficients[idx("grid", t)] = grid_emissions[t]
            coefficients[idx("g_chp", t)] = gas_emissions[t]
            coefficients[idx("g_gb", t)] = gas_emissions[t]
        le(coefficients, float(options.carbon_cap))

    if options.slack_caps is not None:
        for variable, cap in zip(("slack_e", "slack_c", "slack_h"), options.slack_caps):
            coefficients = {idx(variable, t): 1.0 for t in range(horizon)}
            le(coefficients, float(cap))

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
