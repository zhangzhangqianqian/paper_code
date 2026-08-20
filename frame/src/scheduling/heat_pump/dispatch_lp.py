"""Continuous LP teacher for the isolated simulated heat-pump S track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.optimize import linprog

from ..dispatch_lp import DispatchInputs, DispatchSolveOptions, solve_dispatch_lp
from .parameters import HeatPumpParameters
from .schema import HEAT_PUMP_INDEX, HEAT_PUMP_VARIABLES


@dataclass(frozen=True)
class HeatPumpDispatchInputs:
    demand: np.ndarray
    pv_available: np.ndarray
    wt_available: np.ndarray
    parameters: Mapping[str, object]
    heat_pump: HeatPumpParameters
    initial_soc: float = 0.5


@dataclass(frozen=True)
class HeatPumpDispatchSolveOptions:
    objective_mode: str = "weighted_sum"
    carbon_cap: float | None = None
    operating_cost_cap: float | None = None
    slack_caps: tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.objective_mode not in {"weighted_sum", "operating_cost", "physical_carbon"}:
            raise ValueError("unsupported objective_mode")
        for name, value in (("carbon_cap", self.carbon_cap), ("operating_cost_cap", self.operating_cost_cap)):
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.slack_caps is not None:
            if len(self.slack_caps) != 3 or any(not np.isfinite(v) or v < 0.0 for v in self.slack_caps):
                raise ValueError("slack_caps must be finite and follow electricity, cooling, heating")
        if self.objective_mode == "physical_carbon" and self.slack_caps is None:
            raise ValueError("physical_carbon objective requires per-task slack_caps")


@dataclass(frozen=True)
class HeatPumpDispatchResult:
    status: str
    message: str
    objective: float
    values: Mapping[str, np.ndarray]
    balance_residuals: Mapping[str, float]
    simultaneous_charge_discharge: float
    operating_cost: float
    physical_carbon: float

    @property
    def success(self) -> bool:
        return self.status == "optimal"


_TIME_VARYING = {"grid_energy_price", "gas_energy_price", "grid_emission_factor", "gas_emission_factor", "carbon_price", "carbon_price_default"}
_REQUIRED = (
    "grid_import_capacity", "chp_electric_capacity", "chp_heat_capacity", "gas_boiler_capacity",
    "electric_chiller_capacity", "absorption_chiller_capacity", "bess_power_capacity",
    "bess_energy_capacity", "chp_electric_efficiency", "chp_heat_efficiency", "gas_boiler_efficiency",
    "electric_chiller_cop", "absorption_chiller_cop", "bess_roundtrip_efficiency", "bess_throughput_cost",
    "grid_energy_price", "gas_energy_price", "unserved_penalty", "chp_ramp_fraction",
)


def _scalar(parameters: Mapping[str, object], name: str) -> float:
    value = np.asarray(parameters[name], dtype=np.float64)
    if value.ndim != 0 or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def _horizon(parameters: Mapping[str, object], name: str, horizon: int, default: float = 0.0) -> np.ndarray:
    value = np.asarray(parameters.get(name, default), dtype=np.float64)
    if value.ndim == 0:
        value = np.full(horizon, float(value), dtype=np.float64)
    if value.shape != (horizon,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be scalar or have shape [H] and be finite")
    return value


def _validate(inputs: HeatPumpDispatchInputs) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    demand = np.asarray(inputs.demand, dtype=np.float64)
    pv = np.asarray(inputs.pv_available, dtype=np.float64)
    wt = np.asarray(inputs.wt_available, dtype=np.float64)
    if demand.ndim != 2 or demand.shape[1] != 3 or demand.shape[0] <= 0:
        raise ValueError("demand must have shape [H,3]")
    horizon = demand.shape[0]
    if pv.shape != (horizon,) or wt.shape != (horizon,):
        raise ValueError("renewable arrays must have shape [H]")
    if not np.isfinite(demand).all() or not np.isfinite(pv).all() or not np.isfinite(wt).all():
        raise ValueError("dispatch inputs must be finite")
    if (demand < 0).any() or (pv < 0).any() or (wt < 0).any():
        raise ValueError("dispatch inputs must be non-negative")
    if not 0.0 <= float(inputs.initial_soc) <= 1.0:
        raise ValueError("initial_soc must be in [0,1]")
    inputs.heat_pump.validate()
    missing = [name for name in _REQUIRED if name not in inputs.parameters]
    if missing:
        raise ValueError(f"Missing dispatch parameters: {missing}")
    for name in _REQUIRED:
        if name not in _TIME_VARYING:
            _scalar(inputs.parameters, name)
    for name in ("grid_energy_price", "gas_energy_price", "grid_emission_factor", "gas_emission_factor", "carbon_price", "carbon_price_default"):
        if name in inputs.parameters:
            _horizon(inputs.parameters, name, horizon)
    return horizon, demand, pv, wt


def _empty_result(status: str, message: str) -> HeatPumpDispatchResult:
    return HeatPumpDispatchResult(status, message, float("nan"), {}, {}, float("nan"), float("nan"), float("nan"))


def _legacy_zero_capacity(inputs: HeatPumpDispatchInputs) -> HeatPumpDispatchResult:
    legacy = solve_dispatch_lp(
        DispatchInputs(inputs.demand, inputs.pv_available, inputs.wt_available, inputs.parameters, inputs.initial_soc),
        DispatchSolveOptions(objective_mode="weighted_sum"),
    )
    if not legacy.success:
        return _empty_result(legacy.status, legacy.message)
    values = dict(legacy.values)
    horizon = inputs.demand.shape[0]
    values["p_hp"] = np.zeros(horizon, dtype=np.float64)
    values["q_hp"] = np.zeros(horizon, dtype=np.float64)
    cost = physical_operating_cost(values, inputs.parameters, inputs.heat_pump)
    carbon = physical_carbon_emissions(values, inputs.parameters)
    return HeatPumpDispatchResult(legacy.status, legacy.message, legacy.objective, values, legacy.balance_residuals, legacy.simultaneous_charge_discharge, cost, carbon)


def physical_operating_cost(values: Mapping[str, np.ndarray], parameters: Mapping[str, object], heat_pump: HeatPumpParameters) -> float:
    horizon = len(np.asarray(values["grid"]))
    grid_price = _horizon(parameters, "grid_energy_price", horizon)
    gas_price = _horizon(parameters, "gas_energy_price", horizon)
    return float(
        np.dot(values["grid"], grid_price)
        + np.dot(values["g_chp"] + values["g_gb"], gas_price)
        + _scalar(parameters, "bess_throughput_cost") * np.sum(values["p_charge"] + values["p_discharge"])
        + _scalar(parameters, "unserved_penalty") * np.sum(values["slack_e"] + values["slack_c"] + values["slack_h"])
        + heat_pump.variable_om_cost * np.sum(values["q_hp"])
    )


def physical_carbon_emissions(values: Mapping[str, np.ndarray], parameters: Mapping[str, object]) -> float:
    horizon = len(np.asarray(values["grid"]))
    grid_factor = _horizon(parameters, "grid_emission_factor", horizon)
    gas_factor = _horizon(parameters, "gas_emission_factor", horizon)
    return float(np.dot(values["grid"], grid_factor) + np.dot(values["g_chp"] + values["g_gb"], gas_factor))


def solve_heat_pump_dispatch_lp(inputs: HeatPumpDispatchInputs, options: HeatPumpDispatchSolveOptions | None = None) -> HeatPumpDispatchResult:
    """Solve one four-hour (or arbitrary horizon) heat-pump dispatch window."""

    horizon, demand, pv, wt = _validate(inputs)
    options = options or HeatPumpDispatchSolveOptions()
    options.validate()
    if inputs.heat_pump.heat_capacity == 0.0:
        if options.objective_mode != "weighted_sum" or options.carbon_cap is not None or options.operating_cost_cap is not None or options.slack_caps is not None:
            raise ValueError("zero-capacity equivalence is defined for the unconstrained weighted-sum LP")
        return _legacy_zero_capacity(inputs)

    p = inputs.parameters
    eta_bess = _scalar(p, "bess_roundtrip_efficiency") ** 0.5
    n = len(HEAT_PUMP_VARIABLES)
    total = horizon * n
    idx = lambda name, t: t * n + HEAT_PUMP_INDEX[name]
    grid_prices = _horizon(p, "grid_energy_price", horizon)
    gas_prices = _horizon(p, "gas_energy_price", horizon)
    grid_emissions = _horizon(p, "grid_emission_factor", horizon)
    gas_emissions = _horizon(p, "gas_emission_factor", horizon)
    carbon_prices = _horizon(p, "carbon_price", horizon, _scalar(p, "carbon_price_default") if "carbon_price_default" in p else 0.0)
    if options.objective_mode == "weighted_sum":
        grid_cost, gas_cost = grid_prices + carbon_prices * grid_emissions, gas_prices + carbon_prices * gas_emissions
        storage_cost, slack_cost = _scalar(p, "bess_throughput_cost"), _scalar(p, "unserved_penalty")
    elif options.objective_mode == "operating_cost":
        grid_cost, gas_cost = grid_prices, gas_prices
        storage_cost, slack_cost = _scalar(p, "bess_throughput_cost"), _scalar(p, "unserved_penalty")
    else:
        grid_cost, gas_cost = grid_emissions, gas_emissions
        storage_cost, slack_cost = 0.0, 0.0

    c = np.zeros(total, dtype=np.float64)
    for t in range(horizon):
        c[idx("grid", t)] = grid_cost[t]
        c[idx("g_chp", t)] = gas_cost[t]
        c[idx("g_gb", t)] = gas_cost[t]
        c[idx("p_hp", t)] = inputs.heat_pump.variable_om_cost * inputs.heat_pump.cop
        for name in ("slack_e", "slack_c", "slack_h"):
            c[idx(name, t)] = slack_cost
        for name in ("p_charge", "p_discharge"):
            c[idx(name, t)] = storage_cost

    equalities: list[tuple[dict[int, float], float]] = []
    inequalities: list[tuple[dict[int, float], float]] = []
    eq = lambda coefficients, rhs: equalities.append((coefficients, rhs))
    le = lambda coefficients, rhs: inequalities.append((coefficients, rhs))
    for t in range(horizon):
        eq({idx("grid", t): 1, idx("pv_use", t): 1, idx("wt_use", t): 1, idx("p_chp", t): 1, idx("p_discharge", t): 1, idx("slack_e", t): 1, idx("p_ec", t): -1, idx("p_hp", t): -1, idx("p_charge", t): -1}, demand[t, 0])
        eq({idx("q_ec", t): 1, idx("q_ac", t): 1, idx("slack_c", t): 1}, demand[t, 1])
        eq({idx("q_chp", t): 1, idx("q_gb", t): 1, idx("q_hp", t): 1, idx("slack_h", t): 1, idx("q_ac_in", t): -1, idx("q_dump", t): -1}, demand[t, 2])
        eq({idx("pv_use", t): 1, idx("pv_curt", t): 1}, pv[t])
        eq({idx("wt_use", t): 1, idx("wt_curt", t): 1}, wt[t])
        eq({idx("p_chp", t): 1, idx("g_chp", t): -_scalar(p, "chp_electric_efficiency")}, 0.0)
        eq({idx("q_chp", t): 1, idx("g_chp", t): -_scalar(p, "chp_heat_efficiency")}, 0.0)
        eq({idx("q_gb", t): 1, idx("g_gb", t): -_scalar(p, "gas_boiler_efficiency")}, 0.0)
        eq({idx("q_ec", t): 1, idx("p_ec", t): -_scalar(p, "electric_chiller_cop")}, 0.0)
        eq({idx("q_ac", t): 1, idx("q_ac_in", t): -_scalar(p, "absorption_chiller_cop")}, 0.0)
        eq({idx("q_hp", t): 1, idx("p_hp", t): -inputs.heat_pump.cop}, 0.0)
        initial_energy = float(inputs.initial_soc) * _scalar(p, "bess_energy_capacity")
        if t == 0:
            eq({idx("soc", t): 1, idx("p_charge", t): -eta_bess, idx("p_discharge", t): 1.0 / eta_bess}, initial_energy)
        else:
            eq({idx("soc", t): 1, idx("soc", t - 1): -1, idx("p_charge", t): -eta_bess, idx("p_discharge", t): 1.0 / eta_bess}, 0.0)
        if t == horizon - 1:
            eq({idx("soc", t): 1}, initial_energy)
        ramp = _scalar(p, "chp_ramp_fraction") * _scalar(p, "chp_electric_capacity")
        if t == 0:
            le({idx("p_chp", t): 1}, ramp)
            le({idx("p_chp", t): -1}, 0.0)
        else:
            le({idx("p_chp", t): 1, idx("p_chp", t - 1): -1}, ramp)
            le({idx("p_chp", t): -1, idx("p_chp", t - 1): 1}, ramp)

    if options.carbon_cap is not None:
        le({idx("grid", t): grid_emissions[t] for t in range(horizon)} | {idx("g_chp", t): gas_emissions[t] for t in range(horizon)} | {idx("g_gb", t): gas_emissions[t] for t in range(horizon)}, float(options.carbon_cap))
    if options.slack_caps is not None:
        for name, cap in zip(("slack_e", "slack_c", "slack_h"), options.slack_caps):
            le({idx(name, t): 1.0 for t in range(horizon)}, float(cap))
    if options.operating_cost_cap is not None:
        coefficients: dict[int, float] = {}
        for t in range(horizon):
            coefficients[idx("grid", t)] = grid_prices[t]
            coefficients[idx("g_chp", t)] = gas_prices[t]
            coefficients[idx("g_gb", t)] = gas_prices[t]
            coefficients[idx("p_charge", t)] = _scalar(p, "bess_throughput_cost")
            coefficients[idx("p_discharge", t)] = _scalar(p, "bess_throughput_cost")
            coefficients[idx("p_hp", t)] = inputs.heat_pump.variable_om_cost * inputs.heat_pump.cop
            for name in ("slack_e", "slack_c", "slack_h"):
                coefficients[idx(name, t)] = _scalar(p, "unserved_penalty")
        le(coefficients, float(options.operating_cost_cap))

    bounds_by_name = {
        "grid": (0.0, _scalar(p, "grid_import_capacity")),
        "pv_use": (0.0, None), "pv_curt": (0.0, None), "wt_use": (0.0, None), "wt_curt": (0.0, None),
        "g_chp": (0.0, _scalar(p, "chp_electric_capacity") / _scalar(p, "chp_electric_efficiency")),
        "g_gb": (0.0, _scalar(p, "gas_boiler_capacity") / _scalar(p, "gas_boiler_efficiency")),
        "p_chp": (0.0, _scalar(p, "chp_electric_capacity")), "q_chp": (0.0, _scalar(p, "chp_heat_capacity")),
        "q_gb": (0.0, _scalar(p, "gas_boiler_capacity")),
        "p_ec": (0.0, _scalar(p, "electric_chiller_capacity") / _scalar(p, "electric_chiller_cop")),
        "q_ec": (0.0, _scalar(p, "electric_chiller_capacity")),
        "q_ac_in": (0.0, _scalar(p, "absorption_chiller_capacity") / _scalar(p, "absorption_chiller_cop")),
        "q_ac": (0.0, _scalar(p, "absorption_chiller_capacity")),
        "p_charge": (0.0, _scalar(p, "bess_power_capacity")), "p_discharge": (0.0, _scalar(p, "bess_power_capacity")),
        "soc": (0.0, _scalar(p, "bess_energy_capacity")), "slack_e": (0.0, None), "slack_c": (0.0, None), "slack_h": (0.0, None), "q_dump": (0.0, None),
        "p_hp": (0.0, inputs.heat_pump.heat_capacity / inputs.heat_pump.cop), "q_hp": (0.0, inputs.heat_pump.heat_capacity),
    }
    if set(bounds_by_name) != set(HEAT_PUMP_VARIABLES):
        raise AssertionError("heat-pump bounds must cover each named variable exactly once")
    bounds = [bounds_by_name[name] for t in range(horizon) for name in HEAT_PUMP_VARIABLES]

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
        return _empty_result(str(result.status), result.message)
    values = {name: result.x[[idx(name, t) for t in range(horizon)]].copy() for name in HEAT_PUMP_VARIABLES}
    residuals = {
        "electricity": float(np.max(np.abs(values["grid"] + values["pv_use"] + values["wt_use"] + values["p_chp"] + values["p_discharge"] + values["slack_e"] - values["p_ec"] - values["p_hp"] - values["p_charge"] - demand[:, 0]))),
        "cooling": float(np.max(np.abs(values["q_ec"] + values["q_ac"] + values["slack_c"] - demand[:, 1]))),
        "heating": float(np.max(np.abs(values["q_chp"] + values["q_gb"] + values["q_hp"] + values["slack_h"] - values["q_ac_in"] - values["q_dump"] - demand[:, 2]))),
    }
    operating_cost = physical_operating_cost(values, p, inputs.heat_pump)
    carbon = physical_carbon_emissions(values, p)
    return HeatPumpDispatchResult("optimal", result.message, float(result.fun), values, residuals, float(np.max(np.minimum(values["p_charge"], values["p_discharge"]))), operating_cost, carbon)


__all__ = [
    "HeatPumpDispatchInputs", "HeatPumpDispatchResult", "HeatPumpDispatchSolveOptions",
    "physical_carbon_emissions", "physical_operating_cost", "solve_heat_pump_dispatch_lp",
]
