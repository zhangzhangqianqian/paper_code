from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from src.scheduling.heat_pump.dispatch_lp import (
    HeatPumpDispatchInputs,
    HeatPumpDispatchSolveOptions,
    solve_heat_pump_dispatch_lp,
)
from src.scheduling.heat_pump.parameters import HeatPumpParameters


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.000001,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "unserved_penalty": 100.0, "chp_ramp_fraction": 0.5,
}


def _inputs(capacity: float = 20.0) -> HeatPumpDispatchInputs:
    return HeatPumpDispatchInputs(
        demand=np.array([[10.0, 5.0, 12.0], [12.0, 5.0, 12.0], [10.0, 5.0, 12.0], [8.0, 4.0, 10.0]]),
        pv_available=np.array([2.0, 3.0, 1.0, 0.0]),
        wt_available=np.ones(4),
        parameters=PARAMETERS,
        heat_pump=HeatPumpParameters(cop=3.0, heat_capacity=capacity),
    )


def test_heat_pump_conversion_and_balances():
    result = solve_heat_pump_dispatch_lp(_inputs())
    assert result.success, result.message
    np.testing.assert_allclose(result.values["q_hp"], 3.0 * result.values["p_hp"], atol=1e-8)
    assert max(result.balance_residuals.values()) <= 1e-8
    assert result.operating_cost >= 0.0


def test_zero_heat_pump_capacity_matches_legacy_lp():
    hp = solve_heat_pump_dispatch_lp(_inputs(0.0))
    old_inputs = DispatchInputs(_inputs(0.0).demand, _inputs(0.0).pv_available, _inputs(0.0).wt_available, PARAMETERS)
    old = solve_dispatch_lp(old_inputs)
    assert hp.success and old.success
    assert hp.objective == pytest.approx(old.objective, abs=1e-7)
    for name in PARAMETERS:
        assert name in PARAMETERS
    for name, values in old.values.items():
        np.testing.assert_allclose(hp.values[name], values, atol=1e-7)
    np.testing.assert_allclose(hp.values["p_hp"], 0.0)
    np.testing.assert_allclose(hp.values["q_hp"], 0.0)


def test_operating_cost_cap_is_enforced_without_carbon_price():
    inputs = _inputs()
    economic = solve_heat_pump_dispatch_lp(inputs, HeatPumpDispatchSolveOptions(objective_mode="operating_cost"))
    assert economic.success
    constrained = solve_heat_pump_dispatch_lp(
        inputs,
        HeatPumpDispatchSolveOptions(objective_mode="physical_carbon", operating_cost_cap=economic.operating_cost + 1e-7, slack_caps=(100.0, 100.0, 100.0)),
    )
    assert constrained.success, constrained.message
    assert constrained.operating_cost <= economic.operating_cost + 2e-6


def test_physical_carbon_objective_excludes_heat_pump_om_cost():
    low_om_inputs = _inputs()
    high_om_inputs = HeatPumpDispatchInputs(
        demand=low_om_inputs.demand,
        pv_available=low_om_inputs.pv_available,
        wt_available=low_om_inputs.wt_available,
        parameters=low_om_inputs.parameters,
        heat_pump=HeatPumpParameters(cop=3.0, heat_capacity=20.0, variable_om_cost=0.4),
        initial_soc=low_om_inputs.initial_soc,
    )
    options = HeatPumpDispatchSolveOptions(
        objective_mode="physical_carbon",
        slack_caps=(100.0, 100.0, 100.0),
    )
    low_om = solve_heat_pump_dispatch_lp(low_om_inputs, options)
    high_om = solve_heat_pump_dispatch_lp(high_om_inputs, options)
    assert low_om.success and high_om.success
    assert high_om.objective == pytest.approx(low_om.objective, abs=1e-8)
    assert high_om.physical_carbon == pytest.approx(low_om.physical_carbon, abs=1e-8)
    assert high_om.operating_cost >= low_om.operating_cost


def test_heat_pump_options_reject_nonfinite_cost_cap():
    with pytest.raises(ValueError):
        HeatPumpDispatchSolveOptions(operating_cost_cap=float("nan")).validate()
