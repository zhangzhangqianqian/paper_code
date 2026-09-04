from __future__ import annotations

import numpy as np
import pytest
import torch

from src.joint_dispatch.formal_v4_2_rollout import (
    canonical_recourse_once,
    settle_and_advance_v42,
)
from src.joint_dispatch.formal_v4_state import FormalV4ClosedLoopState
from src.scheduling.dispatch_schema import VARIABLES


I = {name: index for index, name in enumerate(VARIABLES)}
PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.2, "unserved_penalty": 100.0,
    "surplus_penalty": 0.1, "chp_ramp_fraction": 0.5,
}


def _state() -> FormalV4ClosedLoopState:
    return FormalV4ClosedLoopState(
        torch.zeros(1, 24, 4), torch.zeros(1, 24, 12), torch.zeros(1, 24, 17), torch.zeros(1, 24, 6),
        torch.full((1, 1), 0.5), torch.zeros(1, 1), np.datetime64("2015-01-01T00:00"), "trajectory-a",
    )


def _plan() -> np.ndarray:
    plan = np.zeros((4, len(VARIABLES)), dtype=np.float64)
    plan[:, I["grid"]] = 5.0
    return plan


def test_rollout_executes_only_first_plan_row_and_carries_settled_state():
    state = _state()
    plan = _plan()
    plan[0, I["p_charge"]] = 3.0
    plan[3, I["p_charge"]] = 10.0
    result = settle_and_advance_v42(state, plan, {"demand": np.array([5.0, 0.0, 0.0]), "renewable": np.zeros(2)}, PARAMETERS)
    expected_soc = (0.5 * 40.0 + np.sqrt(0.9) * 3.0) / 40.0
    assert result.executed_plan_index == 0
    assert result.next_state.initial_soc[0, 0].item() == pytest.approx(expected_soc)
    assert result.next_state.previous_chp[0, 0].item() == pytest.approx(result.settled[I["p_chp"]])


def test_canonical_recourse_returns_first_row_and_not_later_row():
    plan = _plan()
    plan[0, I["grid"]] = 5.0
    plan[3, I["grid"]] = 99.0
    settled, shortage, p_dump, q_dump = canonical_recourse_once(plan, (np.array([5.0, 0.0, 0.0]), np.zeros(2)), PARAMETERS)
    assert settled[I["grid"]] == pytest.approx(5.0)
    assert shortage.shape == (3,)
    assert p_dump >= 0.0 and q_dump >= 0.0


def test_invalid_plan_shape_is_rejected():
    with pytest.raises(ValueError, match="\[4,21\]"):
        canonical_recourse_once(np.zeros((3, len(VARIABLES))), (np.ones(3), np.ones(2)), PARAMETERS)
