from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.joint_dispatch.rollout import (  # noqa: E402
    ClosedLoopState,
    advance_closed_loop_state,
    apply_first_step_recourse,
)


PARAMETERS = {
    "grid_import_capacity": 10.0,
    "gas_energy_price": 1.0,
    "grid_energy_price": 1.0,
    "carbon_price_default": 0.1,
    "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25,
    "bess_throughput_cost": 0.0,
    "unserved_penalty": 100.0,
    "bess_energy_capacity": 10.0,
}


def _plan():
    plan = torch.zeros(1, 21)
    # grid, pv_use, wt_use, p_chp, p_ec, p_charge, p_discharge
    plan[:, 0] = 2.0
    plan[:, 1] = 4.0
    plan[:, 3] = 2.0
    plan[:, 7] = 1.0
    plan[:, 10] = 1.0
    plan[:, 14] = 0.0
    plan[:, 15] = 0.0
    plan[:, 16] = 5.0
    return plan.requires_grad_()


def test_under_forecast_records_shortage_and_overforecast_reduces_sources_in_order():
    base = _plan()
    low = apply_first_step_recourse(base, torch.tensor([[30.0, 0.0, 0.0]]), torch.tensor([4.0]), torch.tensor([2.0]), PARAMETERS)
    assert low.shortage[0, 0] > 0
    high = apply_first_step_recourse(base, torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([4.0]), torch.tensor([2.0]), PARAMETERS)
    assert high.realized_dispatch[0, 0] == 0
    assert high.realized_dispatch[0, 1] == 0
    assert high.realized_dispatch[0, 3] == 0
    assert high.surplus[0, 0] >= 0


def test_cooling_and_heating_mismatch_is_explicit():
    plan = torch.zeros(1, 21)
    plan[:, 11] = 2.0  # q_ec
    plan[:, 13] = 1.0  # q_ac
    plan[:, 8] = 1.0   # q_chp
    plan[:, 9] = 1.0   # q_gb
    result = apply_first_step_recourse(plan, torch.tensor([[0.0, 5.0, 5.0]]), torch.tensor([0.0]), torch.tensor([0.0]), PARAMETERS)
    assert result.shortage[0, 1] == 2
    assert result.shortage[0, 2] == 3


def test_state_advance_rolls_device_history_and_carries_soc_and_chp():
    state = ClosedLoopState(
        soc=torch.tensor([[0.5]]),
        previous_chp=torch.tensor([[0.0]]),
        device_history=torch.zeros(1, 24, 21),
        device_status=torch.zeros(1, 24, 6),
    )
    plan = torch.zeros(1, 21)
    plan[:, 7] = 3.0
    plan[:, 16] = 6.0
    outcome = apply_first_step_recourse(plan, torch.zeros(1, 3), torch.zeros(1), torch.zeros(1), PARAMETERS)
    next_state = advance_closed_loop_state(state, outcome, bess_energy_capacity=10.0)
    assert next_state.device_history.shape == (1, 24, 21)
    assert next_state.device_history[0, -1, 7] == 3.0
    assert next_state.previous_chp[0, 0] == 3.0
    assert next_state.soc[0, 0] == 0.6
