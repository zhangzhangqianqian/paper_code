from __future__ import annotations

import numpy as np
import torch

from src.joint_dispatch.contract import DISPATCH_ORDER, FORECAST_TASK_WEIGHTS
from src.joint_dispatch.formal_v4_objective import (
    STEP_WEIGHTS,
    FormalV4FourHourSettlement,
    clip_formal_v4_gradients,
    fit_training_objective_scale,
    formal_v4_curriculum_weights,
    formal_v4_joint_loss,
    settle_formal_v4_four_hour,
)
from src.joint_dispatch.formal_v4_recourse import settle_first_step_v4
from src.joint_dispatch.model import JointForecastDispatchModel
from src.scheduling.dispatch_schema import VARIABLES


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.2, "unserved_penalty": 100.0,
    "surplus_penalty": 0.0, "chp_ramp_fraction": 0.5, "pv_capacity": 20.0,
}


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(2026)
    return {
        "load_history": torch.rand(batch_size, 24, 4),
        "exog_history": torch.rand(batch_size, 24, 12),
        "device_history": torch.rand(batch_size, 24, 21),
        "device_status": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.cat((torch.rand(batch_size, 4, 5), torch.full((batch_size, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.zeros(batch_size, 1),
        "target_normalized": torch.rand(batch_size, 4, 4),
        "target_physical": torch.rand(batch_size, 4, 4),
        "realized_renewables": torch.rand(batch_size, 4, 2),
        "initial_soc": torch.full((batch_size, 1), 0.5),
        "teacher_dispatch": torch.rand(batch_size, 4, 21),
    }


def test_scale_uses_one_train_only_constant():
    scale = fit_training_objective_scale(np.array([0.1, 10.0, 20.0, 30.0]))
    assert scale.c_ref > 1.0
    assert scale.source_split == "train"
    assert scale.sample_count == 4


def test_scale_uses_fixed_execution_step_weights_for_matrix_input():
    values = np.tile(np.arange(1.0, 5.0), (3, 1))
    scale = fit_training_objective_scale(values)
    expected = float(np.dot(values[0], np.asarray(STEP_WEIGHTS)))
    assert scale.c_ref == max(expected, 1.0)
    assert tuple(STEP_WEIGHTS) == (0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
    assert abs(sum(STEP_WEIGHTS) - 1.0) < 1e-12


def test_near_zero_oracle_does_not_change_v4_gradient():
    model = JointForecastDispatchModel.for_test()
    batch = _batch()
    labels = {key: batch[key] for key in (
        "target_normalized", "target_physical", "realized_renewables", "initial_soc",
        "previous_chp", "teacher_dispatch",
    )}
    output_a = model(**{key: batch[key] for key in (
        "load_history", "exog_history", "device_history", "device_status",
        "scheduler_context", "previous_chp",
    )})
    loss_a = formal_v4_joint_loss(output_a, **labels, parameters=PARAMETERS, c_ref=1000.0,
                                  oracle_diagnostic=torch.zeros(2))
    grad_a = torch.autograd.grad(loss_a.total, tuple(model.parameters()), retain_graph=True, allow_unused=True)
    output_b = model(**{key: batch[key] for key in (
        "load_history", "exog_history", "device_history", "device_status",
        "scheduler_context", "previous_chp",
    )})
    loss_b = formal_v4_joint_loss(output_b, **labels, parameters=PARAMETERS, c_ref=1000.0,
                                  oracle_diagnostic=torch.full((2,), 1e9))
    grad_b = torch.autograd.grad(loss_b.total, tuple(model.parameters()), allow_unused=True)
    assert all(torch.allclose(a, b) for a, b in zip(grad_a, grad_b) if a is not None and b is not None)


def test_curriculum_decision_is_positive_from_first_stage_j_epoch():
    early = formal_v4_curriculum_weights(epoch=0, ramp_epochs=20)
    middle = formal_v4_curriculum_weights(epoch=10, ramp_epochs=20)
    late = formal_v4_curriculum_weights(epoch=20, ramp_epochs=20)
    assert 0.0 < early.decision <= middle.decision <= late.decision
    assert all(np.isfinite(value) and value >= 0.0 for value in (
        early.forecast, early.imitation, early.decision,
        middle.forecast, middle.imitation, middle.decision,
        late.forecast, late.imitation, late.decision,
    ))


def test_formal_v4_1_curriculum_has_zero_imitation_at_epoch_18():
    start = formal_v4_curriculum_weights(epoch=0, ramp_epochs=18)
    boundary = formal_v4_curriculum_weights(epoch=18, ramp_epochs=18)
    after = formal_v4_curriculum_weights(epoch=30, ramp_epochs=18)
    assert (start.forecast, start.imitation, start.decision) == (1.0, 1.0, 0.05)
    assert (boundary.forecast, boundary.imitation, boundary.decision) == (1.0, 0.0, 1.0)
    assert (after.forecast, after.imitation, after.decision) == (1.0, 0.0, 1.0)


def test_forecast_task_weights_keep_gas_at_quarter_weight():
    assert tuple(FORECAST_TASK_WEIGHTS) == (1.0, 1.0, 1.0, 0.25)


def test_settlement_contract_has_reporting_only_shortage():
    ref = torch.ones(2, 4)
    settlement = FormalV4FourHourSettlement(
        per_step_penalized_objective=ref,
        constraint_penalty=torch.zeros(2),
        normalized_shortage=torch.ones(2),
    )
    assert settlement.per_step_penalized_objective.shape == (2, 4)
    assert settlement.normalized_shortage.shape == (2,)


def test_four_hour_constraint_penalty_includes_settlement_only_dump_accounting():
    index = {name: position for position, name in enumerate(VARIABLES)}
    planned = torch.zeros(1, 4, len(VARIABLES))
    planned[..., index["p_chp"]] = PARAMETERS["chp_electric_capacity"]
    demand = torch.zeros(1, 4, 3)
    renewables = torch.zeros(1, 4, 2)
    initial_soc = torch.full((1, 1), 0.5)
    previous_chp = torch.full((1, 1), PARAMETERS["chp_electric_capacity"])

    first = settle_first_step_v4(
        planned[:, 0], demand[:, 0], renewables[:, 0], PARAMETERS,
        initial_soc=initial_soc, previous_chp=previous_chp,
    )
    settled = settle_formal_v4_four_hour(
        planned, demand, renewables, initial_soc, previous_chp, PARAMETERS,
    )

    assert first.p_dump.item() > 0.0
    assert first.balance_residuals.abs().max().item() <= 1.0e-5
    assert first.conversion_residuals.abs().max().item() <= 1.0e-5
    assert settled.constraint_penalty.item() <= 1.0e-5


def test_gradient_clipping_enforces_configured_norm():
    parameter = torch.nn.Parameter(torch.tensor([10.0, -10.0]))
    parameter.grad = torch.tensor([10.0, -10.0])
    norm = clip_formal_v4_gradients([parameter], max_norm=1.0)
    assert norm > 1.0
    assert float(parameter.grad.norm()) <= 1.0 + 1.0e-6
