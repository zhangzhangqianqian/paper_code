from __future__ import annotations

import torch

from src.joint_dispatch.formal_v4_models import RSCPFModel
from src.joint_dispatch.formal_v4_training import (
    configure_stage_j,
    train_stage_j,
    train_stage_p,
    train_stage_s,
)


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.2, "unserved_penalty": 100.0,
    "surplus_penalty": 0.0, "chp_ramp_fraction": 0.5,
}


def _batch(batch_size: int = 2):
    torch.manual_seed(2026)
    return {
        "load_history": torch.rand(batch_size, 24, 4), "exog_history": torch.rand(batch_size, 24, 12),
        "device_history": torch.rand(batch_size, 24, 17), "activity_history": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.cat((torch.rand(batch_size, 4, 5), torch.full((batch_size, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.zeros(batch_size, 1), "target_normalized": torch.rand(batch_size, 4, 4),
        "target_physical": torch.rand(batch_size, 4, 4), "realized_renewables": torch.rand(batch_size, 4, 2),
        "initial_soc": torch.full((batch_size, 1), 0.5), "teacher_dispatch": torch.rand(batch_size, 4, 21),
    }


def _inputs(batch):
    return {key: batch[key] for key in ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")}


def test_stage_p_updates_every_forecaster_parameter():
    model = RSCPFModel(dropout=0.0)
    receipt = train_stage_p(model, _batch(), PARAMETERS, seed=2026)
    assert set(receipt.updated_parameters) == set(receipt.forecaster_parameters)


def test_stage_s_updates_every_scheduler_parameter_and_freezes_forecaster():
    model = RSCPFModel(dropout=0.0)
    receipt = train_stage_s(model, _batch(), PARAMETERS, seed=2026)
    assert set(receipt.updated_parameters) == set(receipt.scheduler_parameters)
    assert receipt.forecaster_gradient_norm == 0.0


def test_stage_j_branches_start_identically():
    source = RSCPFModel(dropout=0.0)
    joint = RSCPFModel(dropout=0.0)
    decoupled = RSCPFModel(dropout=0.0)
    joint.load_state_dict(source.state_dict())
    decoupled.load_state_dict(source.state_dict())
    configure_stage_j(joint, "joint")
    configure_stage_j(decoupled, "decoupled")
    assert all(torch.equal(a, b) for a, b in zip(joint.state_dict().values(), decoupled.state_dict().values()))


def test_decision_gradient_boundary_differs_between_branches():
    batch = _batch()
    joint = RSCPFModel(dropout=0.0)
    decoupled = RSCPFModel(dropout=0.0)
    decoupled.load_state_dict(joint.state_dict())
    joint_receipt = train_stage_j(joint, batch, PARAMETERS, c_ref=1000.0, mode="joint", seed=2026)
    decoupled_receipt = train_stage_j(decoupled, batch, PARAMETERS, c_ref=1000.0, mode="decoupled", seed=2026)
    assert joint_receipt.forecaster_gradient_norm > 0.0
    assert decoupled_receipt.forecaster_gradient_norm == 0.0
    assert joint_receipt.scheduler_gradient_norm > 0.0

