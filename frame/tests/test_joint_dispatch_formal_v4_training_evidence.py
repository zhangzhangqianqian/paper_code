from __future__ import annotations

import torch

from scripts.build_rsc_pf_formal_v4_training_evidence import build_curriculum_receipt, build_gradient_receipt
from src.joint_dispatch.formal_v4_gate0_evidence import validate_gate0_receipt


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


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(2026)
    return {
        "load_history": torch.rand(2, 24, 4),
        "exog_history": torch.rand(2, 24, 12),
        "device_history": torch.rand(2, 24, 17),
        "activity_history": torch.zeros(2, 24, 6),
        "scheduler_context": torch.cat((torch.rand(2, 4, 5), torch.full((2, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.zeros(2, 1),
        "target_normalized": torch.rand(2, 4, 4),
        "target_physical": torch.rand(2, 4, 4),
        "realized_renewables": torch.rand(2, 4, 2),
        "initial_soc": torch.full((2, 1), 0.5),
    }


def test_curriculum_receipt_is_strict_and_decision_active() -> None:
    receipt = build_curriculum_receipt(source_commit="a" * 40)
    validate_gate0_receipt("curriculum", receipt, run_root=".")
    assert receipt["epochs"][0]["decision"] > 0.0


def test_gradient_receipt_proves_joint_boundary_without_checkpoint() -> None:
    receipt = build_gradient_receipt(_batch(), PARAMETERS, c_ref=1000.0, source_commit="a" * 40)
    validate_gate0_receipt("gradient", receipt, run_root=".")
    assert receipt["joint"]["forecaster_decision_gradient_norm"] > 0.0
    assert receipt["decoupled"]["forecaster_decision_gradient_norm"] == 0.0
    assert receipt["checkpoint_created"] is False
