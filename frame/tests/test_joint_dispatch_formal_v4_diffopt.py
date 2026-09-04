from __future__ import annotations

import pytest
import torch

from src.joint_dispatch.formal_v4_diffopt import (
    DIFFOPT_METHOD_ID,
    DifferentiableIESLayer,
    DifferentiableLPGateReceipt,
    DifferentiableLPProblemSpec,
)


PARAMETERS = {
    "grid_import_capacity": 20.0, "chp_electric_capacity": 10.0,
    "chp_heat_capacity": 12.0, "gas_boiler_capacity": 20.0,
    "electric_chiller_capacity": 20.0, "absorption_chiller_capacity": 20.0,
    "bess_power_capacity": 5.0, "bess_energy_capacity": 20.0,
    "chp_electric_efficiency": 0.4, "chp_heat_efficiency": 0.45,
    "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.0,
    "absorption_chiller_cop": 0.8, "bess_roundtrip_efficiency": 0.9,
    "bess_throughput_cost": 1.0e-6, "unserved_penalty": 100.0,
    "chp_ramp_fraction": 1.0,
}


def test_diffopt_problem_declares_predicted_rhs_parameters():
    spec = DifferentiableLPProblemSpec()
    assert spec.predicted_parameters == ("electricity", "cooling", "heating")
    assert spec.horizon == 4
    assert spec.online_optimizer_calls_per_window == 1


def test_diffopt_is_not_reported_as_optimizer_free():
    spec = DifferentiableLPProblemSpec()
    assert spec.deployable is True
    assert spec.optimizer_at_inference is True


def test_diffopt_receipt_fails_closed_until_environment_gate():
    receipt = DifferentiableLPGateReceipt()
    receipt.validate()
    assert receipt.method_id == DIFFOPT_METHOD_ID
    assert receipt.eligible is False


def test_eligible_receipt_requires_all_evidence():
    with pytest.raises(ValueError, match="mandatory gate evidence"):
        DifferentiableLPGateReceipt(eligible=True, dpp_passed=True, finite_solves=99, parity_passed=True, gradient_passed=True, native_probe_passed=True, memory_margin_fraction=0.2, projected_p95_hours=24.0).validate()


def test_diffopt_collapses_column_shaped_state_to_one_scalar_per_window():
    pytest.importorskip("cvxpylayers")
    layer = DifferentiableIESLayer(PARAMETERS)
    dispatch = layer(
        torch.ones(1, 4, 3),
        torch.zeros(1, 4, 2),
        torch.ones(1, 4, 4),
        torch.full((1, 1), 0.5),
        torch.zeros(1, 1),
    )
    assert dispatch.shape == (1, 4, 21)
