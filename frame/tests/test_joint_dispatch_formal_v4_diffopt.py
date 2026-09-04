from __future__ import annotations

import pytest

from src.joint_dispatch.formal_v4_diffopt import (
    DIFFOPT_METHOD_ID,
    DifferentiableLPGateReceipt,
    DifferentiableLPProblemSpec,
)


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
