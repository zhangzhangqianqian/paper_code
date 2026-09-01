from __future__ import annotations

import torch
import pytest

from src.joint_dispatch.external_baseline_data import ExternalNormalization
from src.joint_dispatch.external_baseline_losses import (
    ExternalEvidenceError,
    audit_external_gradients,
    decision_focused_loss,
    forecast_loss,
    policy_imitation_loss,
)
from src.joint_dispatch.external_baselines import DecisionFocusedOnline
from tests.test_rsc_pf_external_baselines import _batch


def _normalization() -> ExternalNormalization:
    return ExternalNormalization(
        load_mean=torch.zeros(4).numpy(), load_scale=torch.ones(4).numpy(),
        exog_mean=torch.zeros(12).numpy(), exog_scale=torch.ones(12).numpy(),
        device_mean=torch.zeros(21).numpy(), device_scale=torch.ones(21).numpy(),
        scheduler_mean=torch.zeros(6).numpy(), scheduler_scale=torch.ones(6).numpy(),
    )


def test_forecast_loss_is_finite_and_differentiable() -> None:
    prediction = torch.randn(2, 4, 4, requires_grad=True)
    target = torch.zeros_like(prediction)
    value = forecast_loss(prediction, target, _normalization())
    assert torch.isfinite(value).item()
    value.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all().item()


def test_policy_imitation_loss_is_finite() -> None:
    prediction = torch.randn(2, 4, 21, requires_grad=True)
    value = policy_imitation_loss(prediction, torch.zeros_like(prediction))
    assert torch.isfinite(value).item()


def test_decision_focused_loss_fails_closed_without_verified_surrogate() -> None:
    model = DecisionFocusedOnline(d_model=32, heads=4, layers=1)
    output = model(_batch())
    with pytest.raises(ExternalEvidenceError, match="verified published surrogate"):
        decision_focused_loss(output, _batch(), {})


def test_decision_focused_loss_accepts_explicit_verified_differentiable_surrogate() -> None:
    model = DecisionFocusedOnline(d_model=32, heads=4, layers=1)
    batch = _batch()
    output = model(batch)
    value = decision_focused_loss(output, batch, {}, surrogate=lambda result, _batch, _params: result.forecast.square().mean())
    value.backward()
    receipt = audit_external_gradients(value, model)
    assert receipt["finite"] is True
    assert receipt["parameters_with_nonzero_grad"] > 0
