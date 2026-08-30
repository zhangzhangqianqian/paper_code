from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.joint_dispatch.model import JointForecastDispatchModel


def test_joint_model_exposes_forecast_bottleneck_and_dispatch():
    model = JointForecastDispatchModel.for_test(exog_dim=12)
    output = model(
        load_history=torch.randn(3, 24, 4),
        exog_history=torch.randn(3, 24, 12),
        device_history=torch.randn(3, 24, 21),
        device_status=torch.zeros(3, 24, 6),
        scheduler_context=torch.ones(3, 4, 6),
        previous_chp=torch.zeros(3, 1),
    )
    assert output.forecast_normalized.shape == (3, 4, 4)
    assert output.forecast_physical.shape == (3, 4, 4)
    assert output.physical_features.shape == (3, 4, 10)
    assert output.control_logits.shape == (3, 15)
    assert output.dispatch.shape == (3, 4, 21)


def test_softplus_physical_conversion_is_smooth_and_finite():
    model = JointForecastDispatchModel.for_test(exog_dim=12)
    raw = torch.tensor([[-2.0, 0.0, 21.0, 100.0]], requires_grad=True)
    physical = model.forecast_to_physical(raw)
    assert physical[0, 0] > 0.0
    assert physical[0, 2].item() == 21.0
    assert physical[0, 3].item() == 100.0
    physical.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_dispatch_gradient_reaches_both_networks():
    model = JointForecastDispatchModel.for_test(exog_dim=12)
    output = model(
        load_history=torch.randn(2, 24, 4),
        exog_history=torch.randn(2, 24, 12),
        device_history=torch.randn(2, 24, 21),
        device_status=torch.zeros(2, 24, 6),
        scheduler_context=torch.ones(2, 4, 6),
        previous_chp=torch.zeros(2, 1),
    )
    output.dispatch[..., 0].sum().backward()
    forecast_grads = [p.grad for p in model.forecaster.parameters() if p.grad is not None]
    scheduler_grads = [p.grad for p in model.scheduler.parameters() if p.grad is not None]
    assert any(torch.isfinite(g).all() and torch.any(g.abs() > 0) for g in forecast_grads)
    assert any(torch.isfinite(g).all() and torch.any(g.abs() > 0) for g in scheduler_grads)


def test_scheduler_uses_forecast_bottleneck():
    torch.manual_seed(7)
    model = JointForecastDispatchModel.for_test(exog_dim=12)
    common = dict(
        load_history=torch.randn(1, 24, 4),
        exog_history=torch.randn(1, 24, 12),
        device_history=torch.randn(1, 24, 21),
        device_status=torch.zeros(1, 24, 6),
        scheduler_context=torch.ones(1, 4, 6),
        previous_chp=torch.zeros(1, 1),
    )
    first = model(**common)
    altered = dict(common, load_history=common["load_history"] + 3.0)
    second = model(**altered)
    assert not torch.allclose(first.control_logits, second.control_logits)
