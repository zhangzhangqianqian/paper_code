from __future__ import annotations

import torch

from src.joint_dispatch.formal_v4_models import RegimeAwareRSCPFModel


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    context = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 0.5]).reshape(1, 1, 6).expand(batch_size, 4, 6).clone()
    return {
        "load_history": torch.randn(batch_size, 24, 4),
        "exog_history": torch.randn(batch_size, 24, 12),
        "device_history": torch.randn(batch_size, 24, 17),
        "activity_history": torch.zeros(batch_size, 24, 6),
        "scheduler_context": context,
        "previous_chp": torch.zeros(batch_size, 1),
    }


def test_regime_aware_model_preserves_four_task_dispatch_contract() -> None:
    model = RegimeAwareRSCPFModel.for_test()
    output = model(**_batch())
    assert output.forecast_normalized.shape == (2, 4, 4)
    assert output.forecast_physical.shape == (2, 4, 4)
    assert output.regime_logits.shape == (2, 4, 3)
    assert output.regime_probabilities.shape == (2, 4, 3)
    assert output.thermal_magnitudes.shape == (2, 4, 2)
    assert output.controls.shape == (2, 4, 15)
    assert output.dispatch.shape == (2, 4, 21)
    assert torch.all(output.forecast_physical >= 0.0)
    torch.testing.assert_close(
        output.regime_probabilities.sum(-1), torch.ones(2, 4), atol=1.0e-6, rtol=0.0,
    )


def test_dispatch_gradient_reaches_regime_head() -> None:
    model = RegimeAwareRSCPFModel.for_test()
    output = model(**_batch())
    output.dispatch[..., 0].sum().backward()
    gradient = model.regime_head.output.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.any(gradient.abs() > 0.0)


def test_decoupled_dispatch_gradient_does_not_reach_regime_head() -> None:
    model = RegimeAwareRSCPFModel.for_test()
    output = model(detach_forecast_for_dispatch=True, **_batch())
    output.dispatch[..., 0].sum().backward()
    gradient = model.regime_head.output.weight.grad
    assert gradient is None or torch.allclose(gradient, torch.zeros_like(gradient))

