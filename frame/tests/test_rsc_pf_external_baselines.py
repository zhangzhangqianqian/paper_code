from __future__ import annotations

import numpy as np
import torch

from src.joint_dispatch.external_baseline_data import ExternalBaselineBatch
from src.joint_dispatch.external_baselines import (
    DecisionFocusedOnline,
    DigitalTwinsPolicy,
    ExternalForecastPTO,
    build_external_baseline,
)
from src.joint_dispatch.contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER


def _batch(batch_size: int = 2) -> ExternalBaselineBatch:
    generator = torch.Generator().manual_seed(3)
    context = torch.rand(batch_size, 4, 6, generator=generator)
    context[..., 0:5] = context[..., 0:5] * 5.0
    context[..., 5] = 0.5
    return ExternalBaselineBatch(
        load_history=torch.rand(batch_size, 24, len(TASK_ORDER), generator=generator),
        exog_history=torch.rand(batch_size, 24, len(EXOG_ORDER), generator=generator),
        device_history=torch.rand(batch_size, 24, len(DISPATCH_ORDER), generator=generator),
        device_status=torch.zeros(batch_size, 24, len(STATUS_ORDER)),
        scheduler_context=context,
        previous_chp=torch.rand(batch_size, 1, generator=generator),
        forecast_target=torch.rand(batch_size, 4, len(TASK_ORDER), generator=generator),
        teacher_dispatch=torch.rand(batch_size, 4, len(DISPATCH_ORDER), generator=generator),
        oracle_first_step_objective=torch.rand(batch_size, generator=generator),
        split="validation",
    )


def test_itransformer_pto_forecast_shape() -> None:
    model = ExternalForecastPTO(d_model=32, heads=4, layers=1)
    batch = _batch()
    prediction = model(batch.load_history, batch.exog_history)
    assert tuple(prediction.shape) == (2, 4, 4)
    assert torch.isfinite(prediction).all().item()


def test_decision_focused_wrapper_exposes_gradient_and_optimizer_role() -> None:
    model = DecisionFocusedOnline(d_model=32, heads=4, layers=1)
    output = model(_batch())
    assert output.optimizer_role == "exact optimizer at inference"
    assert output.forecast.requires_grad
    output.forecast.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_direct_policy_returns_feasible_dispatch_without_exact_lp(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("direct policy must not call an exact LP")

    monkeypatch.setattr("src.scheduling.dispatch_lp.solve_dispatch_lp", fail_if_called)
    model = DigitalTwinsPolicy(hidden_dim=32)
    output = model(_batch())
    assert tuple(output.dispatch.shape) == (2, 4, 21)
    assert output.exact_lp_calls == 0
    assert torch.isfinite(output.dispatch).all().item()


def test_factory_rejects_unknown_method() -> None:
    try:
        build_external_baseline("not-a-frozen-method")
    except ValueError as error:
        assert "unknown external baseline" in str(error)
    else:
        raise AssertionError("unknown method should fail closed")
