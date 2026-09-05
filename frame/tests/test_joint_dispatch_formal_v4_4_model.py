from __future__ import annotations

import torch

from src.joint_dispatch.model import JointForecastDispatchModel
from src.joint_dispatch.formal_v4_4_model import ResidualGatedRSCPFModel


def _model() -> ResidualGatedRSCPFModel:
    return ResidualGatedRSCPFModel(
        transition_probability=torch.full((4, 3, 3), 1.0 / 3.0),
        decoder_parameters=JointForecastDispatchModel._test_parameters(),
        task_mean=torch.zeros(4), task_scale=torch.ones(4),
        physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10),
        dropout=0.0,
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "load_history": torch.ones(batch_size, 24, 4),
        "exog_history": torch.zeros(batch_size, 24, 12),
        "device_history": torch.zeros(batch_size, 24, 17),
        "activity_history": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.cat((torch.zeros(batch_size, 4, 5), torch.full((batch_size, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.ones(batch_size, 1),
        "last_thermal_regime": torch.zeros(batch_size, dtype=torch.long),
    }


def test_zero_residuals_preserve_scheme2r_magnitudes() -> None:
    model = _model().eval()
    with torch.no_grad():
        output = model(**_batch())
        expected = model.core.forecast_to_physical(output.base_forecast_normalized)
    torch.testing.assert_close(output.thermal_magnitudes, expected[..., 1:3])


def test_v44_output_contract() -> None:
    output = _model()(**_batch())
    assert output.forecast_physical.shape == (2, 4, 4)
    assert output.regime_logits.shape == (2, 4, 3)
    assert output.regime_probabilities.shape == (2, 4, 3)
    assert output.thermal_residuals.shape == (2, 4, 2)
    assert output.controls.shape == (2, 4, 15)
    assert output.dispatch.shape == (2, 4, 21)
    assert torch.all(output.forecast_physical >= 0)
    torch.testing.assert_close(output.regime_probabilities.sum(-1), torch.ones(2, 4))
    torch.testing.assert_close(output.forecast_physical[..., [0, 3]], output.base_forecast_physical[..., [0, 3]])


def test_decision_gradient_reaches_all_joint_groups() -> None:
    model = _model()
    loss = model(**_batch()).dispatch.square().mean()
    loss.backward()
    for name, parameters in model.v44_parameter_groups().items():
        norm = sum(float(parameter.grad.norm()) for parameter in parameters if parameter.grad is not None)
        assert norm > 0.0, name
