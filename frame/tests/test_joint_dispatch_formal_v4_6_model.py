from __future__ import annotations

import numpy as np
import pytest
import torch

from src.joint_dispatch.formal_v4_6_model import RiskAdjustedRSCPFModelV46
from src.scheduling.proxy_physics import balance_residuals, conversion_residuals, soc_residuals


def make_v46_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "load_history": torch.zeros(batch_size, 24, 4),
        "exog_history": torch.zeros(batch_size, 24, 12),
        "device_history": torch.zeros(batch_size, 24, 17),
        "activity_history": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.zeros(batch_size, 4, 6),
        "previous_chp": torch.zeros(batch_size, 1),
        "last_thermal_regime": torch.zeros(batch_size, dtype=torch.long),
    }


def make_v46_model_with_known_probabilities() -> RiskAdjustedRSCPFModelV46:
    model = RiskAdjustedRSCPFModelV46.for_test(risk_cap=torch.full((4, 3), 10.0))
    with torch.no_grad():
        model.thermal_head.transition_log_prior.zero_()
        for layer in (model.thermal_head.gate_output, model.thermal_head.magnitude_output):
            layer.weight.zero_()
            layer.bias.zero_()
    return model


def test_v46_exposes_distinct_nominal_and_scheduler_demand() -> None:
    model = RiskAdjustedRSCPFModelV46.for_test(risk_cap=torch.full((4, 3), 10.0))
    output = model(**make_v46_batch(batch_size=2))
    assert output.forecast_nominal_physical.shape == (2, 4, 4)
    assert output.risk_adjustment.shape == (2, 4, 3)
    assert output.risk_cap.shape == (2, 4, 3)
    assert output.scheduler_demand.shape == (2, 4, 4)
    assert output.controls.shape == (2, 4, 15)
    assert output.dispatch.shape == (2, 4, 21)
    assert torch.all(output.risk_adjustment >= 0)
    assert torch.all(output.risk_adjustment <= 10.0 + 1e-6)
    assert torch.equal(output.scheduler_demand[..., 3], output.forecast_nominal_physical[..., 3])


def test_thermal_risk_is_probability_masked_and_decoupled_detaches_forecast() -> None:
    model = make_v46_model_with_known_probabilities()
    joint = model(**make_v46_batch(), detach_forecast_for_dispatch=False)
    decoupled = model(**make_v46_batch(), detach_forecast_for_dispatch=True)
    assert torch.all(joint.risk_adjustment[..., 1] <= joint.risk_cap[..., 1] * joint.regime_probabilities[..., 1] + 1e-6)
    assert torch.all(joint.risk_adjustment[..., 2] <= joint.risk_cap[..., 2] * joint.regime_probabilities[..., 2] + 1e-6)
    grad = torch.autograd.grad(decoupled.dispatch.sum(), model.base_forecaster_parameters(), allow_unused=True)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in grad)


@pytest.mark.parametrize("output_bias", [-40.0, 40.0])
def test_physics_remains_feasible_at_zero_and_maximum_risk(output_bias: float) -> None:
    model = RiskAdjustedRSCPFModelV46.for_test(risk_cap=torch.full((4, 3), 10.0))
    torch.nn.init.zeros_(model.risk_head.output.weight)
    torch.nn.init.constant_(model.risk_head.output.bias, output_bias)
    batch = make_v46_batch(batch_size=2)
    output = model(**batch)
    features = model.core._raw_physical_features(output.scheduler_demand, batch["scheduler_context"])
    soc = soc_residuals(output.dispatch, features, model.core.decoder_parameters)
    residuals = (
        balance_residuals(output.dispatch, features),
        conversion_residuals(output.dispatch, model.core.decoder_parameters),
        soc["state"], soc["terminal"],
    )
    assert max(float(value.abs().max()) for value in residuals) <= 1e-6
