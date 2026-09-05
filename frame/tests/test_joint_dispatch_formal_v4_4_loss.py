from __future__ import annotations

import numpy as np
import torch

from src.joint_dispatch.formal_v4_4_loss import curriculum_weights, forecast_loss_v44
from src.joint_dispatch.formal_v4_4_model import ResidualGatedRSCPFModel
from src.joint_dispatch.formal_v4_4_regime import fit_thermal_prior
from src.joint_dispatch.model import JointForecastDispatchModel


def _receipt():
    target = np.zeros((12, 4, 4), dtype=np.float64)
    target[:, :, 0] = 10.0
    target[4:8, :, 1] = 100.0
    target[8:12, :, 2] = 50.0
    history = np.zeros((12, 24, 4), dtype=np.float64)
    times = np.asarray([f"{year}-01-01" for year in (2015, 2016, 2017, 2018)] * 3, dtype="datetime64[ns]")
    return fit_thermal_prior(target, history, times), target


def _model_and_batch():
    receipt, target = _receipt()
    model = ResidualGatedRSCPFModel(
        transition_probability=torch.as_tensor(receipt.transition_probability, dtype=torch.float32),
        decoder_parameters=JointForecastDispatchModel._test_parameters(),
        task_mean=torch.zeros(4), task_scale=torch.ones(4),
        physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10), dropout=0.0,
    )
    batch = {
        "load_history": torch.ones(2, 24, 4), "exog_history": torch.zeros(2, 24, 12),
        "device_history": torch.zeros(2, 24, 17), "activity_history": torch.zeros(2, 24, 6),
        "scheduler_context": torch.cat((torch.zeros(2, 4, 5), torch.full((2, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.ones(2, 1), "last_thermal_regime": torch.zeros(2, dtype=torch.long),
        "target_normalized": torch.as_tensor(target[:2], dtype=torch.float32),
        "target_physical": torch.as_tensor(target[:2], dtype=torch.float32),
        "thermal_transition_mask": torch.zeros(2),
    }
    return model, batch, receipt


def test_curriculum_is_fixed_and_monotone() -> None:
    early = curriculum_weights(0, ramp_epochs=5, leakage_target=0.25)
    late = curriculum_weights(5, ramp_epochs=5, leakage_target=0.25)
    assert early.point == 0.25 and late.point == 1.0
    assert early.inactive_leakage == 0.10 and late.inactive_leakage == 0.25


def test_forecast_loss_is_finite_and_backpropagates() -> None:
    model, batch, receipt = _model_and_batch()
    output = model(**{k: v for k, v in batch.items() if k not in {"target_normalized", "target_physical", "thermal_transition_mask"}})
    loss = forecast_loss_v44(output, batch, receipt, curriculum_weights(5, 5, 0.25))
    assert torch.isfinite(loss.total)
    loss.total.backward()

