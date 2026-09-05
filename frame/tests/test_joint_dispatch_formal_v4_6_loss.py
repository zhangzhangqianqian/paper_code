from __future__ import annotations

import numpy as np
import torch

from src.joint_dispatch.formal_v4_4_regime import fit_thermal_prior
from src.joint_dispatch.formal_v4_4_training import named_autograd_norms
from src.joint_dispatch.formal_v4_5_loss import JointNormalizationV45, curriculum_weights_v45
from src.joint_dispatch.formal_v4_6_loss import build_j_optimizer_v46, joint_loss_v46
from src.joint_dispatch.formal_v4_6_model import RiskAdjustedRSCPFModelV46
from src.joint_dispatch.model import JointForecastDispatchModel


def _prior():
    target = np.zeros((12, 4, 4), dtype=np.float64)
    target[:, :, 0] = 10.0
    target[4:8, :, 1] = 100.0
    target[8:12, :, 2] = 50.0
    history = np.zeros((12, 24, 4), dtype=np.float64)
    times = np.asarray([f"{year}-01-01" for year in (2015, 2016, 2017, 2018)] * 3, dtype="datetime64[ns]")
    return fit_thermal_prior(target, history, times)


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    target = torch.zeros(batch_size, 4, 4)
    target[..., 0] = 10.0
    target[0, 1, 1] = 4.0
    target[1, 2, 2] = 3.0
    return {
        "load_history": torch.ones(batch_size, 24, 4),
        "exog_history": torch.zeros(batch_size, 24, 12),
        "device_history": torch.zeros(batch_size, 24, 17),
        "activity_history": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.cat((torch.zeros(batch_size, 4, 5), torch.full((batch_size, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.ones(batch_size, 1),
        "last_thermal_regime": torch.zeros(batch_size, dtype=torch.long),
        "target_normalized": target,
        "target_physical": target,
        "teacher_dispatch": torch.zeros(batch_size, 4, 21),
    }


def _model() -> RiskAdjustedRSCPFModelV46:
    return RiskAdjustedRSCPFModelV46(
        transition_probability=torch.full((4, 3, 3), 1.0 / 3.0),
        risk_cap=torch.full((4, 3), 5.0),
        decoder_parameters=JointForecastDispatchModel._test_parameters(),
        task_mean=torch.zeros(4), task_scale=torch.ones(4),
        physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10),
        dropout=0.0,
    )


def test_joint_loss_contains_full_forecast_and_risk_terms() -> None:
    model = _model()
    batch = _batch()
    inputs = {key: value for key, value in batch.items() if key not in {"target_normalized", "target_physical", "teacher_dispatch"}}
    output = model(**inputs)
    terms = joint_loss_v46(
        output=output,
        parent_output=output.as_v44_forecast_output(),
        batch=batch,
        prior=_prior(),
        normalization=JointNormalizationV45(1.0, 1.0, 1.0),
        curriculum=curriculum_weights_v45(3, 5),
        risk_multiplier=1.0,
        decision_loss=output.dispatch.square().mean(),
    )
    assert terms.forecast.regime.item() > 0
    assert terms.forecast.active_magnitude.item() >= 0
    assert terms.forecast.inactive_leakage.item() >= 0
    assert terms.risk_size.item() >= 0
    assert terms.off_risk.item() >= 0
    assert torch.isfinite(terms.total)


def test_joint_decision_gradient_reaches_risk_and_forecast_groups() -> None:
    model = _model()
    batch = _batch()
    inputs = {key: value for key, value in batch.items() if key not in {"target_normalized", "target_physical", "teacher_dispatch"}}
    output = model(**inputs)
    decision = output.dispatch.square().mean()
    norms = named_autograd_norms(decision, model.v46_parameter_groups())
    assert norms["base"] > 0
    assert norms["gate"] > 0
    assert norms["magnitude"] > 0
    assert norms["risk"] > 0
    assert norms["scheduler"] > 0


def test_v46_j_optimizer_has_frozen_group_rates() -> None:
    model = _model()
    contract = {
        "pilot_budget": {"j_forecaster_lr": 2e-4, "j_head_lr": 5e-4, "j_scheduler_lr": 5e-4, "weight_decay": 1e-5},
        "risk_adjustment": {"j_risk_lr": 5e-4},
    }
    joint = build_j_optimizer_v46(model, contract, "joint")
    decoupled = build_j_optimizer_v46(model, contract, "decoupled")
    assert {group["name"] for group in joint.param_groups} == {"base", "head", "risk", "scheduler"}
    assert {group["name"] for group in decoupled.param_groups} == {"risk", "scheduler"}
    assert all(float(group["lr"]) == 5e-4 for group in decoupled.param_groups)
