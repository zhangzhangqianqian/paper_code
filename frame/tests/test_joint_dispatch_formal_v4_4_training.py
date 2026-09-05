from __future__ import annotations

import torch

from src.joint_dispatch.formal_v4_4_model import ResidualGatedRSCPFModel
from src.joint_dispatch.formal_v4_4_training import (
    StageBudgetV44,
    run_stage_j_pair_v44,
    run_stage_p0_v44,
    run_stage_p1_v44,
    run_stage_s_v44,
)
from src.joint_dispatch.model import JointForecastDispatchModel


def _model() -> ResidualGatedRSCPFModel:
    return ResidualGatedRSCPFModel(
        transition_probability=torch.full((4, 3, 3), 1.0 / 3.0),
        decoder_parameters=JointForecastDispatchModel._test_parameters(),
        task_mean=torch.zeros(4), task_scale=torch.ones(4),
        physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10), dropout=0.0,
    )


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    dispatch = torch.zeros(batch_size, 4, 21)
    dispatch[..., 0] = 1.0
    target = torch.ones(batch_size, 4, 4)
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
        "teacher_dispatch": dispatch,
    }


def test_joint_pair_uses_identical_parent_and_matched_steps() -> None:
    budget = StageBudgetV44(max_epochs=1, minimum_epochs=1)
    p0 = run_stage_p0_v44(_model(), [_batch()], budget)
    p1 = run_stage_p1_v44(p0, [_batch()], budget)
    s = run_stage_s_v44(p1, [_batch()], budget)
    pair = run_stage_j_pair_v44(s, [_batch()], budget)
    assert pair.joint.parent_sha256 == pair.decoupled.parent_sha256
    assert pair.joint.optimizer_steps == pair.decoupled.optimizer_steps
    assert pair.joint.gradient_norms["decision_to_gate"] > 0.0
    assert pair.joint.gradient_norms["decision_to_magnitude"] > 0.0
    assert pair.decoupled.gradient_norms["decision_to_base"] <= 1.0e-12
    assert pair.decoupled.gradient_norms["forecast_to_base"] > 0.0


def test_p0_p1_receipts_keep_in_process_parent_models() -> None:
    budget = StageBudgetV44(max_epochs=1, minimum_epochs=1)
    p0 = run_stage_p0_v44(_model(), [_batch()], budget)
    p1 = run_stage_p1_v44(p0, [_batch()], budget)
    assert p0.model is not None and p1.model is not None
    assert p1.parent_sha256 == p0.final_sha256
