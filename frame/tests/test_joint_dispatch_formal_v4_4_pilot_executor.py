from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.joint_dispatch.formal_v4_4_model import ResidualGatedRSCPFModel
from src.joint_dispatch.formal_v4_4_pilot_executor import build_continuous_control_v44, execute_training_stages_v44
from src.joint_dispatch.formal_v4_4_regime import ThermalPriorReceiptV44
from src.joint_dispatch.formal_v4_4_teacher import TeacherReceiptV44
from src.joint_dispatch.model import JointForecastDispatchModel


def _p0() -> ResidualGatedRSCPFModel:
    probability = torch.full((4, 3, 3), 1.0 / 3.0)
    return ResidualGatedRSCPFModel(
        transition_probability=probability,
        decoder_parameters=JointForecastDispatchModel._test_parameters(),
        task_mean=torch.zeros(4), task_scale=torch.ones(4),
        physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10), dropout=0.0,
    )


def test_continuous_control_reuses_p0_without_regime_heads() -> None:
    p0 = _p0()
    control = build_continuous_control_v44(p0)
    groups = control.v44_parameter_groups()
    assert groups["gate"] == ()
    assert groups["magnitude"] == ()
    assert set(control.state_dict()).issubset(set(p0.state_dict()))
    for name, value in control.state_dict().items():
        torch.testing.assert_close(value, p0.state_dict()[name])


def test_continuous_control_rejects_non_residual_parent() -> None:
    import pytest
    with pytest.raises(TypeError, match="residual P0"):
        build_continuous_control_v44(torch.nn.Linear(2, 2))


def test_executor_runs_matched_stage_order_on_two_windows(tmp_path, monkeypatch) -> None:
    from tests.test_joint_dispatch_formal_v4_4_teacher import _teacher_fixture
    sources, materialized = _teacher_fixture(tmp_path, monkeypatch)
    payload = {
        "pilot_budget": {
            "batch_size": 2, "minimum_epochs": 1, "early_stopping_patience": 1,
            "p0_max_epochs": 1, "p1_max_epochs": 1, "s_max_epochs": 1, "j_max_epochs": 1,
            "p0_forecaster_lr": 1.0e-3, "p1_base_lr": 2.0e-4, "p1_head_lr": 1.0e-3,
            "s_scheduler_lr": 5.0e-4, "j_forecaster_lr": 2.0e-4, "j_head_lr": 5.0e-4,
            "j_scheduler_lr": 5.0e-4, "weight_decay": 1.0e-5, "max_grad_norm": 1.0,
            "train_windows": 2,
        },
        "pilot_candidate": {"temperature": 1.0, "inactive_leakage_weight": 0.25},
    }
    contract = SimpleNamespace(validate=lambda: None, payload=payload, pilot_train_windows=2, contract_sha256="a" * 64)
    source = materialized.normalization_source
    prior = ThermalPriorReceiptV44(
        transition_probability=np.full((4, 3, 3), 1.0 / 3.0), transition_count=np.ones((4, 3, 3)),
        active_mean=np.ones(2), active_scale=np.ones(2), active_count=np.ones(2), class_count=np.ones(3, dtype=np.int64),
        years=(2015, 2016, 2017, 2018), target_sha256="a" * 64, history_sha256="b" * 64, timestamp_sha256="c" * 64,
    )
    teacher = TeacherReceiptV44(
        forecast=np.ones((len(materialized.train), 4, 4)), renewable_plan=np.zeros((len(materialized.train), 4, 2)),
        dispatch=np.zeros((len(materialized.train), 4, 21)), objective=np.zeros(len(materialized.train)),
        shortage=np.zeros((len(materialized.train), 3)), status=("optimal",) * len(materialized.train),
        timestamps=materialized.train.timestamps, state_hashes=materialized.train.state_hashes, lineage={"lineage_sha256": "d" * 64},
    )
    bundle = execute_training_stages_v44(
        materialized=materialized, contract=contract, artifact_root=tmp_path / "artifacts", seed=2026,
        parameters=JointForecastDispatchModel._test_parameters(), prior=prior, teacher=teacher,
    )
    assert bundle.p0.stage == "P0"
    assert bundle.p1.stage == "P1"
    assert bundle.continuous_control.mode == "continuous_base"
    assert bundle.s.stage == "S"
    assert bundle.j.joint.mode == "joint"
    assert bundle.j.decoupled.mode == "decoupled"
