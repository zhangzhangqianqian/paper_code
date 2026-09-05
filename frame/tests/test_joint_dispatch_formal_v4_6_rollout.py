from __future__ import annotations

import tempfile

import numpy as np
import torch

from src.joint_dispatch.formal_v4_2_data import fit_train_normalization
from src.joint_dispatch.formal_v4_4_pilot_materializer import V44WindowCollection
from src.joint_dispatch.formal_v4_5_pilot_data import MaterializedPilotV45
from src.joint_dispatch.formal_v4_6_model import RiskAdjustedRSCPFModelV46
from src.joint_dispatch.formal_v4_6_rollout import rollout_v46_2019
from src.joint_dispatch.model import JointForecastDispatchModel
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit


def _split(year: int) -> FormalV4WindowSplit:
    target = np.zeros((1, 4, 4), dtype=np.float64)
    target[..., 0] = 1.0
    timestamp = np.asarray([f"{year}-01-01"], dtype="datetime64[ns]")
    return FormalV4WindowSplit(
        np.zeros((1, 24, 4)), np.zeros((1, 24, 12)), np.zeros((1, 24, 2)), np.zeros((1, 24, 17)), np.zeros((1, 24, 6)),
        target, np.ones((1, 4, 3)), np.zeros((1, 4, 2)), np.zeros((1, 4, 2)), np.zeros((1, 4, 3)),
        np.full((1, 1), 0.5), np.zeros((1, 1)), timestamp, np.asarray(["trajectory"]), np.asarray(["state"]),
        split="train" if year == 2015 else "selection",
    )


def test_rollout_records_nominal_adjusted_and_first_step_state(tmp_path):
    train = V44WindowCollection(_split(2015), "train")
    selection = V44WindowCollection(_split(2019), "selection_full")
    materialized = MaterializedPilotV45(train, train, selection, selection, train, {})
    model = RiskAdjustedRSCPFModelV46.for_test(torch.ones(4, 3))
    result = rollout_v46_2019(
        model=model, materialized=materialized, indices=np.arange(1), normalization=fit_train_normalization(train.split),
        parameters=JointForecastDispatchModel._test_parameters(), artifact_root=tmp_path, method_id="rsc_pf_joint",
    )
    assert result.forecast_nominal.shape == (1, 4, 4)
    assert result.risk_adjustment.shape == (1, 4, 3)
    assert result.scheduler_demand.shape == (1, 4, 4)
    assert result.settled_dispatch.shape == (1, 21)
    assert np.max(result.physical_residual) <= 1e-6
