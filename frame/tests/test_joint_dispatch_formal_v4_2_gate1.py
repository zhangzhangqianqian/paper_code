from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
import torch

from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract
from src.joint_dispatch.formal_v4_2_data import Gate1OriginManifestV42


GATE1 = run_path(str(Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_gate1.py"))


def _manifest(cooling=0.5, heating=0.5):
    return Gate1OriginManifestV42(np.arange(4), np.arange(4).astype("datetime64[h]"), ("cooling_active",) * 4, np.ones(4), {"cooling": cooling, "heating": heating}, "a" * 64)


def _input(tmp_path, manifest=None):
    contract = load_formal_v4_2_contract("configs/joint_forecast_dispatch_formal_v4_2.json")
    return GATE1["Gate1InputV42"](manifest or _manifest(), contract, "b" * 64, "c" * 64, output_path=tmp_path / "GATE1_FREEZE.json")


def test_gate1_rejects_near_zero_activity_manifest(tmp_path):
    with pytest.raises(GATE1["Gate1AuthorizationError"]):
        GATE1["run_gate1_calibration"](_input(tmp_path, _manifest(0.00025, 0.0685)))


def test_gate1_freeze_records_every_gate2_budget(tmp_path):
    receipt = GATE1["run_gate1_calibration"](_input(tmp_path))
    required = {"stage_p_max_epochs", "stage_s_max_epochs", "stage_j_max_epochs", "minimum_epochs", "patience", "validation_interval", "learning_rates", "curriculum", "candidate_values"}
    assert required <= set(receipt["frozen_training"])
    assert receipt["authorized_gate2"] is True
    assert receipt["evaluation_year_accessed"] is False


def test_gate1_rejects_candidates_outside_frozen_contract(tmp_path):
    value = _input(tmp_path)
    with pytest.raises(GATE1["Gate1AuthorizationError"], match="differ from the frozen contract"):
        GATE1["run_gate1_calibration"](replace(value, candidate_values=(1.0, 2.0)))


def test_gate1_cli_exposes_no_calibration_budget_overrides():
    options = {action.dest for action in GATE1["build_gate1_parser"]()._actions}
    assert options.isdisjoint({"epochs", "learning_rate", "candidate_values", "origins"})


def test_gate1_keeps_four_forecast_tasks_but_settles_three_rigid_demands():
    target = torch.arange(2 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4)
    rigid = GATE1["_rigid_demand_target"](target)
    assert rigid.shape == (2, 4, 3)
    assert torch.equal(rigid, target[..., :3])
    assert torch.equal(target[..., 3], torch.tensor([[3.0, 7.0, 11.0, 15.0], [19.0, 23.0, 27.0, 31.0]]))
