from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from runpy import run_path

import numpy as np
import pytest

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
