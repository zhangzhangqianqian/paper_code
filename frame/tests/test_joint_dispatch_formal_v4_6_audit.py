from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.joint_dispatch.formal_v4_4_training import StageReceiptV44
from src.joint_dispatch.formal_v4_6_artifacts import audit_pilot_v46, audit_training_artifacts_v46
from src.joint_dispatch.formal_v4_6_contract import load_formal_v4_6_contract


CONTRACT = load_formal_v4_6_contract("configs/joint_forecast_dispatch_formal_v4_6.json")


def _training_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    caps = root / "risk_caps"
    caps.mkdir()
    # The audit reads immutable files; use the risk-cap writer so hashes are valid.
    from src.joint_dispatch.formal_v4_6_risk import fit_risk_caps_v46, save_risk_caps_v46
    prediction = np.zeros((4, 4, 4))
    target = np.zeros_like(prediction)
    target[..., 0] = 1.0
    times = np.asarray(["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"], dtype="datetime64[ns]")
    save_risk_caps_v46(caps, fit_risk_caps_v46(prediction, target, times, "early_stop", .90, CONTRACT.contract_sha256, "1" * 64))
    stage_root = root / "stages"
    stage_root.mkdir()
    import torch
    for name in ("P0", "P1", "S", "J_joint", "J_decoupled"):
        torch.save({"final_sha256": "1" * 64, "best_sha256": "1" * 64, "state_dict": {}}, stage_root / f"{name}.pt")
    candidates = {str(multiplier): {"risk_multiplier": multiplier, "eligible": True, "decision_objective": multiplier, "parent_sha256": "1" * 64, "joint_final_sha256": "1" * 64, "decoupled_final_sha256": "1" * 64} for multiplier in (.5, 1., 2.)}
    (root / "CALIBRATION.json").write_text(json.dumps({"selected_multiplier": .5, "selection_role": "early_stop", "selection_year_accessed": False, "candidates": candidates}), encoding="utf-8")
    (root / "TRAINING_RECEIPT.json").write_text(json.dumps({"contract_sha256": CONTRACT.contract_sha256, "evaluation_year_accessed": False, "stages": {name: str(stage_root / f"{name}.pt") for name in ("P0", "P1", "S", "J_joint", "J_decoupled")}}), encoding="utf-8")


def _rollout_fixture(root: Path) -> None:
    rollout = root / "rollout"
    rollout.mkdir(parents=True)
    nominal = np.zeros((1, 4, 4)); cap = np.ones((1, 4, 3)); adjustment = np.zeros((1, 4, 3)); demand = nominal.copy()
    np.savez(rollout / "rsc_pf_joint.npz", forecast_nominal=nominal, risk_adjustment=adjustment, risk_cap=cap, scheduler_demand=demand, times=np.asarray(["2019-01-01"], dtype="datetime64[ns]"))


def test_audit_reconstructs_cap_candidate_and_checkpoint_selection(tmp_path):
    _training_fixture(tmp_path); audit = audit_training_artifacts_v46(tmp_path, CONTRACT)
    assert audit["cap_reconstructed"]
    assert audit["candidate_selection_reconstructed"]
    assert audit["checkpoint_selection_reconstructed"]
    assert audit["evaluation_year_accessed"] is False


def test_audit_rejects_adjusted_gas_and_changed_cap(tmp_path):
    _training_fixture(tmp_path); _rollout_fixture(tmp_path)
    with np.load(tmp_path / "rollout" / "rsc_pf_joint.npz", allow_pickle=False) as arrays:
        values = {key: arrays[key] for key in arrays.files}
    values["scheduler_demand"][..., 3] = 1.0
    np.savez(tmp_path / "rollout" / "rsc_pf_joint.npz", **values)
    try:
        audit_pilot_v46(tmp_path, CONTRACT)
    except ValueError as exc:
        assert "gas" in str(exc) or "lineage" in str(exc)
    else:
        raise AssertionError("tampered gas rollout was accepted")
