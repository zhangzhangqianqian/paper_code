"""Immutable v4.6 training/Pilot artifacts and independent audits."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .formal_v4_4_artifacts import canonical_sha256, write_json_once, write_npz_once
from .formal_v4_6_contract import FormalV46Contract
from .formal_v4_6_risk import RiskCapReceiptV46, load_risk_caps_v46, save_risk_caps_v46
from .formal_v4_6_pilot_executor import TrainingBundleV46


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items() if key != "model"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _write_stage(root: Path, name: str, receipt: Any) -> str:
    if receipt.model is None:
        raise ValueError(f"stage {name} has no model")
    path = root / "stages" / f"{name}.pt"
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": receipt.stage, "mode": receipt.mode, "parent_sha256": receipt.parent_sha256,
        "final_sha256": receipt.final_sha256, "best_sha256": receipt.best_sha256,
        "terminal_sha256": receipt.terminal_sha256, "epochs": receipt.epochs,
        "best_epoch": receipt.best_epoch, "optimizer_steps": receipt.optimizer_steps,
        "gradient_norms": dict(receipt.gradient_norms),
        "state_dict": {name: value.detach().cpu().clone() for name, value in receipt.model.state_dict().items()},
    }
    temporary = path.with_name(f".{path.name}.writing")
    torch.save(payload, temporary)
    temporary.replace(path)
    return str(path)


def write_training_artifacts_v46(root: str | Path, bundle: TrainingBundleV46, contract: FormalV46Contract) -> Mapping[str, str]:
    contract.validate()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not isinstance(bundle.risk_caps, RiskCapReceiptV46):
        raise TypeError("bundle.risk_caps must be RiskCapReceiptV46")
    risk_root = root / "risk_caps"
    risk_root.mkdir(parents=True, exist_ok=True)
    save_risk_caps_v46(risk_root, bundle.risk_caps)
    stages = {
        "P0": bundle.p0, "P1": bundle.p1, "S": bundle.s,
        "J_joint": bundle.calibration.selected.pair.joint,
        "J_decoupled": bundle.calibration.selected.pair.decoupled,
    }
    paths = {name: _write_stage(root, name, receipt) for name, receipt in stages.items()}
    calibration_payload = {
        "schema": "formal-v4.6-calibration-v1",
        "selected_multiplier": float(bundle.calibration.selected_multiplier),
        "selection_role": str(bundle.calibration.selection_role),
        "selection_year_accessed": bool(bundle.calibration.selection_year_accessed),
        "candidates": {
            str(key): {
                "risk_multiplier": float(candidate.risk_multiplier),
                "eligible": bool(candidate.validation.eligible),
                "decision_objective": float(candidate.validation.decision_objective),
                "failure_names": list(candidate.validation.failure_names),
                "joint_final_sha256": candidate.pair.joint.final_sha256,
                "decoupled_final_sha256": candidate.pair.decoupled.final_sha256,
                "parent_sha256": candidate.pair.joint.parent_sha256,
            }
            for key, candidate in bundle.calibration.candidates.items()
        },
    }
    calibration_path = root / "CALIBRATION.json"
    write_json_once(calibration_path, calibration_payload)
    receipt_payload = {
        "schema": "formal-v4.6-training-artifacts-v1",
        "contract_sha256": contract.contract_sha256,
        "risk_caps": str(risk_root / "RISK_CAPS.json"),
        "calibration": str(calibration_path),
        "stages": paths,
        "evaluation_year_accessed": False,
    }
    receipt_path = root / "TRAINING_RECEIPT.json"
    write_json_once(receipt_path, receipt_payload)
    return {"training_receipt": str(receipt_path), "calibration": str(calibration_path), "risk_caps": str(risk_root / "RISK_CAPS.json"), **{f"stage_{name}": path for name, path in paths.items()}}


def audit_training_artifacts_v46(root: str | Path, contract: FormalV46Contract) -> Mapping[str, Any]:
    contract.validate()
    root = Path(root)
    receipt = json.loads((root / "TRAINING_RECEIPT.json").read_text(encoding="utf-8"))
    if receipt.get("contract_sha256") != contract.contract_sha256:
        raise ValueError("training artifact contract hash mismatch")
    caps = load_risk_caps_v46(root / "risk_caps")
    calibration = json.loads((root / "CALIBRATION.json").read_text(encoding="utf-8"))
    candidates = calibration.get("candidates", {})
    eligible = [row for row in candidates.values() if bool(row.get("eligible"))]
    if not eligible:
        raise ValueError("no eligible calibration candidate")
    selected_key = str(calibration.get("selected_multiplier"))
    selected = candidates.get(selected_key)
    expected = min(eligible, key=lambda row: float(row["decision_objective"]))
    candidate_ok = selected is not None and float(selected["decision_objective"]) == float(expected["decision_objective"])
    checkpoint_ok = True
    for name, path in receipt.get("stages", {}).items():
        if not Path(path).is_file():
            checkpoint_ok = False
            break
        payload = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_ok = checkpoint_ok and payload.get("final_sha256") == payload.get("best_sha256")
    return {
        "cap_reconstructed": bool(caps.cap.shape == (4, 3) and caps.split_role == "early_stop" and caps.years and all(year in (2015, 2016, 2017, 2018) for year in caps.years)),
        "candidate_selection_reconstructed": bool(candidate_ok),
        "checkpoint_selection_reconstructed": bool(checkpoint_ok),
        "evaluation_year_accessed": bool(receipt.get("evaluation_year_accessed", False)),
        "selection_year_accessed": bool(calibration.get("selection_year_accessed", False)),
        "contract_sha256": contract.contract_sha256,
    }


def audit_pilot_v46(root: str | Path, contract: FormalV46Contract) -> Mapping[str, Any]:
    root = Path(root)
    training_audit = audit_training_artifacts_v46(root, contract)
    rollout_root = root / "rollout"
    files = sorted(rollout_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError("Pilot rollout artifacts are missing")
    gas_ok = True; cap_ok = True; years: set[int] = set()
    for path in files:
        with np.load(path, allow_pickle=False) as arrays:
            required = {"forecast_nominal", "risk_adjustment", "risk_cap", "scheduler_demand", "times"}
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(f"rollout artifact is missing {sorted(missing)[0]}")
            nominal = np.asarray(arrays["forecast_nominal"]); adjustment = np.asarray(arrays["risk_adjustment"]); cap = np.asarray(arrays["risk_cap"]); demand = np.asarray(arrays["scheduler_demand"]); times = np.asarray(arrays["times"], dtype="datetime64[ns]")
            gas_ok = gas_ok and bool(np.array_equal(demand[..., 3], nominal[..., 3]))
            cap_ok = cap_ok and bool(np.all(adjustment <= cap + 1.0e-6) and np.all(adjustment >= -1.0e-8))
            years.update(int(value) for value in times.astype("datetime64[Y]").astype(int) + 1970)
    if not gas_ok:
        raise ValueError("gas semantic or lineage audit failed")
    if not cap_ok:
        raise ValueError("risk cap audit failed")
    return {**training_audit, "gas_semantics_valid": True, "risk_caps_valid": True, "rollout_years": sorted(years), "evaluation_year_accessed": bool(training_audit["evaluation_year_accessed"] or 2020 in years)}


__all__ = ["audit_pilot_v46", "audit_training_artifacts_v46", "write_training_artifacts_v46"]
