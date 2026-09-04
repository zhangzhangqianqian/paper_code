from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path
import shutil

import numpy as np

from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_once_json
from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract
from src.joint_dispatch.formal_v4_2_gate2_artifacts import write_gate2_rollout, write_gate2_row_receipt
from src.joint_dispatch.formal_v4_2_metrics import compute_v42_metrics_from_arrays
from src.joint_dispatch.formal_v4_2_methods import registered_method_rows


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json"
GATE2 = run_path(str(ROOT / "scripts" / "run_rsc_pf_formal_v4_2_gate2.py"))
AUDIT = run_path(str(ROOT / "scripts" / "audit_rsc_pf_formal_v4_2_gate2.py"))
ONLINE = {
    "Scheme2R-PTO", "State-Conditioned-PTO", "Official iTransformer-PTO",
    "Differentiable-LP", "Perfect-Information-MPC", "Seasonal-Naive-PTO",
}


def _arrays(objective: float) -> dict[str, np.ndarray]:
    n = 2
    return {
        "forecast_target": np.ones((n, 4, 4)), "forecast_prediction": np.ones((n, 4, 4)),
        "planned_dispatch": np.zeros((n, 4, 21)), "settled_dispatch": np.zeros((n, 21)),
        "shortage_energy": np.zeros((n, 3)), "p_dump": np.zeros(n), "q_dump": np.zeros(n),
        "operating_cost": np.zeros(n), "physical_carbon": np.zeros(n),
        "penalized_objective": np.full(n, objective),
        "target_times": np.arange(n, dtype=np.int64), "state_hashes": np.asarray(["a", "b"]),
        "residual_balance": np.zeros((n, 3)), "residual_capacity": np.zeros((n, 21)),
        "residual_conversion": np.zeros((n, 5)), "residual_soc": np.zeros((n, 2)),
        "residual_ramp": np.zeros((n, 1)), "residual_exclusivity": np.zeros((n, 1)),
        "residual_renewable_accounting": np.zeros((n, 2)), "residual_finite": np.zeros((n, 1)),
    }


def _complete_root(tmp_path: Path) -> Path:
    contract = load_formal_v4_2_contract(CONFIG)
    root = tmp_path / "run"; protocol = root / "protocol"; protocol.mkdir(parents=True)
    shutil.copyfile(CONFIG, protocol / "formal_v4_2_contract.json")
    lineage = {
        "contract_sha256": contract.contract_sha256, "source_manifest_sha256": "1" * 64,
        "train_manifest_sha256": "2" * 64, "calibration_manifest_sha256": "3" * 64,
        "evaluation_manifest_sha256": "4" * 64, "normalization_sha256": "5" * 64,
    }
    rows = {}
    for expected in registered_method_rows(contract, gate="gate2"):
        seed_name = "deterministic" if expected.seed is None else str(expected.seed)
        directory = root / "gate2" / "rows" / expected.method_id / seed_name
        directory.mkdir(parents=True)
        objective = 1.0 if expected.method_id == "RSC-PF" else (2.0 if expected.method_id == "Decoupled-RSC-PF" else 3.0)
        arrays = _arrays(objective)
        rollout_hash = write_gate2_rollout(directory / "ROLLOUT.npz", arrays, lineage)
        metrics = compute_v42_metrics_from_arrays(arrays, method_id=expected.method_id)
        metrics_payload = metrics.to_dict()
        write_once_json(directory / "METRICS.json", metrics_payload)
        parent = "9" * 64 if expected.method_id in {"RSC-PF", "Decoupled-RSC-PF"} else "not_applicable"
        if expected.seed is not None:
            (directory / "CHECKPOINT.pt").write_bytes(b"trained-checkpoint")
            training = {
                "optimizer_steps": 1, "stage_s_parent_sha256": parent,
                "decision_forecaster_gradient_norm": 1.0 if expected.method_id == "RSC-PF" else 0.0,
                "forecast_loss_applicable": expected.method_id != "Direct-Policy",
            }
            if expected.method_id == "Official iTransformer-PTO":
                training["upstream_commit"] = "c2426e68ca13f74aaec08045c5c724d8ad328124"
            if expected.method_id == "Differentiable-LP":
                training.update({"sample_exposures": 2, "expected_sample_exposures": 2, "failed_solves": 0, "gradient_norm": 1.0})
            write_once_json(directory / "TRAINING_RECEIPT.json", training)
            checkpoint_hash = sha256_file(directory / "CHECKPOINT.pt")
            training_hash = sha256_file(directory / "TRAINING_RECEIPT.json")
        else:
            checkpoint_hash = training_hash = "not_applicable"
        row = {
            **lineage, "method_id": expected.method_id, "seed": expected.seed,
            "checkpoint_sha256": checkpoint_hash, "training_receipt_sha256": training_hash,
            "rollout_sha256": rollout_hash, "metrics_sha256": sha256_file(directory / "METRICS.json"),
            "settled_hours": 2, "optimizer_calls": 2 if expected.method_id in ONLINE else 0,
            "penalized_objective": metrics.penalized_objective,
            "shortage_energy": float(metrics.shortage_energy.sum()),
            "balance_residual_max": 0.0, "capacity_violation_max": 0.0,
            "physical_residual_max": 0.0, "stage_s_parent_sha256": parent,
            "test_set_accessed": False,
        }
        write_gate2_row_receipt(directory, row)
        label = expected.method_id if expected.seed is None else f"{expected.method_id}__seed_{expected.seed}"
        rows[label] = json.loads((directory / "ROW_RECEIPT.json").read_text(encoding="utf-8"))
    write_once_json(root / "gate2" / "rows.json", rows)
    decision = GATE2["authorize_gate2"](list(rows.values()), contract)
    GATE2["write_gate2_decision"](
        root / "gate2" / "GATE2_DECISION.json", decision,
        contract_sha256=contract.contract_sha256,
    )
    return root


def test_audit_accepts_complete_recomputed_matrix(tmp_path: Path) -> None:
    root = _complete_root(tmp_path)
    audit = AUDIT["audit_gate2"](root)
    assert audit.authorized_gate3 is True
    assert audit.recomputed_row_count == 23
    assert (root / "protocol" / "GATE2_TRANSITION.json").is_file()


def test_audit_rejects_metric_not_supported_by_rollout(tmp_path: Path) -> None:
    root = _complete_root(tmp_path)
    path = root / "gate2" / "rows" / "RSC-PF" / "2026" / "METRICS.json"
    payload = json.loads(path.read_text(encoding="utf-8")); payload["penalized_objective"] = -1.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert AUDIT["audit_gate2"](root).authorized_gate3 is False


def test_audit_rejects_wrong_stage_s_ancestor(tmp_path: Path) -> None:
    root = _complete_root(tmp_path)
    path = root / "gate2" / "rows" / "RSC-PF" / "2026" / "ROW_RECEIPT.json"
    payload = json.loads(path.read_text(encoding="utf-8")); payload["stage_s_parent_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert AUDIT["audit_gate2"](root).authorized_gate3 is False
