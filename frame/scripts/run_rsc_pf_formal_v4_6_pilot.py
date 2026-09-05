"""Run one immutable formal-v4.6 Pilot after a passing train-only diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_artifacts import canonical_sha256, sha256_file, write_json_once  # noqa: E402
from src.joint_dispatch.formal_v4_5_pilot_data import load_v45_pilot_cache, load_v45_training_cache  # noqa: E402
from src.joint_dispatch.formal_v4_6_artifacts import audit_pilot_v46, write_training_artifacts_v46  # noqa: E402
from src.joint_dispatch.formal_v4_6_contract import load_formal_v4_6_contract  # noqa: E402
from src.joint_dispatch.formal_v4_6_metrics import forecast_metrics_v46, risk_metrics_v46  # noqa: E402
from src.joint_dispatch.formal_v4_6_pilot_executor import execute_real_pilot_v46  # noqa: E402
from src.joint_dispatch.formal_v4_6_pilot_gate import authorize_pilot_v46  # noqa: E402
from src.joint_dispatch.formal_v4_6_provenance import validate_source_manifest_v46  # noqa: E402


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _pilot_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    rollouts = result["rollouts"]
    joint = rollouts["rsc_pf_joint"]; decoupled = rollouts["fair_decoupled"]
    joint_metrics = forecast_metrics_v46(joint.forecast_nominal, joint.target, joint.probability)
    dec_metrics = forecast_metrics_v46(decoupled.forecast_nominal, decoupled.target, decoupled.probability)
    risk = risk_metrics_v46(joint.risk_adjustment, joint.risk_cap, joint.target, joint.probability)
    ratio = lambda a, b: float(a / b) if np.isfinite(a) and np.isfinite(b) and b > 0 else float("inf")
    wape = joint_metrics.wape; base_wape = dec_metrics.wape
    four = float(np.mean(list(wape.values()))); four_base = float(np.mean(list(base_wape.values())))
    active = float(np.mean([wape["cooling"], wape["heating"]])); active_base = float(np.mean([base_wape["cooling"], base_wape["heating"]]))
    comparisons = {
        "four_task_score_ratio": ratio(four, four_base), "electricity_wape_ratio": ratio(wape["electricity"], base_wape["electricity"]), "gas_wape_ratio": ratio(wape["gas"], base_wape["gas"]), "active_thermal_wape_ratio": ratio(active, active_base), "normalized_inactive_leakage_ratio": ratio(float(np.mean(list(joint_metrics.inactive_leakage.values()))), float(np.mean(list(dec_metrics.inactive_leakage.values())))),
    }
    calibration = result["calibration"]
    selected = calibration.selected
    return {
        **comparisons,
        "joint_decision_objective": float(np.mean(joint.penalized_objective)), "decoupled_decision_objective": float(np.mean(decoupled.penalized_objective)),
        "joint_shortage": float(np.mean(joint.shortage.sum(axis=1))), "decoupled_shortage": float(np.mean(decoupled.shortage.sum(axis=1))),
        "decoupled_decision_gradient": float(max(selected.pair.decoupled.gradient_norms.get(name, 0.0) for name in ("decision_to_base", "decision_to_gate", "decision_to_magnitude"))),
        "cap_violation": float(risk.maximum_cap_violation), "gas_adjustment": float(risk.gas_adjustment), "physical_residual": float(np.max(joint.physical_residual)),
        "macro_f1_drop": float(max(0.0, dec_metrics.macro_f1 - joint_metrics.macro_f1)), "transition_balanced_accuracy_gain": float(joint_metrics.transition_balanced_accuracy - dec_metrics.transition_balanced_accuracy),
        "forecast": {"joint": joint_metrics.__dict__, "decoupled": dec_metrics.__dict__}, "risk": risk.__dict__, "gradients": dict(selected.pair.decoupled.gradient_norms),
    }


def run_formal_v46_pilot(*, config: str | Path, diagnostic_receipt: str | Path, source_manifest: str | Path, materialized_root: str | Path, train_data: str | Path, benchmark: str | Path, capacity_receipt: str | Path, output_root: str | Path, run_id: str) -> Mapping[str, Any]:
    torch.set_num_threads(1)
    contract = load_formal_v4_6_contract(config)
    diagnostic = _read_json(diagnostic_receipt)
    if diagnostic.get("diagnostic_authorized_pilot") is not True:
        raise ValueError("diagnostic receipt does not authorize Pilot")
    manifest = _read_json(source_manifest)
    validate_source_manifest_v46(manifest, FRAME_ROOT, str(run_id), contract.contract_sha256)
    root = Path(output_root).resolve() / str(run_id)
    if root.exists() and any(root.rglob("PILOT_RECEIPT.json")):
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=False)
    pilot_root = root / "pilot"; pilot_root.mkdir()
    training = load_v45_training_cache(materialized_root=materialized_root, train_data=train_data, benchmark=benchmark, capacity_receipt=capacity_receipt)
    def selection_loader():
        return load_v45_pilot_cache(materialized_root=materialized_root, train_data=train_data, benchmark=benchmark, capacity_receipt=capacity_receipt)
    result = execute_real_pilot_v46(materialized_training=training, selection_loader=selection_loader, contract=contract, artifact_root=pilot_root, benchmark=benchmark, capacity_receipt=capacity_receipt, seed=int(contract.payload["pilot_budget"]["seed"]))
    write_training_artifacts_v46(pilot_root, result["bundle"], contract)
    measurements = _pilot_metrics(result)
    measurements.update({"evaluation_year_accessed": False, "hashes_valid": True})
    decision = authorize_pilot_v46(measurements, contract)
    receipt = {"schema": "formal-v4.6-pilot-receipt-v1", "run_id": str(run_id), "contract_sha256": contract.contract_sha256, "lineage": {"diagnostic_receipt_sha256": sha256_file(diagnostic_receipt), "source_manifest_sha256": sha256_file(source_manifest), "train_data_sha256": sha256_file(train_data), "capacity_receipt_sha256": sha256_file(capacity_receipt)}, "accessed_years": [2015, 2016, 2017, 2018, 2019], "evaluation_year_accessed": False, "authorized_gate1": bool(decision.authorized_gate1), "failures": list(decision.failure_names), "measurements": measurements}
    write_json_once(pilot_root / "PILOT_RECEIPT.json", receipt)
    audit = audit_pilot_v46(pilot_root, contract)
    audit.update({"authorized_gate1": bool(decision.authorized_gate1), "receipt_sha256": canonical_sha256(receipt)})
    audit_hash = write_json_once(pilot_root / "PILOT_AUDIT.json", audit)
    write_json_once(root / "PILOT_TRANSITION.json", {"schema": "formal-v4.6-pilot-transition-v1", "run_id": str(run_id), "contract_sha256": contract.contract_sha256, "authorized_gate1": bool(decision.authorized_gate1), "evaluation_year_accessed": False, "audit_sha256": audit_hash})
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "diagnostic-receipt", "source-manifest", "materialized-root", "train-data", "benchmark", "capacity-receipt", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_formal_v46_pilot(config=args.config, diagnostic_receipt=args.diagnostic_receipt, source_manifest=args.source_manifest, materialized_root=args.materialized_root, train_data=args.train_data, benchmark=args.benchmark, capacity_receipt=args.capacity_receipt, output_root=args.output_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"authorized_gate1": False, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False)); return 2
    print(json.dumps({"authorized_gate1": receipt["authorized_gate1"], "failures": receipt["failures"]}, ensure_ascii=False)); return 0 if receipt["authorized_gate1"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
