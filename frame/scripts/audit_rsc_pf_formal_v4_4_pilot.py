"""Independent audit of a formal-v4.4 Pilot run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path: sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_artifacts import canonical_sha256, sha256_file, write_json_once  # noqa: E402
from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract  # noqa: E402
from src.joint_dispatch.formal_v4_4_metrics import compute_forecast_metrics_v44  # noqa: E402
from src.joint_dispatch.formal_v4_4_pilot_gate import PilotDecisionV44, authorize_pilot_v44  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"JSON object required: {path}")
    return value


def audit_run(run_dir: str | Path, contract_path: str | Path) -> PilotDecisionV44:
    root = Path(run_dir).resolve(); contract = load_formal_v4_4_contract(contract_path); pilot = root / "pilot"
    failures: list[str] = []
    try:
        receipt = _json(pilot / "PILOT_RECEIPT.json"); saved_audit = _json(pilot / "PILOT_AUDIT.json")
        with np.load(pilot / "PILOT_ARRAYS.npz", allow_pickle=False) as arrays:
            required = {name: arrays[name] for name in ("prediction", "target", "probability", "prior_probability", "regimes", "times")}
        metrics = compute_forecast_metrics_v44(required["prediction"], required["target"], required["probability"], required["prior_probability"], required["regimes"], required["times"])
        metrics_dict = asdict(metrics)
        if canonical_sha256(receipt.get("metrics", {})) != canonical_sha256(metrics_dict): failures.append("metrics_recompute")
        if saved_audit.get("metrics_sha256") != canonical_sha256(metrics_dict): failures.append("metrics_hash")
        if saved_audit.get("receipt_sha256") != canonical_sha256(receipt): failures.append("receipt_hash")
        split_path = root / "gate0" / "PILOT_SPLIT.npz"
        if split_path.is_file():
            with np.load(split_path, allow_pickle=False) as split:
                values = [np.asarray(split[name], dtype=np.int64) for name in split.files]
            if any(len(value) == 0 or len(np.unique(value)) != len(value) for value in values): failures.append("split_integrity")
            if len(values) >= 2 and set(values[0].tolist()) & set(values[1].tolist()): failures.append("train_eval_overlap")
        else: failures.append("split_missing")
        if receipt.get("accessed_years") != [2015, 2016, 2017, 2018, 2019]: failures.append("evaluation_access")
        recomputed = authorize_pilot_v44(receipt, contract)
        failures.extend(recomputed.failures)
        if bool(receipt.get("authorized_gate1")) != recomputed.authorized_gate1: failures.append("authorization_disagreement")
        result = PilotDecisionV44(not failures and recomputed.authorized_gate1, recomputed.criteria, tuple(sorted(set(failures))), recomputed.measured, tuple(receipt.get("accessed_years", ())), tuple(receipt.get("rows", ())), str(saved_audit.get("receipt_sha256", "")))
    except Exception as exc:
        result = PilotDecisionV44(False, {}, (f"audit_exception:{type(exc).__name__}",), {}, (), (), "")
    audit_payload = {"schema": "formal-v4.4-pilot-independent-audit-v1", "authorized_gate1": result.authorized_gate1, "failures": list(result.failures), "criteria": dict(result.criteria), "audit_sha256": canonical_sha256({"authorized_gate1": result.authorized_gate1, "failures": list(result.failures), "criteria": dict(result.criteria)})}
    try: write_json_once(pilot / "PILOT_INDEPENDENT_AUDIT.json", audit_payload)
    except FileExistsError: pass
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--contract", type=Path, required=True); args = parser.parse_args(argv)
    result = audit_run(args.run_dir, args.contract); print(json.dumps({"authorized_gate1": result.authorized_gate1, "failures": list(result.failures)}, ensure_ascii=False)); return 0 if result.authorized_gate1 else 2


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["audit_run", "main"]
