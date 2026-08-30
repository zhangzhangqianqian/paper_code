"""Independent preflight audit for the joint forecast--dispatch evidence tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.contract import load_joint_training_contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=Path("configs/joint_forecast_dispatch_contract_v1.json"))
    parser.add_argument("--report-root", type=Path, default=None)
    args = parser.parse_args(argv)
    contract = load_joint_training_contract(args.contract)
    root = Path(args.report_root) if args.report_root is not None else contract.path("output_root")
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    resource_path = root / "resource_benchmark" / "resource_benchmark_receipt.json"
    checks["resource_gate_receipt"] = resource_path.exists()
    if resource_path.exists():
        receipt = json.loads(resource_path.read_text(encoding="utf-8"))
        gate = receipt.get("resource_gate", {})
        checks["resource_gate_passed"] = bool(gate.get("gate_passed", False))
        checks["resource_gate_exact_count"] = int(gate.get("solve_count", -1)) == int(contract.resource_gate["benchmark_solves"])
    else:
        reasons.append("missing resource benchmark receipt")
    smoke_receipt = root / "smoke" / "smoke_receipt.json"
    checks["smoke_gate_receipt"] = smoke_receipt.exists()
    if not smoke_receipt.exists():
        reasons.append("smoke gate has not produced a complete receipt")
    elif not bool(json.loads(smoke_receipt.read_text(encoding="utf-8")).get("complete", False)):
        checks["smoke_gate_complete"] = False
        reasons.append("smoke receipt is present but incomplete")
    else:
        checks["smoke_gate_complete"] = True
    data_receipt = root / "data" / "data_build_receipt.json"
    checks["formal_data_receipt"] = data_receipt.exists()
    if data_receipt.exists():
        data_payload = json.loads(data_receipt.read_text(encoding="utf-8"))
        checks["formal_data_complete"] = bool(data_payload.get("formal", False)) and not bool(data_payload.get("test_set_accessed", False))
        checks["origin_keyed_future_context"] = bool(data_payload.get("origin_keyed_future_context", False))
        checks["causal_history_source"] = data_payload.get("history_source") == "causal_lp"
        if not checks["formal_data_complete"]:
            reasons.append("formal data receipt is incomplete or test-contaminated")
    else:
        reasons.append("missing formal data build receipt")
    selection_receipt = root / "selection" / "selection_receipt.json"
    checks["selection_receipt"] = selection_receipt.exists()
    if not selection_receipt.exists():
        reasons.append("validation selection receipt is not frozen")
    elif not bool(json.loads(selection_receipt.read_text(encoding="utf-8")).get("complete", False)):
        checks["selection_complete"] = False
        reasons.append("selection receipt is present but incomplete")
    else:
        checks["selection_complete"] = True
    checks["test_not_accessed_without_selection"] = not ((root / "test").exists() and not selection_receipt.exists())
    if not checks["test_not_accessed_without_selection"]:
        reasons.append("test directory exists without a frozen selection receipt")
    # The sealed test is audited separately from the validation preflight.  It
    # must exist only after selection, use exactly the selected variant, and
    # still make no online LP calls.
    test_receipt_path = root / "test" / "test_receipt.json"
    checks["test_receipt"] = test_receipt_path.exists()
    test_set_accessed = False
    if test_receipt_path.exists() and selection_receipt.exists():
        test_payload = json.loads(test_receipt_path.read_text(encoding="utf-8"))
        selection_payload = json.loads(selection_receipt.read_text(encoding="utf-8"))
        test_set_accessed = bool(test_payload.get("test_set_accessed", False))
        checks["test_complete"] = bool(test_payload.get("complete", False)) and test_set_accessed
        checks["test_matches_selection"] = (
            test_payload.get("selected_variant") == selection_payload.get("selected_variant")
            and int(test_payload.get("selected_seed", -1)) == int(selection_payload.get("selected_seed", -2))
        )
        metrics = test_payload.get("metrics", {})
        checks["test_finite_state"] = bool(metrics.get("state_finite", False) and metrics.get("forecast_finite", False))
        checks["test_online_exact_lp_calls_zero"] = int(test_payload.get("online_exact_lp_calls", -1)) == 0
        if not checks["test_complete"]:
            reasons.append("sealed test receipt is incomplete")
        if not checks["test_matches_selection"]:
            reasons.append("sealed test does not match the frozen selection")
        if not checks["test_finite_state"]:
            reasons.append("sealed test contains non-finite state or forecast metrics")
        if not checks["test_online_exact_lp_calls_zero"]:
            reasons.append("sealed test reports an online LP call")
    else:
        checks["test_complete"] = False
        checks["test_matches_selection"] = False
        checks["test_finite_state"] = False
        checks["test_online_exact_lp_calls_zero"] = False
        reasons.append("missing sealed test receipt or frozen selection")
    checks["online_exact_lp_calls_zero"] = checks.get("test_online_exact_lp_calls_zero", True)
    validation_root = root / "validation"
    joint_receipts = [validation_root / "joint_from_scratch" / f"seed_{seed}" / "validation_receipt.json" for seed in contract.seeds]
    checks["joint_validation_receipts"] = all(path.exists() for path in joint_receipts)
    if checks["joint_validation_receipts"]:
        joint_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in joint_receipts]
        checks["joint_rollin_refresh"] = all(bool(item.get("rollin_refresh", {}).get("implemented", False)) for item in joint_payloads)
        checks["joint_validation_no_online_lp"] = all(int(item.get("metrics", {}).get("online_exact_lp_calls", 1)) == 0 and not bool(item.get("test_set_accessed", True)) for item in joint_payloads)
        if not checks["joint_rollin_refresh"]:
            reasons.append("one or more joint validation candidates lacks the required training-only roll-in refresh")
    else:
        reasons.append("missing one or more five-seed joint validation receipts")
    status = "pass" if all(checks.values()) else "blocked"
    manifest = {
        "status": status,
        "contract": str(args.contract.resolve()),
        "report_root": str(root.resolve()),
        "checks": checks,
        "reasons": reasons,
        "test_set_accessed": test_set_accessed,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if status == "pass" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
