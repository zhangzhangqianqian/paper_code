"""Run validation-only heat-pump LP frontier diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.heat_pump.frontier import load_frontier_contract, solve_scenario_frontier, summarize_frontier, frontier_contract_hash
from src.scheduling.heat_pump.parameters import HeatPumpParameters
from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios, load_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    modes = (args.smoke, args.formal, args.dry_run, args.verify_only)
    if sum(bool(value) for value in modes) != 1:
        parser.error("choose exactly one of --dry-run, --smoke, --formal or --verify-only")
    contract = load_frontier_contract(args.contract)
    output = Path(args.output_dir)
    if not args.verify_only and output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output directory: {output}")
    values = load_benchmark(args.benchmark)
    if args.verify_only:
        summary_path = output / "frontier_summary.json"
        manifest_path = output / "frontier_manifest.json"
        receipt_path = output / "heat_pump_gate_receipt.json"
        if not summary_path.is_file() or not manifest_path.is_file():
            raise SystemExit("verify-only requires frontier_summary.json and frontier_manifest.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_contract_hash = frontier_contract_hash(args.contract)
        checks = {
            "summary_status_complete": summary.get("status") == "complete",
            "manifest_status_complete": manifest.get("status") == "complete",
            "summary_test_set_sealed": summary.get("test_set_accessed") is False,
            "manifest_test_set_sealed": manifest.get("test_set_accessed") is False,
            "contract_hash_matches": summary.get("contract_sha256") == expected_contract_hash,
            "manifest_record_count_matches": manifest.get("records") == summary.get("total_records"),
        }
        verified = all(checks.values())
        payload = {
            "schema_version": "heat-pump-frontier-verification-v1",
            "status": "verified" if verified else "verification_failed",
            "decision_status": summary.get("decision_status"),
            "checks": checks,
            "test_set_accessed": False,
            "summary": str(summary_path.resolve()),
            "manifest": str(manifest_path.resolve()),
        }
        if receipt_path.is_file():
            payload["gate_receipt"] = str(receipt_path.resolve())
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if verified else 2
    if args.dry_run:
        sample_count = 16
        print(json.dumps({"status": "dry_run", "sample_count": sample_count, "configurations": len(contract.cop_values) * len(contract.capacity_multipliers) * len(contract.gas_price_multipliers) * len(contract.variable_om_cost_values), "variable_om_cost_values": list(contract.variable_om_cost_values), "nominal_variable_om_cost": contract.nominal_variable_om_cost, "epsilon_levels": list(contract.epsilon_cost_tolerances), "test_set_accessed": False}, ensure_ascii=False, indent=2))
        return 0
    sample_count = 16 if args.smoke else 2048
    batch = generate_synthetic_scenarios(values, "validation", 2027, sample_count, generator_version="synthetic-scheduling-domain-v2", scenario_generation=contract.scenario_generation)
    output.mkdir(parents=True, exist_ok=False)
    all_records: list[dict[str, object]] = []
    base_capacity = float(values["values"]["gas_boiler_capacity"]) * 0.5
    for cop in contract.cop_values:
        for capacity_multiplier in contract.capacity_multipliers:
            for gas_multiplier in contract.gas_price_multipliers:
                for variable_om_cost in contract.variable_om_cost_values:
                    hp = HeatPumpParameters(
                        cop=float(cop),
                        heat_capacity=base_capacity * float(capacity_multiplier),
                        variable_om_cost=float(variable_om_cost),
                    )
                    for index in range(batch.n_samples):
                        rows = solve_scenario_frontier(batch, index, values["values"], hp, float(gas_multiplier), contract.epsilon_cost_tolerances)
                        for row in rows:
                            all_records.append({"cop": cop, "capacity_multiplier": capacity_multiplier, "gas_price_multiplier": gas_multiplier, "variable_om_cost": variable_om_cost, **row})
    fieldnames = sorted({key for row in all_records for key in row})
    with (output / "frontier_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    summary = summarize_frontier(all_records, contract)
    summary.update({"status": "complete", "split": "validation", "sample_count": batch.n_samples, "test_set_accessed": False, "contract_sha256": frontier_contract_hash(args.contract), "benchmark": str(Path(args.benchmark).resolve())})
    (output / "frontier_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "frontier_manifest.json").write_text(json.dumps({"schema_version": "heat-pump-frontier-manifest-v1", "status": "complete", "test_set_accessed": False, "records": len(all_records), "summary": "frontier_summary.json"}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_bytes = (output / "frontier_summary.json").read_bytes()
    gate_receipt = {
        "schema_version": "heat-pump-frontier-gate-receipt-v1",
        "status": "complete",
        "decision_status": summary["decision_status"],
        "gate_checks": summary["gate_checks"],
        "test_set_accessed": False,
        "split": "validation",
        "mode": "formal" if args.formal else "smoke",
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "contract_sha256": summary["contract_sha256"],
    }
    (output / "heat_pump_gate_receipt.json").write_text(json.dumps(gate_receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["decision_status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
