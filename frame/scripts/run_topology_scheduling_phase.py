"""Validate and route frozen topology forecasts into the existing R/S tracks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology_protocol_contract import load_topology_contract, resolve_phase_b_matrix  # noqa: E402
from src.topology_scheduling_runner import build_routing_manifest, build_scheduling_run_plan  # noqa: E402
from src.topology_test_runner import load_branch_freeze  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch-freeze", required=True)
    parser.add_argument("--prediction-dir", required=True)
    # These paths are accepted and recorded to make the command auditable;
    # the existing LP implementation remains the sole solver owner.
    parser.add_argument("--scheduling-contract", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--renewable-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    contract = load_topology_contract(_resolve(args.contract))
    freeze = load_branch_freeze(_resolve(args.branch_freeze))
    resolve_phase_b_matrix(contract, str(freeze["branch"]))
    plan = build_scheduling_run_plan(freeze)
    if args.dry_run:
        print(json.dumps({
            "stage": "topology_protocol_pilot_task10",
            "status": "dry_run",
            "branch": freeze["branch"],
            "branch_frozen_before_scheduling": True,
            "test_set_accessed": False,
            "lp_equations_modified": False,
            "run_count": len(plan),
            "real_replay_runs": sum(row["track"] == "real_replay" for row in plan),
            "simulated_dispatch_runs": sum(row["track"] == "simulated_dispatch" for row in plan),
            "scheduling_contract": str(_resolve(args.scheduling_contract)),
            "benchmark": str(_resolve(args.benchmark)),
            "ledger": str(_resolve(args.ledger)),
            "renewable_file": str(_resolve(args.renewable_file)),
        }, ensure_ascii=False, indent=2))
        return 0

    for path in (args.scheduling_contract, args.benchmark, args.ledger, args.renewable_file):
        if not _resolve(path).is_file():
            raise FileNotFoundError(_resolve(path))
    manifest = build_routing_manifest(
        freeze, _resolve(args.prediction_dir), output_dir=_resolve(args.output_dir)
    )
    manifest["scheduling_contract"] = str(_resolve(args.scheduling_contract))
    manifest["benchmark"] = str(_resolve(args.benchmark))
    manifest["ledger"] = str(_resolve(args.ledger))
    manifest["renewable_file"] = str(_resolve(args.renewable_file))
    (_resolve(args.output_dir) / "topology_scheduling_routing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "stage": manifest["stage"], "status": manifest["status"],
        "branch": manifest["branch"], "run_count": len(manifest["runs"]),
        "lp_equations_modified": manifest["lp_equations_modified"],
        "output_dir": str(_resolve(args.output_dir)),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
