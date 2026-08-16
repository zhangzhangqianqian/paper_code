"""Run the branch-authorized 2021 prediction phase of the topology protocol."""

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
from src.topology_test_runner import (  # noqa: E402
    build_phase_b_run_plan,
    load_branch_freeze,
    run_phase_b,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--branch-freeze", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-freeze-sha256")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    contract_path = _resolve(args.contract)
    freeze_path = _resolve(args.branch_freeze)
    contract = load_topology_contract(contract_path)
    freeze = load_branch_freeze(freeze_path, expected_sha256=args.expected_freeze_sha256)
    branch = str(freeze["branch"])
    resolve_phase_b_matrix(contract, branch)
    plan = build_phase_b_run_plan(freeze)
    if args.dry_run:
        print(json.dumps({
            "stage": "topology_protocol_pilot_phase_b",
            "status": "dry_run",
            "branch": branch,
            "branch_frozen_before_test": True,
            "test_set_accessed": False,
            "run_count": len(plan),
            "models": sorted({str(row["model"]) for row in plan}),
            "seeds": sorted({int(row["seed"]) for row in plan}),
            "training_years": [2017, 2018, 2019],
            "validation_year": 2020,
            "test_year": 2021,
            "runs": list(plan),
        }, ensure_ascii=False, indent=2))
        return 0

    manifest = run_phase_b(
        _resolve(args.data_dir), freeze_path, _resolve(args.output_dir),
        contract_path=contract_path, smoke=args.smoke, resume=args.resume,
        expected_freeze_sha256=args.expected_freeze_sha256,
    )
    print(json.dumps({
        "stage": manifest["stage"],
        "status": manifest["status"],
        "branch": manifest["branch"],
        "run_count_completed": manifest["run_count_completed"],
        "run_count_expected": manifest["run_count_expected"],
        "test_set_accessed": manifest["test_set_accessed"],
        "output_dir": str(_resolve(args.output_dir)),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
