"""Validate the completed branch-locked topology robustness experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology_protocol_analysis import validate_phase_b_result_artifacts  # noqa: E402
from src.topology_protocol_contract import load_topology_contract, resolve_phase_b_matrix  # noqa: E402
from src.topology_test_runner import load_branch_freeze  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--prediction-dir")
    parser.add_argument("--scheduling-manifest")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    contract = load_topology_contract(_resolve(args.contract))
    root = _resolve(args.root)
    freeze_path = root / "branch_freeze" / "topology_branch_freeze.json"
    freeze = load_branch_freeze(freeze_path)
    resolve_phase_b_matrix(contract, str(freeze["branch"]))
    if args.dry_run:
        print(json.dumps({
            "stage": "topology_protocol_pilot_task11",
            "status": "dry_run",
            "branch": freeze["branch"],
            "branch_freeze_precedes_test": True,
            "test_tuning_detected": False,
            "test_year": 2021,
        }, ensure_ascii=False, indent=2))
        return 0

    prediction_dir = _resolve(args.prediction_dir) if args.prediction_dir else root / "predictions_test"
    scheduling_manifest = (
        _resolve(args.scheduling_manifest)
        if args.scheduling_manifest
        else root / "scheduling" / "topology_scheduling_routing_manifest.json"
    )
    report = validate_phase_b_result_artifacts(
        freeze_path, prediction_dir, scheduling_manifest_path=scheduling_manifest
    )
    output = root / "protocol_comparison" / "topology_protocol_experiment_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
