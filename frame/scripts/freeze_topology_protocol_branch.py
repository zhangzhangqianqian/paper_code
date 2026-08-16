"""Compare the two validation protocols and freeze the Phase-B branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology_protocol_analysis import (  # noqa: E402
    compare_validation_protocols,
    decide_topology_branch,
    freeze_topology_branch,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_effect_rows(path: Path) -> list[dict[str, object]]:
    import csv
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_did_rows(path: Path) -> list[dict[str, object]]:
    return _read_effect_rows(path)


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _phase_b_matrix(contract: dict[str, object], branch: str) -> dict[str, object]:
    phase_a = contract["phase_a"]
    phase_b = contract["phase_b"]
    if branch == "core_conclusion_stable":
        models = [
            {"model": str(item["model"]), "candidate_id": str(item["candidate_id"])}
            for item in phase_a["models"]
        ]
        seeds = [int(value) for value in phase_a["seeds"]]
    else:
        models = [
            {"model": str(item["model"]), "candidate_id": str(item["candidate_id"])}
            for item in phase_b["joint_models"]
        ]
        seeds = [int(value) for value in phase_b["formal_seeds"]]
    return {
        "models": models,
        "seeds": seeds,
        "training_years": [2017, 2018, 2019],
        "validation_year": 2020,
        "test_year": 2021,
        "forecast_years": [2017, 2018, 2019, 2020, 2021],
        "scheduling_tracks": ["real_replay", "simulated_dispatch"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-registry")
    parser.add_argument("--post-output-dir")
    parser.add_argument("--contract")
    parser.add_argument("--audit-dir")
    parser.add_argument("--pilot-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=168)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    contract = None
    if args.contract:
        from src.topology_protocol_contract import load_topology_contract
        contract = load_topology_contract(_resolve(args.contract))
    if args.pilot_dir and not args.post_output_dir:
        args.post_output_dir = args.pilot_dir
    if args.audit_dir and not args.cross_registry:
        args.cross_registry = str(Path(args.audit_dir).parent / "cross_topology_artifact_registry.json")
    if not args.cross_registry or not args.post_output_dir:
        parser.error("--cross-registry and --post-output-dir (or --pilot-dir/--audit-dir aliases) are required")
    if args.dry_run:
        print(json.dumps({
            "stage": "topology_protocol_pilot_task7",
            "status": "dry_run",
            "test_set_accessed": False,
            "bootstrap_replicates": args.bootstrap_replicates,
            "block_length": args.block_length,
            "comparison_output_dir": str(_resolve(args.output_dir)),
        }, ensure_ascii=False, indent=2))
        return 0

    manifest = compare_validation_protocols(
        _resolve(args.cross_registry),
        _resolve(args.post_output_dir),
        _resolve(args.output_dir),
        repo_root=_resolve(args.repo_root),
        bootstrap_replicates=args.bootstrap_replicates,
        block_length=args.block_length,
        seed=args.seed,
    )
    output_root = _resolve(args.output_dir)
    effect_rows = _read_effect_rows(output_root / "protocol_effect_bootstrap.csv")
    did_rows = _read_did_rows(output_root / "model_gap_difference_in_differences.csv")
    branch = decide_topology_branch(effect_rows, did_rows)
    if contract is None:
        raise ValueError("formal branch freeze requires --contract")
    phase_b_matrix = _phase_b_matrix(contract, branch)
    input_hashes: dict[str, object] = {
        "contract": _sha256(_resolve(args.contract)),
        "comparison_manifest": _sha256(output_root / "validation_comparison_manifest.json"),
    }
    if args.audit_dir:
        audit_root = _resolve(args.audit_dir)
        audit_files = sorted(path for path in audit_root.rglob("*") if path.is_file())
        input_hashes["audit_files"] = {
            str(path.relative_to(audit_root)): _sha256(path) for path in audit_files
        }
    freeze = freeze_topology_branch(
        branch,
        manifest,
        output_root / "branch_freeze",
        git_revision=_git_revision(),
        phase_b_matrix=phase_b_matrix,
        input_hashes=input_hashes,
    )
    print(json.dumps({
        "stage": "topology_protocol_pilot_task7",
        "status": "passed",
        "branch": branch,
        "common_origin_count": manifest["common_origin_count"],
        "test_set_accessed": False,
        "freeze_file": str(output_root / "branch_freeze" / ("pilot_invalid.json" if branch == "pilot_invalid" else "topology_branch_freeze.json")),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
