"""Compare the two validation protocols and freeze the Phase-B branch."""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cross-registry", required=True)
    parser.add_argument("--post-output-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=168)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
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
    freeze = freeze_topology_branch(
        branch,
        manifest,
        output_root / "branch_freeze",
        git_revision=_git_revision(),
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
