"""Run the Phase-A post-gas-engine topology validation pilot.

The command reads only 2017--2020 and produces validation-only artifacts.
Use ``--dry-run`` before any data access.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology_validation_runner import (  # noqa: E402
    MAX_EPOCHS,
    PHASE_A_MODELS,
    PHASE_A_SEEDS,
    build_phase_a_run_plan,
    run_validation_pilot,
)


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    runs = build_phase_a_run_plan()
    if args.dry_run:
        print(json.dumps({
            "stage": "topology_protocol_pilot_phase_a",
            "status": "dry_run",
            "data_read": False,
            "years_loaded": [2017, 2018, 2019, 2020],
            "sealed_years": [2021],
            "run_count": len(runs),
            "models": [{"model": model, "candidate_id": candidate} for model, candidate in PHASE_A_MODELS],
            "seeds": list(PHASE_A_SEEDS),
            "runs": list(runs),
        }, ensure_ascii=False, indent=2))
        return 0

    sample_limit = 512 if args.smoke else None
    max_epochs = 2 if args.smoke else MAX_EPOCHS
    result = run_validation_pilot(
        _resolve(args.data_dir),
        _resolve(args.output_dir),
        sample_limit=sample_limit,
        max_epochs=max_epochs,
        resume=args.resume,
    )
    print(json.dumps({
        "stage": result["stage"],
        "status": result["status"],
        "run_count": result["run_count"],
        "output_dir": str(_resolve(args.output_dir)),
        "years_loaded": result["years_loaded"],
        "test_set_accessed": result["test_set_accessed"],
        "sample_counts": result["sample_counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
