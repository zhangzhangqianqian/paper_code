"""CLI for validation-only calibration and formal external-baseline runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_baseline_training import (  # noqa: E402
    METHODS,
    SEEDS,
    run_external_calibration,
    run_external_validation,
)


def _load_config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("implementation config must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen external baseline calibration/validation")
    parser.add_argument("--stage", choices=("calibration", "validation"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--config", type=Path, default=FRAME_ROOT / "configs" / "rsc_pf_external_baseline_implementation_v1.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke-limit", type=int)
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = _load_config(args.config)
    if args.data_root is not None:
        config["data_root"] = str(args.data_root)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.smoke_limit is not None:
        config["smoke_limit"] = int(args.smoke_limit)
        config["smoke_epochs"] = int(args.smoke_epochs)
    config["resume"] = bool(args.resume)
    if args.stage == "calibration":
        if args.all_seeds or args.seed != 2026:
            raise ValueError("calibration is restricted to seed 2026 and does not accept --all-seeds")
        path = run_external_calibration(args.method, 2026, config)
        print(json.dumps({"stage": args.stage, "method": args.method, "seed": 2026, "checkpoint": str(path)}, ensure_ascii=False))
        return 0
    seeds = SEEDS if args.all_seeds else (args.seed,)
    results = []
    for seed in seeds:
        path = run_external_validation(args.method, int(seed), config)
        results.append({"stage": args.stage, "method": args.method, "seed": int(seed), "checkpoint": str(path)})
    print(json.dumps(results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

