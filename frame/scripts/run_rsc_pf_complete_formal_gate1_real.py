"""Run the real complete-v1 Gate 1 training and 2019 rolling evaluation.

This command is intentionally separate from the metadata-only Gate 1 command.
It requires an authorized Gate 0 transition and an explicit v4.2 source run
containing the already materialized 2015--2018 and 2019 windows.  Without
``--smoke`` it performs the full 37-row matrix and can take many hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.complete_formal_gate1 import Gate1RunConfig, run_complete_gate1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json")
    parser.add_argument("--gate0-transition", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True, help="v4.2 run containing Gate 1 train/selection windows")
    parser.add_argument("--output-root", type=Path, default=FRAME_ROOT / "reports" / "rsc_pf_complete_formal")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-from", type=Path, help="read-only prior Gate 1 run whose verified artifacts may be reused")
    parser.add_argument("--smoke", action="store_true", help="run a tiny non-authorizing wiring check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Gate1RunConfig(
        contract_path=args.contract.resolve(),
        gate0_transition_path=args.gate0_transition.resolve(),
        source_run_root=args.source_run.resolve(),
        output_root=args.output_root.resolve(),
        run_id=str(args.run_id),
        smoke=bool(args.smoke),
        resume_from=None if args.resume_from is None else args.resume_from.resolve(),
    )
    try:
        result = run_complete_gate1(config)
    except Exception as exc:
        print({"authorized_gate2": False, "run_id": config.run_id, "error_type": type(exc).__name__, "reason": str(exc)})
        return 2
    print({
        "authorized_gate2": result["authorized_gate2"],
        "audit_status": result["audit_status"],
        "row_count": result["row_count"],
        "transition_path": result["transition_path"],
    })
    return 0 if result["authorized_gate2"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
