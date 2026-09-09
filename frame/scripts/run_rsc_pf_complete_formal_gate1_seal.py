"""Seal a complete-v1 Gate 1 run stopped by an external interruption."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.complete_formal_gate1_seal import seal_interrupted_gate1_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reason", default="run interrupted by system restart before Gate 1 completion; no evaluation was performed")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = seal_interrupted_gate1_run(args.run_root.resolve(), reason=args.reason)
    except Exception as exc:
        print({"sealed": False, "error_type": type(exc).__name__, "reason": str(exc)})
        return 2
    print({"sealed": True, "failure_receipt": str(path)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
