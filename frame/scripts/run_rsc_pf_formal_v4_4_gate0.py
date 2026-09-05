"""Run the explicit formal-v4.4 Gate 0 preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path: sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_gate0 import run_gate0_v44  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "source-manifest", "base-train-data", "base-selection-data", "benchmark", "capacity-receipt", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_gate0_v44(contract_path=args.contract, source_manifest=args.source_manifest, base_train_data=args.base_train_data, base_selection_data=args.base_selection_data, benchmark=args.benchmark, capacity_receipt=args.capacity_receipt, output_root=args.output_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"authorized_pilot": False, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False)); return 2
    print(json.dumps({"authorized_pilot": receipt.authorized_pilot, "run_id": receipt.run_id}, ensure_ascii=False)); return 0 if receipt.authorized_pilot else 3


if __name__ == "__main__": raise SystemExit(main())
