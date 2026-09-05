"""Run the bounded formal-v4.4 Pilot; never starts Gate 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path: sys.path.insert(0, str(FRAME_ROOT))
from src.joint_dispatch.formal_v4_4_pilot import run_pilot_v44  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contract", "gate0-transition", "source-manifest", "base-train-data", "base-selection-data", "benchmark", "capacity-receipt", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--run-id", required=True); args = parser.parse_args(argv)
    try:
        decision = run_pilot_v44(contract_path=args.contract, gate0_transition=args.gate0_transition, source_manifest=args.source_manifest, base_train_data=args.base_train_data, base_selection_data=args.base_selection_data, benchmark=args.benchmark, capacity_receipt=args.capacity_receipt, output_root=args.output_root, run_id=args.run_id)
    except PermissionError as exc:
        print(json.dumps({"authorized_gate1": False, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False)); return 2
    except Exception as exc:
        print(json.dumps({"authorized_gate1": False, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False)); return 2
    print(json.dumps({"authorized_gate1": decision.authorized_gate1, "failures": list(decision.failures), "audit_sha256": decision.audit_sha256}, ensure_ascii=False)); return 0 if decision.authorized_gate1 else 3


if __name__ == "__main__": raise SystemExit(main())
