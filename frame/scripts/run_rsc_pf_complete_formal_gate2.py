"""Run Gate 2 only after an authorized Gate 1 transition exists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.complete_formal_execution import run_gate2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--gate1-transition", required=True)
    parser.add_argument("--run-id", default="complete_formal_gate2_synthetic")
    args = parser.parse_args()
    contract = CompleteFormalContract.from_path(args.contract)
    decision, audit, stage_dir = run_gate2(contract, args.output, args.gate1_transition, run_id=args.run_id)
    print(json.dumps({"authorized": decision.authorized_next_gate, "audit_status": audit.status, "stage_dir": str(stage_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
