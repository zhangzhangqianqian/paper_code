"""Audit an already-produced complete-v1 Gate 1 row receipt.

The module-level gate primitive still exposes a synthetic fixture for tests,
but this command intentionally requires a serialized row mapping.  Running a
command without an explicit receipt must never create a fake Gate 1 result.

Use ``run_rsc_pf_complete_formal_gate1_real.py`` to train and evaluate the
37-row matrix; this compatibility command only audits rows that already exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.complete_formal_execution import run_gate1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", required=True, help="JSON file containing the 37 real Gate 1 row receipts")
    parser.add_argument("--run-id", default="complete_formal_gate1_real")
    args = parser.parse_args()
    contract = CompleteFormalContract.from_path(args.contract)
    rows_path = Path(args.rows).resolve()
    payload = json.loads(rows_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--rows must contain a JSON object keyed by method/seed")
    if any(bool(row.get("synthetic", False)) for row in payload.values() if isinstance(row, dict)):
        raise PermissionError("Gate 1 command refuses synthetic rows")
    decision, audit, stage_dir = run_gate1(contract, args.output, payload, run_id=args.run_id)
    print(json.dumps({"authorized_gate2": decision.authorized_gate2, "audit_status": audit.status, "stage_dir": str(stage_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
