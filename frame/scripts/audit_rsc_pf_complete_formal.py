"""Audit an explicit serialized formal row matrix without discovering data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.complete_formal_execution import audit_complete_formal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"))
    parser.add_argument("--rows", required=True, help="JSON object keyed by Method/seed")
    parser.add_argument("--stage", choices=("gate1", "gate2"), required=True)
    parser.add_argument("--origin-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = CompleteFormalContract.from_path(args.contract)
    rows = json.loads(Path(args.rows).read_text(encoding="utf-8"))
    result = audit_complete_formal(contract, rows, stage=args.stage, origin_count=args.origin_count)
    payload = {"stage": result.stage, "status": result.status, "recomputed_row_count": result.recomputed_row_count, "failures": result.failures, "evaluation_year_accessed": result.evaluation_year_accessed, "excluded_year_accessed": result.excluded_year_accessed}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if result.status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
