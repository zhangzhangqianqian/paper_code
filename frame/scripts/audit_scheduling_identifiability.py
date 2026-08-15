"""运行阶段10.1参数证据与调度可辨识性审计。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.contracts import load_scheduling_contract  # noqa: E402
from src.scheduling.parameter_audit import (  # noqa: E402
    audit_identifiability,
    read_parameter_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    contract = load_scheduling_contract(args.contract)
    ledger = read_parameter_ledger(args.ledger)
    training_stats = {
        "scaling_years": list(contract.train_years),
        "test_year_used_for_scaling": False,
        "source": "frozen_year_protocol",
    }
    report = audit_identifiability(contract, ledger, training_stats)
    result = report.to_dict()
    result.update(
        {
            "stage": "10.1",
            "contract": str(args.contract.resolve()),
            "ledger": str(args.ledger.resolve()),
            "data_directory": str(args.data_dir.resolve()),
            "data_directory_exists": args.data_dir.exists(),
            "training_stats": training_stats,
        }
    )
    if not args.data_dir.exists():
        result["status"] = "fail"
        result.setdefault("issues", []).append("data_directory不存在")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "audit_manifest.json"
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
