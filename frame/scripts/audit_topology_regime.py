"""Run the pre-branch Kitakyushu equipment-regime audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.topology_protocol_contract import load_topology_contract  # noqa: E402
from src.topology_regime_audit import build_topology_regime_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_topology_contract(args.contract)
    audit = build_topology_regime_audit(args.data_dir, args.output_dir)
    print(
        json.dumps(
            {
                "stage": "topology_regime_audit",
                "status": audit["status"],
                "audited_years": audit["audited_years"],
                "test_year_accessed": audit["test_year_accessed"],
                "contract_version": contract["contract_version"],
                "output_dir": str(Path(args.output_dir).resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
