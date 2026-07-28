"""验证阶段5.1外部基线公平性契约。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fairness_contract import DEFAULT_CONTRACT_PATH, load_fairness_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="验证外部基线公平性契约")
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT_PATH))
    args = parser.parse_args()
    contract = load_fairness_contract(args.contract)
    print(
        json.dumps(
            {
                "contract_version": contract.raw["contract_version"],
                "tasks": list(contract.raw["tasks"]),
                "lookback": contract.lookback,
                "horizon": contract.horizon,
                "prediction_shape": ["batch", contract.horizon, contract.task_count],
                "baselines": [spec.name for spec in contract.baseline_specs],
                "status": "valid",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
