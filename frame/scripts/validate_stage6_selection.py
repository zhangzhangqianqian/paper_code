"""校验阶段6.1验证协议和模型选择规则。"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage6_contract import load_stage6_selection_contract  # noqa: E402


def main() -> None:
    contract = load_stage6_selection_contract()
    print(
        json.dumps(
            {
                "contract_version": contract.raw["contract_version"],
                "primary_protocol": contract.primary_protocol,
                "secondary_protocol": contract.secondary_protocol,
                "candidate_models": list(contract.candidate_models),
                "hyperparameter_candidates": [
                    candidate["candidate_id"]
                    for candidate in contract.hyperparameter_candidates
                ],
                "test_set_status": contract.raw["validation_policy"]["test_set_status"],
                "test_metrics_may_be_read": contract.raw["validation_policy"][
                    "test_metrics_may_be_read"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
