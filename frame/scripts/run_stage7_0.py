"""阶段 7.0：验证冻结配置并生成正式运行契约。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage7_contract import write_stage7_contract  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行阶段 7.0 冻结配置验证")
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6_6_kitakyushu/stage6_selected_config.json",
    )
    parser.add_argument("--contract-output", default="frame/configs/stage7_contract.json")
    parser.add_argument(
        "--report-output",
        default="frame/reports/stage7_0_kitakyushu/stage7_0_manifest.json",
    )
    parser.add_argument("--force", action="store_true", help="明确允许覆盖已有阶段 7 契约")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = write_stage7_contract(
        freeze_path=_resolve(args.freeze_config),
        contract_path=_resolve(args.contract_output),
        report_path=_resolve(args.report_output),
        force=args.force,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

