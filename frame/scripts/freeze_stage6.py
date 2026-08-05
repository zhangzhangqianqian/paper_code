"""生成阶段 6.6 的验证结果冻结配置。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage6_freeze import freeze_stage6  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结阶段 6.6 最终方案并交接阶段 7")
    parser.add_argument("--stage6-2-dir", default="frame/reports/stage6_2/kitakyushu_full")
    parser.add_argument("--stage6-3-dir", default="frame/reports/stage6_3/kitakyushu_small_sample")
    parser.add_argument("--stage6-4-dir", default="frame/reports/stage6_4/kitakyushu_full")
    parser.add_argument("--stage6-5-dir", default="frame/reports/stage6_5/kitakyushu_full")
    parser.add_argument("--output-dir", default="frame/reports/stage6_6_kitakyushu")
    parser.add_argument("--force", action="store_true", help="明确允许覆盖非空冻结输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = freeze_stage6(
        output_root=_resolve(args.output_dir),
        stage6_2_root=_resolve(args.stage6_2_dir),
        stage6_3_root=_resolve(args.stage6_3_dir),
        stage6_4_root=_resolve(args.stage6_4_dir),
        stage6_5_root=_resolve(args.stage6_5_dir),
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

