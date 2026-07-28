"""阶段 6.5：验证集迁移收益、负迁移率和 bootstrap 置信区间。

输入为阶段 6.2 或 6.3 的验证结果目录，只读取 predictions_validation.npz。
正式运行示例：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\run_stage6_transfer_analysis.py `
        --input-dir frame\\reports\\stage6_2\\full_v2 `
        --output-dir frame\\reports\\stage6_5\\full_v2

小样本协议使用同一脚本，把 input/output 目录换成阶段 6.3 目录即可。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_analysis import METRICS, analyze_transfer  # noqa: E402


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行阶段6.5验证集迁移收益与负迁移率分析"
    )
    parser.add_argument(
        "--input-dir",
        default="frame/reports/stage6_2/full_v2",
        help="阶段6.2或6.3验证结果目录",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage6_5/full_v2",
        help="阶段6.5输出目录",
    )
    parser.add_argument(
        "--protocol",
        choices=("full", "small_sample"),
        default="full",
        help="输入结果对应的验证协议",
    )
    parser.add_argument(
        "--error-metric",
        choices=METRICS,
        default="WAPE",
        help="用于显著负迁移bootstrap判定的主要误差指标",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=500,
        help="配对bootstrap重复次数；设为0则只计算原始迁移收益",
    )
    parser.add_argument(
        "--bootstrap-granularities",
        default="task",
        help="进行显著性bootstrap的粒度，逗号分隔：task,horizon,season_horizon",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    granularities = tuple(
        value.strip()
        for value in args.bootstrap_granularities.split(",")
        if value.strip()
    )
    manifest = analyze_transfer(
        input_root=_resolve_path(args.input_dir),
        output_root=_resolve_path(args.output_dir),
        protocol=args.protocol,
        error_metric=args.error_metric,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_granularities=granularities,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
