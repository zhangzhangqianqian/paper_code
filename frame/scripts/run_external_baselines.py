"""阶段5.5：统一运行三个外部基线并生成公平性报告。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.external_report import (  # noqa: E402
    MODEL_ORDER,
    load_and_validate_runs,
    write_unified_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统一运行DLinear、MMoE-lite和SOFTS并生成阶段5.5报告"
    )
    parser.add_argument(
        "--protocol", choices=("full", "small_sample"), default="small_sample"
    )
    parser.add_argument("--output-dir", required=True, help="统一报告输出目录")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=512)
    parser.add_argument("--max-validation-samples", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument(
        "--softs-deterministic-pooling",
        action="store_true",
        help="使用确定性STAR核心池化，便于重复冒烟测试",
    )
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> None:
    args = parse_args()
    if args.max_epochs <= 0 or args.patience < 0:
        raise ValueError("max-epochs必须为正数，patience不能为负数")
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_script = PROJECT_ROOT / "scripts" / "train_external_baseline.py"
    common = [
        sys.executable,
        str(train_script),
        "--protocol",
        args.protocol,
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--max-epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--threads",
        str(args.threads),
        "--seed",
        str(args.seed),
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-validation-samples",
        str(args.max_validation_samples),
        "--max-test-samples",
        str(args.max_test_samples),
    ]
    for model in MODEL_ORDER:
        run_dir = output_dir / model.replace("-", "_")
        command = [
            *common,
            "--model",
            model,
            "--output-dir",
            str(run_dir),
        ]
        if model == "softs" and args.softs_deterministic_pooling:
            command.append("--softs-deterministic-pooling")
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)

    runs = load_and_validate_runs(output_dir, args.protocol)
    summary = write_unified_report(output_dir, args.protocol, runs)
    print(json.dumps(summary["consistency_checks"], ensure_ascii=False, indent=2))
    print(f"统一报告已写入：{output_dir}")


if __name__ == "__main__":
    main()
