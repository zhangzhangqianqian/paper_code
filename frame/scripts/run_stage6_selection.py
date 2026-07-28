"""阶段 6.2：全年验证集候选模型筛选（不读取测试集结果）。

正式运行示例：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\run_stage6_selection.py `
        --output-dir frame\\reports\\stage6_2\\full

CPU 冒烟示例：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\run_stage6_selection.py `
        --output-dir frame\\reports\\stage6_2\\smoke --smoke `
        --max-train-samples 512 --max-validation-samples 128 `
        --max-epochs 2 --patience 1
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

from src.data_pipeline import read_heew_canonical  # noqa: E402
from src.stage6_contract import load_stage6_selection_contract  # noqa: E402
from src.validation_selection import run_validation_sweep  # noqa: E402


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在全年训练/验证协议上执行阶段6.2候选模型筛选"
    )
    parser.add_argument(
        "--energy-file",
        default="dataset/HEEW/cleaned_data/Total_energy.csv",
        help="HEEW负荷CSV路径",
    )
    parser.add_argument(
        "--weather-file",
        default="dataset/HEEW/cleaned_data/Total_weather.csv",
        help="HEEW气象CSV路径",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage6_2/full",
        help="阶段6.2输出目录",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="允许覆盖训练规模/轮数，仅用于CPU冒烟，不得作为正式结论",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        name: value
        for name, value in (
            ("max_train_samples", args.max_train_samples),
            ("max_validation_samples", args.max_validation_samples),
            ("max_epochs", args.max_epochs),
            ("patience", args.patience),
            ("batch_size", args.batch_size),
            ("threads", args.threads),
            ("seed", args.seed),
        )
        if value is not None
    }
    if overrides and not args.smoke:
        raise ValueError("覆盖阶段6.1训练协议只能与 --smoke 一起使用")

    contract = load_stage6_selection_contract()
    training = contract.raw["training_policy"]
    energy_path = _resolve_path(args.energy_file)
    weather_path = _resolve_path(args.weather_file)
    frame, _ = read_heew_canonical(energy_path, weather_path)
    completed = run_validation_sweep(
        frame=frame,
        output_root=_resolve_path(args.output_dir),
        hyperparameter_candidates=contract.hyperparameter_candidates,
        model_names=contract.candidate_models,
        batch_size=int(overrides.get("batch_size", training["batch_size"])),
        weight_decay=float(training["weight_decay"]),
        max_epochs=int(overrides.get("max_epochs", training["max_epochs"])),
        early_stopping_patience=int(
            overrides.get("patience", training["early_stopping_patience"])
        ),
        grad_clip_norm=float(training["gradient_clip_norm"]),
        threads=int(overrides.get("threads", 8)),
        seed=int(overrides.get("seed", training["random_seed"])),
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
    )
    print(
        json.dumps(
            {
                "stage": "6.2",
                "protocol": "full",
                "completed_runs": len(completed),
                "output_dir": str(_resolve_path(args.output_dir)),
                "smoke": bool(args.smoke),
                "test_set_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
