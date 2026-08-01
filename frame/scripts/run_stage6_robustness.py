"""阶段 6.3：小样本验证集鲁棒性检查。

该脚本复用阶段 6.2 的五个模型和 H1—H4 配置，在小样本训练区间重新训练，
只评估小样本验证集，不构造或读取小样本测试窗口。

正式运行示例：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\run_stage6_robustness.py `
        --stage6-2-dir frame\\reports\\stage6_2\\full `
        --output-dir frame\\reports\\stage6_3\\small_sample

CPU 冒烟示例：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\run_stage6_robustness.py `
        --stage6-2-dir frame\\reports\\stage6_2\\smoke `
        --output-dir frame\\reports\\stage6_3\\smoke --smoke `
        --max-train-samples 64 --max-validation-samples 32 `
        --max-epochs 1 --patience 1 --batch-size 32 --threads 2
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

from src.data_pipeline import SMALL_SAMPLE_SPLIT, read_heew_canonical, save_json  # noqa: E402
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_TASKS,
    read_kitakyushu_canonical,
)
from src.stage6_contract import load_stage6_selection_contract  # noqa: E402
from src.validation_selection import (  # noqa: E402
    run_protocol_sweep,
    write_stability_comparison,
)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行阶段6.3小样本验证集鲁棒性检查"
    )
    parser.add_argument(
        "--dataset",
        choices=("kitakyushu_energy_station", "heew_total"),
        default="kitakyushu_energy_station",
        help="阶段 6 数据协议；默认使用 Kitakyushu 四任务",
    )
    parser.add_argument(
        "--kitakyushu-data-dir",
        default="Kitakyushu dataset",
        help="Kitakyushu 原始 ZIP/解压文件目录",
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
        "--stage6-2-dir",
        default="frame/reports/stage6_2/kitakyushu/full",
        help="阶段6.2全年验证结果目录，用于排名稳定性对照",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage6_3/kitakyushu/small_sample",
        help="阶段6.3输出目录",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="允许覆盖训练规模/轮数，仅用于CPU冒烟",
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
    if args.dataset == "kitakyushu_energy_station":
        if args.energy_file != "dataset/HEEW/cleaned_data/Total_energy.csv" or args.weather_file != "dataset/HEEW/cleaned_data/Total_weather.csv":
            raise ValueError("Kitakyushu 协议不接受 HEEW 的 --energy-file/--weather-file 参数")
        frame, _ = read_kitakyushu_canonical(
            _resolve_path(args.kitakyushu_data_dir), years=tuple(range(2015, 2022))
        )
        task_names = KITAKYUSHU_TASKS
        exog_columns = KITAKYUSHU_EXOG_COLUMNS
        split_spec = KITAKYUSHU_SMALL_SAMPLE_SPLIT
    else:
        energy_path = _resolve_path(args.energy_file)
        weather_path = _resolve_path(args.weather_file)
        frame, _ = read_heew_canonical(energy_path, weather_path)
        task_names = None
        exog_columns = None
        split_spec = SMALL_SAMPLE_SPLIT
    output_root = _resolve_path(args.output_dir)
    completed = run_protocol_sweep(
        frame=frame,
        output_root=output_root,
        hyperparameter_candidates=contract.hyperparameter_candidates,
        model_names=contract.candidate_models,
        split_spec=split_spec,
        dataset_kind=args.dataset,
        **({"task_names": task_names, "exog_columns": exog_columns} if task_names is not None else {}),
        protocol_name="small_sample",
        stage_name="6.3",
        manifest_name="stage6_3_manifest.json",
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
    stability = write_stability_comparison(
        _resolve_path(args.stage6_2_dir), output_root
    )
    manifest_path = output_root / "stage6_3_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.update(
        {
            "stage6_2_reference": str(_resolve_path(args.stage6_2_dir)),
            "stability_summary": stability,
            "negative_transfer_analysis": "deferred_to_stage6_5",
            "test_set_accessed": False,
        }
    )
    save_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "stage": "6.3",
                "protocol": "small_sample",
                "completed_runs": len(completed),
                "output_dir": str(output_root),
                "rank_order_changed": stability["rank_order_changed"],
                "rank_order_completely_reversed": stability[
                    "rank_order_completely_reversed"
                ],
                "test_set_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
