"""阶段2—4.5：标准化、窗口批处理、统一模型训练、验证和结果导出。

示例（HEEW 全量协议，先用少量样本做 CPU 冒烟训练）：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\train_models.py `
        --model hard_share --protocol full `
        --output-dir frame\\reports\\phase3\\smoke `
        --max-epochs 2 --patience 1 --max-train-samples 512 `
        --max-validation-samples 128 --max-test-samples 256

正式实验时去掉 ``max-*-samples`` 限制，并在固定的 output-dir 中保存配置、
标准化参数、训练历史、最佳 checkpoint 和原始量纲测试预测。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import regression_metrics  # noqa: E402
from src.data_pipeline import (  # noqa: E402
    FULL_SPLIT,
    HEEW_EXOG_COLUMNS,
    SMALL_SAMPLE_SPLIT,
    TASKS,
    build_protocol_windows,
    clean_dataframe,
    read_csv_canonical,
    read_heew_canonical,
    save_json,
    select_training_frame,
)
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.models import (  # noqa: E402
    DynamicDirectedMTLModel,
    DynamicSymmetricMTLModel,
    HardShareMTLModel,
    IndependentSTLModel,
    MODEL_NAMES,
    StaticDirectedMTLModel,
    build_forecasting_model,
)
from src.training import (  # noqa: E402
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    make_dataloader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练统一多能源负荷预测模型")
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="stl",
    )
    parser.add_argument(
        "--protocol", choices=("full", "small_sample"), default="full"
    )
    parser.add_argument(
        "--dataset",
        choices=("kitakyushu_energy_station", "heew_total"),
        default="kitakyushu_energy_station",
        help="数据协议；默认使用 Kitakyushu 四任务数据",
    )
    parser.add_argument(
        "--kitakyushu-data-dir",
        default="Kitakyushu dataset",
        help="Kitakyushu 数据目录（包含三个核心 ZIP）",
    )
    parser.add_argument("--input", help="兼容旧格式的单个规范 CSV 路径")
    parser.add_argument("--energy-file", help="HEEW 负荷 CSV 路径")
    parser.add_argument("--weather-file", help="HEEW 气象 CSV 路径")
    parser.add_argument("--output-dir", required=True, help="模型和结果输出目录")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_input(args: argparse.Namespace):
    if args.dataset == "kitakyushu_energy_station":
        if args.input or args.energy_file or args.weather_file:
            raise ValueError(
                "Kitakyushu 协议使用 --kitakyushu-data-dir，不能同时提供 HEEW 文件参数"
            )
        frame, metadata = read_kitakyushu_canonical(
            _resolve_path(args.kitakyushu_data_dir),
            years=tuple(range(2015, 2022)),
        )
        return frame, "kitakyushu_energy_station", metadata
    if args.input and (args.energy_file or args.weather_file):
        raise ValueError("--input 与 --energy-file/--weather-file 不能同时使用")
    if args.input:
        frame, _ = read_csv_canonical(_resolve_path(args.input))
        return frame, "legacy_single_csv", {}
    if bool(args.energy_file) != bool(args.weather_file):
        raise ValueError("HEEW 输入必须同时提供 --energy-file 和 --weather-file")
    energy_path = _resolve_path(args.energy_file) or (
        REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_energy.csv"
    )
    weather_path = _resolve_path(args.weather_file) or (
        REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_weather.csv"
    )
    return read_heew_canonical(energy_path, weather_path)[0], "heew_total", {}


def _limit_windows(
    windows: Mapping[str, np.ndarray], limit: int | None
) -> Dict[str, np.ndarray]:
    if limit is None:
        return dict(windows)
    if limit <= 0:
        raise ValueError("样本限制必须为正整数")
    return {key: value[:limit] for key, value in windows.items()}


def _make_model(
    args: argparse.Namespace,
    exog_dim: int,
    task_count: int,
):
    common = dict(
        exog_dim=exog_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        dilations=(1, 2),
        dropout=args.dropout,
        horizon=args.horizon,
        task_count=task_count,
    )
    return build_forecasting_model(args.model, **common)


def main() -> None:
    args = parse_args()
    if args.lookback <= 0 or args.horizon <= 0:
        raise ValueError("lookback 和 horizon 必须为正整数")
    if args.horizon != 4:
        raise ValueError("当前阶段的论文协议固定为24→4，horizon必须为4")
    if args.hidden_dim <= 1 or args.kernel_size <= 0:
        raise ValueError("hidden-dim必须大于1，kernel-size必须为正整数")

    output_dir = _resolve_path(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, dataset_kind, dataset_metadata = _read_input(args)
    if dataset_kind == "kitakyushu_energy_station":
        cleaned, cleaning_report = clean_kitakyushu_dataframe(frame)
        task_columns = KITAKYUSHU_TASKS
        exog_columns = tuple(
            column for column in KITAKYUSHU_EXOG_COLUMNS if column in cleaned.columns
        )
        spec = KITAKYUSHU_SPLIT if args.protocol == "full" else KITAKYUSHU_SMALL_SAMPLE_SPLIT
        # 统一使用通用协议构造器，以便 full 和 small_sample 都严格遵守
        # 当前选择的时间切分；适配器中的固定 full 版本仍保留给独立审计脚本使用。
        window_builder = None
    else:
        cleaned, cleaning_report = clean_dataframe(frame)
        # HEEW 使用完整的14维天气+日历输入；旧版单CSV则只使用其中实际存在的列，
        # 从而保持脚本的兼容性而不凭空创建外生变量。
        task_columns = TASKS
        exog_columns = tuple(
            column for column in HEEW_EXOG_COLUMNS if column in cleaned.columns
        )
        spec = FULL_SPLIT if args.protocol == "full" else SMALL_SAMPLE_SPLIT
        window_builder = None

    raw_windows = {}
    for split_name in ("train", "validation", "test"):
        if window_builder is not None:
            raw_windows[split_name] = window_builder(
                cleaned,
                split_name=split_name,
                lookback=args.lookback,
                horizon=args.horizon,
                exog_columns=exog_columns,
            )
        else:
            raw_windows[split_name] = build_protocol_windows(
                cleaned,
                spec,
                split_name=split_name,
                lookback=args.lookback,
                horizon=args.horizon,
                exog_columns=exog_columns,
                task_columns=task_columns,
            )
    windows = {
        "train": _limit_windows(raw_windows["train"], args.max_train_samples),
        "validation": _limit_windows(
            raw_windows["validation"], args.max_validation_samples
        ),
        "test": _limit_windows(raw_windows["test"], args.max_test_samples),
    }
    if any(len(windows[name]["target"]) == 0 for name in windows):
        raise ValueError("至少一个数据切分没有可用滑动窗口")

    # 标准化参数严格只从训练区间拟合，验证/测试仅使用已拟合参数。
    train_frame = select_training_frame(cleaned, spec)
    stats = StandardizationStats.fit(
        train_frame,
        exog_columns,
        task_columns=task_columns,
    )
    standardized = {name: stats.transform_windows(value) for name, value in windows.items()}
    stats.save(output_dir / "normalization_stats.npz")

    train_loader = make_dataloader(standardized["train"], args.batch_size, shuffle=True)
    validation_loader = make_dataloader(
        standardized["validation"], args.batch_size, shuffle=False
    )
    test_loader = make_dataloader(standardized["test"], args.batch_size, shuffle=False)
    model = _make_model(args, len(exog_columns), len(task_columns))
    trainer_config = TrainerConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.patience,
        torch_threads=args.threads,
        seed=args.seed,
    )
    checkpoint_path = output_dir / "best_model.pt"
    fit_started = time.perf_counter()
    history = fit_model(
        model,
        train_loader,
        validation_loader,
        trainer_config,
        checkpoint_path,
    )
    fit_seconds = time.perf_counter() - fit_started
    evaluation_started = time.perf_counter()
    test_loss, prediction_std, target_std = evaluate_model(model, test_loader, "cpu")
    evaluation_seconds = time.perf_counter() - evaluation_started
    prediction = stats.inverse_targets(prediction_std)
    target = stats.inverse_targets(target_std)
    metrics = regression_metrics(target, prediction, task_names=task_columns)
    gate_matrix = None
    dynamic_gate_matrix_file = None
    gate_export = None
    if isinstance(model, StaticDirectedMTLModel):
        gate_matrix = model.get_gate_matrix().detach().cpu().numpy().tolist()
        save_json(
            {
                "row_semantics": "target_task",
                "column_semantics": "source_task",
                "tasks": list(task_columns),
                "gate_matrix": gate_matrix,
            },
            output_dir / "gate_matrix.json",
        )
        gate_export = {
            "kind": "static",
            "file": "gate_matrix.json",
            "shape": [len(task_columns), len(task_columns)],
            "row_semantics": "target_task",
            "column_semantics": "source_task",
        }
    elif isinstance(model, (DynamicSymmetricMTLModel, DynamicDirectedMTLModel)):
        dynamic_gates = []
        model.eval()
        with torch.no_grad():
            for loads, exog, _ in test_loader:
                representations = model.encode_tasks(loads, exog)
                state = model.encode_state(exog, representations)
                if isinstance(model, DynamicSymmetricMTLModel):
                    gates = model.get_gate_matrix(state)
                else:
                    gates = model.get_gate_matrix(state, representations)
                dynamic_gates.append(gates.cpu().numpy())
        dynamic_gate_matrix_file = "gate_matrix_test.npz"
        np.savez_compressed(
            output_dir / dynamic_gate_matrix_file,
            gates=np.concatenate(dynamic_gates, axis=0),
            target_times=windows["test"]["target_times"],
        )
        gate_export = {
            "kind": "dynamic",
            "file": dynamic_gate_matrix_file,
            "shape": [
                int(len(windows["test"]["target_times"])),
                len(task_columns),
                len(task_columns),
            ],
            "row_semantics": "target_task",
            "column_semantics": "source_task",
        }

    np.savez_compressed(
        output_dir / "predictions_test.npz",
        target=target,
        prediction=prediction,
        target_standardized=target_std,
        prediction_standardized=prediction_std,
        target_times=windows["test"]["target_times"],
    )
    save_json(
        {
            "dataset_kind": dataset_kind,
            "dataset_metadata": dataset_metadata,
            "model": args.model,
            "protocol": args.protocol,
            "tasks": list(task_columns),
            "exog_columns": list(exog_columns),
            "window": {"lookback": args.lookback, "horizon": args.horizon},
            "model_interface": {
                "loads": f"[batch, {args.lookback}, {len(task_columns)}]",
                "exog": f"[batch, {args.lookback}, {len(exog_columns)}]",
                "prediction": f"[batch, {args.horizon}, {len(task_columns)}]",
                "device": trainer_config.device,
            },
            "sample_counts": {
                name: int(len(value["target"])) for name, value in windows.items()
            },
            "test_standardized_smooth_l1": test_loss,
            "metrics_original_scale": metrics,
            "static_gate_matrix": gate_matrix,
            "dynamic_gate_matrix_test_file": dynamic_gate_matrix_file,
            "gate_export": gate_export,
            "cleaning": cleaning_report,
            "normalization": stats.summary(),
            "trainer_config": asdict(trainer_config),
            "model_parameter_count": int(sum(p.numel() for p in model.parameters())),
            "runtime_seconds": {
                "fit": float(fit_seconds),
                "test_evaluation": float(evaluation_seconds),
                "test_samples_per_second": float(
                    len(windows["test"]["target"]) / max(evaluation_seconds, 1e-9)
                ),
            },
        },
        output_dir / "metrics_test.json",
    )
    save_json({"history": history}, output_dir / "history.json")
    print(
        json.dumps(
            {
                "model": args.model,
                "protocol": args.protocol,
                "sample_counts": {
                    name: int(len(value["target"])) for name, value in windows.items()
                },
                "checkpoint": str(checkpoint_path),
                "metrics": metrics["overall_equal_task_mean"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
