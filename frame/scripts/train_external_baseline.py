"""阶段5外部基线训练脚本。

当前脚本接入 DLinear、MMoE-lite 和 SOFTS。输入模式由公平性契约显式声明：DLinear
和 SOFTS 使用 ``loads_only``，MMoE-lite 使用 ``loads_and_exog``；所有模型都不会
接收未来天气或日历变量。
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
    read_heew_canonical,
    save_json,
    select_training_frame,
)
from src.external_models import (  # noqa: E402
    DLinearBaseline,
    MMoELiteBaseline,
    SOFTSBaseline,
)
from src.fairness_contract import load_fairness_contract  # noqa: E402
from src.training import (  # noqa: E402
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    make_dataloader,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练外部轻量基线")
    parser.add_argument(
        "--model", choices=("dlinear", "mmoe-lite", "softs"), default="dlinear"
    )
    parser.add_argument(
        "--protocol", choices=("full", "small_sample"), default="full"
    )
    parser.add_argument("--energy-file", help="HEEW负荷CSV路径")
    parser.add_argument("--weather-file", help="HEEW气象CSV路径")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--lookback", type=int, default=24)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--moving-avg", type=int, default=5)
    parser.add_argument("--expert-count", type=int, default=4)
    parser.add_argument("--expert-hidden-dim", type=int, default=32)
    parser.add_argument("--representation-dim", type=int, default=32)
    parser.add_argument("--head-hidden-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-core", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--e-layers", type=int, default=1)
    parser.add_argument(
        "--softs-disable-instance-norm", action="store_true",
        help="SOFTS适配中关闭输入窗口级归一化",
    )
    parser.add_argument(
        "--softs-deterministic-pooling", action="store_true",
        help="SOFTS适配中使用加权核心而不是训练期随机池化",
    )
    parser.add_argument("--batch-size", type=int, default=128)
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


def _limit_windows(
    windows: Mapping[str, np.ndarray], limit: int | None
) -> Dict[str, np.ndarray]:
    if limit is None:
        return dict(windows)
    if limit <= 0:
        raise ValueError("样本限制必须为正整数")
    return {key: value[:limit] for key, value in windows.items()}


def main() -> None:
    args = parse_args()
    contract = load_fairness_contract()
    baseline_name = {
        "dlinear": "DLinear",
        "mmoe-lite": "MMoE-lite",
        "softs": "SOFTS",
    }[args.model]
    input_mode = "loads_and_exog" if args.model == "mmoe-lite" else "loads_only"
    contract.validate_baseline_input(baseline_name, input_mode)
    if args.lookback != contract.lookback or args.horizon != contract.horizon:
        raise ValueError("外部基线必须遵守公平性契约中的24→4窗口")
    if args.model == "dlinear" and (
        args.moving_avg <= 0 or args.moving_avg % 2 == 0
    ):
        raise ValueError("moving-avg必须为正奇数")

    output_dir = _resolve_path(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_path = _resolve_path(args.energy_file) or (
        REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_energy.csv"
    )
    weather_path = _resolve_path(args.weather_file) or (
        REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_weather.csv"
    )
    frame, _ = read_heew_canonical(energy_path, weather_path)
    cleaned, cleaning_report = clean_dataframe(frame)
    exog_columns = tuple(
        column for column in HEEW_EXOG_COLUMNS if column in cleaned.columns
    )
    spec = FULL_SPLIT if args.protocol == "full" else SMALL_SAMPLE_SPLIT
    raw_windows = {
        split_name: build_protocol_windows(
            cleaned,
            spec,
            split_name=split_name,
            lookback=args.lookback,
            horizon=args.horizon,
            exog_columns=exog_columns,
        )
        for split_name in ("train", "validation", "test")
    }
    windows = {
        "train": _limit_windows(raw_windows["train"], args.max_train_samples),
        "validation": _limit_windows(
            raw_windows["validation"], args.max_validation_samples
        ),
        "test": _limit_windows(raw_windows["test"], args.max_test_samples),
    }
    if any(len(windows[name]["target"]) == 0 for name in windows):
        raise ValueError("至少一个数据切分没有可用滑动窗口")

    train_frame = select_training_frame(cleaned, spec)
    stats = StandardizationStats.fit(train_frame, exog_columns)
    standardized = {
        name: stats.transform_windows(value) for name, value in windows.items()
    }
    stats.save(output_dir / "normalization_stats.npz")
    save_json(dict(contract.raw), output_dir / "fairness_contract.json")

    train_loader = make_dataloader(standardized["train"], args.batch_size, shuffle=True)
    validation_loader = make_dataloader(
        standardized["validation"], args.batch_size, shuffle=False
    )
    test_loader = make_dataloader(standardized["test"], args.batch_size, shuffle=False)
    if args.model == "dlinear":
        model = DLinearBaseline(
            lookback=args.lookback,
            horizon=args.horizon,
            task_count=len(TASKS),
            moving_avg=args.moving_avg,
        )
    elif args.model == "mmoe-lite":
        model = MMoELiteBaseline(
            lookback=args.lookback,
            horizon=args.horizon,
            task_count=len(TASKS),
            exog_dim=len(exog_columns),
            expert_count=args.expert_count,
            expert_hidden_dim=args.expert_hidden_dim,
            representation_dim=args.representation_dim,
            head_hidden_dim=args.head_hidden_dim,
            dropout=args.dropout,
        )
    else:
        model = SOFTSBaseline(
            lookback=args.lookback,
            horizon=args.horizon,
            task_count=len(TASKS),
            d_model=args.d_model,
            d_core=args.d_core,
            d_ff=args.d_ff,
            e_layers=args.e_layers,
            dropout=args.dropout,
            use_instance_norm=not args.softs_disable_instance_norm,
            stochastic_pooling=not args.softs_deterministic_pooling,
        )
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
        input_mode=input_mode,
    )
    fit_seconds = time.perf_counter() - fit_started
    evaluation_started = time.perf_counter()
    test_loss, prediction_std, target_std = evaluate_model(
        model,
        test_loader,
        "cpu",
        input_mode=input_mode,
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    prediction = stats.inverse_targets(prediction_std)
    target = stats.inverse_targets(target_std)
    metrics = regression_metrics(target, prediction)

    gate_path = None
    if isinstance(model, MMoELiteBaseline):
        gate_values = []
        model.eval()
        with torch.no_grad():
            for loads_batch, exog_batch, _ in test_loader:
                gate_values.append(
                    model.gate_weights(loads_batch, exog_batch).cpu().numpy()
                )
        gate_path = output_dir / "gate_weights_test.npz"
        np.savez_compressed(
            gate_path,
            gate_weights=np.concatenate(gate_values, axis=0).astype(np.float32),
            target_times=windows["test"]["target_times"],
        )

    if args.model == "dlinear":
        model_config = {
            "lookback": args.lookback,
            "horizon": args.horizon,
            "task_count": len(TASKS),
            "moving_avg": args.moving_avg,
            "channel_shared_linear": True,
        }
        baseline_family = "linear_forecasting"
        exog_used = False
        exog_interface = "not_used"
    elif args.model == "mmoe-lite":
        model_config = {
            "lookback": args.lookback,
            "horizon": args.horizon,
            "task_count": len(TASKS),
            "exog_dim": len(exog_columns),
            "expert_count": args.expert_count,
            "expert_hidden_dim": args.expert_hidden_dim,
            "representation_dim": args.representation_dim,
            "head_hidden_dim": args.head_hidden_dim,
            "dropout": args.dropout,
            "routing": "task_specific_softmax_over_shared_experts",
            "frequency_module": False,
            "stim_module": False,
        }
        baseline_family = "expert_routing"
        exog_used = True
        exog_interface = f"[batch, {args.lookback}, {len(exog_columns)}]"
    else:
        model_config = {
            "lookback": args.lookback,
            "horizon": args.horizon,
            "task_count": len(TASKS),
            "d_model": args.d_model,
            "d_core": args.d_core,
            "d_ff": args.d_ff,
            "e_layers": args.e_layers,
            "dropout": args.dropout,
            "use_instance_norm": not args.softs_disable_instance_norm,
            "stochastic_pooling": not args.softs_deterministic_pooling,
            "implementation_variant": "minimal_star_adapter",
            "official_code_checked": True,
            "frequency_module": False,
            "stim_module": False,
        }
        baseline_family = "multivariate_channel_mixing"
        exog_used = False
        exog_interface = "not_used"

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
            "dataset_kind": "heew_total",
            "model": args.model,
            "baseline_name": baseline_name,
            "baseline_family": baseline_family,
            "protocol": args.protocol,
            "tasks": list(TASKS),
            "input_mode": input_mode,
            "exog_used": exog_used,
            "future_exogenous_used": False,
            "model_config": model_config,
            "window": {"lookback": args.lookback, "horizon": args.horizon},
            "model_interface": {
                "loads": f"[batch, {args.lookback}, {len(TASKS)}]",
                "exog": exog_interface,
                "prediction": f"[batch, {args.horizon}, {len(TASKS)}]",
                "device": trainer_config.device,
            },
            "sample_counts": {
                name: int(len(value["target"])) for name, value in windows.items()
            },
            "test_standardized_smooth_l1": test_loss,
            "metrics_original_scale": metrics,
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
            "gate_weights_test_file": (
                gate_path.name if gate_path is not None else None
            ),
        },
        output_dir / "metrics_test.json",
    )
    save_json({"history": history}, output_dir / "history.json")
    print(
        json.dumps(
            {
                "model": args.model,
                "input_mode": input_mode,
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
