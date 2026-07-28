"""运行 Persistence 和 Seasonal Naive 基线。

HEEW 区域级数据示例：

    D:\anaconda\envs\pytorch\python.exe frame\scripts\run_baselines.py `
        --energy-file dataset\HEEW\cleaned_data\Total_energy.csv `
        --weather-file dataset\HEEW\cleaned_data\Total_weather.csv `
        --output-dir frame\reports\phase2

基线只使用负荷历史，不使用气象外生变量；这保证它们与后续模型的比较
只反映联合建模能力，而不是外生变量接口差异。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import (  # noqa: E402
    persistence_forecast,
    regression_metrics,
    seasonal_naive_forecast,
)
from src.data_pipeline import (  # noqa: E402
    FULL_SPLIT,
    SMALL_SAMPLE_SPLIT,
    clean_dataframe,
    build_protocol_windows,
    read_csv_canonical,
    read_heew_canonical,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行多能源负荷 Persistence 和 Seasonal Naive 基线"
    )
    parser.add_argument("--input", help="兼容旧格式的单个规范 CSV 路径")
    parser.add_argument("--energy-file", help="HEEW 负荷 CSV 路径")
    parser.add_argument("--weather-file", help="HEEW 气象 CSV 路径")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument(
        "--protocol",
        choices=("full", "small_sample", "both"),
        default="both",
    )
    return parser.parse_args()


def _read_input(args: argparse.Namespace):
    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    if args.input and (args.energy_file or args.weather_file):
        raise ValueError("--input 与 --energy-file/--weather-file 不能同时使用")
    if args.input:
        return read_csv_canonical(resolve_path(args.input))[0], "legacy_single_csv"
    if bool(args.energy_file) != bool(args.weather_file):
        raise ValueError("HEEW 输入必须同时提供 --energy-file 和 --weather-file")
    if not args.energy_file:
        default_energy = REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_energy.csv"
        default_weather = REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_weather.csv"
        return read_heew_canonical(default_energy, default_weather)[0], "heew_total"
    return (
        read_heew_canonical(
            resolve_path(args.energy_file), resolve_path(args.weather_file)
        )[0],
        "heew_total",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, dataset_kind = _read_input(args)
    cleaned, cleaning_report = clean_dataframe(frame)
    protocols = {}
    if args.protocol in ("full", "both"):
        protocols["full"] = FULL_SPLIT
    if args.protocol in ("small_sample", "both"):
        protocols["small_sample"] = SMALL_SAMPLE_SPLIT

    for protocol_name, spec in protocols.items():
        windows = build_protocol_windows(
            cleaned,
            spec,
            split_name="test",
            lookback=24,
            horizon=4,
            exog_columns=(),
        )
        history = windows["loads"]
        actual = windows["target"]
        predictions = {
            "persistence": persistence_forecast(history, horizon=4),
            "seasonal_naive": seasonal_naive_forecast(
                history, horizon=4, season_length=24
            ),
        }
        metrics = {
            name: regression_metrics(actual, prediction)
            for name, prediction in predictions.items()
        }
        np.savez_compressed(
            output_dir / f"predictions_{protocol_name}.npz",
            actual=actual,
            **predictions,
        )
        save_json(
            {
                "dataset_kind": dataset_kind,
                "cleaning": cleaning_report,
                "metrics": metrics,
            },
            output_dir / f"metrics_{protocol_name}.json",
        )
        print(
            json.dumps(
                {
                    "protocol": protocol_name,
                    "dataset_kind": dataset_kind,
                    "samples": int(actual.shape[0]),
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
