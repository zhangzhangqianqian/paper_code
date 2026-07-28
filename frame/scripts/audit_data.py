"""运行数据质量审计、最小清洗和时间协议检查。

HEEW 区域级数据示例：

    D:\anaconda\envs\pytorch\python.exe frame\scripts\audit_data.py `
        --energy-file dataset\HEEW\cleaned_data\Total_energy.csv `
        --weather-file dataset\HEEW\cleaned_data\Total_weather.csv `
        --output-dir frame\reports\phase1

旧版单 CSV 数据仍可通过 ``--input`` 读取。
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

from src.data_pipeline import (  # noqa: E402
    FULL_SPLIT,
    HEEW_EXOG_COLUMNS,
    SMALL_SAMPLE_SPLIT,
    TASKS,
    audit_dataframe,
    clean_dataframe,
    read_csv_canonical,
    read_heew_canonical,
    save_json,
    split_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计 HEEW 双文件或兼容的单 CSV 多能源负荷数据"
    )
    parser.add_argument("--input", help="兼容旧格式的单个规范 CSV 路径")
    parser.add_argument("--energy-file", help="HEEW 负荷 CSV 路径")
    parser.add_argument("--weather-file", help="HEEW 气象 CSV 路径")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="质量报告、清洗数据和切分摘要输出目录",
    )
    parser.add_argument(
        "--protocol",
        choices=("full", "small_sample", "both"),
        default="both",
        help="时间切分协议",
    )
    parser.add_argument(
        "--max-interpolation-hours",
        type=int,
        default=3,
        help="内部短缺口的最大插值长度",
    )
    return parser.parse_args()


def _read_input(args: argparse.Namespace):
    def resolve_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    if args.input and (args.energy_file or args.weather_file):
        raise ValueError("--input 与 --energy-file/--weather-file 不能同时使用")
    if args.input:
        return read_csv_canonical(resolve_path(args.input)), "legacy_single_csv"
    if bool(args.energy_file) != bool(args.weather_file):
        raise ValueError("HEEW 输入必须同时提供 --energy-file 和 --weather-file")
    if not args.energy_file:
        default_energy = REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_energy.csv"
        default_weather = REPOSITORY_ROOT / "dataset/HEEW/cleaned_data/Total_weather.csv"
        return read_heew_canonical(default_energy, default_weather), "heew_total"
    return (
        read_heew_canonical(
            resolve_path(args.energy_file), resolve_path(args.weather_file)
        ),
        "heew_total",
    )


def main() -> None:
    args = parse_args()
    if args.max_interpolation_hours < 0:
        raise ValueError("--max-interpolation-hours 不能为负数")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    (frame, resolved), dataset_kind = _read_input(args)
    required_columns = ("timestamp", *TASKS)
    provenance = {}
    if dataset_kind == "heew_total":
        required_columns = ("timestamp", *TASKS, *HEEW_EXOG_COLUMNS)
        manifest_path = REPOSITORY_ROOT / "dataset/HEEW/manifest.json"
        if manifest_path.exists():
            provenance = {
                "manifest_path": str(manifest_path),
                "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            }
    raw_report = audit_dataframe(frame, required_columns=required_columns)
    save_json(
        {
            "dataset_kind": dataset_kind,
            "provenance": provenance,
            "resolved_columns": resolved,
            "raw_audit": raw_report,
        },
        output_dir / "raw_quality_report.json",
    )

    cleaned, clean_report = clean_dataframe(
        frame, max_interpolation_hours=args.max_interpolation_hours
    )
    clean_audit = audit_dataframe(cleaned, required_columns=required_columns)
    save_json(
        {
            "dataset_kind": dataset_kind,
            "provenance": provenance,
            "cleaning": clean_report,
            "cleaned_audit": clean_audit,
        },
        output_dir / "cleaned_quality_report.json",
    )
    cleaned.to_csv(output_dir / "cleaned_canonical.csv", index=False)

    specs = {}
    if args.protocol in ("full", "both"):
        specs["full"] = FULL_SPLIT
    if args.protocol in ("small_sample", "both"):
        specs["small_sample"] = SMALL_SAMPLE_SPLIT

    split_summary = {}
    for name, spec in specs.items():
        splits = split_dataframe(cleaned, spec)
        split_summary[name] = {
            split_name: {
                "rows": int(len(part)),
                "start": part["timestamp"].min().isoformat(),
                "end": part["timestamp"].max().isoformat(),
            }
            for split_name, part in splits.items()
        }
    save_json(split_summary, output_dir / "split_summary.json")

    print(
        json.dumps(
            {"dataset_kind": dataset_kind, "split_summary": split_summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
