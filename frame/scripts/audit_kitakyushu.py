"""审计并生成 Kitakyushu 四任务规范数据。

示例（PowerShell）：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\audit_kitakyushu.py `
        --data-dir "Kitakyushu dataset" `
        --output-dir frame\\reports\\data_audit\\kitakyushu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import save_json, split_dataframe  # noqa: E402
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_SPLIT,
    audit_kitakyushu_dataframe,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="审计并规范化 Kitakyushu Energy Station 四任务数据"
    )
    parser.add_argument(
        "--data-dir",
        default="Kitakyushu dataset",
        help="包含三个核心 ZIP 数据包的目录",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/data_audit/kitakyushu",
        help="审计报告、字段映射和清洗数据输出目录",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=list(range(2015, 2022)),
        help="读取的年份，默认 2015—2021",
    )
    parser.add_argument(
        "--max-interpolation-hours",
        type=int,
        default=3,
        help="气象变量内部短缺口的最大插值长度",
    )
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.max_interpolation_hours < 0:
        raise ValueError("--max-interpolation-hours 不能为负数")
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, metadata = read_kitakyushu_canonical(
        _resolve_path(args.data_dir), years=tuple(args.years)
    )
    raw_audit = audit_kitakyushu_dataframe(frame)
    save_json(
        {
            "dataset_kind": "kitakyushu_energy_station",
            "metadata": metadata,
            "raw_audit": raw_audit,
        },
        output_dir / "raw_quality_report.json",
    )
    save_json(metadata, output_dir / "field_mapping.json")

    cleaned, cleaning = clean_kitakyushu_dataframe(
        frame, max_interpolation_hours=args.max_interpolation_hours
    )
    cleaned_audit = audit_kitakyushu_dataframe(cleaned)
    save_json(
        {
            "dataset_kind": "kitakyushu_energy_station",
            "metadata": metadata,
            "cleaning": cleaning,
            "cleaned_audit": cleaned_audit,
        },
        output_dir / "cleaned_quality_report.json",
    )
    cleaned.to_csv(output_dir / "cleaned_canonical.csv", index=False)
    cleaned_hash = _sha256(output_dir / "cleaned_canonical.csv")

    splits = split_dataframe(cleaned, KITAKYUSHU_SPLIT)
    split_summary = {
        name: {
            "rows": int(len(part)),
            "start": part["timestamp"].min().isoformat(),
            "end": part["timestamp"].max().isoformat(),
        }
        for name, part in splits.items()
    }
    save_json(
        {
            "dataset_kind": "kitakyushu_energy_station",
            "protocol": {
                "train": [KITAKYUSHU_SPLIT.train_start, KITAKYUSHU_SPLIT.train_end],
                "validation": [
                    KITAKYUSHU_SPLIT.validation_start,
                    KITAKYUSHU_SPLIT.validation_end,
                ],
                "test": [KITAKYUSHU_SPLIT.test_start, KITAKYUSHU_SPLIT.test_end],
            },
            "splits": split_summary,
            "cleaned_csv": {
                "path": str(output_dir / "cleaned_canonical.csv"),
                "sha256": cleaned_hash,
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
            },
        },
        output_dir / "split_summary.json",
    )
    print(
        json.dumps(
            {
                "dataset_kind": "kitakyushu_energy_station",
                "rows_raw": int(len(frame)),
                "rows_cleaned": int(len(cleaned)),
                "columns": list(cleaned.columns),
                "split_summary": split_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
