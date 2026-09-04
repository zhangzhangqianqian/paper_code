"""Build the immutable formal-v4.1 Standard IES benchmark from 2015--2018."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_benchmark import (  # noqa: E402
    FORMAL_V4_TRAIN_YEARS,
    build_formal_v4_benchmark,
    write_formal_v4_benchmark_artifacts,
)
from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical  # noqa: E402
from src.scheduling.parameter_audit import read_parameter_ledger  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rules(path: Path, ledger_path: Path) -> dict[str, Any]:
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(rules, dict):
        raise ValueError("benchmark rules must contain an object")
    ledger = read_parameter_ledger(ledger_path)
    rules["ledger_values"] = {record.parameter_id: record.value for record in ledger.records}
    return rules


def build_from_data(
    data_dir: Path,
    rules_path: Path,
    ledger_path: Path,
    run_root: Path,
) -> dict[str, Any]:
    raw, metadata = read_kitakyushu_canonical(data_dir, years=FORMAL_V4_TRAIN_YEARS)
    frame, cleaning_report = clean_kitakyushu_dataframe(raw)
    rules = _load_rules(rules_path, ledger_path)
    benchmark = build_formal_v4_benchmark(frame, rules, _sha256(ledger_path))
    source_files = metadata.get("source_files", {})
    receipt = write_formal_v4_benchmark_artifacts(
        benchmark,
        run_root,
        source_files=source_files if isinstance(source_files, dict) else {},
        cleaning_report=cleaning_report,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Paper\Kitakyushu dataset"))
    parser.add_argument("--rules", type=Path, default=FRAME_ROOT / "configs" / "standard_ies_formal_v4_rules.yaml")
    parser.add_argument("--ledger", type=Path, default=FRAME_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_from_data(args.data_dir, args.rules, args.ledger, args.run_root)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
