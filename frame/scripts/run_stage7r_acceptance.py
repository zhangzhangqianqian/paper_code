"""Legacy Stage 7-R aggregator.

The canonical acceptance path is ``run_stage7_6.py``.  This compatibility
script aggregates only materialized source runs and therefore does not add
the A4 reuse records; it is retained for reading older Stage 7-R directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage7_6 import (  # noqa: E402
    _aggregate_metric_rows,
    _diagnostic_linkage,
    _index_rows,
    _load_metrics,
    _metric_rows,
    _read_json,
    _resource_rows,
    _validate_run_manifest,
    _write_csv,
)
from src.data_pipeline import save_json  # noqa: E402


REVISION_VERSION = "stage7R.1"
EXPECTED_DEFAULT_TOTAL = 114


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def build_acceptance_sources(
    root: Path,
    stage7_4_runs: int = 40,
) -> Tuple[Dict[str, object], ...]:
    return (
        {
            "stage": "7.3",
            "source_group": "main_internal_reproducible",
            "root": root / "stage7_3",
            "manifest": "stage7_3_manifest.json",
            "expected_runs": 20,
        },
        {
            "stage": "7.4",
            "source_group": "ablation_reproducible",
            "root": root / "stage7_4",
            "manifest": "stage7_4_manifest.json",
            "expected_runs": int(stage7_4_runs),
        },
        {
            "stage": "7.5",
            "source_group": "external_baseline_reproducible",
            "root": root / "stage7_5",
            "manifest": "stage7_5_manifest.json",
            "expected_runs": 44,
        },
        {
            "stage": "7R.STL",
            "source_group": "matched_stl_reference",
            "root": root / "stl_reference",
            "manifest": "stage7r_stl_manifest.json",
            "expected_runs": 10,
        },
    )


def _validate_source(item: Mapping[str, object]) -> Dict[str, object]:
    root = Path(item["root"])
    manifest_path = root / str(item["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Stage 7-R source manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    expected = int(item["expected_runs"])
    if manifest.get("status") != "passed":
        raise ValueError(f"source manifest is not passed: {manifest_path}")
    if manifest.get("run_count_expected") != expected:
        raise ValueError(f"source expected-run mismatch: {manifest_path}")
    if manifest.get("run_count_completed") != expected:
        raise ValueError(f"source completed-run mismatch: {manifest_path}")
    if manifest.get("run_count_failed") != 0:
        raise ValueError(f"source contains failed runs: {manifest_path}")
    if manifest.get("strict_seed_control") is not True:
        raise ValueError(f"source lacks strict seed control: {manifest_path}")
    if manifest.get("legacy_results_excluded") is not True:
        raise ValueError(f"source does not exclude legacy results: {manifest_path}")
    return manifest


def _collect_runs(item: Mapping[str, object]) -> List[Dict[str, object]]:
    root = Path(item["root"])
    stage = str(item["stage"])
    records: List[Dict[str, object]] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        manifest = _validate_run_manifest(manifest_path.parent, stage)
        deterministic = manifest.get("baseline_type") == "deterministic"
        reproducibility = manifest.get("reproducibility", {})
        if not deterministic and reproducibility.get(
            "model_initialized_after_seed"
        ) is not True:
            raise ValueError(f"run lacks strict seed evidence: {manifest_path}")
        records.append(
            {
                "stage": stage,
                "source_group": item["source_group"],
                "run_dir": manifest_path.parent,
                "manifest": manifest,
            }
        )
    expected = int(item["expected_runs"])
    if len(records) != expected:
        raise ValueError(
            f"{stage} run-directory count mismatch: {len(records)} != {expected}"
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the 114 materialized Stage 7-R runs"
    )
    parser.add_argument(
        "--stage7r-root",
        default="frame/reports/stage7r_kitakyushu_reproducible",
    )
    parser.add_argument(
        "--revision-contract",
        default="frame/configs/stage7r_reproducibility_contract.json",
    )
    parser.add_argument(
        "--stage6-4-dir", default="frame/reports/stage6_4/kitakyushu_full"
    )
    parser.add_argument(
        "--stage6-5-dir", default="frame/reports/stage6_5/kitakyushu_full"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _resolve(args.stage7r_root)
    contract = _read_json(_resolve(args.revision_contract))
    if contract.get("revision_version") != REVISION_VERSION:
        raise ValueError("Stage 7-R acceptance requires the stage7R.1 contract")
    matrix = contract.get("run_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("revision contract run_matrix is missing")
    stage7_4_runs = int(matrix["stage7_4_ablations"])
    expected_total = int(matrix["materialized_run_count"])
    sources = build_acceptance_sources(root, stage7_4_runs=stage7_4_runs)
    planned = sum(int(item["expected_runs"]) for item in sources)
    if planned != expected_total:
        raise ValueError(
            f"acceptance plan must contain {expected_total} runs, found {planned}"
        )
    output_dir = _resolve(args.output_dir) if args.output_dir else root / "acceptance"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "7-R acceptance",
                    "revision_version": REVISION_VERSION,
                    "status": "dry_run",
                    "expected_formal_runs": expected_total,
                    "stage7r_root": str(root),
                    "output_dir": str(output_dir),
                    "sources": [
                        {
                            "stage": item["stage"],
                            "source_group": item["source_group"],
                            "expected_runs": item["expected_runs"],
                        }
                        for item in sources
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    manifest_path = output_dir / "stage7r_acceptance_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"manifest exists; use --force: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifests = {
        str(item["stage"]): _validate_source(item) for item in sources
    }
    records: List[Dict[str, object]] = []
    for item in sources:
        records.extend(_collect_runs(item))
    if len(records) != expected_total:
        raise ValueError(
            f"expected {expected_total} materialized runs, found {len(records)}"
        )
    if any(record["manifest"].get("test_used_for_selection", False) for record in records):
        raise ValueError("a Stage 7-R run reports test-set selection")

    metric_rows: List[Dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        metrics = _load_metrics(record)
        metric_rows.extend(_metric_rows(record, metrics))
        print(
            json.dumps(
                {
                    "stage": "7-R acceptance",
                    "status": "run_indexed",
                    "indexed": index,
                    "total": len(records),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    index_rows = _index_rows(records)
    aggregate_rows = _aggregate_metric_rows(metric_rows)
    overall_rows = [row for row in aggregate_rows if row["granularity"] == "overall"]
    resource_rows = _resource_rows(records)
    linkage_rows = _diagnostic_linkage(
        _resolve(args.stage6_4_dir), _resolve(args.stage6_5_dir)
    )
    _write_csv(index_rows, output_dir / "raw_result_index.csv")
    _write_csv(metric_rows, output_dir / "metrics_by_run.csv")
    _write_csv(aggregate_rows, output_dir / "detailed_metrics_mean_std.csv")
    _write_csv(overall_rows, output_dir / "overall_comparison_mean_std.csv")
    _write_csv(resource_rows, output_dir / "resource_summary_mean_std.csv")
    _write_csv(linkage_rows, output_dir / "diagnostic_linkage.csv")

    manifest = {
        "stage": "7-R acceptance",
        "revision_version": REVISION_VERSION,
        "status": "passed",
        "stage7r_root": str(root),
        "formal_run_count": len(records),
        "source_run_counts": {
            str(item["stage"]): int(item["expected_runs"]) for item in sources
        },
        "metric_row_count": len(metric_rows),
        "aggregate_row_count": len(aggregate_rows),
        "strict_seed_control": True,
        "legacy_results_excluded": True,
        "test_set_policy": {
            "stage6_6_selection_locked": True,
            "stage7r_test_reading_allowed": True,
            "test_used_for_selection": False,
        },
        "source_status": {
            stage: manifest.get("status")
            for stage, manifest in source_manifests.items()
        },
        "output_files": [
            "raw_result_index.csv",
            "metrics_by_run.csv",
            "detailed_metrics_mean_std.csv",
            "overall_comparison_mean_std.csv",
            "resource_summary_mean_std.csv",
            "diagnostic_linkage.csv",
            "stage7r_acceptance_manifest.json",
        ],
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
