"""Stage 7.6: aggregate formal results and verify the Stage 7 contract.

This script is intentionally read-only with respect to the Stage 7.3--7.5
source directories. It recomputes metrics from saved predictions, aggregates
fixed-seed runs, links the structure-matched STL reference and A4 reuse records,
and creates the tables needed for the final experiment report.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage7_3 import _seasonal_metrics  # noqa: E402
from src.baselines import regression_metrics  # noqa: E402
from src.data_pipeline import save_json  # noqa: E402


TASKS: Tuple[str, ...] = ("electricity", "cooling", "heating", "gas")
METRICS: Tuple[str, ...] = ("MAE", "RMSE", "WAPE", "MAPE")
EXPECTED_STAGE_RUNS = {"7.3": 20, "7.4": 40, "7.5": 34, "7R.STL": 10}
REUSED_A4_RUNS = 10
EXPECTED_TOTAL_RUNS = 114


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_stage7_6_source_plan() -> Tuple[Dict[str, object], ...]:
    """Return the three frozen source groups used by Stage 7.6."""

    return (
        {
            "stage": "7.3",
            "source_group": "main_internal",
            "default_dir": "frame/reports/stage7_3_kitakyushu_formal",
            "manifest": "stage7_3_manifest.json",
            "expected_runs": 20,
        },
        {
            "stage": "7.4",
            "source_group": "ablation",
            "default_dir": "frame/reports/stage7_4_kitakyushu_formal",
            "manifest": "stage7_4_manifest.json",
            "expected_runs": 40,
        },
        {
            "stage": "7.5",
            "source_group": "external_baseline",
            "default_dir": "frame/reports/stage7_5_kitakyushu_formal",
            "manifest": "stage7_5_manifest.json",
            "expected_runs": 34,
        },
        {
            "stage": "7R.STL",
            "source_group": "matched_stl_reference",
            "default_dir": "frame/reports/stage7_stl_reference_kitakyushu_formal",
            "manifest": "stage7r_stl_manifest.json",
            "expected_runs": 10,
        },
    )


def _validate_root_manifest(root: Path, stage: str, manifest_name: str | None = None) -> Dict[str, object]:
    manifest_path = root / (manifest_name or f"stage{stage.replace('.', '_')}_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Stage {stage} manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    expected = EXPECTED_STAGE_RUNS[stage]
    if manifest.get("status") != "passed":
        raise ValueError(f"Stage {stage} manifest is not passed: {manifest_path}")
    if manifest.get("run_count_expected") != expected:
        raise ValueError(
            f"Stage {stage} expected run count mismatch: "
            f"{manifest.get('run_count_expected')} != {expected}"
        )
    if manifest.get("run_count_completed") != expected or manifest.get("run_count_failed") != 0:
        raise ValueError(f"Stage {stage} does not contain all successful runs")
    return manifest


def _validate_run_manifest(run_dir: Path, stage: str) -> Dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    prediction_path = run_dir / "predictions_test.npz"
    if not manifest_path.exists() or not prediction_path.exists():
        raise FileNotFoundError(f"incomplete Stage {stage} run: {run_dir}")
    manifest = _read_json(manifest_path)
    if manifest.get("stage") != stage:
        raise ValueError(
            f"run stage mismatch: {manifest.get('stage')} != {stage}: {manifest_path}"
        )
    if manifest.get("status") != "passed":
        raise ValueError(f"run is not passed: {manifest_path}")
    if tuple(manifest.get("tasks", ())) != TASKS:
        raise ValueError(f"task order mismatch: {manifest_path}")
    window = manifest.get("window", {})
    if window.get("lookback") != 24 or window.get("horizon") != 4:
        raise ValueError(f"window protocol mismatch: {manifest_path}")
    if manifest.get("test_used_for_selection", False):
        raise ValueError(f"run reports test-set selection: {manifest_path}")
    return manifest


def _iter_runs(root: Path, stage: str, source_group: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = _validate_run_manifest(run_dir, stage)
        records.append(
            {
                "stage": stage,
                "source_group": source_group,
                "run_dir": run_dir,
                "manifest": manifest,
            }
        )
    expected = EXPECTED_STAGE_RUNS[stage]
    if len(records) != expected:
        raise ValueError(
            f"Stage {stage} run-directory count mismatch: {len(records)} != {expected}"
        )
    return records


def _add_reused_a4_records(
    records: List[Dict[str, object]],
    stage73_records: Sequence[Mapping[str, object]],
    freeze: Mapping[str, object],
) -> None:
    """Link A4 to Stage 7.3 primary outputs without copying or retraining."""

    primary = freeze.get("primary_model")
    if not isinstance(primary, Mapping):
        raise ValueError("freeze config missing primary_model for A4 reuse")
    primary_model = str(primary.get("model"))
    primary_candidate = str(primary.get("candidate_id"))
    selected = [
        record for record in stage73_records
        if str(record["manifest"].get("model")) == primary_model
        and str(record["manifest"].get("candidate_id")) == primary_candidate
    ]
    if len(selected) != REUSED_A4_RUNS:
        raise ValueError(
            f"A4 reuse requires {REUSED_A4_RUNS} Stage 7.3 primary runs, found {len(selected)}"
        )
    for record in selected:
        reused_manifest = dict(record["manifest"])
        reused_manifest.update(
            {
                "stage": "7.4",
                "model": "A4",
                "candidate_id": primary_candidate,
                "ablation_reused_from": str(record["run_dir"]),
                "reused_without_retraining": True,
            }
        )
        records.append(
            {
                "stage": "7.4",
                "source_group": "ablation_reused",
                "run_dir": record["run_dir"],
                "manifest": reused_manifest,
            }
        )


def _load_metrics(record: Mapping[str, object]) -> Dict[str, object]:
    run_dir = Path(record["run_dir"])
    with np.load(run_dir / "predictions_test.npz") as archive:
        required = {"target", "prediction", "target_times"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"prediction archive is missing {sorted(missing)}: "
                f"{run_dir / 'predictions_test.npz'}"
            )
        target = np.asarray(archive["target"], dtype=np.float64)
        prediction = np.asarray(archive["prediction"], dtype=np.float64)
        target_times = np.asarray(archive["target_times"])
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError(
            f"prediction shape mismatch at {run_dir}: "
            f"target={target.shape}, prediction={prediction.shape}"
        )
    if target.shape[1:] != (4, len(TASKS)):
        raise ValueError(
            f"expected [N, 4, {len(TASKS)}] predictions at {run_dir}, "
            f"found {target.shape}"
        )
    if len(target_times) != target.shape[0]:
        raise ValueError(
            f"target_times length mismatch at {run_dir}: "
            f"{len(target_times)} != {target.shape[0]}"
        )
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError(f"non-finite target/prediction values at {run_dir}")
    metrics = regression_metrics(target, prediction, task_names=TASKS)
    metrics["per_season"] = _seasonal_metrics(
        target, prediction, target_times
    )
    return metrics


def _metric_rows(record: Mapping[str, object], metrics: Mapping[str, object]) -> List[Dict[str, object]]:
    manifest = record["manifest"]
    base = {
        "stage": record["stage"],
        "source_group": record["source_group"],
        "protocol": manifest.get("protocol", ""),
        "model": manifest.get("model", ""),
        "seed": manifest.get("seed") if manifest.get("seed") is not None else "deterministic",
        "run_dir": str(record["run_dir"]),
    }
    rows: List[Dict[str, object]] = []

    def add(granularity: str, label: str, values: Mapping[str, object]) -> None:
        row = dict(base)
        row.update({"granularity": granularity, "label": label})
        row.update({metric: values[metric] for metric in METRICS})
        rows.append(row)

    add("overall", "overall", metrics["overall_equal_task_mean"])
    for task in TASKS:
        add("task", task, metrics["per_task"][task])
    horizon_metrics = metrics.get(
        "per_horizon_equal_task_mean",
        metrics.get("per_horizon_equal_element_mean", {}),
    )
    for step, values in horizon_metrics.items():
        add("horizon", step, values)
    for season, values in metrics["per_season"].items():
        if values.get("sample_count", 0) == 0:
            continue
        add("season", season, values["overall_equal_task_mean"])
    return rows


def _aggregate_metric_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["stage"]),
            str(row["source_group"]),
            str(row["protocol"]),
            str(row["model"]),
            str(row["granularity"]),
            str(row["label"]),
        )
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        stage, source_group, protocol, model, granularity, label = key
        item: Dict[str, object] = {
            "stage": stage,
            "source_group": source_group,
            "protocol": protocol,
            "model": model,
            "granularity": granularity,
            "label": label,
            "run_count": len(group),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _resource_rows(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        manifest = record["manifest"]
        groups[
            (
                str(record["stage"]),
                str(record["source_group"]),
                str(manifest.get("protocol", "")),
                str(manifest.get("model", "")),
            )
        ].append(manifest)
    rows: List[Dict[str, object]] = []
    for key, group in sorted(groups.items()):
        stage, source_group, protocol, model = key
        item: Dict[str, object] = {
            "stage": stage,
            "source_group": source_group,
            "protocol": protocol,
            "model": model,
            "run_count": len(group),
        }
        for field in ("parameter_count",):
            values = np.asarray([float(item.get(field, 0)) for item in group])
            item[f"{field}_mean"] = float(values.mean())
            item[f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for field in ("fit", "validation_evaluation", "test_evaluation"):
            values = np.asarray(
                [float(item.get("runtime_seconds", {}).get(field, 0.0)) for item in group]
            )
            item[f"{field}_seconds_mean"] = float(values.mean())
            item[f"{field}_seconds_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        sample_counts = group[0].get("sample_counts", {})
        item.update(
            {
                "train_samples": sample_counts.get("train", 0),
                "validation_samples": sample_counts.get("validation", 0),
                "test_samples": sample_counts.get("test", 0),
            }
        )
        rows.append(item)
    return rows


def _index_rows(records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for record in records:
        manifest = record["manifest"]
        rows.append(
            {
                "stage": record["stage"],
                "source_group": record["source_group"],
                "protocol": manifest.get("protocol", ""),
                "model": manifest.get("model", ""),
                "seed": manifest.get("seed") if manifest.get("seed") is not None else "deterministic",
                "status": manifest.get("status"),
                "test_set_accessed": manifest.get("test_set_accessed"),
                "test_used_for_selection": manifest.get("test_used_for_selection", False),
                "parameter_count": manifest.get("parameter_count", 0),
                "run_dir": str(record["run_dir"]),
                "prediction_file": str(Path(record["run_dir"]) / "predictions_test.npz"),
            }
        )
    return rows


def _diagnostic_linkage(
    stage6_4_dir: Path, stage6_5_dir: Path
) -> List[Dict[str, object]]:
    stage6_manifests = {
        "6.4": stage6_4_dir / "stage6_4_manifest.json",
        "6.5": stage6_5_dir / "stage6_5_manifest.json",
    }
    for stage, manifest_path in stage6_manifests.items():
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing Stage {stage} manifest: {manifest_path}")
        manifest = _read_json(manifest_path)
        if manifest.get("test_set_accessed") is not False:
            raise ValueError(
                f"Stage {stage} diagnostic manifest must report "
                f"test_set_accessed=false: {manifest_path}"
            )
    sources = [
        ("gate_diagnostics", "6.4", stage6_4_dir, "gate_diagnostics_summary.csv"),
        ("gate_asymmetry", "6.4", stage6_4_dir, "gate_asymmetry.csv"),
        ("transfer_gains", "6.5", stage6_5_dir, "transfer_gains.csv"),
        ("negative_transfer_summary", "6.5", stage6_5_dir, "negative_transfer_summary.csv"),
    ]
    rows = []
    for diagnostic_type, stage, root, filename in sources:
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"missing diagnostic file: {path}")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            row_count = max(sum(1 for _ in handle) - 1, 0)
        rows.append(
            {
                "diagnostic_type": diagnostic_type,
                "stage": stage,
                "source_file": str(path),
                "row_count": row_count,
                "test_set_accessed": False,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Stage 7 formal results")
    parser.add_argument("--stage7-3-dir", default="frame/reports/stage7_3_kitakyushu_formal")
    parser.add_argument("--stage7-4-dir", default="frame/reports/stage7_4_kitakyushu_formal")
    parser.add_argument("--stage7-5-dir", default="frame/reports/stage7_5_kitakyushu_formal")
    parser.add_argument("--stage7-stl-dir", default="frame/reports/stage7_stl_reference_kitakyushu_formal")
    parser.add_argument("--stage6-4-dir", default="frame/reports/stage6_4/kitakyushu_full")
    parser.add_argument("--stage6-5-dir", default="frame/reports/stage6_5/kitakyushu_full")
    parser.add_argument("--freeze-config", default="frame/reports/stage6_6_kitakyushu/stage6_selected_config.json")
    parser.add_argument("--output-dir", default="frame/reports/stage7_6_kitakyushu_acceptance")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_stage7_6_source_plan()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "7.6",
                    "status": "dry_run",
                    "source_stages": [item["stage"] for item in plan],
                    "trained_source_runs": sum(int(item["expected_runs"]) for item in plan),
                    "reused_a4_runs": REUSED_A4_RUNS,
                    "expected_formal_runs": EXPECTED_TOTAL_RUNS,
                    "outputs": [
                        "raw_result_index.csv",
                        "metrics_by_run.csv",
                        "detailed_metrics_mean_std.csv",
                        "overall_comparison_mean_std.csv",
                        "resource_summary_mean_std.csv",
                        "diagnostic_linkage.csv",
                        "stage7_6_manifest.json",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    roots = {
        "7.3": _resolve(args.stage7_3_dir),
        "7.4": _resolve(args.stage7_4_dir),
        "7.5": _resolve(args.stage7_5_dir),
        "7R.STL": _resolve(args.stage7_stl_dir),
    }
    freeze = _read_json(_resolve(args.freeze_config))
    output_dir = _resolve(args.output_dir)
    manifest_path = output_dir / "stage7_6_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"manifest exists; use --force: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifests = {
        str(item["stage"]): _validate_root_manifest(
            roots[str(item["stage"])], str(item["stage"]), str(item["manifest"])
        )
        for item in plan
    }
    records: List[Dict[str, object]] = []
    stage73_records: List[Dict[str, object]] = []
    for item in plan:
        source_records = _iter_runs(
            roots[str(item["stage"])], str(item["stage"]), str(item["source_group"])
        )
        records.extend(source_records)
        if str(item["stage"]) == "7.3":
            stage73_records = source_records
    _add_reused_a4_records(records, stage73_records, freeze)
    if len(records) != EXPECTED_TOTAL_RUNS:
        raise ValueError(f"expected {EXPECTED_TOTAL_RUNS} effective formal runs, found {len(records)}")
    if any(record["manifest"].get("test_used_for_selection", False) for record in records):
        raise ValueError("a Stage 7 run reports test-set selection")

    metric_rows: List[Dict[str, object]] = []
    for index, record in enumerate(records, start=1):
        metrics = _load_metrics(record)
        record["metrics"] = metrics
        metric_rows.extend(_metric_rows(record, metrics))
        print(
            json.dumps(
                {"stage": "7.6", "status": "run_indexed", "indexed": index, "total": len(records)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    index_rows = _index_rows(records)
    resource_rows = _resource_rows(records)
    aggregate_rows = _aggregate_metric_rows(metric_rows)
    overall_rows = [row for row in aggregate_rows if row["granularity"] == "overall"]
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
        "stage": "7.6",
        "status": "passed",
        "source_manifests": {
            str(item["stage"]): str(roots[str(item["stage"])] / str(item["manifest"]))
            for item in plan
        },
        "source_run_counts": {stage: len([r for r in records if r["stage"] == stage]) for stage in roots},
        "formal_run_count": len(records),
        "metric_row_count": len(metric_rows),
        "aggregate_row_count": len(aggregate_rows),
        "test_set_policy": {
            "selection_locked": True,
            "stage7_test_reading_allowed": True,
            "test_used_for_selection": False,
        },
        "stage6_diagnostics": {
            "gate_diagnostics_dir": str(_resolve(args.stage6_4_dir)),
            "transfer_analysis_dir": str(_resolve(args.stage6_5_dir)),
            "test_set_accessed": False,
        },
        "output_files": [
            "raw_result_index.csv",
            "metrics_by_run.csv",
            "detailed_metrics_mean_std.csv",
            "overall_comparison_mean_std.csv",
            "resource_summary_mean_std.csv",
            "diagnostic_linkage.csv",
            "stage7_6_manifest.json",
        ],
        "source_status": {
            stage: source_manifests[stage].get("status", "passed") for stage in roots
        },
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
