"""Run the frozen Stage 8 negative-transfer and gate analysis.

This entry point is inference-only.  It reads accepted Stage 7-R test
predictions and frozen checkpoints, performs the pre-specified post-hoc
analysis, and publishes results only after a complete validation pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_resource_benchmarks import (  # noqa: E402
    _build_benchmark_model,
)
from run_stage7_3 import _standardize_protocol  # noqa: E402
from run_stage7_6 import (  # noqa: E402
    _iter_runs,
    _read_resource_csv,
    build_stage7_6_source_plan,
)
from src.data_pipeline import build_protocol_windows, select_training_frame  # noqa: E402
from src.kitakyushu_pipeline import (  # noqa: E402
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from src.stage8_analysis import (  # noqa: E402
    HORIZON,
    METRICS,
    SEASONS,
    TASKS,
    bh_adjust,
    equal_task_error_metrics,
    error_metrics,
    gain,
    hierarchical_bootstrap_mae_gain,
    negative_transfer_rate,
    paired_block_bootstrap_error_gain,
    season_labels,
    spearman_correlation,
    temperature_bin_labels,
    temperature_thresholds,
    stage8_expected_tables,
    stage8_role_spec,
    validate_joint_record_matrix,
    weekday_labels,
)

# Backward-compatible names used by the original pure-function tests.
from src.stage8_analysis import bh_adjust as _bh_adjust  # noqa: E402
from src.stage8_analysis import block_indices as _block_indices  # noqa: E402
from src.stage8_analysis import gain as _gain  # noqa: E402


SEEDS = (2026, 2027, 2028, 2029, 2030)
LOOKBACK = 24
EXPECTED_FORMAL_RUNS = 124


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_prediction(run_dir: Path) -> Dict[str, np.ndarray]:
    path = run_dir / "predictions_test.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing test prediction: {path}")
    with np.load(path, allow_pickle=False) as values:
        required = {"target", "prediction", "target_times"}
        missing = required.difference(values.files)
        if missing:
            raise ValueError(f"test prediction missing {sorted(missing)}: {path}")
        target = np.asarray(values["target"], dtype=np.float64)
        prediction = np.asarray(values["prediction"], dtype=np.float64)
        target_times = np.asarray(values["target_times"])
    if target.shape != prediction.shape or target.ndim != 3:
        raise ValueError(f"invalid test prediction shape: {path}")
    if target.shape[1:] != (HORIZON, len(TASKS)):
        raise ValueError(f"expected [N,{HORIZON},{len(TASKS)}], got {target.shape}: {path}")
    if len(target_times) != target.shape[0]:
        raise ValueError(f"target_times length mismatch: {path}")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError(f"non-finite test artifact: {path}")
    return {"target": target, "prediction": prediction, "target_times": target_times}


def _validate_freeze(freeze: Mapping[str, object]) -> None:
    comparison = freeze.get("comparison_model")
    scheme_reference = freeze.get("scheme2r_ablation_reference")
    if not isinstance(comparison, Mapping) or comparison.get("model") != "scheme2r" or comparison.get("candidate_id") != "H4":
        raise ValueError("Stage 8 requires frozen Scheme2R-H4 as the comparison model")
    if not isinstance(scheme_reference, Mapping) or scheme_reference.get("candidate_id") != "H4":
        raise ValueError("Stage 8 requires the frozen Scheme2R-H4 reference configuration")


def _load_stage8_windows(data_dir: Path) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], Dict[str, Dict[str, np.ndarray]]]:
    frame, _ = read_kitakyushu_canonical(data_dir, years=tuple(range(2015, 2022)))
    cleaned, _ = clean_kitakyushu_dataframe(frame)
    specs = {"full": KITAKYUSHU_SPLIT, "small_sample": KITAKYUSHU_SMALL_SAMPLE_SPLIT}
    windows: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    context: Dict[str, Dict[str, np.ndarray]] = {}
    timestamp = pd.to_datetime(cleaned["timestamp"])
    temperature_by_time = pd.Series(cleaned["temperature"].to_numpy(dtype=float), index=timestamp)
    for protocol, spec in specs.items():
        standardized, _ = _standardize_protocol(cleaned, spec)
        raw_test = build_protocol_windows(
            cleaned,
            spec,
            split_name="test",
            lookback=LOOKBACK,
            horizon=HORIZON,
            exog_columns=KITAKYUSHU_EXOG_COLUMNS,
            task_columns=KITAKYUSHU_TASKS,
        )
        if not np.array_equal(standardized["test"]["target_times"], raw_test["target_times"]):
            raise ValueError(f"standardized/raw test timestamp mismatch for {protocol}")
        target_times = pd.to_datetime(raw_test["target_times"])
        target_temperature = np.asarray(
            [temperature_by_time.reindex(target_times + pd.to_timedelta(step, unit="h")).to_numpy() for step in range(HORIZON)]
        ).T
        if not np.isfinite(target_temperature).all():
            raise ValueError(f"missing target temperature context for {protocol}")
        train_frame = select_training_frame(cleaned, spec)
        thresholds = temperature_thresholds(train_frame["temperature"].to_numpy(dtype=float))
        context[protocol] = {
            "target_temperature": target_temperature,
            "temperature_thresholds": thresholds,
            "season": season_labels(target_times),
            "temperature_bin": temperature_bin_labels(target_temperature, thresholds),
            "weekday": weekday_labels(target_times),
        }
        windows[protocol] = standardized
    return windows, context


def _run_records(args: argparse.Namespace, freeze: Mapping[str, object]) -> Dict[str, object]:
    roots = {
        "7.3": _resolve(args.stage7_3_dir),
        "7.4": _resolve(args.stage7_4_dir),
        "7.5": _resolve(args.stage7_5_dir),
        "7R.STL": _resolve(args.stage7_stl_dir),
    }
    records: List[Dict[str, object]] = []
    for item in build_stage7_6_source_plan(freeze):
        records.extend(
            _iter_runs(
                roots[str(item["stage"])],
                str(item["stage"]),
                str(item["source_group"]),
                int(item["expected_runs"]),
            )
        )
    if len(records) != EXPECTED_FORMAL_RUNS:
        raise ValueError(f"Stage 8 requires {EXPECTED_FORMAL_RUNS} accepted Stage 7-R runs, got {len(records)}")
    stage7_7_dir = _resolve(args.stage7_7_dir)
    stage7_7_manifest_path = stage7_7_dir / "stage7_7_manifest.json"
    if not stage7_7_manifest_path.exists():
        raise FileNotFoundError(f"missing Stage 7.7 manifest: {stage7_7_manifest_path}")
    stage7_7_manifest = _read_json(stage7_7_manifest_path)
    if (
        stage7_7_manifest.get("status") != "passed"
        or stage7_7_manifest.get("mode") != "formal"
        or int(stage7_7_manifest.get("run_count_completed", -1)) != 30
        or int(stage7_7_manifest.get("run_count_failed", -1)) != 0
        or stage7_7_manifest.get("test_used_for_selection") is not False
    ):
        raise ValueError("Stage 7.7 formal manifest is incomplete or invalid")
    stage7_7_records = _iter_runs(stage7_7_dir, "7.7", "joint_baseline_addendum", 30)

    roles = stage8_role_spec()
    joint_records: Dict[str, List[Dict[str, object]]] = {}
    source_rules = {
        "scheme2r": (records, "7.3"),
        "mmoe-lite": (records, "7.5"),
        "hard_share": (stage7_7_records, "7.7"),
        "dynamic_symmetric": (stage7_7_records, "7.7"),
        "ple-lite": (stage7_7_records, "7.7"),
    }
    for model, candidate_id in roles["joint_main_models"].items():
        source, stage = source_rules[model]
        joint_records[model] = [
            record for record in source
            if record["stage"] == stage
            and record["manifest"].get("model") == model
            and record["manifest"].get("candidate_id") == candidate_id
        ]
    validate_joint_record_matrix(joint_records)

    audit_spec = roles["selection_audit_reference"]
    audit_records = [
        record for record in records
        if record["stage"] == "7.3"
        and record["manifest"].get("model") == audit_spec["model"]
        and record["manifest"].get("candidate_id") == audit_spec["candidate_id"]
    ]
    transfer_spec = roles["matched_transfer_reference"]
    transfer_records = [
        record for record in records
        if record["stage"] == "7R.STL"
        and record["manifest"].get("model") == transfer_spec["model"]
        and record["manifest"].get("candidate_id") == transfer_spec["candidate_id"]
    ]
    if len(audit_records) != 10 or len(transfer_records) != 10:
        raise ValueError(
            "Stage 8 requires 10 STL-H3 audit records and 10 matched STL-H4 transfer records"
        )

    general_records: Dict[str, List[Dict[str, object]]] = {}
    for model, candidate_id in roles["general_baselines"].items():
        general_records[model] = [
            record for record in records
            if record["stage"] == "7.5"
            and record["manifest"].get("model") == model
            and record["manifest"].get("candidate_id") == candidate_id
        ]
        expected = 2 if candidate_id == "deterministic" else 10
        if len(general_records[model]) != expected:
            raise ValueError(
                f"Stage 8 general baseline count mismatch for {model}: "
                f"{len(general_records[model])} != {expected}"
            )
    return {
        "joint": joint_records,
        "selection_audit": audit_records,
        "matched_transfer": transfer_records,
        "general": general_records,
        "stage7_7_manifest": stage7_7_manifest,
    }


def _validate_stage7_6(stage7_6_dir: Path, resource_file: Path) -> Dict[str, object]:
    manifest_path = stage7_6_dir / "stage7_6_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Stage 7.6 manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "passed":
        raise ValueError("Stage 7.6 manifest is not passed")
    if int(manifest.get("formal_run_count", -1)) != EXPECTED_FORMAL_RUNS:
        raise ValueError("Stage 7.6 does not contain exactly 124 valid records")
    if not resource_file.exists():
        raise FileNotFoundError(f"missing resource benchmark file: {resource_file}")
    resource_rows = _read_resource_csv(resource_file)
    if len(resource_rows) != EXPECTED_FORMAL_RUNS:
        raise ValueError(f"resource benchmark count must be {EXPECTED_FORMAL_RUNS}, got {len(resource_rows)}")
    return manifest


def _pair_records(scheme: Sequence[Mapping[str, object]], stl: Sequence[Mapping[str, object]]) -> Dict[Tuple[str, int], Dict[str, Mapping[str, object]]]:
    result: Dict[Tuple[str, int], Dict[str, Mapping[str, object]]] = {}
    for label, records in (("scheme2r", scheme), ("stl_matched", stl)):
        for record in records:
            manifest = record["manifest"]
            key = (str(manifest["protocol"]), int(manifest["seed"]))
            if key in result and label in result[key]:
                raise ValueError(f"duplicate Stage 8 pair: {key}/{label}")
            result.setdefault(key, {})[label] = record
    expected = {(protocol, seed) for protocol in ("full", "small_sample") for seed in SEEDS}
    if set(result) != expected or any(set(value) != {"scheme2r", "stl_matched"} for value in result.values()):
        raise ValueError("Stage 8 does not contain a complete protocol-seed paired index")
    return result


def _index_learned_records(
    records_by_model: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, Dict[Tuple[str, int], Mapping[str, object]]]:
    indexed: Dict[str, Dict[Tuple[str, int], Mapping[str, object]]] = {}
    expected = {(protocol, seed) for protocol in ("full", "small_sample") for seed in SEEDS}
    for model, records in records_by_model.items():
        model_index: Dict[Tuple[str, int], Mapping[str, object]] = {}
        for record in records:
            manifest = record["manifest"]
            key = (str(manifest["protocol"]), int(manifest["seed"]))
            if key in model_index:
                raise ValueError(f"duplicate learned record for {model}/{key}")
            model_index[key] = record
        if set(model_index) != expected:
            raise ValueError(f"incomplete learned record index for {model}")
        indexed[model] = model_index
    return indexed


def _runtime_value(manifest: Mapping[str, object], key: str) -> float:
    runtime = manifest.get("runtime_seconds")
    if not isinstance(runtime, Mapping):
        return float("nan")
    try:
        return float(runtime.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _model_metric_rows(
    records_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    role: str,
    mape_epsilon: float,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for model, records in records_by_model.items():
        for record in records:
            manifest = record["manifest"]
            prediction = _load_prediction(Path(str(record["run_dir"])))
            metrics = equal_task_error_metrics(
                prediction["target"], prediction["prediction"], mape_epsilon=mape_epsilon
            )
            rows.append(
                {
                    "evidence_role": role,
                    "protocol": str(manifest["protocol"]),
                    "model": model,
                    "candidate_id": str(manifest.get("candidate_id", "")),
                    "seed": manifest.get("seed", "deterministic"),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "WAPE": metrics["WAPE"],
                    "MAPE": metrics["MAPE"],
                    "MAPE_valid_count": metrics["MAPE_valid_count"],
                    "MAPE_valid_fraction": metrics["MAPE_valid_fraction"],
                    "WAPE_valid_task_count": metrics["WAPE_valid_task_count"],
                    "WAPE_total_task_count": metrics["WAPE_total_task_count"],
                    "WAPE_valid_task_fraction": metrics["WAPE_valid_task_fraction"],
                    "parameter_count": int(manifest.get("parameter_count", 0) or 0),
                    "fit_seconds": _runtime_value(manifest, "fit"),
                    "test_evaluation_seconds": _runtime_value(manifest, "test_evaluation"),
                    "test_used_for_selection": False,
                }
            )
    return rows


def _aggregate_model_metric_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["evidence_role"]),
            str(row["protocol"]),
            str(row["model"]),
            str(row["candidate_id"]),
        )
        groups[key].append(row)
    output: List[Dict[str, object]] = []
    for (role, protocol, model, candidate_id), group in sorted(groups.items()):
        aggregate: Dict[str, object] = {
            "evidence_role": role,
            "protocol": protocol,
            "model": model,
            "candidate_id": candidate_id,
            "run_count": len(group),
            "test_used_for_selection": False,
        }
        for field in (
            "MAE", "RMSE", "WAPE", "MAPE", "MAPE_valid_count",
            "MAPE_valid_fraction", "WAPE_valid_task_count",
            "WAPE_total_task_count", "WAPE_valid_task_fraction",
            "parameter_count", "fit_seconds",
            "test_evaluation_seconds",
        ):
            values = np.asarray(
                [float(item[field]) for item in group if item[field] is not None and np.isfinite(float(item[field]))],
                dtype=np.float64,
            )
            aggregate[f"{field}_mean"] = float(np.mean(values)) if values.size else None
            aggregate[f"{field}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else None
        output.append(aggregate)
    return output


def _joint_task_comparison_rows(
    records_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    mape_epsilon: float,
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for model, records in records_by_model.items():
        for record in records:
            manifest = record["manifest"]
            values = _load_prediction(Path(str(record["run_dir"])))
            for task_index, task in enumerate(TASKS):
                metrics = error_metrics(
                    values["target"][:, :, task_index],
                    values["prediction"][:, :, task_index],
                    mape_epsilon=mape_epsilon,
                )
                groups[(str(manifest["protocol"]), model, str(manifest["candidate_id"]), task)].append(metrics)
    output: List[Dict[str, object]] = []
    for (protocol, model, candidate_id, task), group in sorted(groups.items()):
        row: Dict[str, object] = {
            "evidence_role": "joint_multitask_main",
            "protocol": protocol,
            "model": model,
            "candidate_id": candidate_id,
            "task": task,
            "run_count": len(group),
            "test_used_for_selection": False,
        }
        for metric in METRICS:
            values = np.asarray(
                [float(item[metric]) for item in group if item[metric] is not None and np.isfinite(float(item[metric]))],
                dtype=np.float64,
            )
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else None
            row[f"{metric}_valid_seed_count"] = int(values.size)
        row["MAPE_valid_element_fraction_mean"] = float(
            np.mean([float(item["MAPE_valid_fraction"]) for item in group])
        )
        output.append(row)
    return output


def _joint_significance_rows(
    records_by_model: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    replicates: int,
    block_size: int,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    index = _index_learned_records(records_by_model)
    rows: List[Dict[str, object]] = []
    for protocol in ("full", "small_sample"):
        scheme_items = [
            _load_prediction(Path(str(index["scheme2r"][(protocol, seed)]["run_dir"])))
            for seed in SEEDS
        ]
        for reference_model in sorted(set(records_by_model).difference({"scheme2r"})):
            reference_items = [
                _load_prediction(Path(str(index[reference_model][(protocol, seed)]["run_dir"])))
                for seed in SEEDS
            ]
            reference_errors = []
            scheme_errors = []
            for seed_position, seed in enumerate(SEEDS):
                scheme = scheme_items[seed_position]
                reference = reference_items[seed_position]
                if (
                    not np.array_equal(scheme["target_times"], reference["target_times"])
                    or not np.allclose(scheme["target"], reference["target"], atol=1e-6, rtol=0.0)
                ):
                    raise ValueError(
                        f"joint comparison targets differ for {protocol}/{reference_model}/seed_{seed}"
                    )
                target = scheme["target"]
                reference_errors.append(
                    np.mean(np.abs(target - reference["prediction"]), axis=(1, 2))
                )
                scheme_errors.append(
                    np.mean(np.abs(target - scheme["prediction"]), axis=(1, 2))
                )
            effect, low, high, p_value = paired_block_bootstrap_error_gain(
                np.stack(reference_errors),
                np.stack(scheme_errors),
                block_size=block_size,
                replicates=replicates,
                rng=rng,
            )
            rows.append(
                {
                    "protocol": protocol,
                    "reference_model": reference_model,
                    "reference_candidate_id": stage8_role_spec()["joint_main_models"][reference_model],
                    "comparison_model": "scheme2r",
                    "comparison_candidate_id": "H4",
                    "MAE_gain_pct": effect,
                    "gain_ci_low": low,
                    "gain_ci_high": high,
                    "bootstrap_p_value": p_value,
                    "bootstrap_alternative": "two_sided",
                    "seed_count": len(SEEDS),
                    "block_size_hours": block_size,
                    "test_used_for_selection": False,
                }
            )
    bh_adjust(rows, p_key="bootstrap_p_value", gain_key="MAE_gain_pct", family_fields=("protocol",))
    for row in rows:
        adjusted = float(row["fdr_adjusted_p_value"])
        effect = float(row["MAE_gain_pct"])
        row["significant_difference"] = bool(np.isfinite(adjusted) and adjusted <= 0.05)
        row["direction"] = "scheme2r_better" if effect > 0 else "scheme2r_worse" if effect < 0 else "tie"
        row.pop("significant_negative_transfer", None)
    return rows


def _base_row(protocol: str, task: str, step: int | str, granularity: str, sample_count: int) -> Dict[str, object]:
    return {
        "protocol": protocol,
        "granularity": granularity,
        "task": task,
        "horizon_step": step,
        "reference_model": "stl_matched",
        "reference_candidate_id": "H4",
        "joint_model": "scheme2r",
        "joint_candidate_id": "H4",
        "sample_count": sample_count,
        "seed_count": len(SEEDS),
        "gain_MAE_pct": float("nan"),
        "gain_ci_low": float("nan"),
        "gain_ci_high": float("nan"),
        "bootstrap_p_value": float("nan"),
        "fdr_adjusted_p_value": float("nan"),
        "significant_negative_transfer": False,
    }


def _metric_summary(
    actual: np.ndarray,
    reference: np.ndarray,
    joint: np.ndarray,
    *,
    protocol: str,
    task: str,
    step: int | str,
    granularity: str,
    mape_epsilon: float = 1e-8,
) -> Dict[str, object]:
    ref = error_metrics(actual, reference, mape_epsilon=mape_epsilon)
    joined = error_metrics(actual, joint, mape_epsilon=mape_epsilon)
    row = _base_row(protocol, task, step, granularity, int(np.asarray(actual).size))
    for metric in METRICS:
        row[f"reference_{metric}"] = ref[metric]
        row[f"joint_{metric}"] = joined[metric]
        row[f"gain_{metric}_pct"] = gain(ref[metric], joined[metric])
    row["MAPE_valid_count"] = joined["MAPE_valid_count"]
    row["MAPE_valid_fraction"] = joined["MAPE_valid_fraction"]
    return row


def _seed_rows(
    pair_index: Mapping[Tuple[str, int], Mapping[str, Mapping[str, object]]],
    *,
    protocol: str,
    task_index: int,
    target: np.ndarray,
    reference_predictions: np.ndarray,
    joint_predictions: np.ndarray,
    context: Mapping[str, np.ndarray],
    mape_epsilon: float = 1e-8,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    by_seed: List[Dict[str, object]] = []
    context_rows: List[Dict[str, object]] = []
    seasons = context["season"]
    temp_bins = context["temperature_bin"]
    weekdays = context["weekday"]
    for seed_index, seed in enumerate(SEEDS):
        actual_seed = target
        ref_seed = reference_predictions[seed_index]
        joint_seed = joint_predictions[seed_index]
        for step in range(HORIZON):
            row = _metric_summary(actual_seed[:, step, task_index], ref_seed[:, step, task_index], joint_seed[:, step, task_index], protocol=protocol, task=TASKS[task_index], step=step + 1, granularity="task_horizon_seed", mape_epsilon=mape_epsilon)
            row["seed"] = seed
            by_seed.append(row)
            for context_type, labels in (("season", seasons), ("temperature_bin", temp_bins), ("weekday_type", weekdays)):
                for label in sorted(set(labels[:, step].tolist())):
                    mask = labels[:, step] == label
                    if not np.any(mask):
                        continue
                    context_row = _metric_summary(actual_seed[mask, step, task_index], ref_seed[mask, step, task_index], joint_seed[mask, step, task_index], protocol=protocol, task=TASKS[task_index], step=step + 1, granularity="context_seed", mape_epsilon=mape_epsilon)
                    context_row.update({"seed": seed, "context_type": context_type, "context_label": label})
                    context_rows.append(context_row)
        row = _metric_summary(actual_seed[:, :, task_index].reshape(-1), ref_seed[:, :, task_index].reshape(-1), joint_seed[:, :, task_index].reshape(-1), protocol=protocol, task=TASKS[task_index], step="all", granularity="task_overall_seed", mape_epsilon=mape_epsilon)
        row["seed"] = seed
        by_seed.append(row)
    return by_seed, context_rows


def _aggregate_rows(
    seed_rows: Sequence[Mapping[str, object]],
    *,
    protocol: str,
    granularity: str,
    task: str,
    step: int | str,
    bootstrap_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray],
    bootstrap_replicates: int,
    bootstrap_block_size: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    matching = [row for row in seed_rows if row.get("protocol") == protocol and row.get("granularity") == granularity and row.get("task") == task and str(row.get("horizon_step")) == str(step)]
    if len(matching) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} seed rows for {protocol}/{granularity}/{task}/{step}, got {len(matching)}")
    actual, reference, joint = bootstrap_arrays
    row = _base_row(protocol, task, step, granularity, int(actual.shape[1]))
    for metric in METRICS:
        ref_values = np.asarray([float(item[f"reference_{metric}"]) for item in matching if item.get(f"reference_{metric}") is not None], dtype=float)
        joint_values = np.asarray([float(item[f"joint_{metric}"]) for item in matching if item.get(f"joint_{metric}") is not None], dtype=float)
        row[f"reference_{metric}_mean"] = float(np.mean(ref_values)) if ref_values.size else None
        row[f"reference_{metric}_std"] = float(np.std(ref_values, ddof=1)) if ref_values.size > 1 else (0.0 if ref_values.size else None)
        row[f"joint_{metric}_mean"] = float(np.mean(joint_values)) if joint_values.size else None
        row[f"joint_{metric}_std"] = float(np.std(joint_values, ddof=1)) if joint_values.size > 1 else (0.0 if joint_values.size else None)
        row[f"gain_{metric}_mean_pct"] = float(np.mean([float(item[f"gain_{metric}_pct"]) for item in matching if np.isfinite(float(item[f"gain_{metric}_pct"]))])) if any(np.isfinite(float(item[f"gain_{metric}_pct"])) for item in matching) else None
    observed, low, high, p_value = hierarchical_bootstrap_mae_gain(
        actual,
        reference,
        joint,
        block_size=bootstrap_block_size,
        replicates=bootstrap_replicates,
        rng=rng,
    )
    row.update({"gain_MAE_pct": observed, "gain_ci_low": low, "gain_ci_high": high, "bootstrap_p_value": p_value})
    row["negative_seed_count"] = int(sum(float(item["gain_MAE_pct"]) < 0 for item in matching if np.isfinite(float(item["gain_MAE_pct"]))))
    return row


def _gate_export(
    record: Mapping[str, object],
    windows: Mapping[str, Mapping[str, np.ndarray]],
    context: Mapping[str, np.ndarray],
    freeze: Mapping[str, object],
    output_dir: Path,
    batch_size: int,
    sample_limit: int | None = None,
) -> Tuple[Path, List[Dict[str, object]], List[Dict[str, object]]]:
    manifest = dict(record["manifest"])
    manifest["run_dir"] = str(record["run_dir"])
    model, input_mode, checkpoint_paths = _build_benchmark_model(manifest, freeze)
    from run_resource_benchmarks import _load_model
    _load_model(model, checkpoint_paths)
    model.eval()
    protocol = str(manifest["protocol"])
    seed = int(manifest["seed"])
    test = windows[protocol]["test"]
    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError("gate sample limit must be positive")
        test = {key: value[:sample_limit] for key, value in test.items()}
    local_context = {key: value[: len(test["loads"])] for key, value in context.items() if isinstance(value, np.ndarray) and value.ndim >= 1}
    rho_parts: List[np.ndarray] = []
    pi_parts: List[np.ndarray] = []
    gate_parts: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(test["loads"]), batch_size):
            stop = min(start + batch_size, len(test["loads"]))
            loads = torch.from_numpy(test["loads"][start:stop])
            exog = torch.from_numpy(test["exog"][start:stop]) if input_mode == "loads_and_exog" else None
            _, details = model.forward_with_details(loads, exog)
            rho_parts.append(details["rho"].cpu().numpy())
            pi_parts.append(details["pi"].cpu().numpy())
            gate_parts.append(details["gates"].cpu().numpy())
    rho = np.concatenate(rho_parts, axis=0).astype(np.float64)
    pi = np.concatenate(pi_parts, axis=0).astype(np.float64)
    gates = np.concatenate(gate_parts, axis=0).astype(np.float64)
    expected = (len(test["loads"]), HORIZON, len(TASKS), len(TASKS))
    if gates.shape != expected or rho.shape != expected[:-1] or pi.shape != expected:
        raise ValueError(f"unexpected Scheme2R gate shape: rho={rho.shape}, pi={pi.shape}, gates={gates.shape}, expected={expected}")
    if not np.allclose(gates, rho[..., None] * pi, atol=1e-5, rtol=1e-5):
        raise ValueError("Scheme2R gate export violates g=rho*pi")
    if not np.allclose(np.diagonal(gates, axis1=2, axis2=3), 0.0, atol=1e-6):
        raise ValueError("Scheme2R gate diagonal must be zero")
    gate_path = output_dir / f"gates_{protocol}_seed_{seed}.npz"
    np.savez_compressed(
        gate_path,
        rho=rho,
        pi=pi,
        gates=gates,
        target_times=test["target_times"],
        season=local_context["season"],
        temperature_bin=local_context["temperature_bin"],
        weekday_type=local_context["weekday"],
    )
    summary: List[Dict[str, object]] = []
    asymmetry: List[Dict[str, object]] = []
    entropy = -np.sum(np.where(pi > 0, pi * np.log(np.maximum(pi, 1e-12)), 0.0), axis=-1)
    labels_by_context = [("overall", np.ones((len(gates), HORIZON), dtype=bool))]
    for context_type, labels in (("season", local_context["season"]), ("temperature_bin", local_context["temperature_bin"]), ("weekday_type", local_context["weekday"])):
        for label in sorted(set(labels.reshape(-1).tolist())):
            labels_by_context.append((f"{context_type}:{label}", labels == label))
    for context_label, mask in labels_by_context:
        for step in range(HORIZON):
            sample_mask = mask[:, step]
            if not np.any(sample_mask):
                continue
            mean_gate = gates[sample_mask, step].mean(axis=0)
            for target_index, target_task in enumerate(TASKS):
                for source_index, source_task in enumerate(TASKS):
                    summary.append({
                        "protocol": protocol,
                        "seed": seed,
                        "context_type": "overall" if context_label == "overall" else context_label.split(":", 1)[0],
                        "context_label": "overall" if context_label == "overall" else context_label.split(":", 1)[1],
                        "horizon_step": step + 1,
                        "target_task": target_task,
                        "source_task": source_task,
                        "sample_count": int(np.sum(sample_mask)),
                        "rho_mean": float(rho[sample_mask, step, target_index].mean()),
                        "pi_mean": float(pi[sample_mask, step, target_index, source_index].mean()),
                        "gate_mean": float(mean_gate[target_index, source_index]),
                        "source_entropy_mean": float(entropy[sample_mask, step, target_index].mean()),
                    })
                    if target_index != source_index:
                        reverse = mean_gate[source_index, target_index]
                        asymmetry.append({
                            "protocol": protocol,
                            "seed": seed,
                            "context_type": "overall" if context_label == "overall" else context_label.split(":", 1)[0],
                            "context_label": "overall" if context_label == "overall" else context_label.split(":", 1)[1],
                            "horizon_step": step + 1,
                            "target_task": target_task,
                            "source_task": source_task,
                            "gate_forward": float(mean_gate[target_index, source_index]),
                            "gate_reverse": float(reverse),
                            "signed_asymmetry": float(mean_gate[target_index, source_index] - reverse),
                            "absolute_asymmetry": float(abs(mean_gate[target_index, source_index] - reverse)),
                        })
    return gate_path, summary, asymmetry


def _gate_error_rows(
    pair_index: Mapping[Tuple[str, int], Mapping[str, Mapping[str, object]]],
    gate_dir: Path,
    gate_sample_limit: int | None = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (protocol, seed), pair in sorted(pair_index.items()):
        ref = _load_prediction(Path(str(pair["stl_matched"]["run_dir"])))
        joint = _load_prediction(Path(str(pair["scheme2r"]["run_dir"])))
        gate_path = gate_dir / f"gates_{protocol}_seed_{seed}.npz"
        with np.load(gate_path, allow_pickle=False) as values:
            gates = np.asarray(values["gates"], dtype=float)
        for task_index, target_task in enumerate(TASKS):
            for step in range(HORIZON):
                sample_count = gates.shape[0]
                delta = np.abs(ref["target"][:sample_count, step, task_index] - ref["prediction"][:sample_count, step, task_index]) - np.abs(ref["target"][:sample_count, step, task_index] - joint["prediction"][:sample_count, step, task_index])
                for source_index, source_task in enumerate(TASKS):
                    if task_index == source_index:
                        continue
                    rows.append({
                        "protocol": protocol,
                        "seed": seed,
                        "horizon_step": step + 1,
                        "target_task": target_task,
                        "source_task": source_task,
                        "sample_count": len(delta),
                        "spearman_rho": spearman_correlation(gates[:, step, task_index, source_index], delta),
                        "interpretation": "association_only",
                    })
    return rows


def _summary_association(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int, str, str], List[float]] = defaultdict(list)
    for row in rows:
        value = row.get("spearman_rho")
        if value is not None and np.isfinite(float(value)):
            grouped[(str(row["protocol"]), int(row["horizon_step"]), str(row["target_task"]), str(row["source_task"]))].append(float(value))
    output: List[Dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        protocol, step, target, source = key
        output.append({
            "protocol": protocol,
            "horizon_step": step,
            "target_task": target,
            "source_task": source,
            "seed_count": len(values),
            "spearman_mean": float(np.mean(values)),
            "spearman_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "interpretation": "association_only",
        })
    return output


def _validate_tables(output_dir: Path) -> Dict[str, object]:
    required = stage8_expected_tables()
    missing = [name for name in required if not (output_dir / name).exists()]
    if missing:
        raise ValueError(f"Stage 8 missing output tables: {missing}")
    lengths = {name: int(len(pd.read_csv(output_dir / name))) for name in required}
    for path in output_dir.glob("*.csv"):
        frame = pd.read_csv(path)
        if frame.empty:
            raise ValueError(f"Stage 8 table is empty: {path.name}")
    joint = pd.read_csv(output_dir / "joint_model_comparison.csv")
    expected_joint_keys = {
        (protocol, model)
        for protocol in ("full", "small_sample")
        for model in stage8_role_spec()["joint_main_models"]
    }
    actual_joint_keys = set(zip(joint["protocol"].astype(str), joint["model"].astype(str)))
    if actual_joint_keys != expected_joint_keys or len(joint) != len(expected_joint_keys):
        raise ValueError("joint model comparison does not have exact protocol/model coverage")
    if not (joint["run_count"].astype(int) == len(SEEDS)).all():
        raise ValueError("every joint-model aggregate must contain five formal seeds")
    for field in ("MAE_mean", "RMSE_mean", "WAPE_mean"):
        values = pd.to_numeric(joint[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"joint model comparison contains non-finite {field}")
    valid_fraction = pd.to_numeric(joint["MAPE_valid_fraction_mean"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(valid_fraction).all() or np.any((valid_fraction < 0.0) | (valid_fraction > 1.0)):
        raise ValueError("joint model MAPE valid fractions are invalid")
    significance = pd.read_csv(output_dir / "joint_model_significance.csv")
    if len(significance) != 8:
        raise ValueError("joint significance table must contain four comparisons per protocol")
    per_task = pd.read_csv(output_dir / "joint_model_per_task.csv")
    if len(per_task) != 40 or not (per_task["run_count"].astype(int) == len(SEEDS)).all():
        raise ValueError("joint per-task table must contain 40 five-seed aggregates")
    for field in ("MAE_mean", "RMSE_mean"):
        values = pd.to_numeric(per_task[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"joint per-task table contains non-finite {field}")
    general = pd.read_csv(output_dir / "general_baseline_comparison.csv")
    if len(general) != 8:
        raise ValueError("general baseline table must contain four baselines per protocol")
    index = pd.read_csv(output_dir / "stage8_input_index.csv")
    if len(index) != 94:
        raise ValueError(f"Stage 8 input index must contain 94 role-labelled records, got {len(index)}")
    return {"required_table_rows": lengths}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Stage 8 test analysis")
    parser.add_argument("--freeze-config", default="frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json")
    parser.add_argument("--kitakyushu-data-dir", default="D:/Paper/Kitakyushu dataset")
    parser.add_argument("--stage7-3-dir", default="frame/reports/stage7r_3_kitakyushu_formal")
    parser.add_argument("--stage7-4-dir", default="frame/reports/stage7r_4_kitakyushu_formal")
    parser.add_argument("--stage7-5-dir", default="frame/reports/stage7r_5_kitakyushu_formal")
    parser.add_argument("--stage7-stl-dir", default="frame/reports/stage7r_stl_reference_kitakyushu_formal")
    parser.add_argument("--stage7-6-dir", default="frame/reports/stage7r_6_kitakyushu_acceptance")
    parser.add_argument("--stage7-7-dir", default="frame/reports/stage7r_7_joint_baselines_formal")
    parser.add_argument("--resource-file", default="frame/reports/stage7r_resource_benchmarks.csv")
    parser.add_argument("--output-dir", default="frame/reports/stage8r_joint_revised")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-block-size", type=int, default=24)
    parser.add_argument("--gate-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--mape-epsilon", type=float, default=1e-8)
    parser.add_argument("--gate-sample-limit", type=int, default=None, help="limit gate inference samples for smoke tests; omit for formal analysis")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates <= 0 or args.bootstrap_block_size <= 0 or args.gate_batch_size <= 0 or args.threads <= 0:
        raise ValueError("bootstrap、gate-batch-size 和 threads 必须为正整数")
    if args.mape_epsilon <= 0:
        raise ValueError("mape-epsilon must be positive")
    torch.set_num_threads(int(args.threads))
    freeze = _read_json(_resolve(args.freeze_config))
    _validate_freeze(freeze)
    stage7_6_manifest = _validate_stage7_6(_resolve(args.stage7_6_dir), _resolve(args.resource_file))
    output_dir = _resolve(args.output_dir)
    if args.dry_run:
        roles = stage8_role_spec()
        stage7_7_manifest = _resolve(args.stage7_7_dir) / "stage7_7_manifest.json"
        print(json.dumps({
            "stage": "8",
            "status": "dry_run",
            "joint_main_models": roles["joint_main_models"],
            "general_baselines": roles["general_baselines"],
            "selection_audit_reference": roles["selection_audit_reference"],
            "matched_transfer_reference": roles["matched_transfer_reference"],
            "protocols": ["full", "small_sample"],
            "seeds": list(SEEDS),
            "accepted_stage7r_runs": EXPECTED_FORMAL_RUNS,
            "expected_stage7_7_runs": 30,
            "stage7_7_formal_manifest_present": stage7_7_manifest.exists(),
            "bootstrap_replicates": args.bootstrap_replicates,
            "test_set_accessed": False,
            "test_used_for_selection": False,
            "retraining": False,
        }, ensure_ascii=False, indent=2))
        return
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output directory exists; choose a new path or pass --overwrite: {output_dir}")
    record_bundle = _run_records(args, freeze)
    joint_records = record_bundle["joint"]
    scheme_records = joint_records["scheme2r"]
    stl_records = record_bundle["matched_transfer"]
    pair_index = _pair_records(scheme_records, stl_records)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f"{output_dir.name}.tmp-", dir=output_dir.parent))
    published = False
    try:
        windows, context = _load_stage8_windows(_resolve(args.kitakyushu_data_dir))
        print(json.dumps({"stage": "8", "status": "data_loaded", "protocol_test_windows": {name: int(len(value["test"]["loads"])) for name, value in windows.items()}}, ensure_ascii=False), flush=True)
        joint_run_metrics = _model_metric_rows(
            joint_records,
            role="joint_multitask_main",
            mape_epsilon=args.mape_epsilon,
        )
        joint_comparison = _aggregate_model_metric_rows(joint_run_metrics)
        _write_csv(joint_comparison, temporary_dir / "joint_model_comparison.csv")
        joint_task_comparison = _joint_task_comparison_rows(
            joint_records, mape_epsilon=args.mape_epsilon
        )
        _write_csv(joint_task_comparison, temporary_dir / "joint_model_per_task.csv")
        joint_significance = _joint_significance_rows(
            joint_records,
            replicates=args.bootstrap_replicates,
            block_size=args.bootstrap_block_size,
            rng=np.random.default_rng(args.seed + 17),
        )
        _write_csv(joint_significance, temporary_dir / "joint_model_significance.csv")
        general_run_metrics = _model_metric_rows(
            record_bundle["general"],
            role="general_baseline_supplementary",
            mape_epsilon=args.mape_epsilon,
        )
        general_comparison = _aggregate_model_metric_rows(general_run_metrics)
        _write_csv(general_comparison, temporary_dir / "general_baseline_comparison.csv")
        reference_by_protocol: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
        joint_by_protocol: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
        per_seed_rows: List[Dict[str, object]] = []
        context_rows: List[Dict[str, object]] = []
        aggregated_overall: List[Dict[str, object]] = []
        aggregated_horizon: List[Dict[str, object]] = []
        rng = np.random.default_rng(args.seed)
        for protocol in ("full", "small_sample"):
            print(json.dumps({"stage": "8", "status": "metrics_protocol_started", "protocol": protocol}, ensure_ascii=False), flush=True)
            for seed in SEEDS:
                ref = _load_prediction(Path(str(pair_index[(protocol, seed)]["stl_matched"]["run_dir"])))
                joint = _load_prediction(Path(str(pair_index[(protocol, seed)]["scheme2r"]["run_dir"])))
                if not np.array_equal(ref["target_times"], joint["target_times"]) or not np.allclose(ref["target"], joint["target"], atol=1e-6, rtol=0.0):
                    raise ValueError(f"paired STL/Scheme2R targets differ for {protocol}/seed_{seed}")
                reference_by_protocol[protocol].append(ref)
                joint_by_protocol[protocol].append(joint)
            target = reference_by_protocol[protocol][0]["target"]
            ref_predictions = np.stack([item["prediction"] for item in reference_by_protocol[protocol]])
            joint_predictions = np.stack([item["prediction"] for item in joint_by_protocol[protocol]])
            for task_index, task in enumerate(TASKS):
                seed_task, seed_context = _seed_rows(pair_index, protocol=protocol, task_index=task_index, target=target, reference_predictions=ref_predictions, joint_predictions=joint_predictions, context=context[protocol], mape_epsilon=args.mape_epsilon)
                per_seed_rows.extend(seed_task)
                context_rows.extend(seed_context)
                actual_h = np.stack([target[:, step, task_index] for step in range(HORIZON)])
                for step in range(HORIZON):
                    actual = np.repeat(target[:, step, task_index][None, :], len(SEEDS), axis=0)
                    aggregated_horizon.append(_aggregate_rows(seed_task, protocol=protocol, granularity="task_horizon_seed", task=task, step=step + 1, bootstrap_arrays=(actual, ref_predictions[:, :, step, task_index], joint_predictions[:, :, step, task_index]), bootstrap_replicates=args.bootstrap_replicates, bootstrap_block_size=args.bootstrap_block_size, rng=rng))
                actual_all = np.repeat(target[:, :, task_index].reshape(1, -1), len(SEEDS), axis=0)
                aggregated_overall.append(_aggregate_rows(seed_task, protocol=protocol, granularity="task_overall_seed", task=task, step="all", bootstrap_arrays=(actual_all, ref_predictions[:, :, :, task_index].reshape(len(SEEDS), -1), joint_predictions[:, :, :, task_index].reshape(len(SEEDS), -1)), bootstrap_replicates=args.bootstrap_replicates, bootstrap_block_size=args.bootstrap_block_size, rng=rng))
        bh_adjust(aggregated_overall, family_fields=("protocol",))
        bh_adjust(aggregated_horizon, family_fields=("protocol",))
        _write_csv(aggregated_overall, temporary_dir / "transfer_task_overall.csv")
        _write_csv(aggregated_horizon, temporary_dir / "transfer_task_horizon.csv")
        _write_csv(context_rows, temporary_dir / "transfer_context.csv")
        rate_rows: List[Dict[str, object]] = []
        for protocol in ("full", "small_sample"):
            for granularity, rows in (("task_overall", [row for row in aggregated_overall if row["protocol"] == protocol]), ("task_horizon", [row for row in aggregated_horizon if row["protocol"] == protocol])):
                for significant in (False, True):
                    rate = negative_transfer_rate(rows, significant=significant)
                    rate.update({"protocol": protocol, "granularity": granularity})
                    rate_rows.append(rate)
        _write_csv(rate_rows, temporary_dir / "negative_transfer_rates.csv")
        threshold_rows = []
        for protocol in ("full", "small_sample"):
            for index, value in enumerate(context[protocol]["temperature_thresholds"], start=1):
                threshold_rows.append({"protocol": protocol, "threshold": f"Q{index}", "temperature": float(value), "fit_split": "training_only"})
        _write_csv(threshold_rows, temporary_dir / "temperature_bin_thresholds.csv")
        gate_rows: List[Dict[str, object]] = []
        asymmetry_rows: List[Dict[str, object]] = []
        gate_files: List[str] = []
        for protocol in ("full", "small_sample"):
            for seed in SEEDS:
                print(json.dumps({"stage": "8", "status": "gate_run_started", "protocol": protocol, "seed": seed}, ensure_ascii=False), flush=True)
                path, summary, asymmetry = _gate_export(pair_index[(protocol, seed)]["scheme2r"], windows, context[protocol], freeze, temporary_dir, args.gate_batch_size, args.gate_sample_limit)
                gate_files.append(path.name)
                gate_rows.extend(summary)
                asymmetry_rows.extend(asymmetry)
                print(json.dumps({"stage": "8", "status": "gate_run_completed", "protocol": protocol, "seed": seed}, ensure_ascii=False), flush=True)
        _write_csv(gate_rows, temporary_dir / "gate_summary.csv")
        _write_csv(asymmetry_rows, temporary_dir / "gate_asymmetry.csv")
        association_rows = _gate_error_rows(pair_index, temporary_dir, args.gate_sample_limit)
        _write_csv(_summary_association(association_rows), temporary_dir / "gate_error_association.csv")
        resource_lookup = _read_resource_csv(_resolve(args.resource_file))
        resource_rows = [dict(row) for row in resource_lookup.values()]
        for row in joint_run_metrics:
            if row["model"] in {"hard_share", "dynamic_symmetric", "ple-lite"}:
                resource_rows.append(
                    {
                        "stage": "7.7",
                        "source_group": "joint_baseline_addendum",
                        "protocol": row["protocol"],
                        "model": row["model"],
                        "candidate_id": row["candidate_id"],
                        "seed": row["seed"],
                        "parameter_count": row["parameter_count"],
                        "fit_seconds": row["fit_seconds"],
                        "test_evaluation_seconds": row["test_evaluation_seconds"],
                        "evidence_role": row["evidence_role"],
                    }
                )
        _write_csv(resource_rows, temporary_dir / "resource_comparison.csv")
        index_rows = []
        role_groups = [
            ("joint_multitask_main", joint_records),
            ("general_baseline_supplementary", record_bundle["general"]),
            ("validation_selection_audit_only", {"stl_matched": record_bundle["selection_audit"]}),
            ("matched_transfer_reference_only", {"stl_matched": record_bundle["matched_transfer"]}),
        ]
        for evidence_role, grouped in role_groups:
            for model, records in grouped.items():
                for record in records:
                    manifest_record = record["manifest"]
                    index_rows.append(
                        {
                            "evidence_role": evidence_role,
                            "protocol": manifest_record["protocol"],
                            "seed": manifest_record.get("seed", "deterministic"),
                            "model": model,
                            "candidate_id": manifest_record.get("candidate_id"),
                            "run_dir": str(record["run_dir"]),
                            "prediction_file": str(Path(str(record["run_dir"])) / "predictions_test.npz"),
                            "test_used_for_selection": False,
                        }
                    )
        _write_csv(index_rows, temporary_dir / "stage8_input_index.csv")
        figure_script = SCRIPTS_DIR / "render_stage8_figures.py"
        render = subprocess.run(
            [sys.executable, str(figure_script), "--input-dir", str(temporary_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        figure_files = sorted(
            str(path.relative_to(temporary_dir))
            for path in (temporary_dir / "figures").glob("*")
            if path.suffix.lower() in {".svg", ".pdf", ".png", ".tiff"}
        )
        table_validation = _validate_tables(temporary_dir)
        manifest = {
            "stage": "8",
            "status": "passed",
            "joint_main_models": stage8_role_spec()["joint_main_models"],
            "general_baselines": stage8_role_spec()["general_baselines"],
            "selection_audit_reference": stage8_role_spec()["selection_audit_reference"],
            "matched_transfer_reference": stage8_role_spec()["matched_transfer_reference"],
            "protocols": ["full", "small_sample"],
            "seeds": list(SEEDS),
            "accepted_stage7r_runs": EXPECTED_FORMAL_RUNS,
            "stage7_7_formal_runs": 30,
            "bootstrap": {"replicates": args.bootstrap_replicates, "block_size_hours": args.bootstrap_block_size, "seed": args.seed, "alternative": "G<0", "p_value_correction": "add_one"},
            "fdr": {"method": "Benjamini-Hochberg", "families": ["protocol/task_overall", "protocol/task_horizon"], "alpha": 0.05},
            "undefined_metric_policy": {"WAPE": "null_when_actual_denominator_is_zero", "MAPE": "nonzero_actuals_only", "mape_epsilon": args.mape_epsilon},
            "stage7_6_manifest": str(_resolve(args.stage7_6_dir) / "stage7_6_manifest.json"),
            "stage7_7_manifest": str(_resolve(args.stage7_7_dir) / "stage7_7_manifest.json"),
            "resource_source_record_count": len(resource_lookup),
            "resource_output_record_count": len(resource_rows),
            "test_set_accessed": True,
            "test_used_for_selection": False,
            "retraining": False,
            "training_started": False,
            "gate_files": gate_files,
            "figure_files": figure_files,
            "table_files": list(stage8_expected_tables()),
            "output_inventory": sorted(
                list(stage8_expected_tables())
                + figure_files
                + gate_files
                + ["stage8_manifest.json"]
            ),
            "table_validation": table_validation,
        }
        (temporary_dir / "stage8_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_dir, output_dir)
        published = True
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except subprocess.CalledProcessError as exc:
        (temporary_dir / "stage8_figure_render_failure.json").write_text(
            json.dumps({"returncode": exc.returncode, "stdout": exc.stdout, "stderr": exc.stderr}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        raise
    except Exception:
        failure = temporary_dir / "stage8_failure.json"
        failure.write_text(json.dumps({"stage": "8", "status": "failed"}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
    finally:
        if published and temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
