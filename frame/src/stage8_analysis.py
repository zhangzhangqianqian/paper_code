"""Pure statistics and context helpers for frozen Stage 8 analysis.

This module deliberately has no model-training dependency.  It operates on
paired predictions already produced by Stage 7-R and keeps the statistical
definitions in one place so that tables, figures and tests share one source
of truth.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


TASKS = ("electricity", "cooling", "heating", "gas")
HORIZON = 4
SEASONS = ("winter", "spring", "summer", "autumn")
SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}
METRICS = ("MAE", "RMSE", "WAPE", "MAPE")
FORMAL_PROTOCOLS = ("full", "small_sample")
FORMAL_SEEDS = (2026, 2027, 2028, 2029, 2030)
JOINT_MAIN_MODELS = {
    "hard_share": "H2",
    "dynamic_symmetric": "H1",
    "mmoe-lite": "external_fixed",
    "ple-lite": "ple_lite_fixed_v1",
    "scheme2r": "H4",
}


def stage8_role_spec() -> Dict[str, object]:
    """Return the preregistered evidence roles used by Stage 8.

    The main leaderboard is deliberately limited to joint multi-task models.
    STL-H3 documents the Stage 6 validation selection only; STL-H4 is used
    solely as the structure-matched reference for transfer diagnostics.
    """

    return {
        "joint_main_models": dict(JOINT_MAIN_MODELS),
        "general_baselines": {
            "dlinear": "external_fixed",
            "softs": "external_fixed",
            "persistence": "deterministic",
            "seasonal_naive": "deterministic",
        },
        "selection_audit_reference": {
            "model": "stl_matched",
            "candidate_id": "H3",
            "role": "validation_selection_audit_only",
        },
        "matched_transfer_reference": {
            "model": "stl_matched",
            "candidate_id": "H4",
            "role": "matched_transfer_reference_only",
        },
    }


def stage8_expected_tables() -> Tuple[str, ...]:
    return (
        "joint_model_comparison.csv",
        "joint_model_per_task.csv",
        "joint_model_significance.csv",
        "general_baseline_comparison.csv",
        "transfer_task_overall.csv",
        "transfer_task_horizon.csv",
        "transfer_context.csv",
        "negative_transfer_rates.csv",
        "temperature_bin_thresholds.csv",
        "gate_summary.csv",
        "gate_asymmetry.csv",
        "gate_error_association.csv",
        "resource_comparison.csv",
        "stage8_input_index.csv",
    )


def equal_task_error_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    mape_epsilon: float = 1e-8,
) -> Dict[str, float | int | None]:
    """Compute task-wise metrics and then average tasks equally.

    Electricity, cooling, heating and gas use different physical units and
    magnitudes.  Flattening them together would let the largest-scale task
    dominate, so the formal leaderboard first computes each task separately.
    """

    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if actual.shape != prediction.shape or actual.ndim != 3 or actual.shape[2] <= 0:
        raise ValueError("actual and prediction must share [sample,horizon,task] shape")
    per_task = [
        error_metrics(actual[:, :, index], prediction[:, :, index], mape_epsilon=mape_epsilon)
        for index in range(actual.shape[2])
    ]
    result: Dict[str, float | int | None] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in per_task if row[metric] is not None and np.isfinite(float(row[metric]))]
        result[metric] = float(np.mean(values)) if values else None
    valid_count = int(sum(int(row["MAPE_valid_count"]) for row in per_task))
    total_count = int(sum(int(row["sample_count"]) for row in per_task))
    wape_valid_count = int(sum(row["WAPE"] is not None for row in per_task))
    result.update(
        {
            "MAPE_valid_count": valid_count,
            "MAPE_valid_fraction": float(valid_count / total_count) if total_count else 0.0,
            "WAPE_valid_task_count": wape_valid_count,
            "WAPE_total_task_count": int(actual.shape[2]),
            "WAPE_valid_task_fraction": float(wape_valid_count / actual.shape[2]),
            "sample_count": int(actual.shape[0]),
            "task_count": int(actual.shape[2]),
        }
    )
    return result


def validate_joint_record_matrix(
    records_by_model: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Validate exact five-model, two-protocol, five-seed coverage."""

    if set(records_by_model) != set(JOINT_MAIN_MODELS):
        raise ValueError("joint model set does not match the Stage 8 role contract")
    expected_keys = {(protocol, seed) for protocol in FORMAL_PROTOCOLS for seed in FORMAL_SEEDS}
    for model, candidate_id in JOINT_MAIN_MODELS.items():
        keys = set()
        for record in records_by_model[model]:
            manifest = record.get("manifest")
            if not isinstance(manifest, Mapping):
                raise ValueError(f"missing manifest for {model}")
            if manifest.get("model") != model or manifest.get("candidate_id") != candidate_id:
                raise ValueError(f"wrong model/candidate in Stage 8 records for {model}")
            if tuple(manifest.get("tasks", ())) != TASKS:
                raise ValueError(f"task order mismatch for {model}")
            if manifest.get("test_used_for_selection", False):
                raise ValueError(f"test-selected record is forbidden for {model}")
            key = (str(manifest.get("protocol")), int(manifest.get("seed")))
            if key in keys:
                raise ValueError(f"duplicate Stage 8 record for {model}/{key}")
            keys.add(key)
        if keys != expected_keys:
            raise ValueError(f"incomplete Stage 8 record matrix for {model}: {len(keys)} != 10")


def error_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    mape_epsilon: float = 1e-8,
) -> Dict[str, float | int | None]:
    """Return zero-safe metrics and explicit MAPE validity metadata."""

    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if actual.shape != prediction.shape or actual.size == 0:
        raise ValueError("actual and prediction must be non-empty vectors of equal shape")
    if mape_epsilon <= 0:
        raise ValueError("mape_epsilon must be positive")
    error = actual - prediction
    denominator = float(np.sum(np.abs(actual)))
    valid = np.abs(actual) > mape_epsilon
    if np.any(valid):
        mape = float(np.mean(np.abs(error[valid]) / np.abs(actual[valid])) * 100.0)
    else:
        mape = None
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "WAPE": float(np.sum(np.abs(error)) / denominator * 100.0)
        if denominator > mape_epsilon
        else None,
        "MAPE": mape,
        "MAPE_valid_count": int(np.sum(valid)),
        "MAPE_valid_fraction": float(np.mean(valid)),
        "sample_count": int(actual.size),
    }


def gain(reference_error: float | None, joint_error: float | None) -> float:
    """Positive values mean the joint model has lower error."""

    if reference_error is None or joint_error is None:
        return float("nan")
    if not np.isfinite(reference_error) or not np.isfinite(joint_error):
        return float("nan")
    if abs(float(reference_error)) <= 1e-12:
        return float("nan")
    return float((reference_error - joint_error) / reference_error * 100.0)


def block_indices(
    n: int,
    block_size: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate circular moving-block indices with shape ``[B, n]``."""

    if n <= 0 or block_size <= 0 or replicates <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    block_size = min(int(block_size), int(n))
    block_count = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(replicates, block_count))
    offsets = np.arange(block_size, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(replicates, -1)[:, :n]


def _sample_block_mean(
    values: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    indices = block_indices(len(values), block_size, 1, rng)[0]
    return float(np.mean(values[indices]))


def hierarchical_bootstrap_mae_gain(
    actual: np.ndarray,
    reference_prediction: np.ndarray,
    joint_prediction: np.ndarray,
    *,
    block_size: int,
    replicates: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Paired seed/time hierarchical bootstrap for MAE gain.

    Inputs have shape ``[seed, time]``.  Seeds are resampled with replacement;
    within each selected seed, circular 24-hour blocks are resampled.  The
    selected seeds are then averaged equally, preventing thousands of
    overlapping windows from masquerading as independent experiments.
    """

    actual = np.asarray(actual, dtype=np.float64)
    reference_prediction = np.asarray(reference_prediction, dtype=np.float64)
    joint_prediction = np.asarray(joint_prediction, dtype=np.float64)
    if actual.ndim != 2 or reference_prediction.shape != actual.shape or joint_prediction.shape != actual.shape:
        raise ValueError("bootstrap inputs must have equal [seed,time] shape")
    seed_count, sample_count = actual.shape
    if seed_count <= 0 or sample_count <= 0:
        raise ValueError("bootstrap inputs cannot be empty")
    observed_reference = float(np.mean(np.mean(np.abs(actual - reference_prediction), axis=1)))
    observed_joint = float(np.mean(np.mean(np.abs(actual - joint_prediction), axis=1)))
    observed = gain(observed_reference, observed_joint)
    values = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        selected = rng.integers(0, seed_count, size=seed_count)
        reference_errors = np.empty(seed_count, dtype=np.float64)
        joint_errors = np.empty(seed_count, dtype=np.float64)
        for position, seed_index in enumerate(selected):
            indices = block_indices(sample_count, block_size, 1, rng)[0]
            reference_errors[position] = np.mean(
                np.abs(actual[seed_index, indices] - reference_prediction[seed_index, indices])
            )
            joint_errors[position] = np.mean(
                np.abs(actual[seed_index, indices] - joint_prediction[seed_index, indices])
            )
        values[replicate] = gain(float(np.mean(reference_errors)), float(np.mean(joint_errors)))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return observed, float("nan"), float("nan"), float("nan")
    # One-sided H0: G >= 0 versus H1: G < 0, with a finite-sample correction.
    p_value = float((1.0 + np.sum(finite >= 0.0)) / (finite.size + 1.0))
    return (
        observed,
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
        p_value,
    )


def paired_block_bootstrap_error_gain(
    reference_absolute_error: np.ndarray,
    joint_absolute_error: np.ndarray,
    *,
    block_size: int,
    replicates: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Bootstrap a paired gain from pre-aggregated `[seed,time]` errors.

    Each time row may already be the equal-weight mean over tasks and forecast
    horizons.  Resampling therefore preserves chronological 24-hour blocks
    instead of flattening task or horizon axes into a false time axis.  The
    returned p value is two-sided for the null hypothesis of zero gain.
    """

    reference = np.asarray(reference_absolute_error, dtype=np.float64)
    joint = np.asarray(joint_absolute_error, dtype=np.float64)
    if reference.ndim != 2 or reference.shape != joint.shape or reference.size == 0:
        raise ValueError("paired error inputs must share non-empty [seed,time] shape")
    seed_count, sample_count = reference.shape
    observed = gain(float(np.mean(reference)), float(np.mean(joint)))
    values = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        selected = rng.integers(0, seed_count, size=seed_count)
        reference_means = np.empty(seed_count, dtype=np.float64)
        joint_means = np.empty(seed_count, dtype=np.float64)
        for position, seed_index in enumerate(selected):
            indices = block_indices(sample_count, block_size, 1, rng)[0]
            reference_means[position] = float(np.mean(reference[seed_index, indices]))
            joint_means[position] = float(np.mean(joint[seed_index, indices]))
        values[replicate] = gain(float(np.mean(reference_means)), float(np.mean(joint_means)))
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return observed, float("nan"), float("nan"), float("nan")
    lower_tail = (1.0 + float(np.sum(finite <= 0.0))) / (finite.size + 1.0)
    upper_tail = (1.0 + float(np.sum(finite >= 0.0))) / (finite.size + 1.0)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return (
        observed,
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
        float(p_value),
    )


def season_labels(target_times: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    timestamps = pd.to_datetime(np.asarray(target_times).astype("datetime64[ns]"))
    result = np.empty((len(timestamps), horizon), dtype=object)
    for step in range(horizon):
        months = (timestamps + pd.to_timedelta(step, unit="h")).month.to_numpy()
        result[:, step] = [
            next(name for name, months_for_season in SEASON_MONTHS.items() if month in months_for_season)
            for month in months
        ]
    return result


def temperature_thresholds(training_temperature: np.ndarray) -> np.ndarray:
    values = np.asarray(training_temperature, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 4:
        raise ValueError("training temperature must contain at least four finite values")
    thresholds = np.quantile(values, [0.25, 0.5, 0.75])
    if np.any(np.diff(thresholds) < 0):
        raise ValueError("temperature thresholds are not monotonic")
    return thresholds.astype(np.float64)


def temperature_bin_labels(
    target_temperature: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    values = np.asarray(target_temperature, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(3)
    return np.asarray(
        [[f"Q{int(np.searchsorted(thresholds, value, side='right')) + 1}" for value in row] for row in values],
        dtype=object,
    )


def weekday_labels(target_times: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    timestamps = pd.to_datetime(np.asarray(target_times).astype("datetime64[ns]"))
    result = np.empty((len(timestamps), horizon), dtype=object)
    for step in range(horizon):
        days = (timestamps + pd.to_timedelta(step, unit="h")).dayofweek.to_numpy()
        result[:, step] = np.where(days < 5, "weekday", "weekend")
    return result


def rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks without requiring SciPy."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return None
    rx = rank_average(x[mask])
    ry = rank_average(y[mask])
    rx_centered = rx - float(np.mean(rx))
    ry_centered = ry - float(np.mean(ry))
    denominator = float(np.sqrt(np.sum(rx_centered**2) * np.sum(ry_centered**2)))
    if denominator <= 1e-12:
        return None
    # Avoid np.corrcoef here: on some Windows CPU environments its BLAS/OpenMP
    # initialization collides with the PyTorch runtime used by gate inference.
    return float(np.sum(rx_centered * ry_centered) / denominator)


def bh_adjust(
    rows: Sequence[Dict[str, object]],
    *,
    p_key: str = "bootstrap_p_value",
    gain_key: str = "gain_MAE_pct",
    family_fields: Sequence[str] = (),
) -> None:
    """Apply BH correction independently inside declared hypothesis families."""

    families: Dict[Tuple[object, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in family_fields)
        try:
            p = float(row.get(p_key, float("nan")))
        except (TypeError, ValueError):
            p = float("nan")
        if np.isfinite(p):
            families[key].append(row)
        row["fdr_adjusted_p_value"] = float("nan")
        row["significant_negative_transfer"] = False
    for family in families.values():
        order = sorted(range(len(family)), key=lambda index: float(family[index][p_key]))
        adjusted = np.ones(len(family), dtype=np.float64)
        running = 1.0
        count = len(family)
        for rank in range(count - 1, -1, -1):
            value = min(running, float(family[order[rank]][p_key]) * count / (rank + 1))
            running = value
            adjusted[order[rank]] = value
        for row, value in zip(family, adjusted):
            row["fdr_adjusted_p_value"] = float(np.clip(value, 0.0, 1.0))
            try:
                effect = float(row[gain_key])
            except (TypeError, ValueError):
                effect = float("nan")
            row["significant_negative_transfer"] = bool(
                np.isfinite(effect) and effect < 0.0 and value <= 0.05
            )


def negative_transfer_rate(rows: Sequence[Mapping[str, object]], *, significant: bool = False) -> Dict[str, object]:
    values = []
    for row in rows:
        try:
            effect = float(row.get("gain_MAE_pct", float("nan")))
        except (TypeError, ValueError):
            effect = float("nan")
        if not np.isfinite(effect):
            continue
        if significant and not bool(row.get("significant_negative_transfer", False)):
            continue
        values.append(effect)
    denominator = len([row for row in rows if np.isfinite(float(row.get("gain_MAE_pct", float("nan"))) if row.get("gain_MAE_pct") is not None else float("nan"))])
    count = int(np.sum(np.asarray(values) < 0.0)) if values else 0
    return {
        "numerator": count,
        "denominator": denominator,
        "rate_pct": float(count / denominator * 100.0) if denominator else None,
        "significant": bool(significant),
    }
