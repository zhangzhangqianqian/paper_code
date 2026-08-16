"""Predeclared paired inference for the scheduling evidence tables."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .statistics import benjamini_hochberg


def paired_day_values(
    daily: pd.DataFrame,
    reference_model: str,
    comparator: str,
    metric: str,
    track: str,
    scenario: str,
) -> np.ndarray:
    subset = daily[(daily["track"] == track) & (daily["scenario"] == scenario) & daily["model"].isin([reference_model, comparator])]
    # Calendar-day inference excludes the final partial day (2021-12-31 has
    # 21 hourly origins); hourly descriptive tables retain it separately.
    if "hour_count" in subset.columns:
        subset = subset[subset["hour_count"] == 24]
    if metric not in subset.columns:
        raise ValueError(f"missing metric: {metric}")
    by_day = subset.groupby(["model", "date"], as_index=False)[metric].mean()
    left = by_day[by_day["model"] == reference_model].set_index("date")[metric]
    right = by_day[by_day["model"] == comparator].set_index("date")[metric]
    if set(left.index) != set(right.index):
        raise ValueError("paired comparison requires identical complete calendar-day sets")
    if len(left) < 2:
        raise ValueError("at least two calendar days are required")
    return (left.sort_index() - right.sort_index()).to_numpy(dtype=float)


def _bootstrap(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[sampled].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_flip_p_value(values: Iterable[float], replicates: int = 2000, seed: int = 2026) -> float:
    differences = np.asarray(list(values), dtype=float)
    if len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("sign-flip test requires at least two finite paired differences")
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(replicates, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((1.0 + np.count_nonzero(null >= observed)) / (replicates + 1.0))


def compare_models(
    daily: pd.DataFrame,
    reference_model: str,
    comparator: str,
    track: str,
    scenario: str,
    metric: str,
    replicates: int = 2000,
    seed: int = 2026,
) -> dict[str, object]:
    differences = paired_day_values(daily, reference_model, comparator, metric, track, scenario)
    subset = daily[(daily["track"] == track) & (daily["scenario"] == scenario) & daily["model"].isin([reference_model, comparator])]
    if "hour_count" in subset.columns:
        subset = subset[subset["hour_count"] == 24]
    means = subset.groupby("model")[metric].mean()
    ci_low, ci_high = _bootstrap(differences, replicates, seed)
    p_value = sign_flip_p_value(differences, replicates, seed)
    comparator_mean = float(means[comparator])
    return {
        "reference_model": reference_model,
        "comparator": comparator,
        "track": track,
        "scenario": scenario,
        "metric": metric,
        "reference_mean": float(means[reference_model]),
        "comparator_mean": comparator_mean,
        "paired_difference": float(differences.mean()),
        "relative_difference_percent": float(100.0 * differences.mean() / comparator_mean) if abs(comparator_mean) > 1e-12 else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "adjusted_p_value": np.nan,
        "significant": False,
        "preferred_model": reference_model if differences.mean() < 0 else comparator,
        "n_days": int(len(differences)),
        "n_seeds": int(subset["seed"].nunique()),
        "metric_direction": "lower_is_better",
    }


def apply_bh(results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    if results.empty:
        return results.copy()
    output = results.copy()
    adjusted = benjamini_hochberg(output["p_value"].to_numpy(), alpha=alpha)
    output["adjusted_p_value"] = adjusted["adjusted_p_values"]
    output["significant"] = adjusted["reject"]
    return output
