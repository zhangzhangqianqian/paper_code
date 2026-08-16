"""Paired protocol statistics for topology-regime sensitivity analyses.

The time axis is the first axis of every prediction array.  Task and forecast
horizon axes are deliberately kept inside each metric calculation and are
never concatenated with the time axis for resampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SUPPORTED_METRICS = ("MAE", "RMSE", "WAPE", "MAPE")


@dataclass(frozen=True)
class BootstrapEstimate:
    """Summary of one paired bootstrap contrast.

    ``estimate`` and the confidence interval use the error contrast
    (right-hand protocol/model minus left-hand protocol/model), so negative
    values indicate lower error for the right-hand quantity.  ``samples`` is
    retained for reproducibility and downstream diagnostic plots.
    """

    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    relative_change: float
    replicates: int
    block_length: int
    n_origins: int
    metric: str
    samples: np.ndarray


def circular_moving_block_indices(
    n_origins: int,
    block_length: int,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return ``[replicates, n_origins]`` circular block indices.

    A sampled replicate is assembled from consecutive blocks on a circular
    time axis.  The implementation intentionally rejects a block longer than
    the available origin sequence: silently shortening it would change the
    declared dependence scale.
    """

    n_origins = int(n_origins)
    block_length = int(block_length)
    replicates = int(replicates)
    if n_origins <= 0 or block_length <= 0 or replicates <= 0:
        raise ValueError("n_origins, block_length and replicates must be positive")
    if n_origins < block_length:
        raise ValueError("n_origins must be at least block_length")
    if not hasattr(rng, "integers"):
        raise TypeError("rng must provide an integers method")

    block_count = int(np.ceil(n_origins / block_length))
    starts = np.asarray(
        rng.integers(0, n_origins, size=(replicates, block_count)), dtype=np.int64
    )
    if starts.shape != (replicates, block_count):
        raise ValueError("rng.integers returned an unexpected shape")
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n_origins
    return indices.reshape(replicates, -1)[:, :n_origins]


def _as_prediction_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2 or array.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [n_origins, ...]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _validate_triplet(
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = _as_prediction_array("left_prediction", left_prediction)
    right = _as_prediction_array("right_prediction", right_prediction)
    actual = _as_prediction_array("target", target)
    if left.shape != right.shape or left.shape != actual.shape:
        raise ValueError("predictions and target must have identical shapes")
    return left, right, actual


def _metric_value(prediction: np.ndarray, target: np.ndarray, metric: str) -> float:
    metric = str(metric).upper()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"unsupported metric: {metric}; choose from {SUPPORTED_METRICS}")
    error = prediction - target
    axes = tuple(range(1, error.ndim))
    if metric == "MAE":
        return float(np.mean(np.abs(error)))
    if metric == "RMSE":
        return float(np.sqrt(np.mean(error**2)))
    if metric == "WAPE":
        denominator = float(np.sum(np.abs(target)))
        if denominator <= 1e-12:
            return float("nan")
        return float(np.sum(np.abs(error)) / denominator * 100.0)
    # The project-wide MAPE convention uses a finite epsilon for zero targets.
    denominator = np.maximum(np.abs(target), 1e-6)
    return float(np.mean(np.abs(error) / denominator, axis=axes).mean() * 100.0)


def _contrast_p_value(samples: np.ndarray) -> float:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan")
    lower_tail = (1.0 + float(np.sum(finite <= 0.0))) / (finite.size + 1.0)
    upper_tail = (1.0 + float(np.sum(finite >= 0.0))) / (finite.size + 1.0)
    return float(min(1.0, 2.0 * min(lower_tail, upper_tail)))


def _relative_change(left_error: float, right_error: float) -> float:
    if not np.isfinite(left_error) or abs(left_error) <= 1e-12:
        return float("nan")
    return float((right_error - left_error) / abs(left_error) * 100.0)


def _bootstrap_contrast(
    left_prediction: np.ndarray,
    right_prediction: np.ndarray,
    target: np.ndarray,
    *,
    metric: str,
    block_length: int,
    replicates: int,
    seed: int,
) -> BootstrapEstimate:
    left, right, actual = _validate_triplet(left_prediction, right_prediction, target)
    metric = str(metric).upper()
    n_origins = int(actual.shape[0])
    block_length = int(block_length)
    replicates = int(replicates)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if n_origins < block_length:
        raise ValueError("n_origins must be at least block_length")

    left_observed = _metric_value(left, actual, metric)
    right_observed = _metric_value(right, actual, metric)
    observed = float(right_observed - left_observed)
    samples = np.empty(replicates, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for replicate in range(replicates):
        indices = circular_moving_block_indices(
            n_origins, block_length, 1, rng
        )[0]
        left_value = _metric_value(left[indices], actual[indices], metric)
        right_value = _metric_value(right[indices], actual[indices], metric)
        samples[replicate] = right_value - left_value

    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        ci_low = ci_high = p_value = float("nan")
    else:
        ci_low = float(np.quantile(finite, 0.025))
        ci_high = float(np.quantile(finite, 0.975))
        p_value = _contrast_p_value(finite)
    return BootstrapEstimate(
        estimate=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        relative_change=_relative_change(left_observed, right_observed),
        replicates=replicates,
        block_length=block_length,
        n_origins=n_origins,
        metric=metric,
        samples=samples,
    )


def paired_protocol_bootstrap(
    cross_prediction: np.ndarray,
    post_prediction: np.ndarray,
    target: np.ndarray,
    *,
    metric: str,
    block_length: int,
    replicates: int,
    seed: int,
) -> BootstrapEstimate:
    """Estimate the post-minus-cross error change on aligned origins."""

    return _bootstrap_contrast(
        cross_prediction,
        post_prediction,
        target,
        metric=metric,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )


def difference_in_differences_bootstrap(
    predictions: Mapping[tuple[str, str], np.ndarray],
    target: np.ndarray,
    *,
    metric: str,
    block_length: int,
    replicates: int,
    seed: int,
) -> BootstrapEstimate:
    """Estimate the protocol change in the Scheme2R--Dynamic-Symmetric gap."""

    required = (
        ("cross_topology", "scheme2r"),
        ("cross_topology", "dynamic_symmetric"),
        ("post_ge_regular_operation", "scheme2r"),
        ("post_ge_regular_operation", "dynamic_symmetric"),
    )
    missing = [key for key in required if key not in predictions]
    if missing:
        raise ValueError(f"predictions missing required keys: {missing}")
    actual = _as_prediction_array("target", target)
    arrays = {key: _as_prediction_array(str(key), value) for key, value in predictions.items()}
    if any(value.shape != actual.shape for value in arrays.values()):
        raise ValueError("all predictions and target must have identical shapes")

    metric = str(metric).upper()
    n_origins = int(actual.shape[0])
    block_length = int(block_length)
    replicates = int(replicates)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if n_origins < block_length:
        raise ValueError("n_origins must be at least block_length")

    def gap(index: np.ndarray) -> float:
        cross_scheme = _metric_value(
            arrays[("cross_topology", "scheme2r")][index], actual[index], metric
        )
        cross_dynamic = _metric_value(
            arrays[("cross_topology", "dynamic_symmetric")][index], actual[index], metric
        )
        post_scheme = _metric_value(
            arrays[("post_ge_regular_operation", "scheme2r")][index], actual[index], metric
        )
        post_dynamic = _metric_value(
            arrays[("post_ge_regular_operation", "dynamic_symmetric")][index], actual[index], metric
        )
        return float((post_scheme - post_dynamic) - (cross_scheme - cross_dynamic))

    observed = gap(np.arange(n_origins, dtype=np.int64))
    samples = np.empty(replicates, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    for replicate in range(replicates):
        indices = circular_moving_block_indices(
            n_origins, block_length, 1, rng
        )[0]
        samples[replicate] = gap(indices)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        ci_low = ci_high = p_value = relative_change = float("nan")
    else:
        ci_low = float(np.quantile(finite, 0.025))
        ci_high = float(np.quantile(finite, 0.975))
        p_value = _contrast_p_value(finite)
        relative_change = float("nan")
    return BootstrapEstimate(
        estimate=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        relative_change=relative_change,
        replicates=replicates,
        block_length=block_length,
        n_origins=n_origins,
        metric=metric,
        samples=samples,
    )


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted p-values in the original order."""

    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if values.size == 0:
        return values.copy()
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.empty_like(ranked)
    running = 1.0
    count = ranked.size
    for position in range(count - 1, -1, -1):
        running = min(running, ranked[position] * count / (position + 1))
        adjusted_ranked[position] = running
    adjusted = np.empty_like(values)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


__all__ = [
    "BootstrapEstimate",
    "SUPPORTED_METRICS",
    "benjamini_hochberg",
    "circular_moving_block_indices",
    "difference_in_differences_bootstrap",
    "paired_protocol_bootstrap",
]
