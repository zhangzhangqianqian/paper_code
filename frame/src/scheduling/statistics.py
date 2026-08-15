"""Paired block bootstrap and multiple-comparison correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    replicates: int
    samples: np.ndarray


def _block_values(data: pd.DataFrame, block: str) -> pd.Series:
    if "value" not in data.columns:
        raise ValueError("bootstrap input must contain a 'value' column")
    if block in data.columns:
        keys = data[block]
    elif block == "day" and "timestamp" in data.columns:
        keys = pd.to_datetime(data["timestamp"]).dt.floor("D")
    else:
        raise ValueError(f"Cannot derive bootstrap block: {block}")
    values = pd.to_numeric(data["value"], errors="raise")
    return pd.DataFrame({"block": keys, "value": values}).groupby("block")["value"].mean()


def paired_block_bootstrap(
    a: pd.DataFrame,
    b: pd.DataFrame,
    block: str = "day",
    replicates: int = 2000,
    seed: int = 2026,
) -> BootstrapResult:
    """Bootstrap paired block means for two model result series."""

    if replicates <= 0:
        raise ValueError("replicates must be positive")
    left = _block_values(a, block)
    right = _block_values(b, block)
    common = left.index.intersection(right.index)
    if len(common) < 2:
        raise ValueError("At least two common blocks are required")
    differences = (left.loc[common] - right.loc[common]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(replicates, len(differences)))
    samples = differences[sampled].mean(axis=1)
    return BootstrapResult(
        estimate=float(differences.mean()),
        ci_low=float(np.quantile(samples, 0.025)),
        ci_high=float(np.quantile(samples, 0.975)),
        replicates=int(replicates),
        samples=samples,
    )


def benjamini_hochberg(p_values: Iterable[float], alpha: float = 0.05) -> Mapping[str, np.ndarray]:
    """Return BH-adjusted p-values and rejection decisions."""

    p = np.asarray(list(p_values), dtype=np.float64)
    if p.ndim != 1 or len(p) == 0 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must be finite numbers in [0,1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    adjusted = np.empty_like(p)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return {"adjusted_p_values": adjusted, "reject": adjusted <= alpha}
