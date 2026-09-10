"""Causal reference forecasts used by the formal baseline adapters.

This module deliberately contains no dataset discovery or file I/O.  The
formal runner passes an already-materialized chronological task matrix and an
origin index, so the helper can only read the preceding 24-hour phase.  Keeping
the small reference implementation here avoids coupling the formal-v4
baseline registry to the older three-task utilities in :mod:`src.baselines`.
"""

from __future__ import annotations

import numpy as np


def seasonal_naive_24h(
    task_values: np.ndarray,
    origin_index: int,
    *,
    horizon: int = 4,
    task_count: int = 3,
) -> np.ndarray:
    """Return the previous-day same-phase values for one causal origin.

    ``task_values`` is a chronological ``[time, task]`` array.  At origin
    ``t`` the forecast for ``t+1:t+horizon`` is copied from
    ``t-23:t-23+horizon`` (the same four phases on the preceding day).  The
    function rejects origins without a complete 24-hour history and never
    indexes future rows.
    """

    values = np.asarray(task_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("task_values must have shape [time, task]")
    if task_count <= 0 or task_count > values.shape[1]:
        raise ValueError("task_count must be within the task dimension")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if origin_index < 24 or origin_index > values.shape[0]:
        raise ValueError("origin does not have a complete causal history")
    start = origin_index - 24
    stop = start + horizon
    # The source interval is strictly before the forecast origin when the
    # origin denotes the last observed row (the convention used by the
    # formal rolling evaluator).
    if stop > origin_index:
        raise ValueError("seasonal reference would read at or after the origin")
    return values[start:stop, :task_count].copy()


__all__ = ["seasonal_naive_24h"]
