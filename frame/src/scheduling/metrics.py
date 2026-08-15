"""Stage 10.10: auditable summaries for real replay and simulated dispatch."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def summarize_real_replay(results: Iterable[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    """Aggregate R-track window results without mixing it with S-track rows."""

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(list(results))
    if frame.empty:
        return frame
    group_columns = [column for column in ("model", "candidate", "seed", "protocol") if column in frame.columns]
    value_columns = [column for column in ("grid_mae_window", "gas_mae_window") if column in frame.columns]
    if not group_columns or not value_columns:
        raise ValueError("R-track results require model identifiers and MAE columns")
    return frame.groupby(group_columns, as_index=False)[value_columns].mean()


def summarize_simulated_dispatch(results: Iterable[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    """Aggregate S-track plans and realized outcomes by model/scenario."""

    frame = results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame(list(results))
    if frame.empty:
        return frame
    group_columns = [column for column in ("model", "scenario", "seed", "protocol") if column in frame.columns]
    if not group_columns:
        raise ValueError("S-track results require at least one model/scenario identifier")
    value_columns = [
        column
        for column in (
            "planned_cost", "realized_cost", "regret", "grid_upward", "grid_downward",
            "gas_upward", "gas_downward", "unserved_electricity", "unserved_cooling",
            "unserved_heating", "carbon", "curtailment", "solver_time_seconds",
        )
        if column in frame.columns
    ]
    if not value_columns:
        raise ValueError("S-track results contain no recognized numeric metrics")
    return frame.groupby(group_columns, as_index=False)[value_columns].agg(["mean", "std"]).reset_index()
