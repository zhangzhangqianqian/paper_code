"""R 轨偏差结算的指标汇总。"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .real_replay import ReplayResult


def summarize_replay(result: ReplayResult) -> Mapping[str, float]:
    """返回可写入 CSV 的归一化回放指标。"""

    return {
        "grid_upward_energy": result.grid_upward_energy,
        "grid_downward_energy": result.grid_downward_energy,
        "gas_upward_energy": result.gas_upward_energy,
        "gas_downward_energy": result.gas_downward_energy,
        "grid_imbalance_cost": result.grid_imbalance_cost,
        "gas_imbalance_cost": result.gas_imbalance_cost,
        "total_imbalance_cost": result.total_imbalance_cost,
        "grid_mae": float(np.mean(np.abs(result.grid_error))),
        "gas_mae": float(np.mean(np.abs(result.gas_error))),
    }
