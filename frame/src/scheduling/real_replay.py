"""阶段10.6：真实能源站购能申报与偏差回放。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class EnergyNomination:
    grid: np.ndarray
    gas: np.ndarray
    origin_times: np.ndarray | None = None


@dataclass(frozen=True)
class ReplayResult:
    grid_error: np.ndarray
    gas_error: np.ndarray
    grid_upward_energy: float
    grid_downward_energy: float
    gas_upward_energy: float
    gas_downward_energy: float
    grid_imbalance_cost: float
    gas_imbalance_cost: float
    total_imbalance_cost: float


def build_energy_nomination(
    forecasts: np.ndarray,
    pv_forecast: np.ndarray,
    origin_times: np.ndarray | None = None,
) -> EnergyNomination:
    """将四任务预测转为 R 轨购电、购气申报。"""

    load_forecast = np.asarray(forecasts, dtype=np.float64)
    pv = np.asarray(pv_forecast, dtype=np.float64)
    if load_forecast.ndim != 3 or tuple(load_forecast.shape[1:]) != (4, 4):
        raise ValueError("forecasts必须为[N,4,4]，任务顺序为e/c/h/g")
    if pv.ndim != 3 or tuple(pv.shape[1:]) != (4, 2):
        raise ValueError("pv_forecast必须为[N,4,2]，第二列为WT预测")
    if load_forecast.shape[0] != pv.shape[0]:
        raise ValueError("负荷与PV/WT预测样本数不一致")
    if origin_times is not None and len(origin_times) != load_forecast.shape[0]:
        raise ValueError("origin_times数量与预测样本数不一致")
    electric = np.maximum(load_forecast[:, :, 0], 0.0)
    gas = np.maximum(load_forecast[:, :, 3], 0.0)
    pv_available = np.maximum(pv[:, :, 0], 0.0)
    grid = np.maximum(electric - pv_available, 0.0)
    return EnergyNomination(grid=grid, gas=gas, origin_times=origin_times)


def settle_real_replay(
    nomination: EnergyNomination,
    actual: Mapping[str, np.ndarray],
    prices: Mapping[str, float],
) -> ReplayResult:
    """按非对称上/下调价格结算真实购能偏差。"""

    actual_grid = np.asarray(actual["actual_grid_import"], dtype=np.float64)
    actual_gas = np.asarray(actual["gas"], dtype=np.float64)
    required_price_keys = ("grid_upward", "grid_downward", "gas_upward", "gas_downward")
    missing_prices = [key for key in required_price_keys if key not in prices]
    if missing_prices:
        raise ValueError(f"Missing settlement prices: {missing_prices}")
    if any(float(prices[key]) < 0.0 or not np.isfinite(float(prices[key])) for key in required_price_keys):
        raise ValueError("Settlement prices must be finite and non-negative")
    if actual_grid.shape != nomination.grid.shape or actual_gas.shape != nomination.gas.shape:
        raise ValueError("实际购能序列与申报形状不一致")
    grid_error = actual_grid - nomination.grid
    gas_error = actual_gas - nomination.gas
    grid_up = np.maximum(grid_error, 0.0)
    grid_down = np.maximum(-grid_error, 0.0)
    gas_up = np.maximum(gas_error, 0.0)
    gas_down = np.maximum(-gas_error, 0.0)
    grid_upward_energy = float(grid_up.sum())
    grid_downward_energy = float(grid_down.sum())
    gas_upward_energy = float(gas_up.sum())
    gas_downward_energy = float(gas_down.sum())
    grid_cost = grid_upward_energy * float(prices["grid_upward"]) + grid_downward_energy * float(prices["grid_downward"])
    gas_cost = gas_upward_energy * float(prices["gas_upward"]) + gas_downward_energy * float(prices["gas_downward"])
    return ReplayResult(
        grid_error=grid_error,
        gas_error=gas_error,
        grid_upward_energy=grid_upward_energy,
        grid_downward_energy=grid_downward_energy,
        gas_upward_energy=gas_upward_energy,
        gas_downward_energy=gas_downward_energy,
        grid_imbalance_cost=float(grid_cost),
        gas_imbalance_cost=float(gas_cost),
        total_imbalance_cost=float(grid_cost + gas_cost),
    )


def settle_first_step_replay(
    nomination: EnergyNomination,
    actual: Mapping[str, np.ndarray],
    prices: Mapping[str, float],
) -> ReplayResult:
    """Settle only the executed first step of each rolling forecast window.

    The full four-step result remains available through :func:`settle_real_replay`
    for window-level forecast diagnostics. This helper is the only settlement
    interface used for executed-energy and imbalance-cost accounting, so
    overlapping rolling windows cannot charge steps 2--4 repeatedly.
    """

    grid = np.asarray(nomination.grid, dtype=np.float64)
    gas = np.asarray(nomination.gas, dtype=np.float64)
    actual_grid = np.asarray(actual["actual_grid_import"], dtype=np.float64)
    actual_gas = np.asarray(actual["gas"], dtype=np.float64)
    if grid.ndim != 2 or gas.ndim != 2 or grid.shape[1] < 1 or gas.shape[1] < 1:
        raise ValueError("rolling nominations must have at least one forecast step")
    if actual_grid.ndim != 2 or actual_gas.ndim != 2:
        raise ValueError("rolling actuals must have shape [N,H]")
    if actual_grid.shape[1] < 1 or actual_gas.shape[1] < 1:
        raise ValueError("rolling actuals must have at least one forecast step")
    return settle_real_replay(
        EnergyNomination(grid=grid[:, :1], gas=gas[:, :1], origin_times=nomination.origin_times),
        {
            "actual_grid_import": actual_grid[:, :1],
            "gas": actual_gas[:, :1],
        },
        prices,
    )
