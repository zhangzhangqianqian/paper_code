"""阶段10.4：透明的 PV/WT 可用出力模型。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def _column(weather: pd.DataFrame | Mapping[str, object], name: str) -> np.ndarray:
    values = weather[name] if isinstance(weather, Mapping) else weather[name]
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name}必须是一维序列")
    return array


def pv_available(
    weather: pd.DataFrame | Mapping[str, object],
    parameters: Mapping[str, float],
) -> np.ndarray:
    """按辐照、温度修正和额定容量生成 PV 可用出力。"""

    irradiance = np.maximum(_column(weather, "solar_irradiance"), 0.0)
    temperature = _column(weather, "temperature")
    capacity = float(parameters["pv_rated_capacity"])
    reference_irradiance = float(parameters["pv_reference_irradiance"])
    efficiency = float(parameters["pv_conversion_efficiency"])
    reference_temperature = float(parameters["pv_reference_temperature"])
    temperature_coefficient = float(parameters["pv_temperature_coefficient"])
    if capacity < 0 or reference_irradiance <= 0 or efficiency < 0:
        raise ValueError("PV容量、参考辐照和效率必须满足非负/正值约束")
    normalized_irradiance = np.clip(irradiance / reference_irradiance, 0.0, 1.0)
    temperature_factor = np.maximum(
        0.0, 1.0 + temperature_coefficient * (temperature - reference_temperature)
    )
    output = capacity * normalized_irradiance * temperature_factor * efficiency
    return np.clip(output, 0.0, capacity)


def wt_available(
    weather: pd.DataFrame | Mapping[str, object],
    parameters: Mapping[str, float],
) -> np.ndarray:
    """按切入、额定和切出风速的分段曲线生成 WT 可用出力。"""

    wind_speed = np.maximum(_column(weather, "wind_speed"), 0.0)
    capacity = float(parameters["wt_rated_capacity"])
    cut_in = float(parameters["wt_cut_in_speed"])
    rated = float(parameters["wt_rated_speed"])
    cut_out = float(parameters["wt_cut_out_speed"])
    if capacity < 0 or not (0.0 < cut_in < rated < cut_out):
        raise ValueError("WT容量和切入/额定/切出风速顺序无效")
    output = np.zeros_like(wind_speed, dtype=np.float64)
    ramp_mask = (wind_speed >= cut_in) & (wind_speed < rated)
    plateau_mask = (wind_speed >= rated) & (wind_speed < cut_out)
    output[ramp_mask] = capacity * (
        (wind_speed[ramp_mask] ** 3 - cut_in**3) / (rated**3 - cut_in**3)
    )
    output[plateau_mask] = capacity
    return np.clip(output, 0.0, capacity)
