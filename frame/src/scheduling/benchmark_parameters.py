"""Freeze the transparent simulated IES parameter set for the S track."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


REQUIRED_LOAD_COLUMNS = ("electricity", "cooling", "heating")


@dataclass(frozen=True)
class SchedulingParameters:
    """Numerical capacities and assumptions used by the simulated dispatch track."""

    values: Mapping[str, float]
    training_statistics: Mapping[str, Mapping[str, float]]
    derivation_rules: Mapping[str, str]
    source_years: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "values": dict(self.values),
            "training_statistics": {k: dict(v) for k, v in self.training_statistics.items()},
            "derivation_rules": dict(self.derivation_rules),
            "source_years": list(self.source_years),
        }


def _finite_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"Missing required training column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"Training column {column} contains empty or non-finite values")
    if (values < 0).any():
        raise ValueError(f"Training load {column} contains negative values")
    return values


def _ledger_value(ledger: Mapping[str, float], key: str) -> float:
    if key not in ledger:
        raise KeyError(f"Missing ledger parameter: {key}")
    value = float(ledger[key])
    if not np.isfinite(value):
        raise ValueError(f"Ledger parameter {key} is not finite")
    return value


def derive_benchmark_parameters(
    train_frame: pd.DataFrame,
    ledger: Mapping[str, float],
    source_years: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019),
) -> SchedulingParameters:
    """Derive all S-track capacities from training-period statistics only.

    ``pv_profile`` and ``wt_profile`` are optional normalized available-power
    columns. If supplied, their capacities are scaled to the ledger's annual
    penetration targets; otherwise the returned renewable capacities remain the
    transparent normalized targets from the ledger.
    """

    loads = {name: _finite_series(train_frame, name) for name in REQUIRED_LOAD_COLUMNS}
    p95 = {name: float(np.percentile(values, 95)) for name, values in loads.items()}
    energy = {name: float(values.sum()) for name, values in loads.items()}

    values = {
        "grid_import_capacity": 1.5 * p95["electricity"],
        "chp_electric_capacity": 0.35 * p95["electricity"],
        "chp_heat_capacity": (0.45 / 0.35) * (0.35 * p95["electricity"]),
        "gas_boiler_capacity": 1.2 * p95["heating"],
        "electric_chiller_capacity": 0.6 * p95["cooling"],
        "absorption_chiller_capacity": 0.6 * p95["cooling"],
        "bess_power_capacity": 0.2 * p95["electricity"],
        "bess_energy_capacity": 4.0 * (0.2 * p95["electricity"]),
        "chp_electric_efficiency": _ledger_value(ledger, "chp_electric_efficiency"),
        "chp_heat_efficiency": _ledger_value(ledger, "chp_heat_efficiency"),
        "gas_boiler_efficiency": _ledger_value(ledger, "gas_boiler_efficiency"),
        "electric_chiller_cop": _ledger_value(ledger, "electric_chiller_cop"),
        "absorption_chiller_cop": _ledger_value(ledger, "absorption_chiller_cop"),
        "bess_roundtrip_efficiency": _ledger_value(ledger, "bess_roundtrip_efficiency"),
        "bess_throughput_cost": _ledger_value(ledger, "bess_throughput_cost"),
        "grid_energy_price": _ledger_value(ledger, "grid_energy_price"),
        "gas_energy_price": _ledger_value(ledger, "gas_energy_price"),
        "grid_emission_factor": _ledger_value(ledger, "grid_emission_factor"),
        "gas_emission_factor": _ledger_value(ledger, "gas_emission_factor"),
        "carbon_price_default": _ledger_value(ledger, "carbon_price_default"),
        "carbon_price_sensitivity": _ledger_value(ledger, "carbon_price_sensitivity"),
        "unserved_penalty": _ledger_value(ledger, "unserved_penalty"),
        "chp_ramp_fraction": _ledger_value(ledger, "chp_ramp_fraction"),
    }
    renewable_targets = {
        "pv_annual_share": _ledger_value(ledger, "pv_rated_capacity"),
        "wt_annual_share": _ledger_value(ledger, "wt_rated_capacity"),
    }
    for name, profile_key, target_key in (
        ("pv_capacity", "pv_profile", "pv_annual_share"),
        ("wt_capacity", "wt_profile", "wt_annual_share"),
    ):
        if profile_key in train_frame.columns:
            profile = pd.to_numeric(train_frame[profile_key], errors="coerce").to_numpy(dtype=np.float64)
            if profile.size != len(train_frame) or not np.isfinite(profile).all() or (profile < 0).any():
                raise ValueError(f"Invalid normalized renewable profile: {profile_key}")
            profile_energy = float(profile.sum())
            if profile_energy <= 0.0:
                raise ValueError(f"Renewable profile has no available energy: {profile_key}")
            values[name] = renewable_targets[target_key] * energy["electricity"] / profile_energy
        else:
            values[name] = renewable_targets[target_key]
    values.update(renewable_targets)

    statistics = {
        name: {
            "count": float(values_for_task.size),
            "p95": p95[name],
            "sum": energy[name],
            "mean": float(values_for_task.mean()),
        }
        for name, values_for_task in loads.items()
    }
    rules = {
        "grid_import_capacity": "1.50 * training electricity P95",
        "chp_electric_capacity": "0.35 * training electricity P95",
        "chp_heat_capacity": "(eta_chp_h / eta_chp_e) * chp electric capacity",
        "gas_boiler_capacity": "1.20 * training heating P95",
        "electric_chiller_capacity": "0.60 * training cooling P95",
        "absorption_chiller_capacity": "0.60 * training cooling P95",
        "bess_power_capacity": "0.20 * training electricity P95",
        "bess_energy_capacity": "4.0 hours * BESS power capacity",
        "pv_capacity": "target annual PV energy share * training electricity energy / normalized PV profile energy",
        "wt_capacity": "target annual WT energy share * training electricity energy / normalized WT profile energy",
    }
    return SchedulingParameters(
        values=values,
        training_statistics=statistics,
        derivation_rules=rules,
        source_years=tuple(source_years),
    )
