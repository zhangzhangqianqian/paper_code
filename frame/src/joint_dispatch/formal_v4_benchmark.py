"""Build and persist the training-only Standard IES benchmark for formal-v4.1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from src.scheduling.renewables import pv_available, wt_available


FORMAL_V4_TRAIN_YEARS = (2015, 2016, 2017, 2018)
BENCHMARK_SCHEMA = "standard-ies-benchmark-v4.1"
RECEIPT_SCHEMA = "formal-v4.1-benchmark-receipt-v1"
REQUIRED_LEDGER_VALUES = (
    "chp_electric_efficiency", "chp_heat_efficiency", "gas_boiler_efficiency",
    "electric_chiller_cop", "absorption_chiller_cop", "bess_roundtrip_efficiency",
    "bess_throughput_cost", "grid_energy_price", "gas_energy_price",
    "grid_emission_factor", "gas_emission_factor", "carbon_price_default",
    "carbon_price_sensitivity", "unserved_penalty", "chp_ramp_fraction",
    "pv_rated_capacity", "wt_rated_capacity", "pv_reference_irradiance",
    "pv_conversion_efficiency", "pv_reference_temperature", "pv_temperature_coefficient",
    "wt_cut_in_speed", "wt_rated_speed", "wt_cut_out_speed",
)


def _finite_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValueError(f"benchmark frame is missing required column: {name}")
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"benchmark column is empty or non-finite: {name}")
    if (values < 0).any() and name in {"electricity", "cooling", "heating", "gas"}:
        raise ValueError(f"benchmark load contains negative values: {name}")
    return values


def _frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update("\x1f".join(str(column) for column in ordered.columns).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(ordered, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _ledger_values(rules: Mapping[str, Any]) -> dict[str, float]:
    raw = rules.get("ledger_values")
    if not isinstance(raw, Mapping):
        raise ValueError("rules must provide ledger_values loaded from the parameter ledger")
    values: dict[str, float] = {}
    for key in REQUIRED_LEDGER_VALUES:
        if key not in raw:
            raise ValueError(f"parameter ledger is missing {key}")
        value = float(raw[key])
        if not np.isfinite(value):
            raise ValueError(f"parameter ledger value is not finite: {key}")
        values[key] = value
    return values


def _validate_rules(rules: Mapping[str, Any]) -> None:
    if rules.get("schema_version") != "standard-ies-benchmark-rules-v4.1":
        raise ValueError("benchmark rules schema is not formal v4.1")
    if tuple(int(year) for year in rules.get("source_years", ())) != FORMAL_V4_TRAIN_YEARS:
        raise ValueError("benchmark rules source_years must equal 2015-2018")
    required = tuple(str(item) for item in rules.get("required_columns", ()))
    if "timestamp" not in required or not {"electricity", "cooling", "heating", "gas"}.issubset(required):
        raise ValueError("benchmark rules must require all four forecast tasks")
    ratios = rules.get("capacity_ratios")
    if not isinstance(ratios, Mapping):
        raise ValueError("benchmark rules capacity_ratios must be an object")
    expected = {
        "grid_import_capacity", "chp_electric_capacity", "gas_boiler_capacity",
        "electric_chiller_capacity", "absorption_chiller_capacity", "bess_power_capacity",
        "bess_energy_hours",
    }
    if set(ratios) != expected or any(float(ratios[key]) <= 0 for key in expected):
        raise ValueError("benchmark rules capacity ratios are invalid")
    shares = rules.get("renewable_energy_shares")
    if not isinstance(shares, Mapping) or set(shares) != {"pv", "wt"} or any(float(shares[key]) <= 0 for key in shares):
        raise ValueError("benchmark rules renewable energy shares are invalid")


@dataclass(frozen=True)
class FormalV4Benchmark:
    values: Mapping[str, float]
    training_statistics: Mapping[str, Mapping[str, float]]
    derivation_rules: Mapping[str, str]
    source_years: tuple[int, ...]
    year_counts: Mapping[str, int]
    source_units: Mapping[str, str]
    source_labels: Mapping[str, str]
    raw_frame_sha256: str
    parameter_ledger_sha256: str
    schema_version: str = BENCHMARK_SCHEMA

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": "kitakyushu_energy_station",
            "track": "simulated_dispatch",
            "data_origin": "transparent simulated topology calibrated only from 2015-2018 training statistics",
            "source_years": list(self.source_years),
            "year_counts": dict(self.year_counts),
            "values": dict(self.values),
            "training_statistics": {key: dict(value) for key, value in self.training_statistics.items()},
            "derivation_rules": dict(self.derivation_rules),
            "source_units": dict(self.source_units),
            "source_labels": dict(self.source_labels),
            "raw_frame_sha256": self.raw_frame_sha256,
            "parameter_ledger_sha256": self.parameter_ledger_sha256,
            "test_year_used_for_scaling": False,
        }


def build_formal_v4_benchmark(
    frame: pd.DataFrame,
    rules: Mapping[str, Any],
    ledger_hash: str,
) -> FormalV4Benchmark:
    """Resolve benchmark capacities from a strictly training-only frame.

    ``rules['ledger_values']`` is injected by the CLI after hashing and
    parsing ``scheduling_parameter_ledger_v2.csv``.  Keeping the hash as a
    separate argument makes the provenance receipt explicit and testable.
    """

    _validate_rules(rules)
    if not isinstance(ledger_hash, str) or not ledger_hash.strip():
        raise ValueError("parameter ledger hash must be non-empty")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("benchmark frame must be a non-empty DataFrame")
    if "timestamp" not in frame.columns:
        raise ValueError("benchmark frame is missing timestamp")
    timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
    if timestamps.isna().any():
        raise ValueError("benchmark frame contains invalid timestamps")
    years = timestamps.dt.year.to_numpy(dtype=np.int64)
    if np.any(~np.isin(years, FORMAL_V4_TRAIN_YEARS)):
        raise PermissionError("benchmark timestamps must be within 2015-2018")
    if set(int(year) for year in years) != set(FORMAL_V4_TRAIN_YEARS):
        raise ValueError("benchmark frame must include every training year 2015-2018")
    if timestamps.duplicated().any():
        raise ValueError("benchmark frame contains duplicate timestamps")
    working = frame.copy()
    working["timestamp"] = timestamps
    working = working.sort_values("timestamp").reset_index(drop=True)
    for name in ("electricity", "cooling", "heating", "gas"):
        _finite_column(working, name)

    ledger = _ledger_values(rules)
    ratios = {key: float(value) for key, value in rules["capacity_ratios"].items()}
    shares = {key: float(value) for key, value in rules["renewable_energy_shares"].items()}
    if abs(shares["pv"] - ledger["pv_rated_capacity"]) > 1e-12 or abs(shares["wt"] - ledger["wt_rated_capacity"]) > 1e-12:
        raise ValueError("renewable shares disagree with the parameter ledger")

    loads = {name: _finite_column(working, name) for name in ("electricity", "cooling", "heating")}
    statistics = {
        name: {
            "count": float(values.size),
            "p95": float(np.percentile(values, 95)),
            "sum": float(values.sum()),
            "mean": float(values.mean()),
        }
        for name, values in loads.items()
    }
    profile_parameters = {
        "pv_rated_capacity": float(rules["renewable_profile"]["pv_profile_rated_capacity"]),
        "pv_reference_irradiance": ledger["pv_reference_irradiance"],
        "pv_conversion_efficiency": ledger["pv_conversion_efficiency"],
        "pv_reference_temperature": ledger["pv_reference_temperature"],
        "pv_temperature_coefficient": ledger["pv_temperature_coefficient"],
        "wt_rated_capacity": float(rules["renewable_profile"]["wt_profile_rated_capacity"]),
        "wt_cut_in_speed": ledger["wt_cut_in_speed"],
        "wt_rated_speed": ledger["wt_rated_speed"],
        "wt_cut_out_speed": ledger["wt_cut_out_speed"],
    }
    if "pv_profile" in working.columns:
        pv_profile = _finite_column(working, "pv_profile")
    else:
        pv_profile = pv_available(working, profile_parameters)
    if "wt_profile" in working.columns:
        wt_profile = _finite_column(working, "wt_profile")
    else:
        wt_profile = wt_available(working, profile_parameters)
    if pv_profile.sum() <= 0 or wt_profile.sum() <= 0:
        raise ValueError("renewable profile has no available energy")

    e_p95 = statistics["electricity"]["p95"]
    c_p95 = statistics["cooling"]["p95"]
    h_p95 = statistics["heating"]["p95"]
    chp_electric = ratios["chp_electric_capacity"] * e_p95
    values = {
        "grid_import_capacity": ratios["grid_import_capacity"] * e_p95,
        "chp_electric_capacity": chp_electric,
        "chp_heat_capacity": (ledger["chp_heat_efficiency"] / ledger["chp_electric_efficiency"]) * chp_electric,
        "gas_boiler_capacity": ratios["gas_boiler_capacity"] * h_p95,
        "electric_chiller_capacity": ratios["electric_chiller_capacity"] * c_p95,
        "absorption_chiller_capacity": ratios["absorption_chiller_capacity"] * c_p95,
        "bess_power_capacity": ratios["bess_power_capacity"] * e_p95,
        "bess_energy_capacity": ratios["bess_energy_hours"] * ratios["bess_power_capacity"] * e_p95,
        "pv_capacity": shares["pv"] * statistics["electricity"]["sum"] / float(pv_profile.sum()),
        "wt_capacity": shares["wt"] * statistics["electricity"]["sum"] / float(wt_profile.sum()),
        "pv_annual_share": shares["pv"],
        "wt_annual_share": shares["wt"],
    }
    values.update({key: ledger[key] for key in REQUIRED_LEDGER_VALUES if key not in values})
    derivation_rules = {
        "grid_import_capacity": f"{ratios['grid_import_capacity']:.2f} * training electricity P95",
        "chp_electric_capacity": f"{ratios['chp_electric_capacity']:.2f} * training electricity P95",
        "chp_heat_capacity": "(eta_chp_h / eta_chp_e) * CHP electric capacity",
        "gas_boiler_capacity": f"{ratios['gas_boiler_capacity']:.2f} * training heating P95",
        "electric_chiller_capacity": f"{ratios['electric_chiller_capacity']:.2f} * training cooling P95",
        "absorption_chiller_capacity": f"{ratios['absorption_chiller_capacity']:.2f} * training cooling P95",
        "bess_power_capacity": f"{ratios['bess_power_capacity']:.2f} * training electricity P95",
        "bess_energy_capacity": f"{ratios['bess_energy_hours']:.1f} hours * BESS power capacity",
        "pv_capacity": "PV annual energy share * training electricity energy / training normalized PV profile energy",
        "wt_capacity": "WT annual energy share * training electricity energy / training normalized WT profile energy",
    }
    source_units = {str(key): str(value) for key, value in rules["source_units"].items()}
    source_labels = {
        key: "derived" if key in derivation_rules or key.endswith("_annual_share") else str(rules["parameter_labels"]["ledger_values"])
        for key in values
    }
    year_counts = {str(year): int((years == year).sum()) for year in FORMAL_V4_TRAIN_YEARS}
    return FormalV4Benchmark(
        values=values,
        training_statistics=statistics,
        derivation_rules=derivation_rules,
        source_years=FORMAL_V4_TRAIN_YEARS,
        year_counts=year_counts,
        source_units=source_units,
        source_labels=source_labels,
        raw_frame_sha256=_frame_hash(working),
        parameter_ledger_sha256=ledger_hash.strip(),
    )


def _atomic_write(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite benchmark artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_formal_v4_benchmark_artifacts(
    benchmark: FormalV4Benchmark,
    run_root: str | Path,
    *,
    source_files: Mapping[str, Any] | None = None,
    cleaning_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the resolved YAML and receipt below ``run_root/gate0/benchmark``."""

    output_dir = Path(run_root).resolve() / "gate0" / "benchmark"
    yaml_path = output_dir / "STANDARD_IES_BENCHMARK.yaml"
    receipt_path = output_dir / "FORMAL_V4_BENCHMARK_RECEIPT.json"
    yaml_bytes = yaml.safe_dump(benchmark.to_payload(), sort_keys=False, allow_unicode=True).encode("utf-8")
    _atomic_write(yaml_path, yaml_bytes)
    yaml_hash = hashlib.sha256(yaml_bytes).hexdigest()
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "complete",
        "resolved_benchmark": str(yaml_path),
        "resolved_benchmark_sha256": yaml_hash,
        "raw_frame_sha256": benchmark.raw_frame_sha256,
        "parameter_ledger_sha256": benchmark.parameter_ledger_sha256,
        "source_years": list(benchmark.source_years),
        "year_counts": dict(benchmark.year_counts),
        "test_year_used_for_scaling": False,
        "selection_year_used_for_scaling": False,
        "source_units": dict(benchmark.source_units),
        "source_labels": dict(benchmark.source_labels),
        "source_files": dict(source_files or {}),
        "cleaning_report": dict(cleaning_report or {}),
    }
    receipt_bytes = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(receipt_path, receipt_bytes)
    return receipt


__all__ = [
    "BENCHMARK_SCHEMA", "FORMAL_V4_TRAIN_YEARS", "FormalV4Benchmark",
    "RECEIPT_SCHEMA", "build_formal_v4_benchmark", "write_formal_v4_benchmark_artifacts",
]
