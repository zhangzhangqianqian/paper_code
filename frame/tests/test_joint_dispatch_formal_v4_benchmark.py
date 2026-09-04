from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.joint_dispatch.formal_v4_benchmark import (
    BENCHMARK_SCHEMA,
    FORMAL_V4_TRAIN_YEARS,
    RECEIPT_SCHEMA,
    build_formal_v4_benchmark,
    write_formal_v4_benchmark_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "configs" / "standard_ies_formal_v4_rules.yaml"
LEGACY_BENCHMARK = Path(r"D:\Paper\standard_ies_benchmark_v1.yaml")

LEDGER = {
    "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45,
    "gas_boiler_efficiency": 0.90,
    "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75,
    "bess_roundtrip_efficiency": 0.90,
    "bess_throughput_cost": 1.0e-6,
    "grid_energy_price": 1.0,
    "gas_energy_price": 0.6,
    "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25,
    "carbon_price_default": 0.0,
    "carbon_price_sensitivity": 0.2,
    "unserved_penalty": 100.0,
    "chp_ramp_fraction": 0.5,
    "pv_rated_capacity": 0.15,
    "wt_rated_capacity": 0.10,
    "pv_reference_irradiance": 1000.0,
    "pv_conversion_efficiency": 0.20,
    "pv_reference_temperature": 25.0,
    "pv_temperature_coefficient": -0.004,
    "wt_cut_in_speed": 3.0,
    "wt_rated_speed": 12.0,
    "wt_cut_out_speed": 25.0,
}


def _rules() -> dict[str, object]:
    rules = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    rules["ledger_values"] = dict(LEDGER)
    return rules


def _frame() -> pd.DataFrame:
    rows = []
    for year, offset in zip(FORMAL_V4_TRAIN_YEARS, (0.0, 10.0, 20.0, 30.0)):
        for hour in (0, 12):
            rows.append(
                {
                    "timestamp": pd.Timestamp(year=year, month=1, day=1, hour=hour),
                    "electricity": 100.0 + offset + hour,
                    "cooling": 40.0 + offset / 2 + hour / 10,
                    "heating": 30.0 + offset / 3 + hour / 10,
                    "gas": 5.0 + offset / 10,
                    "temperature": 15.0,
                    "solar_irradiance": 500.0,
                    "wind_speed": 8.0,
                    "pv_profile": 0.2 + offset / 1000,
                    "wt_profile": 0.3 + offset / 1000,
                }
            )
    return pd.DataFrame(rows)


def test_benchmark_rejects_selection_year() -> None:
    bad = pd.concat(
        [_frame(), pd.DataFrame([{"timestamp": "2019-01-01", "electricity": 1, "cooling": 1, "heating": 1, "gas": 1}])],
        ignore_index=True,
    )
    with pytest.raises(PermissionError, match="2015-2018"):
        build_formal_v4_benchmark(bad, _rules(), "ledger-test")


def test_benchmark_rejects_evaluation_year() -> None:
    bad = pd.concat(
        [_frame(), pd.DataFrame([{"timestamp": "2020-01-01", "electricity": 1, "cooling": 1, "heating": 1, "gas": 1}])],
        ignore_index=True,
    )
    with pytest.raises(PermissionError, match="2015-2018"):
        build_formal_v4_benchmark(bad, _rules(), "ledger-test")


def test_benchmark_derives_capacities_from_training_p95_and_profiles() -> None:
    frame = _frame()
    benchmark = build_formal_v4_benchmark(frame, _rules(), "ledger-sha256")
    electricity = frame["electricity"].to_numpy()
    cooling = frame["cooling"].to_numpy()
    heating = frame["heating"].to_numpy()
    assert benchmark.schema_version == BENCHMARK_SCHEMA
    assert benchmark.source_years == FORMAL_V4_TRAIN_YEARS
    assert benchmark.year_counts == {str(year): 2 for year in FORMAL_V4_TRAIN_YEARS}
    assert benchmark.values["grid_import_capacity"] == pytest.approx(1.5 * np.percentile(electricity, 95))
    assert benchmark.values["electric_chiller_capacity"] == pytest.approx(0.6 * np.percentile(cooling, 95))
    assert benchmark.values["gas_boiler_capacity"] == pytest.approx(1.2 * np.percentile(heating, 95))
    assert benchmark.values["pv_capacity"] == pytest.approx(0.15 * electricity.sum() / frame["pv_profile"].sum())
    assert benchmark.values["wt_capacity"] == pytest.approx(0.10 * electricity.sum() / frame["wt_profile"].sum())
    assert benchmark.source_labels["grid_import_capacity"] == "derived"
    assert benchmark.source_labels["grid_energy_price"] == "simulated"
    assert benchmark.parameter_ledger_sha256 == "ledger-sha256"
    assert benchmark.to_payload()["test_year_used_for_scaling"] is False


def test_benchmark_requires_all_training_years_and_valid_units() -> None:
    incomplete = _frame().loc[lambda value: value["timestamp"].dt.year != 2018]
    with pytest.raises(ValueError, match="every training year"):
        build_formal_v4_benchmark(incomplete, _rules(), "ledger-test")
    rules = _rules()
    rules["source_units"] = {"electricity": "wrong"}
    benchmark = build_formal_v4_benchmark(_frame(), rules, "ledger-test")
    assert benchmark.source_units["electricity"] == "wrong"


def test_resolved_rules_contain_no_legacy_statistics() -> None:
    rules = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))
    assert "values" not in rules
    assert "training_statistics" not in rules
    assert rules["source_years"] == list(FORMAL_V4_TRAIN_YEARS)


def test_artifacts_are_atomic_and_legacy_benchmark_is_untouched(tmp_path: Path) -> None:
    before = LEGACY_BENCHMARK.read_bytes() if LEGACY_BENCHMARK.exists() else None
    receipt = write_formal_v4_benchmark_artifacts(build_formal_v4_benchmark(_frame(), _rules(), "ledger-test"), tmp_path)
    assert receipt["schema_version"] == RECEIPT_SCHEMA
    assert receipt["source_years"] == list(FORMAL_V4_TRAIN_YEARS)
    assert receipt["test_year_used_for_scaling"] is False
    if before is not None:
        assert LEGACY_BENCHMARK.read_bytes() == before
    with pytest.raises(FileExistsError, match="overwrite"):
        write_formal_v4_benchmark_artifacts(build_formal_v4_benchmark(_frame(), _rules(), "ledger-test"), tmp_path)
