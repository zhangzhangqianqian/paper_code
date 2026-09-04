from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.build_rsc_pf_formal_v4_data import _resolve_benchmark_path
from scripts.run_rsc_pf_formal_v4_capacity_audit import build_capacity_freeze_payload
from src.joint_dispatch.formal_protocol_v4 import load_formal_v4_spec
from src.joint_dispatch.formal_v4_capacity import CapacityOriginManifest


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "configs" / "joint_forecast_dispatch_formal_v4.json"
V41 = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_1.json"


def _manifest() -> CapacityOriginManifest:
    indices = np.arange(24, 524, dtype=np.int64)
    timestamps = np.arange(indices.size, dtype="timedelta64[h]") + np.datetime64("2015-01-01T00:00")
    summaries = {
        "electricity_sum": np.ones(indices.size),
        "cooling_sum": np.ones(indices.size),
        "heating_sum": np.ones(indices.size),
        "gas_sum": np.ones(indices.size),
        "cooling_max": np.ones(indices.size),
        "weekday": np.zeros(indices.size),
        "hour_bin": np.zeros(indices.size),
    }
    return CapacityOriginManifest(
        origin_indices=indices,
        origin_timestamps=timestamps,
        strata=np.full(indices.size, "uniform_remaining"),
        demand_summaries=summaries,
        source_base_sha256="a" * 64,
        selection_config={"total": 500, "year_season": 320, "cooling_top_decile": 60, "heating_top_decile": 40, "electricity_top_decile": 40, "uniform_remaining": 40},
    )


def _audit() -> SimpleNamespace:
    diagnostic = {"multiplier": 2.7, "meets_threshold": True, "cooling_shortage_energy_ratio": 0.004, "cooling_shortage_hour_rate": 0.005}
    chronological = {"multiplier": 2.7, "meets_threshold": True, "cooling_shortage_energy_ratio": 0.001, "cooling_shortage_hour_rate": 0.001}
    return SimpleNamespace(
        status="pass",
        selected={"multiplier": 2.7, "meets_threshold": True, "diagnostic_row": diagnostic, **chronological},
        stage_one=(diagnostic,),
        stage_two=(chronological,),
        origin_manifest_sha256="b" * 64,
        source_base_sha256="c" * 64,
        resolved_parameter_sha256="d" * 64,
        diagnostic_initial_state={"initial_soc": 0.5, "previous_chp": 0.0},
        full_state_reset_count=2,
        full_timestamp_start="2015-01-01T00:00:00",
        full_timestamp_end="2018-12-31T23:00:00",
        solver_identity="solver",
        candidate_multipliers=(2.7,),
        thresholds={"cooling_shortage_energy_ratio_max": 0.005, "cooling_shortage_hour_rate_max": 0.01},
        capacity_scenario_hash="e" * 64,
    )


def test_v41_benchmark_path_is_run_root_bound(tmp_path: Path) -> None:
    spec = load_formal_v4_spec(V41)
    benchmark = tmp_path / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    benchmark.parent.mkdir(parents=True)
    benchmark.write_text(yaml.safe_dump({"schema_version": "standard-ies-benchmark-v4.1", "source_years": [2015, 2016, 2017, 2018]}), encoding="utf-8")
    assert _resolve_benchmark_path(spec, run_root=tmp_path, benchmark_path=None) == benchmark.resolve()


def test_legacy_v4_benchmark_default_remains_unchanged() -> None:
    spec = load_formal_v4_spec(V4)
    resolved = _resolve_benchmark_path(spec, run_root=None, benchmark_path=None)
    assert resolved.name == "standard_ies_benchmark_v1.yaml"


def test_capacity_freeze_payload_contains_two_stage_evidence() -> None:
    payload = build_capacity_freeze_payload(
        _audit(),
        {"bess_energy_capacity": 1029.6},
        _manifest(),
        run_id="run",
        benchmark_receipt_sha256="f" * 64,
    )
    assert payload["gate0_authorized"] is True
    audit = payload["capacity_audit"]
    assert audit["status"] == "pass"
    assert audit["diagnostic"]["status"] == "pass"
    assert audit["chronological"]["status"] == "pass"
    assert audit["selected"]["multiplier"] == pytest.approx(2.7)
    assert audit["candidates"][0]["meets_threshold"] is True


def test_capacity_freeze_payload_rejects_unselected_audit() -> None:
    audit = _audit()
    audit.status = "fail"
    audit.selected = None
    with pytest.raises(ValueError, match="without a passing selection"):
        build_capacity_freeze_payload(audit, {"bess_energy_capacity": 1.0}, _manifest(), run_id="run", benchmark_receipt_sha256="f" * 64)
