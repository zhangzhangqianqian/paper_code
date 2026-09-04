from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.joint_dispatch.formal_v4_2_contract import FormalV42Contract
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_capacity_audit.py"
SPEC = importlib.util.spec_from_file_location("formal_v42_capacity_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _contract() -> FormalV42Contract:
    return FormalV42Contract(
        {
            "train_years": [2015, 2016, 2017, 2018],
            "selection_year": 2019,
            "capacity": {
                "candidate_multipliers": [1.0],
                "cooling_shortage_energy_ratio_max": 0.005,
                "cooling_shortage_hour_rate_max": 0.01,
            },
        },
        "a" * 64,
        Path("formal_v4_2_test.json"),
    )


def _base(split: str) -> FormalV4BaseSeries:
    if split == "train":
        timestamps = pd.date_range("2015-01-01", "2018-12-31 23:00", freq="h").to_numpy(dtype="datetime64[ns]")
    else:
        timestamps = pd.date_range("2019-01-01", periods=72, freq="h").to_numpy(dtype="datetime64[ns]")
    n = len(timestamps)
    hour = np.arange(n)
    months = pd.DatetimeIndex(timestamps).month.to_numpy()
    electricity = 100.0 + 10.0 * np.sin(2.0 * np.pi * hour / 24.0)
    cooling = np.where(np.isin(months, [5, 6, 7, 8, 9]), 60.0, 5.0)
    heating = np.where(np.isin(months, [12, 1, 2]), 50.0, 10.0)
    gas = np.full(n, 5.0)
    load_and_exog = np.column_stack((electricity, cooling, heating, gas, np.zeros((n, 12))))
    renewable = np.full((n, 2), 0.5)
    prices = np.ones((n, 3))
    return FormalV4BaseSeries(load_and_exog, renewable, renewable, prices, timestamps, split)


class _FakeResult:
    success = True
    message = "optimal"

    def __init__(self, shortage: float) -> None:
        self.values = {
            "slack_c": np.full(4, shortage),
            "soc": np.full(4, 5.0),
            "p_chp": np.full(4, 0.1),
        }


class _FakeSolver:
    def __init__(self, shortage: float = 0.0) -> None:
        self.shortage = shortage

    def __call__(self, inputs):
        return _FakeResult(self.shortage)


PARAMETERS = {
    "electric_chiller_capacity": 10.0,
    "absorption_chiller_capacity": 10.0,
    "bess_energy_capacity": 10.0,
}


def test_capacity_year_mapping_denies_evaluation():
    contract = _contract()
    assert MODULE.capacity_years_for_split(contract, "train") == (2015, 2016, 2017, 2018)
    assert MODULE.capacity_years_for_split(contract, "selection") == (2019,)
    with pytest.raises(MODULE.CapacityEvidenceError, match="forbidden"):
        MODULE.capacity_years_for_split(contract, "evaluation")


def test_capacity_builder_uses_train_only_and_writes_v42_freeze(tmp_path):
    receipt = MODULE.build_capacity_evidence_from_bases(
        _base("train"),
        _base("selection"),
        PARAMETERS,
        _contract(),
        tmp_path / "formal_v4_2_test",
        "b" * 64,
        solver=_FakeSolver(),
    )
    assert receipt["status"] == "pass"
    assert receipt["fit_years"] == [2015, 2016, 2017, 2018]
    assert receipt["selection_years_materialized"] == [2019]
    assert receipt["selection_influenced_capacity"] is False
    assert receipt["evaluation_year_accessed"] is False
    assert receipt["selected"]["multiplier"] == 1.0


def test_capacity_freeze_is_not_written_when_audit_fails(tmp_path):
    root = tmp_path / "formal_v4_2_test_fail"
    with pytest.raises(MODULE.CapacityEvidenceError, match="diagnostic"):
        MODULE.build_capacity_evidence_from_bases(
            _base("train"),
            _base("selection"),
            PARAMETERS,
            _contract(),
            root,
            "b" * 64,
            solver=_FakeSolver(shortage=1.0),
        )
    assert (root / "gate0" / "CAPACITY_AUDIT_RESULT.json").is_file()
    assert not (root / "gate0" / "CAPACITY_FREEZE.json").exists()
