from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_4_contract import FormalV44Contract, load_formal_v4_4_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_4.json"


def test_v44_contract_freezes_scientific_boundaries() -> None:
    contract = load_formal_v4_4_contract(CONTRACT)
    assert isinstance(contract, FormalV44Contract)
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.excluded_years == (2021,)
    assert contract.forecast_tasks == ("electricity", "cooling", "heating", "gas")
    assert contract.rigid_demands == ("electricity", "cooling", "heating")
    assert contract.control_dim == 15 and contract.dispatch_dim == 21
    assert contract.pilot_train_windows == 4096
    assert len(contract.candidate_grid) == 4


def test_v44_contract_rejects_calendar_mask(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["thermal_regime"]["hard_calendar_mask"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="calendar"):
        load_formal_v4_4_contract(path)


def test_v44_contract_rejects_gas_as_rigid_demand(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["rigid_demand_order"] = ["electricity", "cooling", "heating", "gas"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="gas"):
        load_formal_v4_4_contract(path)
