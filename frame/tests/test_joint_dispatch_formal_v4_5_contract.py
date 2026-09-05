from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_5_contract import FormalV45Contract, load_formal_v4_5_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_5.json"


def test_v45_contract_freezes_scientific_boundaries() -> None:
    contract = load_formal_v4_5_contract(CONTRACT)
    assert isinstance(contract, FormalV45Contract)
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.forecast_tasks == ("electricity", "cooling", "heating", "gas")
    assert contract.rigid_demands == ("electricity", "cooling", "heating")
    assert contract.control_dim == 15 and contract.dispatch_dim == 21
    assert contract.joint_curriculum["decision_start"] == 0.05
    assert contract.forecast_guardrails["max_macro_f1_drop"] == 0.02


def test_v45_contract_rejects_threshold_change(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["pilot_thresholds"]["maximum_physical_residual"] = 1.0
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        load_formal_v4_5_contract(path)


def test_v45_contract_rejects_future_evaluation_access(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["allow_evaluation_access_before_gate2"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation"):
        load_formal_v4_5_contract(path)
