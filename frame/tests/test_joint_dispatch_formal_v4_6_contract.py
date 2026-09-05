from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_6_contract import load_formal_v4_6_contract


CONFIG = Path("configs/joint_forecast_dispatch_formal_v4_6.json")


def test_v46_contract_freezes_risk_semantics_and_years() -> None:
    contract = load_formal_v4_6_contract(CONFIG)
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.risk_adjustment == {
        "tasks": ["electricity", "cooling", "heating"],
        "gas_adjustment": 0.0,
        "cap_quantile": 0.90,
        "hidden_width": 64,
        "initial_output_bias": -6.0,
        "regularization_multipliers": [0.5, 1.0, 2.0],
        "risk_size_base_weight": 0.10,
        "off_risk_base_weight": 0.50,
        "j_risk_lr": 0.0005,
    }


def test_v46_contract_rejects_gas_adjustment_and_evaluation_access(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["risk_adjustment"]["gas_adjustment"] = 0.1
    payload["allow_evaluation_access_before_gate2"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="gas|evaluation"):
        load_formal_v4_6_contract(path)
