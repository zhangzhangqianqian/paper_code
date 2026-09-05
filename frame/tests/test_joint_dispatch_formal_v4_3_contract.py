from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_3_contract import (
    FormalV43Contract,
    assert_gate_transition,
    load_formal_v4_3_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_3.json"


def test_v43_contract_freezes_data_and_output_boundaries() -> None:
    contract = load_formal_v4_3_contract(CONFIG)
    assert isinstance(contract, FormalV43Contract)
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.excluded_years == (2021,)
    assert contract.task_order == ("electricity", "cooling", "heating", "gas")
    assert contract.rigid_demands == ("electricity", "cooling", "heating")
    assert contract.latent_control_dim == 15
    assert len(contract.dispatch_order) == 21
    assert contract.gate1_candidate_grid() == tuple(
        (left, right) for left in (0.5, 1.0, 2.0) for right in (0.25, 0.5, 1.0)
    )


def test_v43_gate_order_is_fail_closed() -> None:
    contract = load_formal_v4_3_contract(CONFIG)
    assert_gate_transition(contract, "gate0", "pilot")
    assert_gate_transition(contract, "pilot", "gate1")
    assert_gate_transition(contract, "gate1", "gate2")
    with pytest.raises(ValueError, match="invalid formal-v4.3 gate transition"):
        assert_gate_transition(contract, "gate0", "gate2")


def test_v43_contract_rejects_hard_calendar_mask(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["thermal_regime"]["hard_calendar_mask"] = True
    changed = tmp_path / "bad.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hard calendar mask"):
        load_formal_v4_3_contract(changed)


def test_v43_contract_hash_changes_when_contract_bytes_change(tmp_path: Path) -> None:
    original = load_formal_v4_3_contract(CONFIG)
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    revised = load_formal_v4_3_contract(changed)
    assert original.contract_sha256 != revised.contract_sha256
