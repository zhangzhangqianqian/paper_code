from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_2_contract import (
    FormalV42Contract,
    assert_gate_transition,
    load_formal_v4_2_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json"
INVALID_REGISTRY = ROOT / "configs" / "joint_dispatch_invalid_runs_v4.json"


def test_v42_contract_freezes_information_and_budget_boundaries() -> None:
    contract = load_formal_v4_2_contract(CONFIG)
    assert isinstance(contract, FormalV42Contract)
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.excluded_years == (2021,)
    assert contract.lookback == 24
    assert contract.horizon == 4
    assert contract.gate2_seeds == (2026, 2027, 2028)
    assert contract.seed_extension_seeds == (2029, 2030)
    assert contract.gate3_seeds == (2026, 2027, 2028, 2029, 2030)
    assert contract.gas_semantics == "station_side_auxiliary_prior"
    assert contract.training["stage_order"] == ["P", "teacher", "S", "clone", "J"]
    assert contract.pilot["seed"] == 2026
    assert contract.pilot["windows"] == 128
    assert contract.pilot["rollout_windows"] == 24
    assert contract.latent_control_dim == 15
    assert len(contract.dispatch_order) == 21
    assert contract.allow_future_binary_decisions is False
    assert len(contract.methods) == 9


def test_v42_gate_order_is_fail_closed() -> None:
    contract = load_formal_v4_2_contract(CONFIG)
    assert_gate_transition(contract, "gate0", "pilot")
    assert_gate_transition(contract, "pilot", "gate1")
    assert_gate_transition(contract, "gate1", "gate2")
    assert_gate_transition(contract, "gate2_audit", "seed_extension")
    assert_gate_transition(contract, "seed_extension", "gate3")
    with pytest.raises(ValueError, match="invalid formal-v4.2 gate transition"):
        assert_gate_transition(contract, "gate0", "gate2")


def test_invalid_v41_gate2_is_diagnostic_only() -> None:
    registry = json.loads(INVALID_REGISTRY.read_text(encoding="utf-8"))
    row = next(
        item
        for item in registry["runs"]
        if item["run_root"] == "formal_v4_1_gate2_20260904_104000"
    )
    assert row["status"] == "invalid_diagnostic"
    assert row["allowed_for_formal_results"] is False
    assert {
        "incomplete_method_matrix",
        "realized_future_teacher",
        "open_loop_overlap_evaluation",
        "nonpersistent_optimizer",
        "missing_checkpoint_lineage",
    } <= set(row["reasons"])


def test_contract_hash_changes_when_contract_bytes_change(tmp_path: Path) -> None:
    original = load_formal_v4_2_contract(CONFIG)
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    revised = load_formal_v4_2_contract(changed)
    assert original.contract_sha256 != revised.contract_sha256
