"""Contract tests for the end-to-end joint forecast/dispatch experiment."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.contract import (
    load_joint_training_contract,
    validate_joint_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_contract_v1.json"


def _payload() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def test_valid_contract_freezes_joint_training_boundaries() -> None:
    contract = load_joint_training_contract(CONTRACT)
    assert contract.schema_version == "joint-forecast-dispatch-v1"
    assert contract.lookback == 24
    assert contract.horizon == 4
    assert contract.task_order == ("electricity", "cooling", "heating", "gas")
    assert contract.forecast_task_weights == (1.0, 1.0, 1.0, 0.25)
    assert contract.curriculum.rollin_start_fraction == 0.4
    assert contract.curriculum.model_history_fraction == 0.5
    assert contract.selection["bootstrap_block_hours"] == 168
    assert contract.safety["online_exact_lp_calls"] == 0


def test_contract_freezes_variant_trainability() -> None:
    contract = load_joint_training_contract(CONTRACT)
    variants = {variant.name: variant for variant in contract.variants}
    assert variants["joint_from_scratch"].forecasting_trainable
    assert variants["joint_from_scratch"].scheduling_trainable
    assert variants["warm_started_joint"].initialization == "warm_start"
    assert not variants["frozen_pto"].forecasting_trainable
    assert not variants["frozen_pto"].scheduling_trainable


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(horizon=3), "horizon"),
        (lambda value: value.update(task_order=["cooling", "electricity", "heating", "gas"]), "task_order"),
        (lambda value: value.update(forecast_task_weights=[1.0, 1.0, 1.0, 1.0]), "forecast_task_weights"),
        (lambda value: value["curriculum"].update(decision_start_weight=0.0), "decision_start_weight"),
        (lambda value: value["curriculum"].update(rollin_start_fraction=0.3), "rollin_start_fraction"),
        (lambda value: value["selection"].update(bootstrap_block_hours=24), "bootstrap_block_hours"),
        (lambda value: value["safety"].update(primary_carbon_loss=True), "carbon"),
    ],
)
def test_contract_rejects_frozen_policy_mutations(mutate, match: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        validate_joint_contract(payload)


def test_contract_rejects_invalid_variant_flags() -> None:
    payload = _payload()
    payload["variants"][0]["scheduling_trainable"] = False
    with pytest.raises(ValueError, match="joint_from_scratch"):
        validate_joint_contract(payload)


def test_contract_rejects_unapproved_warm_start_prefix() -> None:
    payload = _payload()
    payload["warm_start_key_prefixes"]["forecast"].append("base.head.bad.")
    with pytest.raises(ValueError, match="warm_start_key_prefixes"):
        validate_joint_contract(payload)


def test_contract_rejects_test_based_selection_and_bad_resource_gate() -> None:
    payload = _payload()
    payload["selection"]["selection_split"] = "test"
    with pytest.raises(ValueError, match="selection"):
        validate_joint_contract(payload)

    payload = _payload()
    payload["resource_gate"]["benchmark_solves"] = 499
    with pytest.raises(ValueError, match="benchmark_solves"):
        validate_joint_contract(payload)
