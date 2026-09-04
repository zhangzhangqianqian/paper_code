from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_protocol_v4 import (
    FORMAL_V4_METHODS,
    SEEDS,
    build_formal_v4_run_matrix,
    load_formal_v4_spec,
    validate_formal_v4_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_formal_v4.json"


def test_v4_method_matrix_and_test_boundary() -> None:
    spec = load_formal_v4_spec(CONTRACT)
    assert spec.methods == FORMAL_V4_METHODS
    assert spec.train_years == (2015, 2016, 2017, 2018)
    assert spec.selection_year == 2019
    assert spec.evaluation_year == 2020
    assert spec.allow_test_access is False
    assert spec.allow_evaluation_access_before_gate3 is False
    assert spec.claims["price_carbon_response"] is False
    assert spec.method("RSC-PF").online_exact_lp is False


def test_state_ledger_excludes_balance_diagnostics() -> None:
    spec = load_formal_v4_spec(CONTRACT)
    assert spec.state_features.continuous_features == (
        "grid", "pv_use", "pv_curt", "wt_use", "wt_curt", "g_chp", "g_gb",
        "p_chp", "q_chp", "q_gb", "p_ec", "q_ec", "q_ac_in", "q_ac",
        "p_charge", "p_discharge", "soc",
    )
    assert spec.state_features.activity_features == (
        "chp_on", "gas_boiler_on", "electric_chiller_on",
        "absorption_chiller_on", "bess_charge_on", "bess_discharge_on",
    )
    assert set(spec.state_features.excluded_features) == {
        "slack_e", "slack_c", "slack_h", "q_dump"
    }


def test_search_budget_is_explicit_for_every_trainable_method() -> None:
    spec = load_formal_v4_spec(CONTRACT)
    assert set(spec.search_budgets) == set(spec.trainable_methods)
    assert all(item.max_trials == 4 for item in spec.search_budgets.values())


def test_v4_matrix_has_all_methods_and_no_ablations() -> None:
    spec = load_formal_v4_spec(CONTRACT)
    rows = build_formal_v4_run_matrix(spec, phase="selection", run_id="unit")
    assert {row["method"] for row in rows} == set(FORMAL_V4_METHODS)
    assert {row["seed"] for row in rows if row["stochastic"]} == set(SEEDS)
    assert all(row["phase"] == "selection" for row in rows)
    assert not any(row["method"].startswith("A") for row in rows)


def test_v4_validator_rejects_unknown_method_and_test_access() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["methods"][0]["name"] = "DigitalTwins-Policy"
    with pytest.raises(ValueError, match="frozen order"):
        validate_formal_v4_payload(changed)

    changed = copy.deepcopy(payload)
    changed["allow_test_access"] = True
    with pytest.raises(ValueError, match="allow_test_access"):
        validate_formal_v4_payload(changed)


def test_v4_validator_rejects_unknown_keys_and_bad_split() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    changed = copy.deepcopy(payload)
    changed["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        validate_formal_v4_payload(changed)

    changed = copy.deepcopy(payload)
    changed["train_years"] = [2015, 2016, 2017, 2018, 2019]
    with pytest.raises(ValueError, match="train_years"):
        validate_formal_v4_payload(changed)

