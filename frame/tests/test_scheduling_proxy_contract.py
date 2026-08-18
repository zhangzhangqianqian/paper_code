from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.dispatch_lp import VARIABLES
from src.scheduling.proxy_contract import FEATURE_ORDER, load_contract, validate_contract


BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"
V2_CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v2.json"


def test_contract_freezes_orders_splits_and_gas_semantics():
    contract = load_contract(CONTRACT, BENCHMARK)
    assert contract.horizon == 4
    assert contract.feature_order == FEATURE_ORDER
    assert contract.label_order == tuple(VARIABLES)
    assert contract.split("train", smoke=True).size == 64
    assert contract.split("validation", smoke=True).size == 16
    assert contract.split("test", smoke=True).size == 16
    assert contract.split("train").size == 8192
    assert contract.gas_prior["is_balance_equation"] is False


def test_contract_rejects_duplicate_feature_order():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data["feature_order"] = list(data["feature_order"])
    data["feature_order"][0] = data["feature_order"][1]
    with pytest.raises(ValueError):
        validate_contract(data, BENCHMARK)


def test_contract_rejects_benchmark_hash_mismatch():
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    data["benchmark_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_contract(data, BENCHMARK)


@pytest.mark.parametrize("field_path", [
    ("gas_prior", "noise_fraction"),
    ("gas_prior", "additive_noise_scale"),
    ("gas_prior", "mask_fraction"),
    ("model", "dropout"),
    ("loss_weights", "dispatch"),
    ("normalization", "epsilon"),
    ("training", "learning_rate"),
    ("safety", "feasibility_tolerance"),
])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_contract_rejects_nonfinite_nested_numeric_settings(field_path, bad_value):
    data = json.loads(CONTRACT.read_text(encoding="utf-8"))
    section, name = field_path
    data[section][name] = bad_value
    with pytest.raises(ValueError):
        validate_contract(data, BENCHMARK)


def test_v2_contract_freezes_feasible_decoder_and_no_fallback():
    contract = load_contract(V2_CONTRACT, BENCHMARK)
    assert contract.is_v2
    assert contract.schema_version == "scheduling-proxy-contract-v2"
    assert contract.model["decision_dim"] == 15
    assert contract.model["decision_groups"] == {
        "cooling": [0, 4], "chp": [4, 8], "soc": [8, 11], "renewable_pv": [11, 15],
    }
    assert contract.safety["allow_exact_fallback"] is False
    assert contract.safety["inference_exact_lp_calls"] == 0


def test_contract_rejects_wrong_v2_group_coverage(tmp_path):
    payload = json.loads(V2_CONTRACT.read_text(encoding="utf-8"))
    payload["model"]["decision_groups"]["renewable_pv"] = [11, 14]
    invalid = tmp_path / "invalid-v2.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decision"):
        load_contract(invalid, BENCHMARK)
