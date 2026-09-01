from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_rsc_pf_external_baseline_sources import (
    load_external_implementation_config,
    prepare_sources,
)


ROOT = Path(__file__).parents[1]
REGISTRY = ROOT / "configs" / "rsc_pf_external_baselines_v1.json"
CONFIG = ROOT / "configs" / "rsc_pf_external_baseline_implementation_v1.json"


def test_implementation_config_matches_frozen_contract() -> None:
    config = load_external_implementation_config(CONFIG)
    assert config["lookback"] == 24
    assert config["horizon"] == 4
    assert config["task_order"] == ["electricity", "cooling", "heating", "gas"]
    assert config["dispatch_dim"] == 21 and config["status_dim"] == 6
    assert config["seeds"] == [2026, 2027, 2028, 2029, 2030]
    assert config["validation_only"] is True and config["test_set_accessed"] is False


def test_prepare_sources_writes_three_method_receipt_without_test_access(tmp_path: Path) -> None:
    result = prepare_sources(REGISTRY, tmp_path / "sources")
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert result["implementation_ready"] is True
    assert result["test_set_accessed"] is False
    assert {item["method_id"] for item in receipt["methods"]} == {
        "iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy",
    }
    assert all(len(item["evidence_sha256"]) == 64 for item in receipt["methods"])
    assert receipt["validation_only"] is True and receipt["test_set_accessed"] is False


def test_prepare_sources_rejects_sealed_test_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed test-set path"):
        prepare_sources(REGISTRY, tmp_path / "sealed_test" / "sources")


def test_prepare_sources_rejects_malformed_evidence_hash(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    payload["methods"][0]["evidence_sha256"] = "not-a-sha256"
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence_sha256"):
        prepare_sources(registry, tmp_path / "sources")
