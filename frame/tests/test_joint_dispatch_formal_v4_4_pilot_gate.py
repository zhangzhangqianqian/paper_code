from __future__ import annotations

import copy
import json
from pathlib import Path

from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract
from src.joint_dispatch.formal_v4_4_pilot_gate import authorize_pilot_v44


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_formal_v4_4_contract(ROOT / "configs" / "joint_forecast_dispatch_formal_v4_4.json")


def _receipt() -> dict:
    digest = "a" * 64
    return {
        "lineage": {"source_manifest_sha256": digest, "train_data_sha256": digest, "selection_data_sha256": digest, "contract_sha256": digest},
        "accessed_years": [2015, 2016, 2017, 2018, 2019], "finite_values": [0.0],
        "comparisons": {"leakage_ratio": {"cooling": 0.5, "heating": 0.5}, "active_wape_ratio": {"cooling": 1.0, "heating": 1.0}, "electricity_gas_wape_ratio": {"electricity": 1.0, "gas": 1.0}, "four_task_score_ratio": 1.0},
        "regime": {"transition_balanced_accuracy_gain": 0.02, "macro_f1": 0.8, "prior_macro_f1": 0.7},
        "joint": {"penalized_objective": 1.0, "shortage": 0.1, "gradient_norms": {"decision_to_gate": 1.0, "decision_to_magnitude": 1.0, "decision_to_scheduler": 1.0}},
        "decoupled": {"penalized_objective": 1.2, "shortage": 0.2, "gradient_norms": {"decision_to_base": 0.0}},
        "physics": {"max_residual": 1.0e-8},
    }


def test_valid_receipt_authorizes_pilot() -> None:
    decision = authorize_pilot_v44(_receipt(), CONTRACT)
    assert decision.authorized_gate1
    assert not decision.failures


def test_pilot_denies_corrupt_receipts() -> None:
    receipt = _receipt(); receipt["accessed_years"].append(2020)
    decision = authorize_pilot_v44(receipt, CONTRACT)
    assert not decision.authorized_gate1
    assert "integrity" in decision.failures

    receipt = _receipt(); receipt["joint"]["gradient_norms"]["decision_to_gate"] = 0.0
    decision = authorize_pilot_v44(receipt, CONTRACT)
    assert not decision.authorized_gate1
    assert "gradient" in decision.failures
