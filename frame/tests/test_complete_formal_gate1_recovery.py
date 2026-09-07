from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.complete_formal_gate1 import Gate1RunConfig, load_gate1_data
from src.joint_dispatch.complete_formal_gate1_recovery import (
    expected_recovery_candidates,
    inspect_recovery_source,
)


FRAME_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"
GATE0_TRANSITION = FRAME_ROOT / "reports" / "rsc_pf_complete_formal" / "complete_formal_gate0_20260906_i" / "gate0" / "GATE0_TRANSITION.json"
SOURCE_RUN = FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2" / "formal_v4_2_20260905_j"
FAILED_RUN = FRAME_ROOT / "reports" / "rsc_pf_complete_formal" / "complete_formal_gate1_20260906_b"


@pytest.fixture(scope="module")
def contract():
    return CompleteFormalContract.from_path(CONTRACT_PATH)


@pytest.fixture(scope="module")
def gate1_data(contract):
    return load_gate1_data(
        Gate1RunConfig(CONTRACT_PATH, GATE0_TRANSITION, SOURCE_RUN, FRAME_ROOT / "reports", "recovery-test"),
        contract,
    )


def test_expected_recovery_candidates_match_frozen_search_grid(contract):
    keys = expected_recovery_candidates(contract)
    assert [(key.family, key.value) for key in keys] == [
        ("RSC-PF", 1.0), ("RSC-PF", 1.5), ("RSC-PF", 2.0), ("RSC-PF", 3.0),
        ("Differentiable-LP", 1e-5), ("Differentiable-LP", 3e-5),
        ("Differentiable-LP", 1e-4), ("Differentiable-LP", 3e-4),
    ]


def test_real_failed_run_is_read_only_inspectable(contract, gate1_data):
    inspection = inspect_recovery_source(FAILED_RUN, contract, gate1_data, GATE0_TRANSITION)
    assert len(inspection.candidates) == 8
    states = {(item.key.family, item.key.value): item.state for item in inspection.candidates}
    assert states[("RSC-PF", 1.0)] == "reusable-complete"
    assert states[("RSC-PF", 3.0)] == "reusable-complete"
    assert states[("Differentiable-LP", 1e-5)] == "reusable-checkpoint"
    assert states[("Differentiable-LP", 3e-5)] == "retrain-required"
    assert inspection.failure_receipt_sha256


def test_recovery_rejects_lineage_contract_mismatch(tmp_path, contract, gate1_data):
    source = tmp_path / "prior"
    gate1 = source / "gate1"
    gate1.mkdir(parents=True)
    lineage = json.loads((FAILED_RUN / "gate1" / "DATA_LINEAGE.json").read_text(encoding="utf-8"))
    lineage["contract_sha256"] = "0" * 64
    (gate1 / "DATA_LINEAGE.json").write_text(json.dumps(lineage), encoding="utf-8")
    (gate1 / "GATE1_FAILURE.json").write_bytes((FAILED_RUN / "gate1" / "GATE1_FAILURE.json").read_bytes())
    with pytest.raises(PermissionError, match="contract"):
        inspect_recovery_source(source, contract, gate1_data, GATE0_TRANSITION)


def test_recovery_rejects_gate0_hash_mismatch(tmp_path, contract, gate1_data):
    source = tmp_path / "prior"
    gate1 = source / "gate1"
    gate1.mkdir(parents=True)
    (gate1 / "DATA_LINEAGE.json").write_bytes((FAILED_RUN / "gate1" / "DATA_LINEAGE.json").read_bytes())
    (gate1 / "GATE1_FAILURE.json").write_bytes((FAILED_RUN / "gate1" / "GATE1_FAILURE.json").read_bytes())
    wrong_gate0 = tmp_path / "wrong_gate0.json"
    wrong_gate0.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="gate0_transition"):
        inspect_recovery_source(source, contract, gate1_data, wrong_gate0)
