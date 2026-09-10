from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract, MethodSeedKey
from src.joint_dispatch.complete_formal_gate1 import (
    Gate1RunConfig,
    expected_gate1_rows,
    load_gate1_data,
    select_gate1_hyperparameters,
)


FRAME_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"
GATE0_TRANSITION = FRAME_ROOT / "reports" / "rsc_pf_complete_formal" / "complete_formal_gate0_20260906_i" / "gate0" / "GATE0_TRANSITION.json"
SOURCE_RUN = FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2" / "formal_v4_2_20260905_j"


def test_expected_gate1_rows_are_the_frozen_37_row_matrix():
    contract = CompleteFormalContract.from_path(CONTRACT_PATH)
    rows = expected_gate1_rows(contract)
    assert len(rows) == 37
    assert rows == contract.expected_rows("gate1")
    assert MethodSeedKey("RSC-PF", 2026) in rows
    assert MethodSeedKey("Seasonal-Naive-PTO", None) in rows
    assert MethodSeedKey("Perfect-Information-MPC", None) in rows


def test_real_gate1_loader_reads_only_train_and_2019_selection():
    contract = CompleteFormalContract.from_path(CONTRACT_PATH)
    data = load_gate1_data(
        Gate1RunConfig(CONTRACT_PATH, GATE0_TRANSITION, SOURCE_RUN, FRAME_ROOT / "reports", "test", True),
        contract,
    )
    assert len(data.train) == 34959
    assert len(data.selection) == 8709
    assert tuple(int(value) for value in np.unique(data.train.target_times.astype("datetime64[Y]").astype(int) + 1970)) == (2015, 2016, 2017, 2018)
    assert tuple(int(value) for value in np.unique(data.selection.target_times.astype("datetime64[Y]").astype(int) + 1970)) == (2019,)
    assert data.lineage["source_manifest_sha256"]


def test_gate1_loader_rejects_gate0_without_authorization(tmp_path):
    contract = CompleteFormalContract.from_path(CONTRACT_PATH)
    transition = tmp_path / "GATE0_TRANSITION.json"
    transition.write_text(json.dumps({"contract_sha256": contract.contract_sha256, "authorized_gate1_training": False}), encoding="utf-8")
    with pytest.raises(PermissionError, match="did not authorize"):
        load_gate1_data(Gate1RunConfig(CONTRACT_PATH, transition, SOURCE_RUN, tmp_path, "bad", True), contract)


def test_smoke_search_is_explicitly_skipped(tmp_path):
    # Keep this check independent of the long matrix while still exercising
    # the complete-v1 search receipt contract against real validated windows.
    contract = CompleteFormalContract.from_path(CONTRACT_PATH)
    data = load_gate1_data(
        Gate1RunConfig(CONTRACT_PATH, GATE0_TRANSITION, SOURCE_RUN, tmp_path, "search-test", True),
        contract,
    )
    receipt = select_gate1_hyperparameters(data, contract, tmp_path, smoke=True)
    assert receipt["status"] == "smoke-skipped"
    assert (tmp_path / "GATE1_SEARCH.json").is_file()
