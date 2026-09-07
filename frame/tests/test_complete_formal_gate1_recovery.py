from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.complete_formal_gate1 import Gate1RunConfig, load_gate1_data
import src.joint_dispatch.complete_formal_gate1 as gate1
from src.joint_dispatch.complete_formal_gate1_recovery import (
    build_recovery_manifest,
    expected_recovery_candidates,
    inspect_candidate,
    inspect_recovery_source,
    materialize_candidate,
    restore_differentiable_lp_artifact,
)
from scripts.run_rsc_pf_complete_formal_gate1_real import build_parser


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


def test_materialize_verified_rsc_candidate_without_changing_source(tmp_path, contract, gate1_data):
    inspection = inspect_recovery_source(FAILED_RUN, contract, gate1_data, GATE0_TRANSITION)
    candidate = inspection.by_key[next(key for key in inspection.by_key if key.family == "RSC-PF" and key.value == 1.0)]
    before = {
        path.relative_to(FAILED_RUN): path.stat().st_mtime_ns
        for path in FAILED_RUN.rglob("*") if path.is_file()
    }
    destination = materialize_candidate(candidate, tmp_path / "rsc_multiplier_1")
    assert (destination / "CHECKPOINT.pt").is_file()
    assert (destination / "COMPLETE_GATE1_ROW_RECEIPT.json").is_file()
    after = {
        path.relative_to(FAILED_RUN): path.stat().st_mtime_ns
        for path in FAILED_RUN.rglob("*") if path.is_file()
    }
    assert before == after


def test_materialize_verified_difflp_checkpoint_to_canonical_row(tmp_path, contract, gate1_data):
    inspection = inspect_recovery_source(FAILED_RUN, contract, gate1_data, GATE0_TRANSITION)
    key = next(key for key in inspection.by_key if key.family == "Differentiable-LP" and key.value == 1e-5)
    candidate = inspection.by_key[key]
    destination = materialize_candidate(candidate, tmp_path / "difflp_lr_1e-05")
    assert candidate.state == "reusable-checkpoint"
    assert destination == (tmp_path / "difflp_lr_1e-05" / "rows" / "Differentiable-LP" / "2026").resolve()
    assert (destination / "CHECKPOINT.pt").is_file()
    assert (destination / "TRAINING_RECEIPT.json").is_file()


def test_restore_difflp_reproduces_checkpoint_parameters_without_optimizer_step(
    tmp_path, contract, gate1_data, monkeypatch
):
    inspection = inspect_recovery_source(FAILED_RUN, contract, gate1_data, GATE0_TRANSITION)
    key = next(key for key in inspection.by_key if key.family == "Differentiable-LP" and key.value == 1e-5)
    destination = materialize_candidate(inspection.by_key[key], tmp_path / "difflp_lr_1e-05")
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("restore must not optimize")))
    artifact = restore_differentiable_lp_artifact(destination, gate1_data, contract)
    payload = torch.load(destination / "CHECKPOINT.pt", map_location="cpu", weights_only=False)
    assert artifact.method_id == "Differentiable-LP"
    assert artifact.seed == 2026
    for name, tensor in artifact.model.state_dict().items():
        assert torch.equal(tensor.cpu(), payload["model"][name].cpu())


def test_search_reuses_completed_trials_and_trains_only_missing(
    tmp_path, contract, gate1_data, monkeypatch
):
    inspection = inspect_recovery_source(FAILED_RUN, contract, gate1_data, GATE0_TRANSITION)
    trained = []

    def fail_rsc(*args, **kwargs):
        raise AssertionError("completed RSC-PF candidates must not be retrained")

    def record_difflp(*args, **kwargs):
        trained.append(float(kwargs["budget"].forecaster_lr))
        return None

    monkeypatch.setattr(gate1, "train_rsc_family", fail_rsc)
    monkeypatch.setattr(gate1, "train_differentiable_lp", record_difflp)
    monkeypatch.setattr(
        gate1, "evaluate_gate1_row",
        lambda *args, **kwargs: {
            "penalized_objective": 1.0,
            "finite": True,
            "physical_feasible": True,
        },
    )
    monkeypatch.setattr(
        gate1, "_itransformer_receipt",
        lambda data, output_dir: ({"source_root": "frame/third_party/iTransformer_source"}, output_dir / "ITRANSFORMER_ADAPTER_RECEIPT.json", SOURCE_RUN / "protocol" / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json"),
    )
    result = gate1.select_gate1_hyperparameters(
        gate1_data, contract, tmp_path, recovery=inspection,
    )
    assert np.allclose(trained, [3e-5, 1e-4, 3e-4])
    assert result["recovery_used"] is True
    assert result["reused_candidate_count"] == 5


def test_cli_accepts_explicit_recovery_source():
    args = build_parser().parse_args([
        "--gate0-transition", str(GATE0_TRANSITION),
        "--source-run", str(SOURCE_RUN),
        "--run-id", "recovered-run",
        "--resume-from", str(FAILED_RUN),
    ])
    assert args.resume_from.resolve() == FAILED_RUN.resolve()


def test_recovery_manifest_records_reused_training_time(contract, gate1_data, tmp_path):
    inspection = inspect_recovery_source(FAILED_RUN, contract, gate1_data, GATE0_TRANSITION)
    payload = build_recovery_manifest(
        inspection, tmp_path / "destination", contract, gate1_data, GATE0_TRANSITION,
        {"trials": [
            {"family": item.key.family, "value": item.key.value,
             "action": "reused-complete" if item.state == "reusable-complete" else "reused-training-reran-evaluation" if item.state == "reusable-checkpoint" else "trained",
             "training_reused": item.state != "retrain-required",
             "evaluation_reused": item.state == "reusable-complete"}
            for item in inspection.candidates
        ]},
    )
    assert payload["source_modified"] is False
    assert payload["candidate_count"] == 8
    assert payload["reused_candidate_count"] == 5
    assert payload["reused_training_runtime_seconds"] > 0
    assert payload["evaluation_year_accessed"] is False
