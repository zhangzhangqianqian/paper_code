"""Read-only inspection primitives for recovering a failed complete-v1 Gate 1 run.

This module deliberately does not train or mutate a prior run.  It classifies
the files that are already present so the continuation runner can later decide
which candidates to copy, re-evaluate, or retrain.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import hashlib
import shutil
from typing import TYPE_CHECKING, Any, Literal, Mapping

import torch

from .complete_formal_contract import CompleteFormalContract
from .formal_v4_2_artifacts import sha256_file
from .formal_v4_2_checkpoint import load_training_checkpoint
from .formal_v4_2_gate2_training import (
    TrainedMethodArtifact,
    build_direct_policy_model,
    build_rsc_model,
)
from .formal_v4_2_training import StageBudgetV42
from ..models import Scheme2RModel

if TYPE_CHECKING:  # pragma: no cover - imported only for static type checking
    from .complete_formal_gate1 import Gate1DataBundle


RecoveryState = Literal["reusable-complete", "reusable-checkpoint", "retrain-required"]


@dataclass(frozen=True)
class RecoveryCandidateKey:
    family: str
    value: float
    method_id: str
    seed: int = 2026


@dataclass(frozen=True)
class CandidatePaths:
    trial_root: Path
    canonical_row: Path
    legacy_row: Path


@dataclass(frozen=True)
class RecoveryCandidateEvidence:
    key: RecoveryCandidateKey
    state: RecoveryState
    source_trial_root: Path
    checkpoint_path: Path | None
    training_receipt_path: Path | None
    complete_receipt_path: Path | None
    reason: str
    file_sha256: Mapping[str, str]
    runtime_seconds_reused: float


FINAL_REUSABLE_METHOD_IDS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "State-Conditioned-PTO",
    "Direct-Policy",
    "Scheme2R-PTO",
)


@dataclass(frozen=True)
class FinalRowKey:
    method_id: str
    seed: int


@dataclass(frozen=True)
class FinalRowEvidence:
    key: FinalRowKey
    state: RecoveryState
    source_row: Path
    checkpoint_path: Path | None
    training_receipt_path: Path | None
    reason: str
    file_sha256: Mapping[str, str]
    runtime_seconds_reused: float


@dataclass(frozen=True)
class Gate1RecoveryInspection:
    source_root: Path
    failure_receipt_sha256: str
    candidates: tuple[RecoveryCandidateEvidence, ...]
    final_rows: tuple[FinalRowEvidence, ...] = ()

    @property
    def by_key(self) -> Mapping[RecoveryCandidateKey, RecoveryCandidateEvidence]:
        return {candidate.key: candidate for candidate in self.candidates}

    @property
    def final_by_key(self) -> Mapping[FinalRowKey, FinalRowEvidence]:
        return {row.key: row for row in self.final_rows}


class CandidateValidationError(ValueError):
    """A candidate artifact is invalid and must be retrained."""


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"recovery JSON artifact is unreadable: {source}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"recovery JSON artifact must be an object: {source}")
    return payload


def expected_recovery_candidates(contract: CompleteFormalContract) -> tuple[RecoveryCandidateKey, ...]:
    """Enumerate the exact eight Gate 1 search candidates from the contract."""

    search = contract.payload["training"]["search"]
    rsc = tuple(float(value) for value in search["rsc_pf_decision_multiplier_grid"])
    difflp = tuple(float(value) for value in search["difflp_learning_rate_grid"])
    return tuple(
        [RecoveryCandidateKey("RSC-PF", value, "RSC-PF") for value in rsc]
        + [RecoveryCandidateKey("Differentiable-LP", value, "Differentiable-LP") for value in difflp]
    )


def candidate_paths(gate1_root: str | Path, key: RecoveryCandidateKey) -> CandidatePaths:
    root = Path(gate1_root).resolve()
    if key.family == "RSC-PF":
        label = f"rsc_multiplier_{key.value:g}"
        trial = root / "search" / label
        canonical = trial / "rows" / "RSC-PF" / str(key.seed)
        return CandidatePaths(trial, canonical, canonical)
    if key.family == "Differentiable-LP":
        label = f"difflp_lr_{key.value:g}"
        trial = root / "search" / label
        canonical = trial / "rows" / "Differentiable-LP" / str(key.seed)
        # The failed run wrote the first DiffLP candidate one level above the
        # canonical row.  This legacy path is accepted only for that exact
        # family/seed layout and is never selected recursively.
        legacy = trial / "rows"
        return CandidatePaths(trial, canonical, legacy)
    raise ValueError(f"unsupported recovery candidate family: {key.family}")


def _candidate_evidence(gate1_root: Path, key: RecoveryCandidateKey) -> RecoveryCandidateEvidence:
    paths = candidate_paths(gate1_root, key)
    training_root = paths.canonical_row
    receipt_path = training_root / "TRAINING_RECEIPT.json"
    checkpoint_path = training_root / "CHECKPOINT.pt"
    if key.family == "Differentiable-LP" and (not receipt_path.is_file() or not checkpoint_path.is_file()):
        legacy_receipt = paths.legacy_row / "TRAINING_RECEIPT.json"
        legacy_checkpoint = paths.legacy_row / "CHECKPOINT.pt"
        if legacy_receipt.is_file() and legacy_checkpoint.is_file():
            receipt_path = legacy_receipt
            checkpoint_path = legacy_checkpoint
            training_root = paths.legacy_row

    complete_path = paths.canonical_row / "COMPLETE_GATE1_ROW_RECEIPT.json"
    files: dict[str, str] = {}
    runtime = 0.0
    if receipt_path.is_file():
        receipt = load_json_object(receipt_path)
        runtime = float(receipt.get("runtime_seconds", 0.0))
        files["TRAINING_RECEIPT.json"] = sha256_file(receipt_path)
    if checkpoint_path.is_file():
        files["CHECKPOINT.pt"] = sha256_file(checkpoint_path)
    if complete_path.is_file():
        complete = load_json_object(complete_path)
        rollout = paths.canonical_row / "ROLLOUT.npz"
        metrics = paths.canonical_row / "METRICS.json"
        if rollout.is_file():
            files["ROLLOUT.npz"] = sha256_file(rollout)
        if metrics.is_file():
            files["METRICS.json"] = sha256_file(metrics)
        if complete.get("status") == "complete" and complete.get("complete") is True:
            return RecoveryCandidateEvidence(
                key, "reusable-complete", paths.trial_root, checkpoint_path if checkpoint_path.is_file() else None,
                receipt_path if receipt_path.is_file() else None, complete_path,
                "complete training and evaluation receipt present", files, runtime,
            )
    if receipt_path.is_file() and checkpoint_path.is_file():
        return RecoveryCandidateEvidence(
            key, "reusable-checkpoint", paths.trial_root, checkpoint_path, receipt_path, None,
            "training checkpoint present; evaluation receipt incomplete", files, runtime,
        )
    return RecoveryCandidateEvidence(
        key, "retrain-required", paths.trial_root, None, None, None,
        "training checkpoint or receipt absent", files, runtime,
    )


def _expected_checkpoint_lineage(contract: CompleteFormalContract, data: "Gate1DataBundle") -> dict[str, str]:
    return {
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": data.lineage["source_manifest_sha256"],
        "data_sha256": data.lineage["train_manifest_sha256"],
        "normalization_sha256": data.lineage["normalization_sha256"],
    }


def _training_paths(gate1_root: Path, key: RecoveryCandidateKey) -> tuple[Path, Path, Path]:
    paths = candidate_paths(gate1_root, key)
    receipt = paths.canonical_row / "TRAINING_RECEIPT.json"
    checkpoint = paths.canonical_row / "CHECKPOINT.pt"
    row_root = paths.canonical_row
    if key.family == "Differentiable-LP" and (not receipt.is_file() or not checkpoint.is_file()):
        receipt = paths.legacy_row / "TRAINING_RECEIPT.json"
        checkpoint = paths.legacy_row / "CHECKPOINT.pt"
        row_root = paths.legacy_row
    return row_root, receipt, checkpoint


def _validate_training_artifact(
    gate1_root: Path,
    key: RecoveryCandidateKey,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
) -> tuple[Path, Path, dict[str, Any]]:
    row_root, receipt_path, checkpoint_path = _training_paths(gate1_root, key)
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        raise CandidateValidationError("training checkpoint or receipt absent")
    receipt = load_json_object(receipt_path)
    required = {
        "method_id": key.method_id,
        "seed": key.seed,
        "epochs": 30,
        "test_set_accessed": False,
    }
    for name, expected in required.items():
        if receipt.get(name) != expected:
            raise CandidateValidationError(f"training receipt field mismatch: {name}")
    if receipt.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise CandidateValidationError("checkpoint hash mismatch")
    try:
        import torch
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CandidateValidationError("checkpoint is unreadable") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "formal-v4.2-training-checkpoint-v1":
        raise CandidateValidationError("checkpoint schema mismatch")
    if int(payload.get("epoch", -1)) != 29:
        raise CandidateValidationError("checkpoint epoch mismatch")
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise CandidateValidationError("checkpoint lineage is missing")
    for name, expected in _expected_checkpoint_lineage(contract, data).items():
        if lineage.get(name) != expected:
            raise CandidateValidationError(f"checkpoint lineage mismatch: {name}")
    return row_root, checkpoint_path, receipt


def _final_row_path(source_root: Path, key: FinalRowKey) -> Path:
    return source_root / "gate1" / "rows" / key.method_id / str(key.seed)


def _final_epochs(contract: CompleteFormalContract) -> int:
    training = contract.payload.get("training", {})
    common = training.get("common", {}) if isinstance(training, Mapping) else {}
    return int(common.get("max_epochs_per_stage", 30))


def _validate_final_row(
    source_root: Path,
    key: FinalRowKey,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
) -> FinalRowEvidence:
    row = _final_row_path(source_root, key)
    receipt_path = row / "TRAINING_RECEIPT.json"
    checkpoint_path = row / "CHECKPOINT.pt"
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        return FinalRowEvidence(
            key, "retrain-required", row, None, None,
            "training checkpoint or receipt absent", {}, 0.0,
        )
    try:
        receipt = load_json_object(receipt_path)
        expected_receipt = {
            "method_id": key.method_id,
            "seed": key.seed,
            "epochs": _final_epochs(contract),
            "test_set_accessed": False,
            "forecast_loss_applicable": key.method_id != "Direct-Policy",
        }
        for name, expected in expected_receipt.items():
            if receipt.get(name) != expected:
                raise CandidateValidationError(f"final row receipt field mismatch: {name}")
        if receipt.get("checkpoint_sha256") != sha256_file(checkpoint_path):
            raise CandidateValidationError("final row checkpoint hash mismatch")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or payload.get("schema") != "formal-v4.2-training-checkpoint-v1":
            raise CandidateValidationError("final row checkpoint schema mismatch")
        if int(payload.get("epoch", -1)) != _final_epochs(contract) - 1:
            raise CandidateValidationError("final row checkpoint epoch mismatch")
        lineage = payload.get("lineage")
        if not isinstance(lineage, Mapping):
            raise CandidateValidationError("final row checkpoint lineage is missing")
        for name, expected in _expected_checkpoint_lineage(contract, data).items():
            if lineage.get(name) != expected:
                raise CandidateValidationError(f"final row checkpoint lineage mismatch: {name}")
        runtime = float(receipt.get("runtime_seconds", 0.0))
        return FinalRowEvidence(
            key, "reusable-checkpoint", row, checkpoint_path, receipt_path,
            "validated final training checkpoint", {
                "CHECKPOINT.pt": sha256_file(checkpoint_path),
                "TRAINING_RECEIPT.json": sha256_file(receipt_path),
            }, runtime,
        )
    except Exception as exc:
        return FinalRowEvidence(
            key, "retrain-required", row, None, None, str(exc), {}, 0.0,
        )


def inspect_final_rows(
    source_root: str | Path,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
) -> tuple[FinalRowEvidence, ...]:
    """Classify final matrix checkpoints already present in a partial run."""

    source = Path(source_root).resolve()
    return tuple(
        _validate_final_row(source, FinalRowKey(method_id, 2026), contract, data)
        for method_id in FINAL_REUSABLE_METHOD_IDS
    )


def _restore_optimizer(model: torch.nn.Module, method_id: str) -> torch.optim.Optimizer:
    """Recreate the optimizer parameter groups used by the saved artifact."""

    weight_decay = 1.0e-4
    if method_id == "RSC-PF":
        return torch.optim.AdamW([
            {"params": tuple(model.forecaster_parameters()), "lr": 1.0e-5},
            {"params": tuple(model.scheduler_parameters()), "lr": 1.0e-3},
        ], weight_decay=weight_decay)
    if method_id == "Decoupled-RSC-PF":
        return torch.optim.AdamW(tuple(model.scheduler_parameters()), lr=1.0e-3, weight_decay=weight_decay)
    if method_id == "State-Conditioned-PTO":
        return torch.optim.AdamW(tuple(model.forecaster_parameters()), lr=1.0e-5, weight_decay=weight_decay)
    if method_id == "Direct-Policy":
        return torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=weight_decay)
    if method_id == "Scheme2R-PTO":
        return torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=weight_decay)
    raise ValueError(f"unsupported final row restoration method: {method_id}")


def restore_final_artifact(
    evidence: FinalRowEvidence,
    data: "Gate1DataBundle",
    contract: CompleteFormalContract,
    output_row: str | Path,
    source_info: Mapping[str, Any] | None = None,
) -> TrainedMethodArtifact:
    """Copy and restore one validated final checkpoint without an update step."""

    if evidence.state != "reusable-checkpoint" or evidence.checkpoint_path is None or evidence.training_receipt_path is None:
        raise CandidateValidationError("final row is not reusable")
    destination = Path(output_row).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite recovery row: {destination}")
    destination.mkdir(parents=True)
    _copy_checked(
        evidence.checkpoint_path,
        destination / "CHECKPOINT.pt",
        evidence.file_sha256["CHECKPOINT.pt"],
    )
    _copy_checked(
        evidence.training_receipt_path,
        destination / "TRAINING_RECEIPT.json",
        evidence.file_sha256["TRAINING_RECEIPT.json"],
    )
    if evidence.key.method_id in {"RSC-PF", "Decoupled-RSC-PF", "State-Conditioned-PTO"}:
        model = build_rsc_model(data.legacy, data.parameters)
    elif evidence.key.method_id == "Direct-Policy":
        model = build_direct_policy_model(data.legacy, data.parameters)
    elif evidence.key.method_id == "Scheme2R-PTO":
        model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0)
    else:  # pragma: no cover - guarded by FINAL_REUSABLE_METHOD_IDS
        raise ValueError(f"unsupported final row restoration method: {evidence.key.method_id}")
    optimizer = _restore_optimizer(model, evidence.key.method_id)
    checkpoint = load_training_checkpoint(
        destination / "CHECKPOINT.pt",
        model=model,
        optimizer=optimizer,
        expected_lineage=_expected_checkpoint_lineage(contract, data),
    )
    receipt = load_json_object(destination / "TRAINING_RECEIPT.json")
    return TrainedMethodArtifact(
        method_id=evidence.key.method_id,
        seed=evidence.key.seed,
        model=model,
        checkpoint_path=checkpoint.path,
        checkpoint_sha256=checkpoint.model_sha256,
        training_receipt=receipt,
        stage_s_parent_sha256=str(receipt.get("stage_s_parent_sha256", "")),
        decision_forecaster_gradient_norm=float(receipt.get("decision_forecaster_gradient_norm", 0.0)),
    )


def _validate_complete_evaluation(
    gate1_root: Path,
    key: RecoveryCandidateKey,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
    checkpoint_path: Path,
    training_receipt_path: Path,
) -> tuple[Path, dict[str, str]]:
    paths = candidate_paths(gate1_root, key)
    receipt_path = paths.canonical_row / "COMPLETE_GATE1_ROW_RECEIPT.json"
    rollout_path = paths.canonical_row / "ROLLOUT.npz"
    metrics_path = paths.canonical_row / "METRICS.json"
    if not receipt_path.is_file() or not rollout_path.is_file() or not metrics_path.is_file():
        raise CandidateValidationError("complete evaluation artifacts are absent")
    receipt = load_json_object(receipt_path)
    required = {
        "method_id": key.method_id,
        "seed": key.seed,
        "status": "complete",
        "complete": True,
        "synthetic": False,
        "paper_result": False,
        "origin_count": int(contract.selection_origin_count),
        "chronological": True,
        "finite": True,
        "physical_feasible": True,
        "evaluation_year_accessed": False,
        "test_set_accessed": False,
        "contract_sha256": contract.contract_sha256,
        "selection_windows_sha256": data.lineage["selection_windows_sha256"],
        "normalization_sha256": data.lineage["normalization_sha256"],
        "source_manifest_sha256": data.lineage["source_manifest_sha256"],
        "train_windows_sha256": data.lineage["train_windows_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_receipt_sha256": sha256_file(training_receipt_path),
        "metrics_sha256": sha256_file(metrics_path),
        "rollout_sha256": sha256_file(rollout_path),
    }
    for name, expected in required.items():
        if receipt.get(name) != expected:
            raise CandidateValidationError(f"complete receipt field mismatch: {name}")
    return receipt_path, {
        "CHECKPOINT.pt": sha256_file(checkpoint_path),
        "TRAINING_RECEIPT.json": sha256_file(training_receipt_path),
        "COMPLETE_GATE1_ROW_RECEIPT.json": sha256_file(receipt_path),
        "METRICS.json": sha256_file(metrics_path),
        "ROLLOUT.npz": sha256_file(rollout_path),
    }


def inspect_candidate(
    gate1_root: str | Path,
    key: RecoveryCandidateKey,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
) -> RecoveryCandidateEvidence:
    """Validate one search candidate and classify its recoverability."""

    root = Path(gate1_root).resolve()
    paths = candidate_paths(root, key)
    try:
        row_root, checkpoint_path, training_receipt = _validate_training_artifact(root, key, contract, data)
    except CandidateValidationError as exc:
        return RecoveryCandidateEvidence(
            key, "retrain-required", paths.trial_root, None, None, None,
            str(exc), {}, 0.0,
        )
    training_receipt_path = row_root / "TRAINING_RECEIPT.json"
    runtime = float(training_receipt.get("runtime_seconds", 0.0))
    try:
        complete_path, hashes = _validate_complete_evaluation(
            root, key, contract, data, checkpoint_path, training_receipt_path,
        )
        return RecoveryCandidateEvidence(
            key, "reusable-complete", paths.trial_root, checkpoint_path, training_receipt_path,
            complete_path, "complete training and evaluation evidence validated", hashes, runtime,
        )
    except CandidateValidationError as exc:
        return RecoveryCandidateEvidence(
            key, "reusable-checkpoint", paths.trial_root, checkpoint_path, training_receipt_path,
            None, str(exc), {
                "CHECKPOINT.pt": sha256_file(checkpoint_path),
                "TRAINING_RECEIPT.json": sha256_file(training_receipt_path),
            }, runtime,
        )


def _validate_source_lineage(
    source_root: Path,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
    gate0_transition_path: Path,
) -> str:
    gate1 = source_root / "gate1"
    lineage_path = gate1 / "DATA_LINEAGE.json"
    failure_path = gate1 / "GATE1_FAILURE.json"
    if not lineage_path.is_file():
        raise FileNotFoundError(f"recovery source lineage is missing: {lineage_path}")
    if not failure_path.is_file():
        raise FileNotFoundError(f"recovery source failure receipt is missing: {failure_path}")
    lineage = load_json_object(lineage_path)
    expected = {
        "contract_sha256": contract.contract_sha256,
        "gate0_transition_sha256": sha256_file(gate0_transition_path),
        "train_windows_sha256": data.lineage["train_windows_sha256"],
        "selection_windows_sha256": data.lineage["selection_windows_sha256"],
        "normalization_sha256": data.lineage["normalization_sha256"],
        "source_manifest_sha256": data.lineage["source_manifest_sha256"],
    }
    for field, value in expected.items():
        if lineage.get(field) != value:
            raise PermissionError(f"recovery source {field} mismatch")
    if lineage.get("evaluation_year_accessed") is not False:
        raise PermissionError("recovery source accessed the evaluation year")
    failure = load_json_object(failure_path)
    if failure.get("contract_sha256") != contract.contract_sha256:
        raise PermissionError("recovery failure receipt contract mismatch")
    if failure.get("evaluation_year_accessed") is not False:
        raise PermissionError("recovery failure receipt reports evaluation-year access")
    return sha256_file(failure_path)


def inspect_recovery_source(
    source_root: str | Path,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
    gate0_transition_path: str | Path,
) -> Gate1RecoveryInspection:
    """Inspect a failed Gate 1 run without writing to it or loading new data."""

    source = Path(source_root).resolve()
    failure_hash = _validate_source_lineage(
        source, contract, data, Path(gate0_transition_path).resolve(),
    )
    gate1_root = source / "gate1"
    candidates = tuple(
        inspect_candidate(gate1_root, key, contract, data)
        for key in expected_recovery_candidates(contract)
    )
    final_rows = inspect_final_rows(source, contract, data)
    return Gate1RecoveryInspection(source, failure_hash, candidates, final_rows)


def _copy_checked(source: Path, destination: Path, expected_hash: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    actual = sha256_file(destination)
    if actual != expected_hash:
        raise IOError(f"copied recovery artifact hash mismatch: {destination.name}")


def materialize_candidate(evidence: RecoveryCandidateEvidence, destination_trial: str | Path) -> Path:
    """Copy one validated candidate into its canonical destination row."""

    if evidence.state == "retrain-required":
        raise CandidateValidationError("cannot materialize a candidate marked retrain-required")
    destination = Path(destination_trial).resolve()
    destination_row = destination / "rows" / evidence.key.method_id / str(evidence.key.seed)
    if destination_row.exists():
        raise FileExistsError(f"refusing to overwrite recovery row: {destination_row}")
    destination_row.mkdir(parents=True)
    source_row = evidence.checkpoint_path.parent if evidence.checkpoint_path is not None else None
    if source_row is None:
        raise CandidateValidationError("validated candidate has no source row")
    for name, expected_hash in evidence.file_sha256.items():
        source = source_row / name
        if not source.is_file():
            # A complete RSC receipt and its rollout/metrics are always beside
            # the canonical checkpoint.  Missing files indicate stale evidence.
            raise CandidateValidationError(f"source recovery artifact is missing: {source}")
        _copy_checked(source, destination_row / name, expected_hash)
    # RSC-PF's Stage-S parent is shared by the family and is needed for a
    # complete provenance tree.  Copy it when it exists; its hash is checked by
    # the copied RSC training receipt's parent hash during later audit.
    shared_source = evidence.source_trial_root / "rows" / "_shared" / str(evidence.key.seed)
    if evidence.key.family == "RSC-PF" and shared_source.is_dir():
        shared_destination = destination / "rows" / "_shared" / str(evidence.key.seed)
        shutil.copytree(shared_source, shared_destination, dirs_exist_ok=False)
    return destination_row


def build_recovery_manifest(
    inspection: Gate1RecoveryInspection,
    destination_run_root: str | Path,
    contract: CompleteFormalContract,
    data: "Gate1DataBundle",
    gate0_transition_path: str | Path,
    search_payload: Mapping[str, Any],
    matrix_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable provenance envelope for a completed recovery run."""

    actions = {
        (str(item.get("family")), float(item.get("value"))): item
        for item in search_payload.get("trials", [])
        if isinstance(item, Mapping)
    }
    candidates: list[dict[str, Any]] = []
    for evidence in inspection.candidates:
        trial = actions.get((evidence.key.family, float(evidence.key.value)), {})
        candidates.append({
            "family": evidence.key.family,
            "value": evidence.key.value,
            "method_id": evidence.key.method_id,
            "seed": evidence.key.seed,
            "validation_state": evidence.state,
            "action": trial.get("action", "not-run"),
            "training_reused": bool(trial.get("training_reused", False)),
            "evaluation_reused": bool(trial.get("evaluation_reused", False)),
            "source_trial_root": str(evidence.source_trial_root),
            "artifact_hashes": dict(evidence.file_sha256),
            "runtime_seconds_reused": float(evidence.runtime_seconds_reused),
            "reason": evidence.reason,
        })
    matrix_actions = {}
    if isinstance(matrix_payload, Mapping):
        for item in matrix_payload.get("actions", []):
            if isinstance(item, Mapping):
                matrix_actions[(str(item.get("method_id")), int(item.get("seed")))] = str(item.get("action", "trained"))
    final_rows = [
        {
            "method_id": evidence.key.method_id,
            "seed": evidence.key.seed,
            "validation_state": evidence.state,
            "action": matrix_actions.get((evidence.key.method_id, evidence.key.seed), "trained"),
            "training_reused": matrix_actions.get((evidence.key.method_id, evidence.key.seed)) == "reused-checkpoint",
            "source_row": str(evidence.source_row),
            "artifact_hashes": dict(evidence.file_sha256),
            "runtime_seconds_reused": float(evidence.runtime_seconds_reused),
            "reason": evidence.reason,
        }
        for evidence in inspection.final_rows
    ]
    return {
        "schema_version": "rsc-pf-complete-formal-gate1-recovery-v1",
        "status": "complete",
        "source_run_id": inspection.source_root.name,
        "source_run_root": str(inspection.source_root),
        "destination_run_id": Path(destination_run_root).resolve().name,
        "destination_run_root": str(Path(destination_run_root).resolve()),
        "failure_receipt_sha256": inspection.failure_receipt_sha256,
        "contract_sha256": contract.contract_sha256,
        "gate0_transition_sha256": sha256_file(gate0_transition_path),
        "source_manifest_sha256": data.lineage["source_manifest_sha256"],
        "train_windows_sha256": data.lineage["train_windows_sha256"],
        "selection_windows_sha256": data.lineage["selection_windows_sha256"],
        "normalization_sha256": data.lineage["normalization_sha256"],
        "candidate_count": len(candidates),
        "reused_candidate_count": sum(1 for item in candidates if item["training_reused"]),
        "reused_training_runtime_seconds": sum(float(item["runtime_seconds_reused"]) for item in candidates if item["training_reused"]),
        "candidates": candidates,
        "final_row_count": len(final_rows),
        "reused_final_row_count": sum(1 for item in final_rows if item["training_reused"]),
        "reused_final_training_runtime_seconds": sum(float(item["runtime_seconds_reused"]) for item in final_rows if item["training_reused"]),
        "final_rows": final_rows,
        "source_modified": False,
        "evaluation_year_accessed": False,
        "test_set_accessed": False,
    }


def restore_differentiable_lp_artifact(
    row_dir: str | Path,
    data: "Gate1DataBundle",
    contract: CompleteFormalContract,
) -> TrainedMethodArtifact:
    """Restore a verified DiffLP model and optimizer without an update step."""

    row = Path(row_dir).resolve()
    receipt_path = row / "TRAINING_RECEIPT.json"
    checkpoint_path = row / "CHECKPOINT.pt"
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        raise CandidateValidationError("canonical Differentiable-LP checkpoint row is incomplete")
    receipt = load_json_object(receipt_path)
    if receipt.get("method_id") != "Differentiable-LP" or receipt.get("seed") != 2026 or receipt.get("epochs") != 30:
        raise CandidateValidationError("Differentiable-LP restoration receipt is invalid")
    if receipt.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise CandidateValidationError("Differentiable-LP restoration checkpoint hash mismatch")
    model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0)
    budget = StageBudgetV42(forecaster_lr=1.0e-5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=budget.forecaster_lr, weight_decay=budget.weight_decay,
    )
    checkpoint = load_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        expected_lineage=_expected_checkpoint_lineage(contract, data),
    )
    return TrainedMethodArtifact(
        method_id="Differentiable-LP",
        seed=2026,
        model=model,
        checkpoint_path=checkpoint.path,
        checkpoint_sha256=checkpoint.model_sha256,
        training_receipt=receipt,
        decision_forecaster_gradient_norm=float(receipt.get("decision_forecaster_gradient_norm", 0.0)),
    )


__all__ = [
    "CandidateValidationError",
    "CandidatePaths",
    "FINAL_REUSABLE_METHOD_IDS",
    "FinalRowEvidence",
    "FinalRowKey",
    "Gate1RecoveryInspection",
    "RecoveryCandidateEvidence",
    "RecoveryCandidateKey",
    "RecoveryState",
    "candidate_paths",
    "expected_recovery_candidates",
    "inspect_candidate",
    "inspect_final_rows",
    "inspect_recovery_source",
    "load_json_object",
    "materialize_candidate",
    "build_recovery_manifest",
    "restore_differentiable_lp_artifact",
    "restore_final_artifact",
]
