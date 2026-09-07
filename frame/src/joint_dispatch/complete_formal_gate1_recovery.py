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

from .complete_formal_contract import CompleteFormalContract
from .formal_v4_2_artifacts import sha256_file

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


@dataclass(frozen=True)
class Gate1RecoveryInspection:
    source_root: Path
    failure_receipt_sha256: str
    candidates: tuple[RecoveryCandidateEvidence, ...]

    @property
    def by_key(self) -> Mapping[RecoveryCandidateKey, RecoveryCandidateEvidence]:
        return {candidate.key: candidate for candidate in self.candidates}


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
    return Gate1RecoveryInspection(source, failure_hash, candidates)


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


__all__ = [
    "CandidateValidationError",
    "CandidatePaths",
    "Gate1RecoveryInspection",
    "RecoveryCandidateEvidence",
    "RecoveryCandidateKey",
    "RecoveryState",
    "candidate_paths",
    "expected_recovery_candidates",
    "inspect_candidate",
    "inspect_recovery_source",
    "load_json_object",
    "materialize_candidate",
]
