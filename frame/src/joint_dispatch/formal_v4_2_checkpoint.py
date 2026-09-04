"""Persistent optimizer/checkpoint utilities for formal-v4.2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import os
from pathlib import Path
import random
import shutil
from typing import Any, Mapping

import numpy as np
import torch

from .formal_v4_2_artifacts import LineageError, sha256_file


CHECKPOINT_SCHEMA = "formal-v4.2-training-checkpoint-v1"


@dataclass(frozen=True)
class TrainingCheckpointV42:
    path: Path
    epoch: int
    early_stopping: Mapping[str, Any]
    lineage: Mapping[str, Any]
    model_sha256: str


def optimizer_step_value(optimizer: torch.optim.Optimizer) -> int:
    steps: list[int] = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            steps.append(int(value.detach().cpu().item()))
        else:
            steps.append(int(value))
    return max(steps, default=0)


def _rng_payload() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _restore_rng(payload: Mapping[str, Any]) -> None:
    torch.set_rng_state(payload["torch"])
    np.random.set_state(tuple(payload["numpy"]))
    random.setstate(tuple(payload["python"]))


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    if temporary.exists():
        raise FileExistsError(f"stale checkpoint write exists: {temporary}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _torch_bytes(payload: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(payload), buffer)
    return buffer.getvalue()


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    early_stopping: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> TrainingCheckpointV42:
    if int(epoch) < 0:
        raise ValueError("checkpoint epoch must be non-negative")
    required = {"contract_sha256", "source_manifest_sha256", "data_sha256"}
    missing = sorted(required - set(lineage))
    if missing:
        raise LineageError(f"checkpoint lineage is missing {missing[0]}")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "early_stopping": dict(early_stopping),
        "rng": _rng_payload(),
        "lineage": dict(lineage),
    }
    destination = Path(path)
    if destination.exists():
        if destination.read_bytes() != _torch_bytes(payload):
            raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    else:
        _atomic_torch_save(destination, payload)
    return TrainingCheckpointV42(
        destination,
        int(epoch),
        dict(early_stopping),
        dict(lineage),
        sha256_file(destination),
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_lineage: Mapping[str, Any],
) -> TrainingCheckpointV42:
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise LineageError("checkpoint schema mismatch")
    for field, expected in expected_lineage.items():
        if payload.get("lineage", {}).get(field) != expected:
            raise LineageError(f"checkpoint {field} lineage mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng(payload["rng"])
    return TrainingCheckpointV42(
        source,
        int(payload["epoch"]),
        dict(payload["early_stopping"]),
        dict(payload["lineage"]),
        sha256_file(source),
    )


def clone_stage_s_branches(
    source: str | Path,
    joint_destination: str | Path,
    decoupled_destination: str | Path,
) -> tuple[Path, Path]:
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != sha256_file(source_path):
        raise LineageError("Stage S source changed while cloning")
    destinations = (Path(joint_destination), Path(decoupled_destination))
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_bytes() != source_bytes:
                raise FileExistsError(f"existing Stage S clone differs: {destination}")
            continue
        temporary = destination.with_name(f".{destination.name}.writing")
        if temporary.exists():
            raise FileExistsError(f"stale Stage S clone write exists: {temporary}")
        temporary.write_bytes(source_bytes)
        os.replace(temporary, destination)
    if sha256_file(destinations[0]) != sha256_file(destinations[1]):
        raise LineageError("Stage S branch clone hashes differ")
    return destinations


__all__ = [
    "CHECKPOINT_SCHEMA",
    "TrainingCheckpointV42",
    "clone_stage_s_branches",
    "load_training_checkpoint",
    "optimizer_step_value",
    "save_training_checkpoint",
]
