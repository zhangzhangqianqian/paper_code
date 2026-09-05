"""Explicit train/early-stop loader adapter for formal-v4.5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .formal_v4_4_artifacts import canonical_sha256, sha256_file
from .formal_v4_4_pilot_materializer import (
    V44WindowCollection, _capacity_multiplier, _capacity_payload, _load_base,
    _load_benchmark, _subset,
)
from .formal_v4_4_pilot_data import build_v44_batches
from .formal_v4_data import materialize_state_windows
from .formal_v4_history import generate_settled_device_trajectory


@dataclass(frozen=True)
class MaterializedTrainingV45:
    train: V44WindowCollection
    early_stop: V44WindowCollection
    normalization_source: V44WindowCollection
    lineage: Mapping[str, Any]

    def roles(self) -> tuple[str, ...]:
        return ("train", "early_stop")


def _teacher_dispatch(teacher: Any, role: str, expected: int) -> np.ndarray | None:
    if teacher is None:
        return None
    value = teacher.get(role) if isinstance(teacher, Mapping) else getattr(teacher, f"{role}_dispatch", None)
    if value is None and hasattr(teacher, "dispatch"):
        value = getattr(teacher, "dispatch")
    if value is None:
        raise ValueError(f"v4.5 teacher is missing {role} dispatch")
    array = np.asarray(value)
    if array.shape != (expected, 4, 21):
        raise ValueError(f"v4.5 {role} teacher dispatch has shape {array.shape}, expected {(expected, 4, 21)}")
    return array


def _with_origin_ids(batches: list[dict[str, Any]], offset: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for batch in batches:
        item = dict(batch)
        sample_indices = np.asarray(item["sample_indices"], dtype=np.int64)
        item["origin_index"] = sample_indices + int(offset)
        result.append(item)
    return result


def build_v45_loaders(
    materialized: Any,
    normalization: Any,
    *,
    batch_size: int,
    teacher: Any | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build separate causal train and early-stop batches.

    The role offset is an artifact-local identifier only; it is never used as a
    model feature. It makes accidental train/validation mixing auditable.
    """

    train = materialized.train
    early_stop = materialized.early_stop
    train_teacher = _teacher_dispatch(teacher, "train", len(train))
    early_teacher = _teacher_dispatch(teacher, "early_stop", len(early_stop))
    train_batches = build_v44_batches(
        train, normalization, np.arange(len(train), dtype=np.int64),
        batch_size=batch_size, teacher_dispatch=train_teacher,
    )
    early_batches = build_v44_batches(
        early_stop, normalization, np.arange(len(early_stop), dtype=np.int64),
        batch_size=batch_size, teacher_dispatch=early_teacher,
    )
    return {
        "train": _with_origin_ids(train_batches, 0),
        "early_stop": _with_origin_ids(early_batches, len(train)),
    }


def materialize_v45_training_only(
    *,
    train_data: str | Path,
    benchmark: str | Path,
    capacity_receipt: str | Path,
    split: Mapping[str, np.ndarray],
    artifact_root: str | Path,
) -> MaterializedTrainingV45:
    """Materialize only 2015--2018; never open a selection archive."""

    required = {"train", "early_stop"}
    if set(split) != required:
        raise ValueError("v4.5 training-only split must contain train and early_stop")
    base = _load_base(train_data)
    years = tuple(sorted(set(base.timestamps.astype("datetime64[Y]").astype(int) + 1970)))
    if years != (2015, 2016, 2017, 2018):
        raise ValueError("v4.5 training archive must contain exactly 2015--2018")
    benchmark_payload = _load_benchmark(benchmark)
    capacity_payload = _capacity_payload(capacity_receipt)
    parameters = dict(benchmark_payload["values"])
    multiplier = _capacity_multiplier(capacity_payload)
    trajectory = generate_settled_device_trajectory(
        base, parameters, capacity_receipt=capacity_receipt,
        trajectory_id="formal-v4.5-training-only",
    )
    windows = materialize_state_windows(
        base, trajectory.settled_dispatch, capacity_receipt=capacity_receipt,
        split="train", bess_energy_capacity=float(parameters["bess_energy_capacity"]) * multiplier,
        settled_mask=trajectory.settled_mask, trajectory_hash=trajectory.trajectory_sha256,
    )
    train_indices = np.asarray(split["train"], dtype=np.int64)
    early_indices = np.asarray(split["early_stop"], dtype=np.int64)
    for name, indices in (("train", train_indices), ("early_stop", early_indices)):
        if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(windows)):
            raise ValueError(f"v4.5 {name} split is outside the training windows")
    lineage = {
        "schema": "formal-v4.5-training-only-lineage-v1",
        "train_data_sha256": sha256_file(train_data),
        "benchmark_sha256": sha256_file(benchmark),
        "capacity_receipt_sha256": sha256_file(capacity_receipt),
        "years": list(years),
        "evaluation_year_accessed": False,
        "selection_year_accessed": False,
        "split": {"train": train_indices.tolist(), "early_stop": early_indices.tolist()},
    }
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return MaterializedTrainingV45(
        train=_subset(windows, train_indices, "train"),
        early_stop=_subset(windows, early_indices, "early_stop"),
        normalization_source=V44WindowCollection(windows, "train"),
        lineage=lineage,
    )


__all__ = ["MaterializedTrainingV45", "build_v45_loaders", "materialize_v45_training_only"]
