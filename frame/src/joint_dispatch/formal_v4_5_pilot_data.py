"""Explicit train/early-stop loader adapter for formal-v4.5."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .formal_v4_4_artifacts import canonical_sha256, sha256_file
from .formal_v4_4_pilot_materializer import (
    V44WindowCollection, _capacity_multiplier, _capacity_payload, _load_base,
    _load_benchmark, _load_collection, _subset,
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


@dataclass(frozen=True)
class MaterializedPilotV45:
    """Train-only artifacts plus the permitted 2019 selection view."""

    train: V44WindowCollection
    early_stop: V44WindowCollection
    selection_full: V44WindowCollection
    selection_stress: V44WindowCollection
    normalization_source: V44WindowCollection
    lineage: Mapping[str, Any]

    def roles(self) -> tuple[str, ...]:
        return ("train", "early_stop", "selection_full", "selection_stress")


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


def load_v45_training_cache(
    *,
    materialized_root: str | Path,
    train_data: str | Path | None = None,
    benchmark: str | Path | None = None,
    capacity_receipt: str | Path | None = None,
) -> MaterializedTrainingV45:
    """Load only immutable train/early-stop artifacts from a prior cache.

    The v4.4 materializer already solved the expensive causal LP trajectory.
    Reusing its train-only artifacts is equivalent data, but this loader never
    opens selection files and records that fact in the returned lineage.
    """

    root = Path(materialized_root)
    data_root = root / "data"
    required = ("train", "early_stop", "normalization_source")
    missing = [name for name in required if not (data_root / f"{name}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"training cache is missing {missing[0]}.npz")
    train = _load_collection(data_root, "train", "train")
    early_stop = _load_collection(data_root, "early_stop", "early_stop")
    normalization_source = _load_collection(data_root, "normalization_source", "train")

    stored_path = data_root / "MATERIALIZED_LINEAGE.json"
    stored = json.loads(stored_path.read_text(encoding="utf-8")) if stored_path.is_file() else {}
    if not isinstance(stored, Mapping):
        raise ValueError("training cache lineage must be a mapping")
    checks = {
        "train_data": train_data,
        "benchmark": benchmark,
        "capacity_receipt": capacity_receipt,
    }
    for name, path in checks.items():
        if path is None:
            continue
        key = f"{name}_sha256"
        if key in stored and stored[key] != sha256_file(path):
            raise ValueError(f"training cache {name} hash does not match requested source")
    years = tuple(sorted(set(normalization_source.timestamps.astype("datetime64[Y]").astype(int) + 1970)))
    if years != (2015, 2016, 2017, 2018):
        raise ValueError("training cache contains years outside 2015--2018")
    lineage = {
        "schema": "formal-v4.5-training-cache-lineage-v1",
        "source_materialized_lineage_sha256": stored.get("lineage_sha256"),
        "source_materialized_root": str(root),
        "years": list(years),
        "selection_files_read": [],
        "selection_year_accessed": False,
        "evaluation_year_accessed": False,
    }
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return MaterializedTrainingV45(
        train=train,
        early_stop=early_stop,
        normalization_source=normalization_source,
        lineage=lineage,
    )


def load_v45_pilot_cache(
    *,
    materialized_root: str | Path,
    train_data: str | Path | None = None,
    benchmark: str | Path | None = None,
    capacity_receipt: str | Path | None = None,
) -> MaterializedPilotV45:
    """Load the verified train cache and the explicitly permitted 2019 view."""

    training = load_v45_training_cache(
        materialized_root=materialized_root, train_data=train_data,
        benchmark=benchmark, capacity_receipt=capacity_receipt,
    )
    data_root = Path(materialized_root) / "data"
    for name in ("selection_full", "selection_stress"):
        if not (data_root / f"{name}.npz").is_file():
            raise FileNotFoundError(data_root / f"{name}.npz")
    selection_full = _load_collection(data_root, "selection_full", "selection_full")
    selection_stress = _load_collection(data_root, "selection_stress", "selection_stress")
    selection_years = tuple(sorted(set(selection_full.timestamps.astype("datetime64[Y]").astype(int) + 1970)))
    if selection_years != (2019,):
        raise ValueError("Pilot selection cache must contain 2019 only")
    lineage = dict(training.lineage)
    lineage.update({
        "schema": "formal-v4.5-pilot-cache-lineage-v1",
        "selection_files_read": ["selection_full.npz", "selection_stress.npz"],
        "selection_year_accessed": True,
        "evaluation_year_accessed": False,
        "selection_years": list(selection_years),
    })
    lineage["lineage_sha256"] = canonical_sha256(lineage)
    return MaterializedPilotV45(
        train=training.train, early_stop=training.early_stop,
        selection_full=selection_full, selection_stress=selection_stress,
        normalization_source=training.normalization_source, lineage=lineage,
    )


__all__ = [
    "MaterializedPilotV45", "MaterializedTrainingV45", "build_v45_loaders", "load_v45_pilot_cache", "load_v45_training_cache",
    "materialize_v45_training_only",
]
