"""Explicit train/early-stop loader adapter for formal-v4.5."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .formal_v4_4_pilot_data import build_v44_batches


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


__all__ = ["build_v45_loaders"]
