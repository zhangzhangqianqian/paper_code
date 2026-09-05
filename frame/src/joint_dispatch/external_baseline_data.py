"""Causal, method-neutral data adapters for external RSC-PF baselines.

The adapter deliberately exposes labels separately from inference features. It
does not know anything about a particular external model and never reads a
test split. Normalization is fitted from the training split only and is
explicitly carried as a value object so a later runner can record its hash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal, Sequence

import numpy as np
import torch
from torch import Tensor

from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER
from .data import JointNormalization, JointWindowSplit, fit_joint_normalization, load_joint_split


@dataclass(frozen=True)
class ExternalBaselineBatch:
    """One causal batch shared by all external baseline wrappers."""

    load_history: Tensor
    exog_history: Tensor
    device_history: Tensor
    device_status: Tensor
    scheduler_context: Tensor
    previous_chp: Tensor
    forecast_target: Tensor
    teacher_dispatch: Tensor
    oracle_first_step_objective: Tensor
    split: Literal["train", "validation"]

    def __post_init__(self) -> None:
        tensors = {
            "load_history": self.load_history,
            "exog_history": self.exog_history,
            "device_history": self.device_history,
            "device_status": self.device_status,
            "scheduler_context": self.scheduler_context,
            "previous_chp": self.previous_chp,
            "forecast_target": self.forecast_target,
            "teacher_dispatch": self.teacher_dispatch,
            "oracle_first_step_objective": self.oracle_first_step_objective,
        }
        for name, value in tensors.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if not torch.is_floating_point(value):
                raise ValueError(f"{name} must be floating point")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain finite values")
        n = int(self.load_history.shape[0])
        expected = {
            "load_history": (24, len(TASK_ORDER)),
            "exog_history": (24, len(EXOG_ORDER)),
            "device_status": (24, len(STATUS_ORDER)),
            "forecast_target": (4, len(TASK_ORDER)),
            "scheduler_context": (4, 6),
            "teacher_dispatch": (4, len(DISPATCH_ORDER)),
        }
        for name, tail in expected.items():
            value = tensors[name]
            if value.ndim != 3 or tuple(value.shape[1:]) != tail:
                raise ValueError(f"{name} must have shape [B,{tail[0]},{tail[1]}]")
            if int(value.shape[0]) != n:
                raise ValueError(f"{name} has a different batch size")
        if self.device_history.ndim != 3 or tuple(self.device_history.shape[1:]) not in {
            (24, 17), (24, len(DISPATCH_ORDER))
        }:
            raise ValueError("device_history must have shape [B,24,17] for v4.6 or legacy [B,24,21]")
        if int(self.device_history.shape[0]) != n:
            raise ValueError("device_history has a different batch size")
        if self.previous_chp.ndim != 2 or tuple(self.previous_chp.shape) != (n, 1):
            raise ValueError("previous_chp must have shape [B,1]")
        if self.oracle_first_step_objective.ndim != 1 or int(self.oracle_first_step_objective.shape[0]) != n:
            raise ValueError("oracle_first_step_objective must have shape [B]")
        if self.split not in {"train", "validation"}:
            raise ValueError("external baselines accept train or validation only")
        if not torch.all((self.device_status == 0.0) | (self.device_status == 1.0)).item():
            raise ValueError("device_status must be binary")

    @classmethod
    def from_split(cls, split: JointWindowSplit, indices: Sequence[int] | np.ndarray | None = None) -> "ExternalBaselineBatch":
        """Convert a causal numpy split (or a subset) to tensors."""

        if split.split not in {"train", "validation"}:
            raise ValueError("external baselines cannot load the test split")
        index = None if indices is None else np.asarray(indices, dtype=np.int64)
        take = lambda value: value if index is None else value[index]
        return cls(
            load_history=torch.as_tensor(take(split.load_history), dtype=torch.float32),
            exog_history=torch.as_tensor(take(split.exog_history), dtype=torch.float32),
            device_history=torch.as_tensor(take(split.device_history), dtype=torch.float32),
            device_status=torch.as_tensor(take(split.device_status), dtype=torch.float32),
            scheduler_context=torch.as_tensor(take(split.scheduler_context), dtype=torch.float32),
            previous_chp=torch.as_tensor(take(split.previous_chp), dtype=torch.float32),
            forecast_target=torch.as_tensor(take(split.forecast_target), dtype=torch.float32),
            teacher_dispatch=torch.as_tensor(take(split.teacher_dispatch), dtype=torch.float32),
            oracle_first_step_objective=torch.as_tensor(take(split.oracle_first_step_objective), dtype=torch.float32),
            split=split.split,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ExternalNormalization:
    """Train-only affine statistics used by external adapters."""

    load_mean: np.ndarray
    load_scale: np.ndarray
    exog_mean: np.ndarray
    exog_scale: np.ndarray
    device_mean: np.ndarray
    device_scale: np.ndarray
    scheduler_mean: np.ndarray
    scheduler_scale: np.ndarray
    fitted_split: str = "train"

    def __post_init__(self) -> None:
        if self.fitted_split != "train":
            raise ValueError("external normalization must be fitted on train")
        for name in (
            "load_mean", "load_scale", "exog_mean", "exog_scale", "device_mean", "device_scale",
            "scheduler_mean", "scheduler_scale",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            if "scale" in name and np.any(value <= 0.0):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @classmethod
    def from_joint(cls, normalization: JointNormalization) -> "ExternalNormalization":
        if normalization.fitted_split != "train":
            raise ValueError("external normalization must be fitted on train")
        return cls(
            load_mean=normalization.load_mean, load_scale=normalization.load_scale,
            exog_mean=normalization.exog_mean, exog_scale=normalization.exog_scale,
            device_mean=normalization.device_mean, device_scale=normalization.device_scale,
            scheduler_mean=normalization.scheduler_mean, scheduler_scale=normalization.scheduler_scale,
        )

    def transform(self, split: JointWindowSplit) -> JointWindowSplit:
        if split.split not in {"train", "validation"}:
            raise ValueError("external normalization cannot transform the test split")
        apply = lambda value, mean, scale: ((value - mean.reshape((1, 1, -1))) / scale.reshape((1, 1, -1))).astype(np.float32)
        return JointWindowSplit(
            load_history=apply(split.load_history, self.load_mean, self.load_scale),
            exog_history=apply(split.exog_history, self.exog_mean, self.exog_scale),
            device_history=apply(split.device_history, self.device_mean, self.device_scale),
            device_status=split.device_status,
            forecast_target=apply(split.forecast_target, self.load_mean, self.load_scale),
            scheduler_context=apply(split.scheduler_context, self.scheduler_mean, self.scheduler_scale),
            previous_chp=split.previous_chp,
            teacher_dispatch=split.teacher_dispatch,
            oracle_first_step_objective=split.oracle_first_step_objective,
            target_times=split.target_times,
            split=split.split,
            history_source=split.history_source,
        )


def fit_external_normalization(train: JointWindowSplit) -> ExternalNormalization:
    """Fit all external statistics from the train split only."""

    return ExternalNormalization.from_joint(fit_joint_normalization(train))


def build_causal_error_history(split: JointWindowSplit) -> np.ndarray:
    """Return a history-only innovation proxy for policy adapters.

    The original Digital-Twins formulation uses historical forecast errors. The
    frozen 24-hour window artifact does not contain previous model forecasts, so
    this adapter exposes the reproducible causal quantity available to every
    method: the one-step load innovation within the observed history. It is
    explicitly not computed from ``forecast_target`` and is recorded as a
    proxy, not mislabeled as a future forecast error.
    """

    if split.split not in {"train", "validation"}:
        raise ValueError("causal error history cannot be built from the test split")
    history = np.asarray(split.load_history, dtype=np.float32)
    if history.ndim != 3 or history.shape[1:] != (24, len(TASK_ORDER)):
        raise ValueError("load_history must have shape [N,24,4]")
    result = np.zeros_like(history, dtype=np.float32)
    if history.shape[1] > 1:
        result[:, 1:, :] = history[:, 1:, :] - history[:, :-1, :]
    if not np.isfinite(result).all():
        raise ValueError("causal error history is not finite")
    return result


def assert_external_batch_causal(batch: ExternalBaselineBatch) -> None:
    """Re-run the public batch contract and reject test-like splits."""

    ExternalBaselineBatch(
        load_history=batch.load_history, exog_history=batch.exog_history,
        device_history=batch.device_history, device_status=batch.device_status,
        scheduler_context=batch.scheduler_context, previous_chp=batch.previous_chp,
        forecast_target=batch.forecast_target, teacher_dispatch=batch.teacher_dispatch,
        oracle_first_step_objective=batch.oracle_first_step_objective, split=batch.split,
    )


def load_external_batches(
    split_path: str | Path,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterable[ExternalBaselineBatch]:
    """Yield deterministic batches from a saved train/validation split."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    split, _normalization, _metadata = load_joint_split(split_path)
    if split.split not in {"train", "validation"}:
        raise ValueError("external baselines cannot load the test split")
    generator = np.random.default_rng(int(seed))
    indices = np.arange(len(split), dtype=np.int64)
    if shuffle:
        generator.shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        batch = ExternalBaselineBatch.from_split(split, indices[start : start + batch_size])
        assert_external_batch_causal(batch)
        yield batch


__all__ = [
    "ExternalBaselineBatch", "ExternalNormalization", "assert_external_batch_causal",
    "build_causal_error_history", "fit_external_normalization", "load_external_batches",
]
