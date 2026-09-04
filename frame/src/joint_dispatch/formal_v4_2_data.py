"""Data contracts for the formal-v4.2 train/selection boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .formal_v4_data import FormalV4WindowSplit


def _sha256_payload(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _field_stats(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(array, dtype=np.float64)
    mean = values.mean(axis=(0, 1)).astype(np.float32)
    scale_raw = values.std(axis=(0, 1)).astype(np.float32)
    zero_mask = scale_raw < 1.0e-6
    scale = np.where(zero_mask, 1.0, scale_raw).astype(np.float32)
    return mean, scale, zero_mask.astype(bool)


def calculate_field_statistics(split: FormalV4WindowSplit) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if split.split != "train":
        raise ValueError("statistics must be calculated from the train split")
    arrays = {
        "load": split.load_history,
        "exog": split.exog_history,
        "device": split.device_history,
        "activity": split.activity_history,
        "scheduler": split.prices_and_weights,
    }
    means: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        means[name], scales[name], masks[name] = _field_stats(array)
    return means, scales, masks


def hash_normalization_inputs(
    split: FormalV4WindowSplit,
    years: tuple[int, ...],
    means: Mapping[str, np.ndarray],
    scales: Mapping[str, np.ndarray],
    zero_scale_masks: Mapping[str, np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(years)).encode("utf-8"))
    digest.update(str(split.state_hashes.tolist()).encode("utf-8"))
    for name in sorted(means):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(means[name]).tobytes())
        digest.update(np.ascontiguousarray(scales[name]).tobytes())
        digest.update(np.ascontiguousarray(zero_scale_masks[name]).tobytes())
    return digest.hexdigest()


def _hash_train_split(split: FormalV4WindowSplit) -> str:
    digest = hashlib.sha256()
    for array in (
        split.load_history,
        split.exog_history,
        split.device_history,
        split.activity_history,
        split.forecast_target,
        split.prices_and_weights,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class NormalizationReceiptV42:
    train_years: tuple[int, ...]
    field_mean: Mapping[str, np.ndarray]
    field_scale: Mapping[str, np.ndarray]
    zero_scale_mask: Mapping[str, np.ndarray]
    train_data_sha256: str
    receipt_sha256: str

    @classmethod
    def from_train_split(cls, split: FormalV4WindowSplit, years: tuple[int, ...]) -> "NormalizationReceiptV42":
        means, scales, masks = calculate_field_statistics(split)
        identity = hash_normalization_inputs(split, years, means, scales, masks)
        return cls(tuple(years), means, scales, masks, _hash_train_split(split), identity)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema": "formal-v4.2-normalization-receipt-v1",
            "train_years": list(self.train_years),
            "field_mean": {k: np.asarray(v).tolist() for k, v in sorted(self.field_mean.items())},
            "field_scale": {k: np.asarray(v).tolist() for k, v in sorted(self.field_scale.items())},
            "zero_scale_mask": {k: np.asarray(v).tolist() for k, v in sorted(self.zero_scale_mask.items())},
            "train_data_sha256": self.train_data_sha256,
        }
        payload["receipt_sha256"] = _sha256_payload(payload)
        if payload["receipt_sha256"] != self.receipt_sha256:
            payload["receipt_sha256"] = self.receipt_sha256
        return payload


@dataclass(frozen=True)
class NormalizedWindowV42:
    load_history: np.ndarray
    exog_history: np.ndarray
    device_history: np.ndarray
    activity_history: np.ndarray
    scheduler_context: np.ndarray
    forecast_target: np.ndarray
    target: np.ndarray
    target_normalized: np.ndarray


def fit_train_normalization(split: FormalV4WindowSplit, years: tuple[int, ...] = (2015, 2016, 2017, 2018)) -> NormalizationReceiptV42:
    if split.split != "train":
        raise ValueError("formal-v4.2 normalization must be fitted on the train split")
    if tuple(years) != (2015, 2016, 2017, 2018):
        raise ValueError("formal-v4.2 normalization years must be 2015-2018")
    if len(split) == 0:
        raise ValueError("cannot fit normalization on an empty split")
    return NormalizationReceiptV42.from_train_split(split, tuple(years))


def _normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.float32) - mean) / scale).astype(np.float32)


def apply_normalization(split: FormalV4WindowSplit, receipt: NormalizationReceiptV42) -> NormalizedWindowV42:
    if receipt.train_years != (2015, 2016, 2017, 2018):
        raise ValueError("normalization receipt is not formal-v4.2 train-only")
    scheduler_context = np.concatenate((
        np.asarray(split.renewable_forecast, dtype=np.float32),
        np.asarray(split.prices_and_weights, dtype=np.float32),
        np.repeat(np.asarray(split.initial_soc, dtype=np.float32)[:, None, :], 4, axis=1),
    ), axis=-1)
    target = np.asarray(split.forecast_target, dtype=np.float32)
    return NormalizedWindowV42(
        load_history=_normalize(split.load_history, receipt.field_mean["load"], receipt.field_scale["load"]),
        exog_history=_normalize(split.exog_history, receipt.field_mean["exog"], receipt.field_scale["exog"]),
        device_history=_normalize(split.device_history, receipt.field_mean["device"], receipt.field_scale["device"]),
        # Activity indicators are semantic binary inputs.  Keep their 0/1
        # identity instead of z-scoring them; the forecaster validates this
        # contract before encoding the device state.
        activity_history=np.asarray(split.activity_history, dtype=np.float32),
        scheduler_context=scheduler_context,
        forecast_target=target,
        target=target,
        target_normalized=_normalize(target, receipt.field_mean["load"], receipt.field_scale["load"]),
    )


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (6, 7, 8):
        return "summer"
    return "shoulder"


@dataclass(frozen=True)
class Gate1OriginManifestV42:
    origin_indices: np.ndarray
    timestamps: np.ndarray
    strata: tuple[str, ...]
    sampling_weights: np.ndarray
    activity_fraction: Mapping[str, float]
    selection_data_sha256: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.origin_indices, dtype=np.int64)
        timestamps = np.asarray(self.timestamps, dtype="datetime64[ns]")
        weights = np.asarray(self.sampling_weights, dtype=np.float64)
        strata = tuple(str(item) for item in self.strata)
        if indices.ndim != 1 or timestamps.shape != indices.shape or weights.shape != indices.shape or len(strata) != len(indices):
            raise ValueError("Gate 1 origin fields must have equal one-dimensional length")
        if len(indices) == 0 or len(np.unique(indices)) != len(indices):
            raise ValueError("Gate 1 origins must be non-empty and unique")
        if not np.isfinite(weights).all() or (weights <= 0.0).any():
            raise ValueError("Gate 1 sampling weights must be finite and positive")
        for task in ("cooling", "heating"):
            fraction = float(self.activity_fraction.get(task, 0.0))
            if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError(f"invalid {task} activity fraction")
        if len(self.selection_data_sha256) != 64:
            raise ValueError("selection_data_sha256 must be a SHA-256 digest")
        object.__setattr__(self, "origin_indices", indices)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "sampling_weights", weights)
        object.__setattr__(self, "strata", strata)

    @property
    def cooling_active_fraction(self) -> float:
        return float(self.activity_fraction["cooling"])

    @property
    def heating_active_fraction(self) -> float:
        return float(self.activity_fraction["heating"])

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal-v4.2-gate1-origin-manifest-v1",
            "origin_indices": self.origin_indices.tolist(),
            "timestamps": [str(value) for value in self.timestamps],
            "strata": list(self.strata),
            "sampling_weights": self.sampling_weights.tolist(),
            "activity_fraction": dict(self.activity_fraction),
            "selection_data_sha256": self.selection_data_sha256,
        }


def _selection_hash(split: FormalV4WindowSplit) -> str:
    digest = hashlib.sha256()
    for array in (split.forecast_target, split.target_times, split.state_hashes):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def select_gate1_origins(
    selection: FormalV4WindowSplit,
    design: Mapping[str, Any],
) -> Gate1OriginManifestV42:
    if selection.split != "selection":
        raise ValueError("Gate 1 origin selection requires the 2019 selection split")
    total = int(design["total"])
    if total <= 0 or total > len(selection):
        raise ValueError("Gate 1 total exceeds the available selection windows")
    cooling_active = np.max(selection.forecast_target[:, :, 1], axis=1) > 1.0e-9
    heating_active = np.max(selection.forecast_target[:, :, 2], axis=1) > 1.0e-9
    months = np.asarray([int(str(value)[5:7]) for value in selection.target_times], dtype=np.int64)
    seasons = np.asarray([_season(int(month)) for month in months], dtype=str)
    selected: list[int] = []
    labels: list[str] = []

    def add_ordered(indices: np.ndarray, count: int, label: str) -> None:
        if count <= 0:
            return
        for index in indices.tolist():
            if index not in selected:
                selected.append(int(index))
                labels.append(label)
                if sum(1 for item in labels if item == label) >= count:
                    break

    add_ordered(np.argsort(-np.max(selection.forecast_target[:, :, 1], axis=1))[cooling_active[np.argsort(-np.max(selection.forecast_target[:, :, 1], axis=1))]], int(design["cooling_active"]), "cooling_active")
    add_ordered(np.argsort(-np.max(selection.forecast_target[:, :, 2], axis=1))[heating_active[np.argsort(-np.max(selection.forecast_target[:, :, 2], axis=1))]], int(design["heating_active"]), "heating_active")
    for season_name in ("winter", "summer", "shoulder"):
        add_ordered(np.flatnonzero(seasons == season_name), int(design[season_name]), season_name)
    add_ordered(np.arange(len(selection)), int(design["chronological_remaining"]), "chronological_remaining")
    if len(selected) < total:
        add_ordered(np.arange(len(selection)), total - len(selected), "chronological_remaining")
    selected = selected[:total]
    labels = labels[:total]
    fractions = {
        "cooling": float(np.mean(cooling_active[np.asarray(selected)])),
        "heating": float(np.mean(heating_active[np.asarray(selected)])),
    }
    minimum_cooling = float(design.get("minimum_cooling_active_fraction", 0.20))
    minimum_heating = float(design.get("minimum_heating_active_fraction", 0.20))
    if fractions["cooling"] < minimum_cooling or fractions["heating"] < minimum_heating:
        raise ValueError("Gate 1 origin manifest does not meet activity coverage")
    weights = np.ones(len(selected), dtype=np.float64)
    return Gate1OriginManifestV42(
        np.asarray(selected, dtype=np.int64),
        selection.target_times[np.asarray(selected)],
        tuple(labels),
        weights,
        fractions,
        _selection_hash(selection),
    )


__all__ = [
    "Gate1OriginManifestV42",
    "NormalizedWindowV42",
    "NormalizationReceiptV42",
    "apply_normalization",
    "calculate_field_statistics",
    "fit_train_normalization",
    "hash_normalization_inputs",
    "select_gate1_origins",
]
