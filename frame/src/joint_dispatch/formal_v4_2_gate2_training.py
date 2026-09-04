"""Frozen data manifests and training batches for formal-v4.2 Gate 2."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from .formal_v4_2_artifacts import canonical_sha256, sha256_file
from .formal_v4_2_contract import FormalV42Contract
from .formal_v4_2_data import NormalizationReceiptV42, apply_normalization
from .formal_v4_data import FormalV4WindowSplit


_WINDOW_FIELDS = (
    "load_history", "exog_history", "renewable_history", "device_history",
    "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
    "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
    "target_times", "trajectory_ids", "state_hashes",
)


@dataclass(frozen=True)
class Gate2DataBundle:
    train: FormalV4WindowSplit
    calibration: FormalV4WindowSplit
    evaluation: FormalV4WindowSplit
    normalization: NormalizationReceiptV42
    train_manifest_sha256: str
    calibration_manifest_sha256: str
    evaluation_manifest_sha256: str
    normalization_sha256: str


@dataclass(frozen=True)
class Gate2BatchPart:
    batch: Mapping[str, torch.Tensor]
    effective_batch_index: int
    accumulation_weight: float
    indices: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def load_window_split(path: str | Path) -> FormalV4WindowSplit:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        split = str(np.asarray(payload["split"]).item())
        values = {name: payload[name] for name in _WINDOW_FIELDS}
    return FormalV4WindowSplit(**values, split=split)


def _load_normalization(path: Path) -> NormalizationReceiptV42:
    payload = _load_json(path)
    if payload.get("schema") != "formal-v4.2-normalization-receipt-v1":
        raise ValueError("Gate 2 normalization schema is invalid")
    return NormalizationReceiptV42(
        train_years=tuple(int(value) for value in payload["train_years"]),
        field_mean={key: np.asarray(value, dtype=np.float32) for key, value in payload["field_mean"].items()},
        field_scale={key: np.asarray(value, dtype=np.float32) for key, value in payload["field_scale"].items()},
        zero_scale_mask={key: np.asarray(value, dtype=bool) for key, value in payload["zero_scale_mask"].items()},
        train_data_sha256=str(payload["train_data_sha256"]),
        receipt_sha256=str(payload["receipt_sha256"]),
    )


def _subset(split: FormalV4WindowSplit, indices: np.ndarray) -> FormalV4WindowSplit:
    return replace(split, **{name: getattr(split, name)[indices] for name in _WINDOW_FIELDS})


def _manifest_hash(split: FormalV4WindowSplit, scope: str) -> str:
    return canonical_sha256({
        "scope": scope,
        "count": len(split),
        "target_times": split.target_times.astype("datetime64[ns]").astype(np.int64).tolist(),
        "trajectory_ids": split.trajectory_ids.tolist(),
        "state_hashes": split.state_hashes.tolist(),
    })


def _years(split: FormalV4WindowSplit) -> tuple[int, ...]:
    values = np.datetime_as_string(split.target_times, unit="Y").astype(np.int64)
    return tuple(int(value) for value in np.unique(values))


def load_gate2_data(run_root: str | Path, contract: FormalV42Contract) -> Gate2DataBundle:
    """Load the Gate 1 materialization without sampling formal train/evaluation."""

    contract.validate()
    root = Path(run_root).resolve()
    transition = _load_json(root / "protocol" / "GATE1_TRANSITION.json")
    if transition.get("authorized_gate2") is not True:
        raise PermissionError("Gate 2 requires an authorized Gate 1 transition")
    if transition.get("contract_sha256") != contract.contract_sha256:
        raise PermissionError("Gate 1 contract lineage differs from Gate 2")
    gate1 = root / "gate1"
    evidence_path = gate1 / "GATE1_EVIDENCE.json"
    evidence = _load_json(evidence_path)
    if transition.get("gate1_evidence_sha256") != sha256_file(evidence_path):
        raise PermissionError("Gate 1 evidence hash mismatch")
    if evidence.get("evaluation_year_accessed") is not False:
        raise PermissionError("Gate 1 accessed the evaluation year")
    train_path = gate1 / "TRAIN_WINDOWS.npz"
    selection_path = gate1 / "SELECTION_WINDOWS.npz"
    normalization_path = gate1 / "NORMALIZATION.json"
    if evidence.get("train_windows_sha256") != sha256_file(train_path):
        raise PermissionError("Gate 1 train windows hash mismatch")
    if evidence.get("selection_windows_sha256") != sha256_file(selection_path):
        raise PermissionError("Gate 1 selection windows hash mismatch")
    train = load_window_split(train_path)
    selection = load_window_split(selection_path)
    if _years(train) != contract.train_years:
        raise PermissionError("Gate 2 train manifest does not cover exactly 2015-2018")
    if _years(selection) != (contract.selection_year,):
        raise PermissionError("Gate 2 evaluation manifest is not exactly 2019")
    if len(selection) > 1 and not np.all(np.diff(selection.target_times) > np.timedelta64(0, "s")):
        raise ValueError("Gate 2 evaluation must be chronological")
    origin_path = gate1 / "GATE1_ORIGIN_MANIFEST.json"
    origin_payload = _load_json(origin_path)
    origin_indices = np.asarray(origin_payload["origin_indices"], dtype=np.int64)
    if len(origin_indices) != int(contract.gate2_budget["calibration_origin_count"]):
        raise ValueError("Gate 1 calibration origin count differs from the frozen budget")
    if origin_indices.min(initial=0) < 0 or origin_indices.max(initial=-1) >= len(selection):
        raise ValueError("Gate 1 calibration origin index is outside selection data")
    calibration = _subset(selection, origin_indices)
    normalization = _load_normalization(normalization_path)
    if normalization.train_years != contract.train_years:
        raise PermissionError("Gate 2 normalization is not train-only")
    return Gate2DataBundle(
        train=train,
        calibration=calibration,
        evaluation=selection,
        normalization=normalization,
        train_manifest_sha256=_manifest_hash(train, "all_eligible_2015_2018"),
        calibration_manifest_sha256=sha256_file(origin_path),
        evaluation_manifest_sha256=_manifest_hash(selection, "all_eligible_2019_chronology"),
        normalization_sha256=sha256_file(normalization_path),
    )


def _batch(split: FormalV4WindowSplit, normalization: NormalizationReceiptV42, indices: np.ndarray) -> dict[str, torch.Tensor]:
    normalized = apply_normalization(_subset(split, indices), normalization)
    selected = _subset(split, indices)
    return {
        "load_history": torch.from_numpy(normalized.load_history),
        "exog_history": torch.from_numpy(normalized.exog_history),
        "device_history": torch.from_numpy(normalized.device_history),
        "activity_history": torch.from_numpy(normalized.activity_history),
        "scheduler_context": torch.from_numpy(normalized.scheduler_context),
        "previous_chp": torch.as_tensor(selected.previous_chp, dtype=torch.float64),
        "initial_soc": torch.as_tensor(selected.initial_soc, dtype=torch.float64),
        "target_normalized": torch.from_numpy(normalized.target_normalized),
        "target_physical": torch.as_tensor(selected.forecast_target, dtype=torch.float64),
        "realized_renewables": torch.as_tensor(selected.renewable_realized, dtype=torch.float64),
    }


def iter_gate2_batches(
    split: FormalV4WindowSplit,
    normalization: NormalizationReceiptV42,
    *,
    effective_batch_size: int = 64,
    micro_batch_size: int | None = None,
) -> Iterator[Gate2BatchPart]:
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    part_size = effective_batch_size if micro_batch_size is None else int(micro_batch_size)
    if part_size <= 0 or effective_batch_size % part_size != 0:
        raise ValueError("micro_batch_size must be a positive divisor of effective_batch_size")
    effective_index = 0
    for start in range(0, len(split), effective_batch_size):
        stop = min(start + effective_batch_size, len(split))
        group = np.arange(start, stop, dtype=np.int64)
        parts = [group[offset:offset + part_size] for offset in range(0, len(group), part_size)]
        for indices in parts:
            yield Gate2BatchPart(
                batch=_batch(split, normalization, indices),
                effective_batch_index=effective_index,
                accumulation_weight=float(len(indices) / len(group)),
                indices=indices,
            )
        effective_index += 1


def full_gate2_batches(
    split: FormalV4WindowSplit,
    normalization: NormalizationReceiptV42,
    *,
    batch_size: int = 64,
) -> list[Mapping[str, torch.Tensor]]:
    return [
        part.batch for part in iter_gate2_batches(
            split, normalization, effective_batch_size=batch_size,
        )
    ]


__all__ = [
    "Gate2BatchPart",
    "Gate2DataBundle",
    "full_gate2_batches",
    "iter_gate2_batches",
    "load_gate2_data",
    "load_window_split",
]

