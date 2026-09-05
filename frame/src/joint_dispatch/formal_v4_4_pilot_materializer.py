"""Causal data materialization and immutable cache for the formal-v4.4 Pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .formal_v4_4_artifacts import canonical_sha256, sha256_file, write_json_once, write_npz_once
from .formal_v4_4_contract import FormalV44Contract
from .formal_v4_data import FormalV4BaseSeries, FormalV4WindowSplit, materialize_state_windows
from .formal_v4_history import generate_settled_device_trajectory


_WINDOW_FIELDS = (
    "load_history", "exog_history", "renewable_history", "device_history",
    "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
    "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
    "target_times", "trajectory_ids", "state_hashes",
)
_ROLES = ("train", "early_stop", "selection_full", "selection_stress")


def _years(times: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(times, dtype="datetime64[Y]").astype(int) + 1970
    return tuple(sorted(set(int(value) for value in values)))


def _load_base(path: str | Path) -> FormalV4BaseSeries:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        required = {"load_and_exog", "renewable_forecast", "renewable_realized", "prices_and_weights", "timestamps", "split"}
        missing = sorted(required.difference(payload.files))
        if missing:
            raise ValueError(f"base archive is missing {missing[0]}")
        split = str(np.asarray(payload["split"]).item())
        return FormalV4BaseSeries(
            payload["load_and_exog"], payload["renewable_forecast"],
            payload["renewable_realized"], payload["prices_and_weights"],
            payload["timestamps"], split,
        )


def load_v44_base_series(train_data: str | Path, selection_data: str | Path) -> tuple[FormalV4BaseSeries, FormalV4BaseSeries]:
    """Load only the frozen 2015--2018 train and 2019 selection archives."""

    train = _load_base(train_data)
    selection = _load_base(selection_data)
    if train.split != "train" or selection.split != "selection":
        raise ValueError("formal-v4.4 base archives must be labelled train and selection")
    if _years(train.timestamps) != (2015, 2016, 2017, 2018):
        raise ValueError("training archive must contain exactly 2015-2018")
    if _years(selection.timestamps) != (2019,):
        raise ValueError("selection archive must contain exactly 2019")
    return train, selection


@dataclass(frozen=True)
class V44WindowCollection:
    """A materialized role with the fields consumed by v4.4 stages."""

    split: FormalV4WindowSplit
    role: str

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"unsupported v4.4 materialized role: {self.role}")
        if self.split.split != ("selection" if self.role.startswith("selection") else "train"):
            raise ValueError("window role and split label disagree")

    def __len__(self) -> int:
        return len(self.split)

    def __getattr__(self, name: str) -> Any:
        # Keep the public collection surface explicit while allowing the
        # existing normalization and batching adapters to consume it.
        try:
            return getattr(self.split, name)
        except AttributeError as exc:
            raise AttributeError(name) from exc

    @property
    def timestamps(self) -> np.ndarray:
        return self.split.target_times

    @property
    def scheduler_context(self) -> np.ndarray:
        return np.concatenate((
            self.split.renewable_forecast,
            self.split.prices_and_weights,
            np.repeat(self.split.initial_soc[:, None, :], 4, axis=1),
        ), axis=-1)


@dataclass(frozen=True)
class MaterializedV44PilotData:
    train: V44WindowCollection
    early_stop: V44WindowCollection
    selection_full: V44WindowCollection
    selection_stress: V44WindowCollection
    normalization_source: V44WindowCollection
    lineage: Mapping[str, Any]

    @property
    def selection(self) -> V44WindowCollection:
        return self.selection_full

    def roles(self) -> tuple[str, ...]:
        return _ROLES


def _subset(split: FormalV4WindowSplit, indices: np.ndarray, role: str) -> V44WindowCollection:
    values = {name: getattr(split, name)[indices] for name in _WINDOW_FIELDS}
    return V44WindowCollection(FormalV4WindowSplit(**values, split=split.split, history_source=split.history_source), role)


def _load_benchmark(path: str | Path) -> Mapping[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("values"), Mapping):
        raise ValueError("standard-IES benchmark must contain a values mapping")
    return payload


def _capacity_multiplier(payload: Mapping[str, Any]) -> float:
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        audit = payload.get("capacity_audit")
        selected = audit.get("selected") if isinstance(audit, Mapping) else None
    if not isinstance(selected, Mapping):
        value = payload.get("capacity_multiplier")
    else:
        value = selected.get("multiplier")
    multiplier = float(1.0 if value is None else value)
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("capacity receipt has no positive selected multiplier")
    return multiplier


def _capacity_payload(path: str | Path) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("capacity receipt must be a JSON object")
    status = payload.get("status") == "pass" or payload.get("gate0_authorized") is True
    if not status:
        raise PermissionError("formal-v4.4 materialization requires a passing capacity receipt")
    _capacity_multiplier(payload)
    return payload


def _lineage(
    *, train_data: Path, selection_data: Path, benchmark: Path, capacity_receipt: Path,
    split: Mapping[str, np.ndarray], contract: FormalV44Contract,
) -> dict[str, Any]:
    split_payload = {name: np.asarray(split[name], dtype=np.int64).tolist() for name in sorted(split)}
    payload: dict[str, Any] = {
        "schema": "formal-v4.4-materialized-lineage-v1",
        "contract_sha256": contract.contract_sha256,
        "train_data_sha256": sha256_file(train_data),
        "selection_data_sha256": sha256_file(selection_data),
        "benchmark_sha256": sha256_file(benchmark),
        "capacity_receipt_sha256": sha256_file(capacity_receipt),
        "split": split_payload,
        "evaluation_year_accessed": False,
    }
    payload["lineage_sha256"] = canonical_sha256(payload)
    return payload


def _arrays(collection: V44WindowCollection) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(collection.split, name)) for name in _WINDOW_FIELDS}


def _save_collection(root: Path, name: str, collection: V44WindowCollection) -> str:
    return write_npz_once(root / f"{name}.npz", _arrays(collection))


def _load_collection(root: Path, name: str, role: str) -> V44WindowCollection:
    path = root / f"{name}.npz"
    with np.load(path, allow_pickle=False) as payload:
        values = {field: payload[field] for field in _WINDOW_FIELDS}
    split_name = "selection" if role.startswith("selection") else "train"
    return V44WindowCollection(FormalV4WindowSplit(**values, split=split_name), role)


def save_or_load_materialized_v44(
    *, artifact_root: str | Path, lineage: Mapping[str, Any], built: MaterializedV44PilotData | None = None,
) -> MaterializedV44PilotData:
    """Persist or reuse an immutable cache, rejecting every lineage mismatch."""

    root = Path(artifact_root) / "data"
    receipt_path = root / "MATERIALIZED_LINEAGE.json"
    if receipt_path.exists():
        stored = json.loads(receipt_path.read_text(encoding="utf-8"))
        if stored != dict(lineage):
            raise ValueError("materialized cache lineage mismatch; use a new run ID")
        required = {name: root / f"{name}.npz" for name in _ROLES}
        required["normalization_source"] = root / "normalization_source.npz"
        if not all(path.is_file() for path in required.values()):
            raise ValueError("materialized cache is incomplete")
        return MaterializedV44PilotData(
            train=_load_collection(root, "train", "train"),
            early_stop=_load_collection(root, "early_stop", "early_stop"),
            selection_full=_load_collection(root, "selection_full", "selection_full"),
            selection_stress=_load_collection(root, "selection_stress", "selection_stress"),
            normalization_source=_load_collection(root, "normalization_source", "train"),
            lineage=stored,
        )
    if built is None:
        raise ValueError("materialized cache is absent and no build result was supplied")
    root.mkdir(parents=True, exist_ok=True)
    for name in _ROLES:
        _save_collection(root, name, getattr(built, name))
    _save_collection(root, "normalization_source", built.normalization_source)
    write_json_once(receipt_path, dict(lineage))
    return built


def materialize_v44_pilot_data(
    *, train_data: str | Path, selection_data: str | Path, benchmark: str | Path,
    capacity_receipt: str | Path, split: Mapping[str, np.ndarray], artifact_root: str | Path,
    contract: FormalV44Contract,
) -> MaterializedV44PilotData:
    """Generate causal device histories and apply the frozen Pilot indices."""

    contract.validate()
    required_split = {"train", "early_stop", "selection_full", "selection_stress"}
    if set(split) != required_split:
        raise ValueError("Pilot split must contain train, early_stop, selection_full, and selection_stress")
    train_base, selection_base = load_v44_base_series(train_data, selection_data)
    benchmark_payload = _load_benchmark(benchmark)
    capacity_payload = _capacity_payload(capacity_receipt)
    lineage = _lineage(
        train_data=Path(train_data), selection_data=Path(selection_data), benchmark=Path(benchmark),
        capacity_receipt=Path(capacity_receipt), split=split, contract=contract,
    )
    root = Path(artifact_root) / "data"
    if (root / "MATERIALIZED_LINEAGE.json").exists():
        return save_or_load_materialized_v44(artifact_root=artifact_root, lineage=lineage)

    parameters = dict(benchmark_payload["values"])
    multiplier = _capacity_multiplier(capacity_payload)
    trajectory_train = generate_settled_device_trajectory(
        train_base, parameters, capacity_receipt=capacity_receipt, trajectory_id="formal-v4.4-train",
    )
    trajectory_selection = generate_settled_device_trajectory(
        selection_base, parameters, capacity_receipt=capacity_receipt, trajectory_id="formal-v4.4-selection",
    )
    train_windows = materialize_state_windows(
        train_base, trajectory_train.settled_dispatch, capacity_receipt=capacity_receipt,
        split="train", bess_energy_capacity=float(parameters["bess_energy_capacity"]) * multiplier,
        settled_mask=trajectory_train.settled_mask, trajectory_hash=trajectory_train.trajectory_sha256,
    )
    selection_windows = materialize_state_windows(
        selection_base, trajectory_selection.settled_dispatch, capacity_receipt=capacity_receipt,
        split="selection", bess_energy_capacity=float(parameters["bess_energy_capacity"]) * multiplier,
        settled_mask=trajectory_selection.settled_mask, trajectory_hash=trajectory_selection.trajectory_sha256,
    )
    indices = {name: np.asarray(values, dtype=np.int64) for name, values in split.items()}
    if len(indices["train"]) != contract.pilot_train_windows:
        raise ValueError("formal-v4.4 Pilot training split must contain exactly 4096 windows")
    for name, values in indices.items():
        source = train_windows if name in {"train", "early_stop"} else selection_windows
        if values.ndim != 1 or len(values) == 0 or np.any(values < 0) or np.any(values >= len(source)):
            raise ValueError(f"Pilot split {name} is outside the materialized source")
    built = MaterializedV44PilotData(
        train=_subset(train_windows, indices["train"], "train"),
        early_stop=_subset(train_windows, indices["early_stop"], "early_stop"),
        selection_full=_subset(selection_windows, indices["selection_full"], "selection_full"),
        selection_stress=_subset(selection_windows, indices["selection_stress"], "selection_stress"),
        normalization_source=V44WindowCollection(train_windows, "train"),
        lineage=lineage,
    )
    return save_or_load_materialized_v44(artifact_root=artifact_root, lineage=lineage, built=built)


__all__ = [
    "MaterializedV44PilotData", "V44WindowCollection", "load_v44_base_series",
    "materialize_v44_pilot_data", "save_or_load_materialized_v44",
]
