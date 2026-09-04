"""Immutable artifact and provenance receipts for formal-v4.1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .formal_v4_objective import STEP_WEIGHTS, TrainingObjectiveScale, fit_training_objective_scale
from .formal_v4_data import FormalV4Normalization, FormalV4WindowSplit
from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from .contract import DISPATCH_ORDER


ARTIFACT_MANIFEST_SCHEMA = "formal-v4.1-artifact-manifest-v1"
NORMALIZATION_RECEIPT_SCHEMA = "formal-v4.1-normalization-receipt-v1"
C_REF_RECEIPT_SCHEMA = "formal-v4.1-c-ref-receipt-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal-v4 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class ArtifactManifest:
    files: Mapping[str, str]
    lineage: Mapping[str, Any]
    schema_version: str = ARTIFACT_MANIFEST_SCHEMA
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        files = {str(key): str(value) for key, value in self.files.items()}
        if not files or any(len(value) != 64 for value in files.values()):
            raise ValueError("artifact manifest must contain SHA-256 hashes")
        object.__setattr__(self, "files", files)
        payload = {"schema_version": self.schema_version, "files": dict(sorted(files.items())), "lineage": dict(self.lineage)}
        digest = canonical_sha256(payload)
        if self.manifest_sha256 and self.manifest_sha256 != digest:
            raise ValueError("artifact manifest hash does not match payload")
        object.__setattr__(self, "manifest_sha256", digest)

    def to_payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "files": dict(sorted(self.files.items())), "lineage": dict(self.lineage), "manifest_sha256": self.manifest_sha256}

    def save(self, path: str | Path) -> None:
        _write_once(Path(path), self.to_payload())


def build_artifact_manifest(root: str | Path, files: Sequence[str | Path], *, lineage: Mapping[str, Any] | None = None) -> ArtifactManifest:
    root_path = Path(root).resolve()
    hashes: dict[str, str] = {}
    for raw in files:
        path = Path(raw)
        if not path.is_absolute():
            path = root_path / path
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root_path)
        except ValueError as exc:
            raise ValueError("artifact path escapes run root") from exc
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        key = relative.as_posix()
        if key in hashes:
            raise ValueError("artifact manifest contains duplicate paths")
        hashes[key] = sha256_file(resolved)
    return ArtifactManifest(hashes, dict(lineage or {}))


def build_normalization_receipt(
    normalization: FormalV4Normalization,
    *,
    train_archive_sha256: str,
    capacity_receipt_sha256: str,
    train_timestamps: np.ndarray,
) -> dict[str, Any]:
    timestamps = np.asarray(train_timestamps, dtype="datetime64[ns]")
    if timestamps.ndim != 1 or timestamps.size == 0:
        raise ValueError("normalization receipt requires non-empty train timestamps")
    if not train_archive_sha256 or not capacity_receipt_sha256:
        raise ValueError("normalization receipt requires archive and capacity hashes")
    payload: dict[str, Any] = {
        "schema_version": NORMALIZATION_RECEIPT_SCHEMA,
        "fitted_split": normalization.fitted_split,
        "train_timestamp_start": str(timestamps[0]),
        "train_timestamp_end": str(timestamps[-1]),
        "train_archive_sha256": str(train_archive_sha256),
        "capacity_receipt_sha256": str(capacity_receipt_sha256),
        "feature_order": {"load": ["electricity", "cooling", "heating", "gas"], "exog": list(range(12)), "device": list(range(17)), "activity": list(range(6)), "scheduler": list(range(6))},
        "statistics": {name: np.asarray(getattr(normalization, name)).tolist() for name in (
            "load_mean", "load_scale", "exog_mean", "exog_scale", "device_mean", "device_scale", "activity_mean", "activity_scale", "scheduler_mean", "scheduler_scale",
        )},
    }
    payload["normalization_sha256"] = canonical_sha256(payload)
    return payload


def build_c_ref_receipt(
    train_objectives: np.ndarray,
    *,
    train_archive_sha256: str,
    capacity_receipt_sha256: str,
    capacity_scenario_hash: str = "",
    objective_implementation_sha256: str = "",
) -> tuple[TrainingObjectiveScale, dict[str, Any]]:
    values = np.asarray(train_objectives, dtype=np.float64)
    scale = fit_training_objective_scale(values, capacity_scenario_hash=capacity_scenario_hash, source_hash=train_archive_sha256)
    payload: dict[str, Any] = {
        "schema_version": C_REF_RECEIPT_SCHEMA,
        "source_split": "train",
        "train_archive_sha256": str(train_archive_sha256),
        "capacity_receipt_sha256": str(capacity_receipt_sha256),
        "capacity_scenario_hash": str(capacity_scenario_hash),
        "objective_implementation_sha256": str(objective_implementation_sha256),
        "step_weights": list(STEP_WEIGHTS),
        "raw_mean": scale.raw_mean,
        "raw_median": scale.raw_median,
        "raw_iqr": scale.raw_iqr,
        "sample_count": scale.sample_count,
        "c_ref": scale.c_ref,
    }
    payload["c_ref_sha256"] = canonical_sha256(payload)
    return scale, payload


def fit_c_ref_from_train_split(
    train: FormalV4WindowSplit,
    parameters: Mapping[str, Any],
    *,
    train_archive_sha256: str,
    capacity_receipt_sha256: str,
    capacity_scenario_hash: str = "",
    objective_implementation_sha256: str = "",
    solver: Any = solve_dispatch_lp,
) -> tuple[TrainingObjectiveScale, dict[str, Any]]:
    """Evaluate every train origin with PI-LP and persist a train-only C_ref.

    The split already carries the frozen rolling state and realized four-hour
    renewables, so no selection/evaluation timestamp can enter this calculation.
    """

    if train.split != "train":
        raise ValueError("C_ref fitting requires the train split")
    objectives: list[float] = []
    for index in range(len(train)):
        context = dict(parameters)
        context["grid_energy_price"] = train.prices_and_weights[index, :, 0]
        context["gas_energy_price"] = train.prices_and_weights[index, :, 1]
        context["carbon_price"] = train.prices_and_weights[index, :, 2]
        solved = solver(DispatchInputs(
            demand=train.rigid_demand[index],
            pv_available=train.renewable_realized[index, :, 0],
            wt_available=train.renewable_realized[index, :, 1],
            parameters=context,
            initial_soc=float(train.initial_soc[index, 0]),
            previous_chp=float(max(train.previous_chp[index, 0], 0.0)),
        ))
        if not solved.success or not np.isfinite(float(solved.objective)) or float(solved.objective) < 0.0:
            raise RuntimeError(f"training PI-LP failed at origin {index}")
        objectives.append(float(solved.objective))
    return build_c_ref_receipt(
        np.asarray(objectives, dtype=np.float64),
        train_archive_sha256=train_archive_sha256,
        capacity_receipt_sha256=capacity_receipt_sha256,
        capacity_scenario_hash=capacity_scenario_hash,
        objective_implementation_sha256=objective_implementation_sha256,
    )


def persist_train_selection_artifacts(
    train: FormalV4WindowSplit,
    selection: FormalV4WindowSplit,
    *,
    run_root: str | Path,
    capacity_receipt: str | Path | Mapping[str, Any],
    capacity_scenario_hash: str = "",
) -> ArtifactManifest:
    """Persist immutable train/selection archives and normalization receipt."""

    root = Path(run_root)
    if not root.is_absolute():
        raise ValueError("formal-v4 run_root must be an explicit absolute path")
    data_root = root / "data"
    if data_root.exists() and any(data_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite data artifacts in {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)

    norm = FormalV4Normalization.fit(train)

    def save(split: FormalV4WindowSplit, path: Path, normalization: FormalV4Normalization | None = None) -> None:
        payload: dict[str, Any] = dict(
            load_history=split.load_history, exog_history=split.exog_history,
            renewable_history=split.renewable_history, device_history=split.device_history,
            activity_history=split.activity_history, forecast_target=split.forecast_target,
            rigid_demand=split.rigid_demand, renewable_forecast=split.renewable_forecast,
            renewable_realized=split.renewable_realized, prices_and_weights=split.prices_and_weights,
            initial_soc=split.initial_soc, previous_chp=split.previous_chp,
            target_times=split.target_times, trajectory_ids=split.trajectory_ids,
            state_hashes=split.state_hashes, split=np.asarray(split.split), history_source=np.asarray(split.history_source),
        )
        if normalization is not None:
            for name in ("load_mean", "load_scale", "exog_mean", "exog_scale", "device_mean", "device_scale", "activity_mean", "activity_scale", "scheduler_mean", "scheduler_scale"):
                payload[f"normalization_{name}"] = getattr(normalization, name)
            payload["normalization_fitted_split"] = np.asarray(normalization.fitted_split)
        np.savez_compressed(
            path,
            **payload,
        )

    save(train, data_root / "train.npz", norm)
    save(selection, data_root / "selection.npz")
    receipt_hash = sha256_file(capacity_receipt) if isinstance(capacity_receipt, (str, Path)) else canonical_sha256(dict(capacity_receipt))
    norm_payload = build_normalization_receipt(norm, train_archive_sha256=sha256_file(data_root / "train.npz"), capacity_receipt_sha256=receipt_hash, train_timestamps=train.target_times)
    _write_once(root / "NORMALIZATION_RECEIPT.json", norm_payload)
    manifest = build_artifact_manifest(root, ["data/train.npz", "data/selection.npz", "NORMALIZATION_RECEIPT.json"], lineage={"capacity_receipt_sha256": receipt_hash, "capacity_scenario_hash": capacity_scenario_hash})
    manifest.save(root / "ARTIFACT_MANIFEST.json")
    return manifest


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA", "ArtifactManifest", "C_REF_RECEIPT_SCHEMA", "NORMALIZATION_RECEIPT_SCHEMA",
    "build_artifact_manifest", "build_c_ref_receipt", "build_normalization_receipt", "canonical_sha256", "fit_c_ref_from_train_split",
    "persist_train_selection_artifacts", "sha256_file",
]
