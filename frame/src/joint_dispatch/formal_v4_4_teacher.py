"""Same-information offline LP teacher for formal-v4.4 Pilot stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from ..scheduling.dispatch_schema import VARIABLES
from .formal_v4_2_data import apply_normalization, fit_train_normalization
from .formal_v4_4_artifacts import canonical_sha256, sha256_file, write_json_once, write_npz_once
from .formal_v4_4_contract import FormalV44Contract
from .formal_v4_4_regime import derive_last_observed_regime
from .formal_v4_4_pilot_materializer import MaterializedV44PilotData, V44WindowCollection


@dataclass(frozen=True)
class TeacherReceiptV44:
    forecast: np.ndarray
    renewable_plan: np.ndarray
    dispatch: np.ndarray
    objective: np.ndarray
    shortage: np.ndarray
    status: tuple[str, ...]
    timestamps: np.ndarray
    state_hashes: np.ndarray
    lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        forecast = np.asarray(self.forecast, dtype=np.float64)
        renewable = np.asarray(self.renewable_plan, dtype=np.float64)
        dispatch = np.asarray(self.dispatch, dtype=np.float64)
        objective = np.asarray(self.objective, dtype=np.float64)
        shortage = np.asarray(self.shortage, dtype=np.float64)
        timestamps = np.asarray(self.timestamps, dtype="datetime64[ns]")
        states = np.asarray(self.state_hashes, dtype=str)
        n = forecast.shape[0] if forecast.ndim == 3 else -1
        if forecast.shape != (n, 4, 4) or renewable.shape != (n, 4, 2) or dispatch.shape != (n, 4, 21):
            raise ValueError("teacher arrays must have shapes [N,4,4], [N,4,2], and [N,4,21]")
        if objective.shape != (n,) or shortage.shape != (n, 3) or timestamps.shape != (n,) or states.shape != (n,) or len(self.status) != n:
            raise ValueError("teacher arrays are not aligned")
        for value in (forecast, renewable, dispatch, objective, shortage):
            if not np.isfinite(value).all():
                raise ValueError("teacher arrays must be finite")
        if (forecast < 0).any() or (renewable < 0).any() or (dispatch < 0).any():
            raise ValueError("teacher arrays must be non-negative")
        object.__setattr__(self, "forecast", forecast)
        object.__setattr__(self, "renewable_plan", renewable)
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "shortage", shortage)
        object.__setattr__(self, "status", tuple(str(value) for value in self.status))
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "state_hashes", states)

    @property
    def predicted_rigid(self) -> np.ndarray:
        return self.forecast[..., :3]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal-v4.4-teacher-receipt-v1",
            "sample_count": int(len(self.status)),
            "status": list(self.status),
            "timestamps": [str(value) for value in self.timestamps],
            "state_hashes": self.state_hashes.tolist(),
            "lineage": dict(self.lineage),
        }


def _model_hash(model: Any) -> str:
    digest = hashlib.sha256()
    state = model.state_dict() if hasattr(model, "state_dict") else {}
    for name in sorted(state):
        value = state[name]
        if torch.is_tensor(value):
            digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(name.encode("utf-8")); digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def _window_collection(materialized: MaterializedV44PilotData | V44WindowCollection) -> V44WindowCollection:
    if isinstance(materialized, V44WindowCollection):
        return materialized
    if isinstance(materialized, MaterializedV44PilotData):
        return materialized.train
    raise TypeError("materialized must be MaterializedV44PilotData or V44WindowCollection")


def _forecast_from_output(output: Any, batch_size: int) -> np.ndarray:
    value = output.get("forecast_physical") if isinstance(output, Mapping) else getattr(output, "forecast_physical", None)
    if value is None:
        value = output.get("forecast") if isinstance(output, Mapping) else getattr(output, "forecast", None)
    if value is None:
        raise ValueError("Stage P1 model output has no physical forecast")
    array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (batch_size, 4, 4) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Stage P1 forecast must be finite, non-negative, and shaped [B,4,4]")
    return array


def _scaled_parameters(benchmark: Mapping[str, Any], capacity: Mapping[str, Any]) -> dict[str, Any]:
    values = benchmark.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("benchmark values are missing")
    parameters = dict(values)
    selected = capacity.get("selected")
    if not isinstance(selected, Mapping):
        audit = capacity.get("capacity_audit")
        selected = audit.get("selected") if isinstance(audit, Mapping) else None
    multiplier = float(selected.get("multiplier", capacity.get("capacity_multiplier", 1.0))) if isinstance(selected, Mapping) else float(capacity.get("capacity_multiplier", 1.0))
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("capacity multiplier is invalid")
    for name in ("electric_chiller_capacity", "absorption_chiller_capacity"):
        if name in parameters:
            parameters[name] = float(parameters[name]) * multiplier
    return parameters


def _lineage(materialized_lineage: Mapping[str, Any], benchmark: Path, capacity: Path, contract: FormalV44Contract, seed: int, model_hash: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "formal-v4.4-teacher-lineage-v1",
        "contract_sha256": contract.contract_sha256,
        "benchmark_sha256": sha256_file(benchmark),
        "capacity_receipt_sha256": sha256_file(capacity),
        "materialized_lineage_sha256": materialized_lineage.get("lineage_sha256", ""),
        "model_sha256": model_hash,
        "seed": int(seed),
        "evaluation_year_accessed": False,
    }
    payload["lineage_sha256"] = canonical_sha256(payload)
    return payload


def _save_teacher(root: Path, receipt: TeacherReceiptV44) -> None:
    root.mkdir(parents=True, exist_ok=True)
    write_npz_once(root / "TEACHER.npz", {
        "forecast": receipt.forecast, "renewable_plan": receipt.renewable_plan,
        "dispatch": receipt.dispatch, "objective": receipt.objective,
        "shortage": receipt.shortage, "timestamps": receipt.timestamps.astype("datetime64[ns]").astype(np.int64),
        "state_hashes": receipt.state_hashes,
    })
    write_json_once(root / "TEACHER.json", receipt.to_payload())


def _load_teacher(root: Path, lineage: Mapping[str, Any]) -> TeacherReceiptV44:
    metadata = json.loads((root / "TEACHER.json").read_text(encoding="utf-8"))
    if metadata.get("lineage") != dict(lineage):
        raise ValueError("teacher cache lineage mismatch; use a new run ID")
    with np.load(root / "TEACHER.npz", allow_pickle=False) as arrays:
        return TeacherReceiptV44(
            forecast=arrays["forecast"], renewable_plan=arrays["renewable_plan"], dispatch=arrays["dispatch"],
            objective=arrays["objective"], shortage=arrays["shortage"], status=tuple(metadata["status"]),
            timestamps=arrays["timestamps"].astype("datetime64[ns]"), state_hashes=arrays["state_hashes"], lineage=metadata["lineage"],
        )


def build_same_information_teacher_v44(
    *, model: Any, materialized: MaterializedV44PilotData | V44WindowCollection, indices: np.ndarray,
    benchmark: str | Path, capacity_receipt: str | Path, artifact_root: str | Path,
    contract: FormalV44Contract, seed: int,
) -> TeacherReceiptV44:
    """Solve one LP per window using only deployed forecasts and carried state."""

    contract.validate()
    collection = _window_collection(materialized)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(collection)):
        raise ValueError("teacher indices are invalid")
    benchmark_path = Path(benchmark); capacity_path = Path(capacity_receipt)
    model_hash = _model_hash(model)
    lineage_source = materialized.lineage if isinstance(materialized, MaterializedV44PilotData) else {"lineage_sha256": ""}
    lineage = _lineage(lineage_source, benchmark_path, capacity_path, contract, seed, model_hash)
    root = Path(artifact_root) / "teacher"
    if (root / "TEACHER.json").exists() or (root / "TEACHER.npz").exists():
        if not (root / "TEACHER.json").is_file() or not (root / "TEACHER.npz").is_file():
            raise ValueError("teacher cache is incomplete")
        return _load_teacher(root, lineage)
    normalization_source = materialized.normalization_source.split if isinstance(materialized, MaterializedV44PilotData) else collection.split
    normalization = fit_train_normalization(normalization_source)
    # Build inference inputs directly instead of using the training batch
    # helper: that helper also derives supervised future regime labels, which
    # are intentionally unavailable to an offline deployed-information
    # teacher.
    normalized = apply_normalization(collection, normalization)
    raw_history = np.asarray(collection.load_history)[indices]
    last_regime = derive_last_observed_regime(raw_history)
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(indices), 64):
        stop = min(start + 64, len(indices))
        local = indices[start:stop]
        batches.append({
            "load_history": torch.from_numpy(normalized.load_history[local]),
            "exog_history": torch.from_numpy(normalized.exog_history[local]),
            "device_history": torch.from_numpy(normalized.device_history[local]),
            "activity_history": torch.from_numpy(normalized.activity_history[local]),
            "scheduler_context": torch.from_numpy(normalized.scheduler_context[local]),
            "previous_chp": torch.as_tensor(collection.previous_chp[local], dtype=torch.float32),
            "last_thermal_regime": torch.as_tensor(last_regime[start:stop], dtype=torch.long),
        })
    predictions: list[np.ndarray] = []
    previous_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    with torch.no_grad():
        for batch in batches:
            model_inputs = {name: batch[name] for name in (
                "load_history", "exog_history", "device_history", "activity_history",
                "scheduler_context", "previous_chp", "last_thermal_regime",
            )}
            output = model(**model_inputs)
            predictions.append(_forecast_from_output(output, len(batch["load_history"])))
    if previous_training and hasattr(model, "train"):
        model.train()
    forecast = np.concatenate(predictions, axis=0)
    selected = {name: np.asarray(getattr(collection, name))[indices] for name in ("renewable_history", "prices_and_weights", "initial_soc", "previous_chp", "target_times", "state_hashes")}
    benchmark_payload = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    capacity_payload = json.loads(capacity_path.read_text(encoding="utf-8"))
    parameters = _scaled_parameters(benchmark_payload, capacity_payload)
    renewable_rows: list[np.ndarray] = []
    dispatch_rows: list[np.ndarray] = []
    objective_rows: list[float] = []
    shortage_rows: list[list[float]] = []
    statuses: list[str] = []
    for row_index, (forecast_row, history, prices, soc, previous) in enumerate(zip(
        forecast, selected["renewable_history"], selected["prices_and_weights"], selected["initial_soc"], selected["previous_chp"],
    )):
        renew = np.repeat(np.asarray(history[-1:, :], dtype=np.float64), 4, axis=0)
        context = dict(parameters)
        context["grid_energy_price"] = np.asarray(prices)[:, 0]
        context["gas_energy_price"] = np.asarray(prices)[:, 1]
        context["carbon_price"] = np.asarray(prices)[:, 2]
        result = solve_dispatch_lp(DispatchInputs(
            demand=forecast_row[:, :3], pv_available=renew[:, 0], wt_available=renew[:, 1],
            parameters=context, initial_soc=float(np.asarray(soc).reshape(-1)[0]), previous_chp=float(np.asarray(previous).reshape(-1)[0]),
        ))
        if result is None or not getattr(result, "success", False):
            message = "no result" if result is None else str(getattr(result, "message", "unknown LP failure"))
            raise RuntimeError(f"same-information teacher LP failed at row {row_index}: {message}")
        dispatch = np.column_stack([np.asarray(result.values[name], dtype=np.float64) for name in VARIABLES])
        if dispatch.shape != (4, 21) or not np.isfinite(dispatch).all() or float(dispatch.min()) < -1.0e-6:
            raise RuntimeError(f"same-information teacher LP returned invalid dispatch at row {row_index}")
        renewable_rows.append(renew)
        dispatch_rows.append(np.maximum(dispatch, 0.0))
        objective_rows.append(float(result.objective))
        shortage_rows.append([float(np.sum(result.values[name])) for name in ("slack_e", "slack_c", "slack_h")])
        statuses.append(str(result.status))
    receipt = TeacherReceiptV44(
        forecast=forecast, renewable_plan=np.stack(renewable_rows), dispatch=np.stack(dispatch_rows),
        objective=np.asarray(objective_rows), shortage=np.asarray(shortage_rows), status=tuple(statuses),
        timestamps=selected["target_times"], state_hashes=selected["state_hashes"], lineage=lineage,
    )
    _save_teacher(root, receipt)
    return receipt


__all__ = ["TeacherReceiptV44", "build_same_information_teacher_v44"]
