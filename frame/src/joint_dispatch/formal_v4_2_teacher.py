"""Same-information LP teacher overlays for the formal-v4.2 protocol.

The teacher is deliberately an *offline* artifact.  It receives the forecast
produced from the causal state at an origin and solves one LP using only the
forecast, last-observed renewable availability, prices and the carried state.
Realised future demand and renewable output are kept on the window only for
later settlement; they are never read while a teacher overlay is built.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..scheduling.dispatch_lp import DispatchInputs, DispatchResult, solve_dispatch_lp
from ..scheduling.dispatch_schema import VARIABLES
from .formal_v4_data import FormalV4WindowSplit


_SHA256 = set("0123456789abcdef")


class TeacherCacheMismatch(ValueError):
    """Raised when a cached teacher is not from the requested lineage."""


def _digest_arrays(*arrays: object, extra: str = "") -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.asarray(value)
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(str(extra).encode("utf-8"))
    return digest.hexdigest()


def _valid_hash(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in _SHA256 for char in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True)
class TeacherKeyV42:
    """All identities that make a teacher overlay reproducible."""

    seed: int
    split: str
    timestamp_sha256: str
    state_sha256: str
    capacity_sha256: str
    benchmark_sha256: str
    normalization_sha256: str
    stage_p_checkpoint_sha256: str
    implementation_sha256: str
    source_manifest_sha256: str
    # Compatibility spelling used by early v4.2 callers/tests.  The
    # canonical serialized field remains ``stage_p_checkpoint_sha256``.
    checkpoint_sha256: str = ""

    def __post_init__(self) -> None:
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if self.split not in {"train", "selection", "evaluation"}:
            raise ValueError("teacher split must be train, selection, or evaluation")
        object.__setattr__(self, "seed", int(self.seed))
        alias = str(self.checkpoint_sha256 or "")
        if alias:
            object.__setattr__(self, "stage_p_checkpoint_sha256", alias)
        for name in (
            "timestamp_sha256", "state_sha256", "capacity_sha256",
            "benchmark_sha256", "normalization_sha256",
            "stage_p_checkpoint_sha256", "implementation_sha256",
            "source_manifest_sha256",
        ):
            object.__setattr__(self, name, _valid_hash(getattr(self, name), name))
        object.__setattr__(self, "checkpoint_sha256", self.stage_p_checkpoint_sha256)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal-v4.2-teacher-key-v1",
            "seed": self.seed,
            "split": self.split,
            "timestamp_sha256": self.timestamp_sha256,
            "state_sha256": self.state_sha256,
            "capacity_sha256": self.capacity_sha256,
            "benchmark_sha256": self.benchmark_sha256,
            "normalization_sha256": self.normalization_sha256,
            "stage_p_checkpoint_sha256": self.stage_p_checkpoint_sha256,
            "implementation_sha256": self.implementation_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
        }


@dataclass(frozen=True)
class TeacherOverlayV42:
    """Predicted inputs and LP labels bound to one frozen teacher key."""

    key: TeacherKeyV42
    predicted_forecast: np.ndarray  # [N,4,4], including the auxiliary gas prior
    rigid_demand: np.ndarray  # [N,4,3]
    gas_prior: np.ndarray  # [N,4,1]
    renewable_plan: np.ndarray  # [N,4,2], persistence forecast
    dispatch: np.ndarray  # [N,4,21]
    objective: np.ndarray  # [N]
    shortage: np.ndarray  # [N,3], cumulative electric/cooling/heating slack
    status: tuple[str, ...]
    target_times: np.ndarray
    state_hashes: np.ndarray
    arrays_sha256: str = ""

    def __post_init__(self) -> None:
        predicted = np.asarray(self.predicted_forecast, dtype=np.float64)
        rigid = np.asarray(self.rigid_demand, dtype=np.float64)
        gas = np.asarray(self.gas_prior, dtype=np.float64)
        renew = np.asarray(self.renewable_plan, dtype=np.float64)
        dispatch = np.asarray(self.dispatch, dtype=np.float64)
        objective = np.asarray(self.objective, dtype=np.float64)
        shortage = np.asarray(self.shortage, dtype=np.float64)
        n = predicted.shape[0] if predicted.ndim == 3 else -1
        if predicted.shape != (n, 4, 4) or rigid.shape != (n, 4, 3) or gas.shape != (n, 4, 1):
            raise ValueError("teacher forecasts must have shapes [N,4,4], [N,4,3], [N,4,1]")
        if renew.shape != (n, 4, 2) or dispatch.shape != (n, 4, len(VARIABLES)):
            raise ValueError("teacher renewable/dispatch arrays have invalid shapes")
        if objective.shape != (n,) or shortage.shape != (n, 3) or len(self.status) != n:
            raise ValueError("teacher metrics must align with the sample count")
        times = np.asarray(self.target_times, dtype="datetime64[ns]")
        states = np.asarray(self.state_hashes, dtype=str)
        if times.shape != (n,) or states.shape != (n,):
            raise ValueError("teacher alignment fields must have shape [N]")
        for value, name in ((predicted, "predicted_forecast"), (rigid, "rigid_demand"),
                            (gas, "gas_prior"), (renew, "renewable_plan"),
                            (dispatch, "dispatch"), (objective, "objective"), (shortage, "shortage")):
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if (predicted < 0).any() or (rigid < 0).any() or (gas < 0).any() or (renew < 0).any() or (dispatch < 0).any():
            raise ValueError("teacher arrays must be non-negative")
        if not np.array_equal(predicted[..., :3], rigid) or not np.array_equal(predicted[..., 3:4], gas):
            raise ValueError("rigid_demand and gas_prior must be slices of predicted_forecast")
        calculated = _digest_arrays(predicted, rigid, gas, renew, dispatch, objective, shortage, times, states, extra="formal-v4.2-teacher-arrays-v1")
        if self.arrays_sha256 and self.arrays_sha256 != calculated:
            raise TeacherCacheMismatch("teacher array hash mismatch")
        object.__setattr__(self, "predicted_forecast", predicted)
        object.__setattr__(self, "rigid_demand", rigid)
        object.__setattr__(self, "gas_prior", gas)
        object.__setattr__(self, "renewable_plan", renew)
        object.__setattr__(self, "dispatch", dispatch)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "shortage", shortage)
        object.__setattr__(self, "target_times", times)
        object.__setattr__(self, "state_hashes", states)
        object.__setattr__(self, "status", tuple(str(item) for item in self.status))
        object.__setattr__(self, "arrays_sha256", calculated)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "formal-v4.2-teacher-overlay-v1",
            "key": self.key.to_payload(),
            "arrays_sha256": self.arrays_sha256,
            "sample_count": int(self.dispatch.shape[0]),
            "target_times": [str(value) for value in self.target_times],
            "state_hashes": self.state_hashes.tolist(),
            "status": list(self.status),
        }


def _as_window_list(windows: Any) -> list[Any]:
    if isinstance(windows, FormalV4WindowSplit):
        # Keep the production split as a convenient batch input, but expose
        # one-row views to the causal predictor/LP so each state is bound to
        # its own timestamp and carried SOC/CHP values.
        if len(windows) == 1:
            return [windows]
        fields = (
            "load_history", "exog_history", "renewable_history", "device_history",
            "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
            "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
            "target_times", "trajectory_ids", "state_hashes",
        )
        rows: list[FormalV4WindowSplit] = []
        for index in range(len(windows)):
            updates = {name: getattr(windows, name)[index:index + 1] for name in fields}
            rows.append(replace(windows, **updates))
        return rows
    if isinstance(windows, (str, bytes)):
        raise TypeError("windows must be a FormalV4WindowSplit or iterable of windows")
    return list(windows)


def _window_value(window: Any, name: str, default: Any = None) -> Any:
    if isinstance(window, FormalV4WindowSplit):
        if len(window) != 1:
            raise ValueError("a FormalV4WindowSplit passed as one teacher window must contain one row")
        mapping = {
            "predicted_target": window.forecast_target[0],
            "renewable_history": window.renewable_history[0],
            "renewable_forecast": window.renewable_forecast[0],
            "prices_and_weights": window.prices_and_weights[0],
            "initial_soc": float(window.initial_soc[0, 0]),
            "previous_chp": float(window.previous_chp[0, 0]),
            "target_time": window.target_times[0],
            "state_hash": str(window.state_hashes[0]),
            "split": window.split,
        }
        return mapping.get(name, default)
    value = getattr(window, name, default)
    if value is not default:
        return value
    if isinstance(window, Mapping):
        return window.get(name, default)
    return default


def _stage_prediction(stage_p: Any, window: Any) -> np.ndarray:
    value: Any = None
    if callable(stage_p):
        value = stage_p(window)
    elif hasattr(stage_p, "predict") and callable(stage_p.predict):
        value = stage_p.predict(window)
    elif isinstance(stage_p, Mapping):
        for name in ("predicted_forecast", "forecast", "prediction", "predicted_rigid"):
            if name in stage_p:
                value = stage_p[name]
                break
    else:
        for name in ("predicted_forecast", "forecast", "prediction", "predicted_rigid"):
            if hasattr(stage_p, name):
                value = getattr(stage_p, name)
                break
    if value is None:
        raise TypeError("stage_p must be callable or expose a forecast/prediction")
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape == (4, 3):
        gas = _window_value(window, "gas_prior", None)
        if gas is None:
            target = _window_value(window, "predicted_target", None)
            gas = np.asarray(target, dtype=np.float64)[:, 3] if target is not None else np.zeros(4)
        array = np.column_stack((array, np.asarray(gas, dtype=np.float64).reshape(4)))
    if array.shape != (4, 4):
        raise ValueError("Stage P prediction must have shape [4,4] or [4,3]")
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Stage P prediction must be finite and non-negative")
    return array


def _parameters_for_window(parameters: Mapping[str, Any] | None, window: Any, horizon: int) -> dict[str, Any]:
    base = parameters
    if base is None:
        base = _window_value(window, "parameters", None)
    if base is None:
        # A custom solver used by protocol tests may only inspect the causal
        # inputs.  The production HiGHS solver will still fail closed below
        # if the required benchmark parameters are absent.
        base = {}
    context = dict(base)
    prices = np.asarray(_window_value(window, "prices_and_weights"), dtype=np.float64)
    if prices.shape != (horizon, 3):
        raise ValueError("prices_and_weights must have shape [4,3]")
    context["grid_energy_price"] = prices[:, 0]
    context["gas_energy_price"] = prices[:, 1]
    context["carbon_price"] = prices[:, 2]
    return context


def build_same_information_teacher_v42(
    stage_p: Any,
    windows: Iterable[Any] | FormalV4WindowSplit,
    *,
    seed: int,
    parameters: Mapping[str, Any] | None = None,
    capacity_sha256: str = "0" * 64,
    benchmark_sha256: str = "0" * 64,
    normalization_sha256: str = "0" * 64,
    stage_p_checkpoint_sha256: str | None = None,
    implementation_sha256: str = "0" * 64,
    source_manifest_sha256: str = "0" * 64,
    solver: Callable[[DispatchInputs], DispatchResult] = solve_dispatch_lp,
) -> TeacherOverlayV42:
    """Build a teacher from causal predictions and persistence renewables.

    ``stage_p`` may be a callable, an object with ``predict`` or a small test
    record exposing ``forecast``/``predicted_forecast``.  This keeps the
    artifact builder independent of a particular neural framework while the
    production runner can pass its frozen Stage-P model.
    """

    rows = _as_window_list(windows)
    if not rows:
        raise ValueError("teacher overlay requires at least one window")
    first_split = _window_value(rows[0], "split", "train")
    timestamp_values = np.asarray([_window_value(row, "target_time") for row in rows], dtype="datetime64[ns]")
    state_values = np.asarray([str(_window_value(row, "state_hash")) for row in rows], dtype=str)
    if any(value == "None" for value in state_values):
        raise ValueError("each teacher window must provide a state hash")
    checkpoint = stage_p_checkpoint_sha256
    if checkpoint is None:
        checkpoint = _window_value(stage_p, "checkpoint_sha256", None)
    if checkpoint is None:
        checkpoint = _window_value(stage_p, "stage_p_checkpoint_sha256", None)
    if checkpoint is None:
        raise ValueError("teacher overlay requires the Stage P checkpoint hash")
    predicted = np.stack([_stage_prediction(stage_p, row) for row in rows], axis=0)
    rigid = predicted[..., :3]
    gas = predicted[..., 3:4]
    renewable_plan = []
    dispatch_rows = []
    objectives = []
    shortages = []
    statuses = []
    for index, (row, forecast) in enumerate(zip(rows, predicted)):
        history = np.asarray(_window_value(row, "renewable_history"), dtype=np.float64)
        if history.ndim == 3 and history.shape[0] == 1:
            history = history[0]
        if history.shape != (24, 2):
            raise ValueError("renewable_history must have shape [24,2]")
        # This is the only renewable signal visible to the planner.
        renew = np.repeat(history[-1:, :], 4, axis=0)
        renewable_plan.append(renew)
        context = _parameters_for_window(parameters, row, 4)
        initial_soc = float(_window_value(row, "initial_soc"))
        previous_chp = float(_window_value(row, "previous_chp"))
        result = solver(DispatchInputs(
            demand=forecast[:, :3], pv_available=renew[:, 0], wt_available=renew[:, 1],
            parameters=context, initial_soc=initial_soc, previous_chp=previous_chp,
        ))
        if not result.success:
            raise RuntimeError(f"same-information teacher LP failed at row {index}: {result.message}")
        dispatch_rows.append(np.column_stack([result.values[name] for name in VARIABLES]))
        objectives.append(float(result.objective))
        shortages.append([
            float(np.sum(result.values["slack_e"])),
            float(np.sum(result.values["slack_c"])),
            float(np.sum(result.values["slack_h"])),
        ])
        statuses.append(str(result.status))
    key = TeacherKeyV42(
        seed=int(seed), split=str(first_split),
        timestamp_sha256=_digest_arrays(timestamp_values, extra="formal-v4.2-teacher-timestamps-v1"),
        state_sha256=_digest_arrays(state_values, extra="formal-v4.2-teacher-states-v1"),
        capacity_sha256=_valid_hash(capacity_sha256, "capacity_sha256"),
        benchmark_sha256=_valid_hash(benchmark_sha256, "benchmark_sha256"),
        normalization_sha256=_valid_hash(normalization_sha256, "normalization_sha256"),
        stage_p_checkpoint_sha256=_valid_hash(checkpoint, "stage_p_checkpoint_sha256"),
        implementation_sha256=_valid_hash(implementation_sha256, "implementation_sha256"),
        source_manifest_sha256=_valid_hash(source_manifest_sha256, "source_manifest_sha256"),
    )
    return TeacherOverlayV42(
        key=key, predicted_forecast=predicted, rigid_demand=rigid, gas_prior=gas,
        renewable_plan=np.stack(renewable_plan), dispatch=np.stack(dispatch_rows),
        objective=np.asarray(objectives), shortage=np.asarray(shortages), status=tuple(statuses),
        target_times=timestamp_values, state_hashes=state_values,
    )


def _key_payload(key: TeacherKeyV42) -> dict[str, Any]:
    return key.to_payload()


def save_teacher_overlay(directory: str | Path, overlay: TeacherOverlayV42) -> tuple[Path, Path]:
    """Write a teacher NPZ and immutable JSON metadata sidecar."""

    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays_path = root / "TEACHER.npz"
    metadata_path = root / "TEACHER.json"
    with __import__("io").BytesIO() as buffer:
        np.savez_compressed(
            buffer, predicted_forecast=overlay.predicted_forecast, rigid_demand=overlay.rigid_demand,
            gas_prior=overlay.gas_prior, renewable_plan=overlay.renewable_plan,
            dispatch=overlay.dispatch, objective=overlay.objective, shortage=overlay.shortage,
            target_times=overlay.target_times.astype("datetime64[ns]").astype("int64"),
            state_hashes=overlay.state_hashes,
        )
        encoded = buffer.getvalue()
    if arrays_path.exists() and arrays_path.read_bytes() != encoded:
        raise FileExistsError(f"immutable teacher artifact differs: {arrays_path}")
    if not arrays_path.exists():
        temporary = arrays_path.with_name(f".{arrays_path.name}.writing")
        temporary.write_bytes(encoded)
        temporary.replace(arrays_path)
    metadata = {**overlay.to_payload(), "key": _key_payload(overlay.key), "arrays_file_sha256": hashlib.sha256(encoded).hexdigest()}
    from .formal_v4_2_artifacts import write_once_json
    write_once_json(metadata_path, metadata)
    return arrays_path, metadata_path


def load_teacher_overlay(directory: str | Path, expected_key: TeacherKeyV42 | None = None) -> TeacherOverlayV42:
    root = Path(directory)
    arrays_path = root / "TEACHER.npz"
    metadata_path = root / "TEACHER.json"
    if not arrays_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("teacher overlay is incomplete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    key_payload = metadata.get("key")
    if not isinstance(key_payload, Mapping):
        raise TeacherCacheMismatch("teacher metadata has no key")
    try:
        key = TeacherKeyV42(**{name: key_payload[name] for name in (
            "seed", "split", "timestamp_sha256", "state_sha256", "capacity_sha256",
            "benchmark_sha256", "normalization_sha256", "stage_p_checkpoint_sha256",
            "implementation_sha256", "source_manifest_sha256",
        )})
    except (KeyError, TypeError, ValueError) as exc:
        raise TeacherCacheMismatch("teacher metadata key is invalid") from exc
    if expected_key is not None and key != expected_key:
        raise TeacherCacheMismatch("teacher key mismatch")
    encoded = arrays_path.read_bytes()
    if metadata.get("arrays_file_sha256") != hashlib.sha256(encoded).hexdigest():
        raise TeacherCacheMismatch("teacher NPZ hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        states = arrays["state_hashes"].astype(str)
        overlay = TeacherOverlayV42(
            key=key, predicted_forecast=arrays["predicted_forecast"], rigid_demand=arrays["rigid_demand"],
            gas_prior=arrays["gas_prior"], renewable_plan=arrays["renewable_plan"], dispatch=arrays["dispatch"],
            objective=arrays["objective"], shortage=arrays["shortage"], status=tuple(metadata.get("status", ())),
            target_times=arrays["target_times"].astype("datetime64[ns]"), state_hashes=states,
            arrays_sha256=str(metadata.get("arrays_sha256", "")),
        )
    if int(metadata.get("sample_count", -1)) != len(overlay.status):
        raise TeacherCacheMismatch("teacher sample count mismatch")
    return overlay


__all__ = [
    "TeacherCacheMismatch", "TeacherKeyV42", "TeacherOverlayV42",
    "build_same_information_teacher_v42", "load_teacher_overlay", "save_teacher_overlay",
]
