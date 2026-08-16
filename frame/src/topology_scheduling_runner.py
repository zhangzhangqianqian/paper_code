"""Branch-aware routing from frozen 2021 forecasts into the R/S schedulers.

This layer deliberately contains no optimization equations.  It validates the
frozen topology branch, aligns prediction artifacts, and exposes the channel
contract required by the existing real-replay and simulated-dispatch entry
points.  The S track receives electricity/cooling/heating as rigid demands;
gas remains a device-side procurement/replay reference on the R track.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .topology_test_runner import (
    JOINT_MODELS,
    PHASE_A_MODELS,
    PHASE_A_SEEDS,
    PHASE_B_FORMAL_SEEDS,
    build_phase_b_run_plan,
    validate_phase_b_authorization,
)


SIMULATED_SCENARIOS = (
    "core",
    "no_bess",
    "no_chp",
    "no_renewables",
    "single_hour",
    "carbon_price_sensitivity",
)
TASKS = ("electricity", "cooling", "heating", "gas")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_scheduling_run_plan(freeze: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """Expand the frozen branch matrix into the R/S scheduling runs."""

    branch = validate_phase_b_authorization(freeze)
    forecast_runs = build_phase_b_run_plan(freeze)
    rows: list[dict[str, object]] = []
    for forecast in forecast_runs:
        rows.append({
            **forecast,
            "track": "real_replay",
            "scenario": "real_replay",
            "gas_role": "device_side_procurement_reference",
        })
        include_s = branch == "core_prediction_changed" or (
            forecast["model"] in {"scheme2r", "dynamic_symmetric"}
            and int(forecast["seed"]) in set(PHASE_A_SEEDS)
        )
        if include_s:
            scenarios = SIMULATED_SCENARIOS if branch == "core_prediction_changed" else ("core",)
            for scenario in scenarios:
                rows.append({
                    **forecast,
                    "track": "simulated_dispatch",
                    "scenario": scenario,
                    "gas_role": "not_a_rigid_demand_balance",
                })
    expected = {"core_prediction_changed": 175, "gas_only_changed": 31, "core_conclusion_stable": 12}[branch]
    if len(rows) != expected:
        raise RuntimeError(f"unexpected frozen scheduling matrix for {branch}: {len(rows)}")
    return tuple(rows)


def validate_prediction_artifact(path: str | Path) -> dict[str, object]:
    """Load one immutable test prediction artifact and validate its schema."""

    artifact = Path(path)
    if artifact.name != "predictions_test.npz" or not artifact.is_file():
        raise FileNotFoundError(artifact)
    with np.load(artifact, allow_pickle=False) as payload:
        if "prediction" not in payload.files or "target_times" not in payload.files:
            raise ValueError("test prediction artifact must contain prediction and target_times")
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
        target_times = np.asarray(payload["target_times"], dtype="datetime64[ns]")
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (4, 4):
        raise ValueError(f"prediction must have shape [N,4,4], got {prediction.shape}")
    if len(target_times) != len(prediction) or len(target_times) == 0:
        raise ValueError("prediction and target_times must have equal non-zero length")
    if np.any(target_times.astype("datetime64[Y]").astype(int) + 1970 != 2021):
        raise ValueError("topology scheduling accepts only 2021 test origins")
    if len(np.unique(target_times)) != len(target_times) or np.any(np.diff(target_times) <= np.timedelta64(0, "ns")):
        raise ValueError("test target_times must be unique and strictly increasing")
    if not np.isfinite(prediction).all():
        raise ValueError("test prediction must be finite")
    return {
        "prediction": prediction,
        "target_times": target_times,
        "source_path": str(artifact),
        "sha256": _sha256(artifact),
    }


def route_prediction_channels(prediction: np.ndarray, track: str) -> dict[str, object]:
    """Translate the four-task forecast into the existing scheduler contract."""

    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 3 or tuple(values.shape[1:]) != (4, 4):
        raise ValueError("forecast must have shape [N,4,4] in [electricity,cooling,heating,gas] order")
    if track == "real_replay":
        return {
            "full_loads": np.maximum(values, 0.0),
            "rigid_demand_tasks": list(TASKS[:3]),
            "gas_reference": np.maximum(values[:, :, 3], 0.0),
            "gas_role": "device_side_procurement_reference",
        }
    if track == "simulated_dispatch":
        return {
            "demand": np.maximum(values[:, :, :3], 0.0),
            "rigid_demand_tasks": list(TASKS[:3]),
            "gas_reference": np.maximum(values[:, :, 3], 0.0),
            "gas_role": "not_a_rigid_demand_balance",
        }
    raise ValueError(f"unknown scheduling track: {track}")


def build_routing_manifest(
    freeze: Mapping[str, object], prediction_root: str | Path,
    *, output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Validate all frozen forecast files and write a routing-only manifest."""

    plan = build_scheduling_run_plan(freeze)
    root = Path(prediction_root)
    records = []
    for run in plan:
        if run["track"] == "simulated_dispatch" and run["scenario"] != "core":
            # The same forecast is intentionally reused across S scenarios.
            pass
        path = root / str(run["model"]) / str(run["candidate_id"]) / f"seed_{run['seed']}" / "predictions_test.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact = validate_prediction_artifact(path)
        routed = route_prediction_channels(artifact["prediction"], str(run["track"]))
        records.append({
            **{key: value for key, value in run.items() if key not in {"branch"}},
            "prediction_path": str(path),
            "prediction_sha256": artifact["sha256"],
            "prediction_shape": list(artifact["prediction"].shape),
            "rigid_demand_tasks": routed["rigid_demand_tasks"],
            "gas_role": routed["gas_role"],
        })
    manifest = {
        "stage": "topology_protocol_pilot_task10",
        "status": "passed",
        "branch": freeze["branch"],
        "branch_frozen_before_scheduling": True,
        "test_year": 2021,
        "test_set_accessed": True,
        "lp_equations_modified": False,
        "runs": records,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "topology_scheduling_routing_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return manifest


__all__ = [
    "SIMULATED_SCENARIOS", "build_routing_manifest", "build_scheduling_run_plan",
    "route_prediction_channels", "validate_prediction_artifact",
]
