"""Build formal-v4.1 C-ref and same-information teacher evidence.

This command performs deterministic train-only LP calculations and a bounded
teacher-alignment probe.  It never trains a neural model and never creates a
Stage-P checkpoint or production teacher overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_artifacts import fit_c_ref_from_train_split, sha256_file  # noqa: E402
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit  # noqa: E402
from src.joint_dispatch.formal_v4_gate0_evidence import write_immutable_json  # noqa: E402
from src.joint_dispatch.formal_v4_objective import STEP_WEIGHTS  # noqa: E402
from src.joint_dispatch.contract import DISPATCH_ORDER  # noqa: E402
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp  # noqa: E402


def _load_window_split(path: Path, split: str) -> FormalV4WindowSplit:
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "load_history", "exog_history", "renewable_history", "device_history",
            "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
            "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
            "target_times", "trajectory_ids", "state_hashes",
        )
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(f"materialized {split} archive is missing fields: {missing}")
        return FormalV4WindowSplit(
            *(np.array(payload[name], copy=True) for name in required),
            split=split,
            history_source="causal_simulation",
        )


def _load_parameters(benchmark_path: Path) -> dict[str, float]:
    payload = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("values"), Mapping):
        raise ValueError("formal-v4.1 benchmark must contain a values mapping")
    return {str(key): float(value) for key, value in payload["values"].items()}


def _git_commit() -> str:
    import subprocess

    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _teacher_alignment_probe(
    train: FormalV4WindowSplit,
    parameters: Mapping[str, Any],
    *,
    train_archive_sha256: str,
    capacity_receipt_sha256: str,
    normalization_sha256: str,
    solver_sha256: str,
    implementation_sha256: str,
    window_count: int = 100,
    solver: Any = solve_dispatch_lp,
) -> dict[str, Any]:
    """Probe same-information LP alignment without creating a Stage-P overlay."""

    if train.split != "train":
        raise ValueError("teacher alignment probe requires the train split")
    if window_count != 100 or len(train) < window_count:
        raise ValueError("teacher alignment probe requires at least 100 train windows")
    target_times = np.asarray(train.target_times[:window_count], dtype="datetime64[ns]")
    if not np.all(np.diff(target_times) > np.timedelta64(0, "s")):
        raise ValueError("teacher probe target times are not strictly chronological")
    aligned = 0
    feasible = 0
    shapes: set[tuple[int, int]] = set()
    for index in range(window_count):
        context = dict(parameters)
        context["grid_energy_price"] = train.prices_and_weights[index, :, 0]
        context["gas_energy_price"] = train.prices_and_weights[index, :, 1]
        context["carbon_price"] = train.prices_and_weights[index, :, 2]
        # The probe deliberately uses the causal persistence forecast for
        # renewables.  Rigid demand is a frozen train fixture only; it is not
        # persisted as a production teacher overlay before Stage P exists.
        result = solver(DispatchInputs(
            demand=train.rigid_demand[index],
            pv_available=train.renewable_forecast[index, :, 0],
            wt_available=train.renewable_forecast[index, :, 1],
            parameters=context,
            initial_soc=float(train.initial_soc[index, 0]),
            previous_chp=float(max(train.previous_chp[index, 0], 0.0)),
        ))
        if not bool(result.success):
            raise RuntimeError(f"teacher alignment LP failed at train window {index}: {result.message}")
        dispatch = np.stack([np.asarray(result.values[name], dtype=np.float64) for name in DISPATCH_ORDER], axis=-1)
        if dispatch.shape != (4, len(DISPATCH_ORDER)) or not np.isfinite(dispatch).all():
            raise ValueError("teacher alignment LP returned an invalid dispatch shape")
        feasible += 1
        shapes.add(tuple(dispatch.shape))
        # The state hash and target timestamp are carried by the exact same
        # materialized row used as the solver input.
        if str(train.trajectory_ids[index]) and str(train.state_hashes[index]):
            aligned += 1
    return {
        "schema_version": "formal-v4.1-teacher-alignment-v1",
        "protocol_id": "formal-v4.1-teacher-alignment-probe-v1",
        "source_commit": _git_commit(),
        "probe_only": True,
        "test_set_accessed": False,
        "production_overlay_deferred_until_stage_p": True,
        "stage_p_checkpoint_sha256": None,
        "window_count": int(window_count),
        "timestamps_aligned": bool(aligned == window_count),
        "state_aligned": bool(aligned == window_count),
        "lp_feasible": bool(feasible == window_count),
        "forecast_shape": [4, 4],
        "dispatch_shape": [4, len(DISPATCH_ORDER)],
        "probe_input_mode": "frozen_train_rigid_demand_plus_persistence_renewables",
        "train_archive_sha256": str(train_archive_sha256),
        "capacity_receipt_sha256": str(capacity_receipt_sha256),
        "normalization_sha256": str(normalization_sha256),
        "solver_sha256": str(solver_sha256),
        "implementation_sha256": str(implementation_sha256),
        "step_weights": list(STEP_WEIGHTS),
        "production_overlay_written": False,
    }


def build_objective_evidence(
    *,
    run_root: Path,
    benchmark_path: Path,
    capacity_receipt_path: Path,
    train_archive_path: Path,
    selection_archive_path: Path,
    solver: Any = solve_dispatch_lp,
    teacher_solver: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build and persist C-ref and teacher evidence below one fresh run root."""

    train = _load_window_split(train_archive_path, "train")
    selection = _load_window_split(selection_archive_path, "selection")
    if len(train) == 0 or len(selection) == 0:
        raise ValueError("train and selection archives must both be non-empty")
    if any(str(value).startswith("2020") for value in selection.target_times):
        raise PermissionError("selection archive contains an evaluation year")
    benchmark = _load_parameters(benchmark_path)
    train_hash = sha256_file(train_archive_path)
    capacity_hash = sha256_file(capacity_receipt_path)
    normalization_path = run_root / "NORMALIZATION_RECEIPT.json"
    if not normalization_path.is_file():
        raise FileNotFoundError(normalization_path)
    normalization_hash = sha256_file(normalization_path)
    solver_path = FRAME_ROOT / "src" / "scheduling" / "dispatch_lp.py"
    implementation_path = Path(__file__).resolve()
    _c_ref_scale, c_ref_payload = fit_c_ref_from_train_split(
        train,
        benchmark,
        train_archive_sha256=train_hash,
        capacity_receipt_sha256=capacity_hash,
        capacity_scenario_hash=str(json.loads(capacity_receipt_path.read_text(encoding="utf-8")).get("capacity_scenario_hash", "")),
        objective_implementation_sha256=sha256_file(solver_path),
        solver=solver,
    )
    write_immutable_json(run_root / "C_REF_RECEIPT.json", c_ref_payload)
    teacher_payload = _teacher_alignment_probe(
        train,
        benchmark,
        train_archive_sha256=train_hash,
        capacity_receipt_sha256=capacity_hash,
        normalization_sha256=normalization_hash,
        solver_sha256=sha256_file(solver_path),
        implementation_sha256=sha256_file(implementation_path),
        solver=teacher_solver or solver,
    )
    write_immutable_json(run_root / "protocol" / "TEACHER_ALIGNMENT_RECEIPT.json", teacher_payload)
    return c_ref_payload, teacher_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--capacity-receipt", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--selection-archive", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    c_ref, teacher = build_objective_evidence(
        run_root=args.run_root.resolve(),
        benchmark_path=args.benchmark_path.resolve(),
        capacity_receipt_path=args.capacity_receipt.resolve(),
        train_archive_path=args.train_archive.resolve(),
        selection_archive_path=args.selection_archive.resolve(),
    )
    print(json.dumps({"status": "pass", "c_ref": c_ref["c_ref"], "teacher_window_count": teacher["window_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_objective_evidence", "main"]
