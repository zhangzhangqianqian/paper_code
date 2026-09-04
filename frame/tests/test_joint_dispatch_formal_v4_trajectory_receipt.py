from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.build_rsc_pf_formal_v4_data import _save_trajectory, build_trajectory_receipt
from src.joint_dispatch.formal_v4_gate0_evidence import validate_gate0_receipt
from src.joint_dispatch.formal_v4_history import SettledTrajectory, TrajectoryAudit


def _trajectory(n: int = 30) -> SettledTrajectory:
    return SettledTrajectory(
        settled_dispatch=np.zeros((n, 21), dtype=np.float64),
        activity_indicators=np.zeros((n, 6), dtype=np.float64),
        initial_soc=np.full((n, 1), 0.5),
        previous_chp=np.zeros((n, 1)),
        next_soc=np.full((n, 1), 0.5),
        next_previous_chp=np.zeros((n, 1)),
        settled_mask=np.ones(n, dtype=bool),
        target_times=np.datetime64("2015-01-01T00:00") + np.arange(n).astype("timedelta64[h]"),
        trajectory_id="trajectory-test",
        audit=TrajectoryAudit(
            solved_hours=n,
            settled_hours=n,
            warmup_hours=24,
            future_label_reads=0,
            max_balance_residual=0.0,
            max_conversion_residual=0.0,
            max_soc_recursion_residual=0.0,
            max_chp_ramp_violation=0.0,
            max_renewable_availability_violation=0.0,
            max_soc_bound_violation=0.0,
            reset_indices=(0,),
            first_model_origin_index=24,
            segment_first_model_origins=(24,),
        ),
    )


def test_trajectory_receipt_binds_all_materialized_files(tmp_path: Path) -> None:
    run_root = tmp_path
    capacity = run_root / "gate0" / "CAPACITY_FREEZE.json"
    benchmark = run_root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    train = run_root / "data" / "train.npz"
    selection = run_root / "data" / "selection.npz"
    capacity.parent.mkdir(parents=True)
    benchmark.parent.mkdir(parents=True)
    train.parent.mkdir(parents=True)
    capacity.write_bytes(b"capacity")
    benchmark.write_bytes(b"benchmark")
    train.write_bytes(b"train")
    selection.write_bytes(b"selection")
    trajectory = _trajectory()
    trajectory_path = run_root / "data" / "trajectory.npz"
    _save_trajectory(trajectory, trajectory_path)

    payload = build_trajectory_receipt(
        run_root=run_root,
        settled_trajectory=trajectory,
        train_archive_path=train,
        selection_archive_path=selection,
        capacity_receipt_path=capacity,
        benchmark_path=benchmark,
    )
    validate_gate0_receipt("trajectory", payload, run_root=run_root)
    assert payload["trajectory_sha256"] == trajectory.trajectory_sha256
    assert payload["trajectory_file_sha256"]

    train.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match"):
        validate_gate0_receipt("trajectory", payload, run_root=run_root)
