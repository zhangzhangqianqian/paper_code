from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.build_rsc_pf_formal_v4_objective_evidence import build_objective_evidence
from src.joint_dispatch.formal_v4_gate0_evidence import validate_gate0_receipt
from src.scheduling.dispatch_lp import DispatchResult
from src.joint_dispatch.contract import DISPATCH_ORDER


def _write_split(path: Path, split: str, n: int = 100) -> None:
    rng = np.random.default_rng(7)
    history_renewable = np.ones((n, 24, 2), dtype=np.float64)
    payload = {
        "load_history": np.ones((n, 24, 4)),
        "exog_history": np.ones((n, 24, 12)),
        "renewable_history": history_renewable,
        "device_history": np.zeros((n, 24, 17)),
        "activity_history": np.zeros((n, 24, 6)),
        "forecast_target": np.ones((n, 4, 4)),
        "rigid_demand": np.ones((n, 4, 3)),
        "renewable_forecast": np.repeat(history_renewable[:, -1:, :], 4, axis=1),
        "renewable_realized": np.ones((n, 4, 2)),
        "prices_and_weights": np.ones((n, 4, 3)),
        "initial_soc": np.full((n, 1), 0.5),
        "previous_chp": np.zeros((n, 1)),
        "target_times": np.datetime64("2015-01-01") + np.arange(n).astype("timedelta64[h]"),
        "trajectory_ids": np.asarray(["trajectory"] * n),
        "state_hashes": np.asarray(["state"] * n),
    }
    if split == "selection":
        payload["target_times"] = np.datetime64("2019-01-01") + np.arange(n).astype("timedelta64[h]")
    np.savez_compressed(path, **payload)


def _solver(_inputs) -> DispatchResult:
    values = {name: np.zeros(4, dtype=np.float64) for name in DISPATCH_ORDER}
    return DispatchResult("optimal", "stub", 2.0, values, {}, 0.0)


def test_objective_evidence_is_train_only_and_teacher_is_probe_only(tmp_path: Path) -> None:
    run_root = tmp_path
    benchmark = run_root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    capacity = run_root / "gate0" / "CAPACITY_FREEZE.json"
    train = run_root / "data" / "train.npz"
    selection = run_root / "data" / "selection.npz"
    benchmark.parent.mkdir(parents=True)
    train.parent.mkdir(parents=True)
    benchmark.write_text("schema_version: standard-ies-benchmark-v4.1\nvalues: {}\n", encoding="utf-8")
    capacity.write_text(json.dumps({"capacity_scenario_hash": "c"}), encoding="utf-8")
    (run_root / "NORMALIZATION_RECEIPT.json").write_text("{}", encoding="utf-8")
    _write_split(train, "train")
    _write_split(selection, "selection")

    c_ref, teacher = build_objective_evidence(
        run_root=run_root,
        benchmark_path=benchmark,
        capacity_receipt_path=capacity,
        train_archive_path=train,
        selection_archive_path=selection,
        solver=_solver,
        teacher_solver=_solver,
    )
    assert c_ref["source_split"] == "train"
    assert teacher["probe_only"] is True
    assert teacher["stage_p_checkpoint_sha256"] is None
    assert teacher["production_overlay_written"] is False
    validate_gate0_receipt("teacher_alignment", teacher, run_root=run_root)
    assert (run_root / "C_REF_RECEIPT.json").exists()
    assert (run_root / "protocol" / "TEACHER_ALIGNMENT_RECEIPT.json").exists()
