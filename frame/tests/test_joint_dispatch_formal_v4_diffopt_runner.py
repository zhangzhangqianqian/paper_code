from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_rsc_pf_formal_v4_diffopt_gate as gate
from src.joint_dispatch.formal_v4_diffopt import validate_diffopt_gate_receipt


def _archive(path: Path, split: str = "train", rows: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rigid_demand=np.ones((rows, 4, 3)),
        renewable_forecast=np.ones((rows, 4, 2)),
        prices_and_weights=np.zeros((rows, 4, 3)),
        initial_soc=np.full((rows, 1), 0.5),
        previous_chp=np.zeros((rows, 1)),
        split=np.asarray(split),
        target_times=np.datetime64("2015-01-01T00", "h") + np.arange(rows).astype("timedelta64[h]"),
    )


def test_diffopt_inputs_are_bound_to_run_root_and_train_only(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    inside = root / "data" / "train.npz"
    _archive(inside, rows=4)
    assert gate._resolve_inside(root, Path("data/train.npz"), "train") == inside.resolve()
    with pytest.raises(ValueError, match="inside"):
        gate._resolve_inside(root, tmp_path / "legacy" / "standard_ies_benchmark_v1.yaml", "benchmark")
    demand, renew, prices, soc, chp, origins = gate._fixed_training_windows(inside, {"grid_emission_factor": 0.5, "gas_emission_factor": 0.25}, 3)
    assert demand.shape == (3, 4, 3)
    assert renew.shape == (3, 4, 2)
    assert prices.shape == (3, 4, 4)
    assert soc.shape == (3,) and chp.shape == (3,) and origins.tolist() == [0, 1, 2]
    selection = root / "data" / "selection.npz"
    _archive(selection, split="selection", rows=4)
    with pytest.raises(ValueError, match="train-only"):
        gate._fixed_training_windows(selection, {}, 3)


def test_diffopt_gate_requires_the_isolated_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"python_executable": str(Path(__file__).resolve().parents[2] / "envs" / "rsc_pf_diffopt_v4" / "python.exe"), "native_layer_probe": {"eligible_for_gate0": True}}), encoding="utf-8")
    monkeypatch.setattr(gate.sys, "executable", str(Path("D:/wrong/python.exe")))
    with pytest.raises(RuntimeError, match="isolated environment"):
        gate._require_isolated_environment(lock)


def test_diffopt_receipt_hashes_are_bound_to_current_run_root(tmp_path: Path):
    root = tmp_path / "run"
    (root / "protocol").mkdir(parents=True)
    benchmark = root / "benchmark.yaml"
    archive = root / "train.npz"
    benchmark.write_text("values: {}\n", encoding="utf-8")
    _archive(archive, rows=100)
    import hashlib
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "schema_version": "formal-v4.1-differentiable-lp-gate-v1",
        "test_set_accessed": False, "windows": 100, "training_years": [2015, 2016, 2017, 2018],
        "benchmark_path": "benchmark.yaml", "train_archive_path": "train.npz",
        "benchmark_sha256": digest(benchmark), "train_archive_sha256": digest(archive),
    }
    validate_diffopt_gate_receipt(payload, run_root=root)
    payload["benchmark_path"] = "../legacy.yaml"
    with pytest.raises(ValueError, match="escapes"):
        validate_diffopt_gate_receipt(payload, run_root=root)
