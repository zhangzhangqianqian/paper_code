from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_once_json
from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract
from src.joint_dispatch.formal_v4_2_data import fit_train_normalization
from src.joint_dispatch.formal_v4_2_gate2_training import (
    iter_gate2_batches,
    load_gate2_data,
)
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json"
FIELDS = (
    "load_history", "exog_history", "renewable_history", "device_history",
    "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
    "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
    "target_times", "trajectory_ids", "state_hashes",
)


def _split(times: np.ndarray, split: str) -> FormalV4WindowSplit:
    n = len(times)
    renewable_history = np.ones((n, 24, 2), dtype=np.float64)
    return FormalV4WindowSplit(
        load_history=np.ones((n, 24, 4)),
        exog_history=np.ones((n, 24, 12)),
        renewable_history=renewable_history,
        device_history=np.ones((n, 24, 17)),
        activity_history=np.zeros((n, 24, 6)),
        forecast_target=np.ones((n, 4, 4)),
        rigid_demand=np.ones((n, 4, 3)),
        renewable_forecast=np.ones((n, 4, 2)),
        renewable_realized=np.ones((n, 4, 2)),
        prices_and_weights=np.ones((n, 4, 3)),
        initial_soc=np.full((n, 1), 0.5),
        previous_chp=np.zeros((n, 1)),
        target_times=times,
        trajectory_ids=np.asarray([f"{split}-{i}" for i in range(n)]),
        state_hashes=np.asarray([f"state-{split}-{i}" for i in range(n)]),
        split=split,
    )


def _save_split(path: Path, split: FormalV4WindowSplit) -> None:
    np.savez_compressed(path, **{name: getattr(split, name) for name in FIELDS}, split=np.asarray(split.split))


def _run_root(tmp_path: Path):
    contract = load_formal_v4_2_contract(CONFIG)
    root = tmp_path / "run"
    gate1 = root / "gate1"
    protocol = root / "protocol"
    gate1.mkdir(parents=True)
    protocol.mkdir(parents=True)
    train_times = np.asarray([
        "2015-01-01T00:00", "2016-01-01T00:00",
        "2017-01-01T00:00", "2018-01-01T00:00",
    ], dtype="datetime64[ns]")
    selection_times = np.datetime64("2019-01-01T00:00") + np.arange(1000).astype("timedelta64[h]")
    train = _split(train_times, "train")
    selection = _split(selection_times, "selection")
    _save_split(gate1 / "TRAIN_WINDOWS.npz", train)
    _save_split(gate1 / "SELECTION_WINDOWS.npz", selection)
    normalization = fit_train_normalization(train)
    write_once_json(gate1 / "NORMALIZATION.json", normalization.to_payload())
    write_once_json(gate1 / "GATE1_ORIGIN_MANIFEST.json", {"origin_indices": list(range(1000))})
    evidence = {
        "evaluation_year_accessed": False,
        "train_windows_sha256": sha256_file(gate1 / "TRAIN_WINDOWS.npz"),
        "selection_windows_sha256": sha256_file(gate1 / "SELECTION_WINDOWS.npz"),
    }
    write_once_json(gate1 / "GATE1_EVIDENCE.json", evidence)
    write_once_json(protocol / "GATE1_TRANSITION.json", {
        "authorized_gate2": True,
        "contract_sha256": contract.contract_sha256,
        "gate1_evidence_sha256": sha256_file(gate1 / "GATE1_EVIDENCE.json"),
    })
    return root, contract


def test_gate2_uses_all_eligible_train_and_evaluation_windows(tmp_path: Path) -> None:
    root, contract = _run_root(tmp_path)
    bundle = load_gate2_data(root, contract)
    assert len(bundle.train) == 4
    assert len(bundle.calibration) == 1000
    assert len(bundle.evaluation) == 1000
    assert np.all(np.diff(bundle.evaluation.target_times) == np.timedelta64(1, "h"))


def test_micro_batches_preserve_effective_batch(tmp_path: Path) -> None:
    root, contract = _run_root(tmp_path)
    bundle = load_gate2_data(root, contract)
    parts = list(iter_gate2_batches(
        bundle.evaluation,
        bundle.normalization,
        effective_batch_size=64,
        micro_batch_size=8,
    ))
    first = [part for part in parts if part.effective_batch_index == 0]
    assert len(first) == 8
    assert sum(part.accumulation_weight for part in first) == pytest.approx(1.0)
    assert sum(len(part.indices) for part in parts) == len(bundle.evaluation)


def test_gate2_rejects_gate1_contract_mismatch(tmp_path: Path) -> None:
    root, contract = _run_root(tmp_path)
    transition = root / "protocol" / "GATE1_TRANSITION.json"
    payload = json.loads(transition.read_text(encoding="utf-8"))
    payload["contract_sha256"] = "0" * 64
    transition.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="contract lineage"):
        load_gate2_data(root, contract)

