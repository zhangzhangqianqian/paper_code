from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_artifacts import (
    build_artifact_manifest,
    build_c_ref_receipt,
    build_normalization_receipt,
    canonical_sha256,
    persist_train_selection_artifacts,
    sha256_file,
)
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries, FormalV4Normalization, materialize_state_windows
from src.joint_dispatch.formal_v4_objective import validate_c_ref_receipt
from src.joint_dispatch.contract import DISPATCH_ORDER


def _split(split: str):
    n = 60
    ts = np.datetime64("2018-01-01") + np.arange(n).astype("timedelta64[h]")
    if split == "selection":
        ts = np.datetime64("2019-01-01") + np.arange(n).astype("timedelta64[h]")
    tasks = np.ones((n, 4), dtype=np.float64)
    load = np.concatenate((tasks, np.zeros((n, 12))), axis=1)
    renew = np.ones((n, 2), dtype=np.float64)
    base = FormalV4BaseSeries(load, renew, renew, np.ones((n, 3)), ts, "train" if split == "train" else "selection")
    receipt = {"gate0_authorized": True, "capacity_scenario_hash": "cap"}
    return materialize_state_windows(base, np.zeros((n, len(DISPATCH_ORDER))), capacity_receipt=receipt)


def test_artifact_manifest_is_immutable_and_detects_tampering(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    manifest = build_artifact_manifest(tmp_path, ["a.txt"], lineage={"capacity": "cap"})
    assert manifest.files["a.txt"] == sha256_file(tmp_path / "a.txt")
    manifest.save(tmp_path / "manifest.json")
    with pytest.raises(FileExistsError):
        manifest.save(tmp_path / "manifest.json")
    (tmp_path / "a.txt").write_text("tampered", encoding="utf-8")
    current = build_artifact_manifest(tmp_path, ["a.txt"], lineage={"capacity": "cap"})
    assert current.files != manifest.files


def test_normalization_receipt_is_train_only_and_hash_bound(tmp_path: Path) -> None:
    train = _split("train")
    norm = FormalV4Normalization.fit(train)
    receipt = build_normalization_receipt(norm, train_archive_sha256="a" * 64, capacity_receipt_sha256="b" * 64, train_timestamps=train.target_times)
    assert receipt["fitted_split"] == "train"
    assert receipt["train_archive_sha256"] == "a" * 64
    assert len(receipt["normalization_sha256"]) == 64
    assert receipt["statistics"]["load_mean"]


def test_c_ref_receipt_rejects_tampered_scale() -> None:
    _, receipt = build_c_ref_receipt(np.array([1.0, 2.0, 3.0]), train_archive_sha256="a" * 64, capacity_receipt_sha256="b" * 64, capacity_scenario_hash="cap")
    scale = validate_c_ref_receipt(receipt, train_archive_sha256="a" * 64, capacity_receipt_sha256="b" * 64)
    assert scale.c_ref == 2.0
    broken = dict(receipt)
    broken["c_ref"] = 999.0
    with pytest.raises(ValueError, match="hash"):
        validate_c_ref_receipt(broken)


def test_persist_train_selection_requires_empty_run_root(tmp_path: Path) -> None:
    train, selection = _split("train"), _split("selection")
    receipt = tmp_path / "capacity.json"
    receipt.write_text(json.dumps({"gate0_authorized": True}), encoding="utf-8")
    root = tmp_path / "run"
    manifest = persist_train_selection_artifacts(train, selection, run_root=root, capacity_receipt=receipt)
    assert (root / "data" / "train.npz").exists()
    assert (root / "NORMALIZATION_RECEIPT.json").exists()
    assert (root / "ARTIFACT_MANIFEST.json").exists()
    with pytest.raises(FileExistsError):
        persist_train_selection_artifacts(train, selection, run_root=root, capacity_receipt=receipt)
