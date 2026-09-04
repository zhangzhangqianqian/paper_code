from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.run_rsc_pf_formal_v4_resource_gate import build_resource_projection
from src.joint_dispatch.formal_v4_gate0_evidence import validate_gate0_receipt


def _archive(path: Path, rows: int = 500, split: str = "train") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rigid_demand=np.ones((rows, 4, 3)),
        renewable_forecast=np.ones((rows, 4, 2)),
        prices_and_weights=np.zeros((rows, 4, 3)),
        split=np.asarray(split),
    )


def test_resource_gate_covers_all_nine_methods_with_train_only_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "run"
    root.mkdir()
    archive = root / "data" / "train.npz"
    _archive(archive)
    monkeypatch.setattr("scripts.run_rsc_pf_formal_v4_resource_gate.shutil.disk_usage", lambda path: type("Usage", (), {"free": 80, "total": 100})())
    payload = build_resource_projection(run_root=root, train_archive=archive)
    assert payload["probe_only"] is True
    assert payload["test_set_accessed"] is False
    assert len(payload["rows"]) == 9
    assert {row["sample_count"] for row in payload["rows"]} == {500}
    validate_gate0_receipt("resource_projection", payload, run_root=root)


def test_resource_gate_rejects_selection_archive(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    archive = root / "data" / "selection.npz"
    _archive(archive, split="selection")
    with pytest.raises(ValueError, match="train-only"):
        build_resource_projection(run_root=root, train_archive=archive)
