from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_rsc_pf_external_baselines import (
    _dispatch_metrics,
    write_external_validation_manifest,
)
from src.joint_dispatch.data import load_joint_split
from src.joint_dispatch.external_baseline_training import METHODS, SEEDS


ROOT = Path(__file__).parents[1]


def test_dispatch_metric_receipt_keeps_explicit_lp_accounting() -> None:
    split, _, _ = load_joint_split(ROOT / "reports" / "joint_forecast_dispatch_v1" / "data" / "validation.npz")
    split = split.take(np.arange(2))
    dispatch = np.zeros((2, 4, 21), dtype=np.float32)
    summary, raw = _dispatch_metrics(dispatch, split, lp_calls=2)
    assert summary["exact_lp_calls"] == 2
    assert summary["windows"] == 2
    assert raw["regret_vs_oracle"].shape == (2,)
    assert 0.0 <= summary["feasibility_rate"] <= 1.0


def test_validation_manifest_requires_all_method_seed_receipts(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    for method in METHODS:
        safe = method.replace("-", "_")
        for seed in SEEDS:
            directory = root / "validation" / safe / f"seed_{seed}" / "evaluation"
            directory.mkdir(parents=True)
            (directory / "evaluation_receipt.json").write_text(json.dumps({
                "method_id": method, "seed": seed, "test_set_accessed": False,
                "metrics": {"forecast_mae_mean": 0.1},
            }), encoding="utf-8")
    manifest = write_external_validation_manifest(root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 15
    assert payload["test_set_accessed"] is False
    assert payload["optimizer_roles"]["DecisionFocused-Online"] == "exact optimizer at inference"


def test_validation_manifest_rejects_incomplete_or_test_receipt(tmp_path: Path) -> None:
    root = tmp_path / "outputs"
    directory = root / "validation" / "iTransformer_PTO" / "seed_2026" / "evaluation"
    directory.mkdir(parents=True)
    (directory / "evaluation_receipt.json").write_text(json.dumps({"test_set_accessed": True}), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        write_external_validation_manifest(root)
