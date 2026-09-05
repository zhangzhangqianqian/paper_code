from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_rsc_pf_external_baselines import (
    _dispatch_metrics,
    write_external_validation_manifest,
)
from scripts.freeze_rsc_pf_external_validation import freeze_external_validation
from src.joint_dispatch.external_baseline_training import METHODS, SEEDS, _decoder_parameters, _external_lp_parameters
from src.joint_dispatch.external_v46_data import build_external_v46_oracle, load_external_v46_split


ROOT = Path(__file__).parents[1]


def test_dispatch_metric_receipt_keeps_explicit_lp_accounting() -> None:
    split = load_external_v46_split(
        ROOT / "reports" / "joint_forecast_dispatch_formal_v4_4" / "formal_v4_4_20260905_f" / "pilot" / "data" / "early_stop.npz",
        "validation",
    ).take(np.arange(2))
    split = split.__class__(
        load_history=split.load_history, exog_history=split.exog_history,
        device_history=split.device_history, device_status=split.device_status,
        scheduler_context=split.scheduler_context, previous_chp=split.previous_chp,
        forecast_target=split.forecast_target, renewable_realized=split.renewable_realized,
        target_times=split.target_times, split="validation", teacher_dispatch=np.zeros((2, 4, 21), dtype=np.float32),
        oracle_first_step_objective=build_external_v46_oracle(split, _external_lp_parameters()), source_path=split.source_path,
    )
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


def test_freeze_rejects_missing_validation_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        freeze_external_validation(tmp_path / "missing", ROOT / "configs" / "rsc_pf_external_baselines_v1.json")


def test_current_v46_output_is_not_frozen_before_formal_validation() -> None:
    output = ROOT / "reports" / "rsc_pf_external_baselines_v46" / "implementation"
    with pytest.raises(FileNotFoundError):
        freeze_external_validation(output, ROOT / "configs" / "rsc_pf_external_baselines_v1.json")
