from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.joint_dispatch.external_baseline_training import (
    load_external_checkpoint,
    run_external_calibration,
    run_external_validation,
)


ROOT = Path(__file__).parents[1]


def _config(tmp_path: Path, *, smoke_limit: int = 2) -> dict[str, object]:
    return {
        "source_root": str(ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation" / "sources"),
        "registry_path": str(ROOT / "configs" / "rsc_pf_external_baselines_v1.json"),
        "data_root": str(ROOT / "reports" / "joint_forecast_dispatch_v1" / "data"),
        "output_root": str(tmp_path / "outputs"),
        "smoke_limit": smoke_limit,
        "smoke_epochs": 1,
    }


def test_unknown_method_and_seed_are_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="unknown external method"):
        run_external_calibration("unknown", 2026, config)
    with pytest.raises(ValueError, match="seed must be one"):
        run_external_calibration("iTransformer-PTO", 7, config)


def test_calibration_writes_provenance_checked_checkpoint_and_resume_is_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path, smoke_limit=2)
    first = run_external_calibration("iTransformer-PTO", 2026, config)
    payload = load_external_checkpoint(first, expected_method="iTransformer-PTO", expected_seed=2026)
    assert payload["test_set_accessed"] is False
    assert payload["stage"] == "calibration"
    assert payload["split_manifest"]["train"].endswith("train.npz")
    expected = float(payload["best_validation_loss"])
    config["resume"] = True
    resumed = run_external_calibration("iTransformer-PTO", 2026, config)
    resumed_payload = load_external_checkpoint(resumed, expected_method="iTransformer-PTO", expected_seed=2026)
    assert resumed_payload["best_validation_loss"] == expected


def test_checkpoint_rejects_other_method_and_test_path(tmp_path: Path) -> None:
    config = _config(tmp_path, smoke_limit=2)
    checkpoint = run_external_calibration("iTransformer-PTO", 2026, config)
    with pytest.raises(ValueError, match="method/seed"):
        load_external_checkpoint(checkpoint, expected_method="DecisionFocused-Online", expected_seed=2026)
    test_path = tmp_path / "sealed_test" / "checkpoint.pt"
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(checkpoint.read_bytes())
    with pytest.raises(ValueError, match="sealed test-set"):
        load_external_checkpoint(test_path, expected_method="iTransformer-PTO", expected_seed=2026)


def test_validation_requires_calibration_gate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="calibration gate"):
        run_external_validation("iTransformer-PTO", 2026, config)

