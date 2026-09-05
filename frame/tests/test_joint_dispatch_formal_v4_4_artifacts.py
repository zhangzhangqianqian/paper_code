from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_4_artifacts import canonical_sha256, recompute_pilot_receipt, write_json_once, write_npz_once
from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_formal_v4_4_contract(ROOT / "configs" / "joint_forecast_dispatch_formal_v4_4.json")


def test_json_writer_rejects_nonfinite_and_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="finite"): write_json_once(path, {"x": float("inf")})
    write_json_once(path, {"x": 1.0})
    with pytest.raises(FileExistsError): write_json_once(path, {"x": 2.0})


def test_recomputed_receipt_uses_saved_arrays(tmp_path: Path) -> None:
    target = np.zeros((1, 2, 4)); target[..., 0] = 1.0; target[..., 3] = 1.0; target[0, 0, 1] = 1.0; target[0, 1, 2] = 1.0
    probability = np.zeros((1, 2, 3)); probability[..., 0] = 1.0; probability[0, 0, 1] = 1.0; probability[0, 1, 2] = 1.0
    arrays = {"prediction": target.copy(), "target": target, "probability": probability, "prior_probability": probability, "regimes": np.array([[1, 2]]), "times": np.arange(2)}
    write_npz_once(tmp_path / "PILOT_ARRAYS.npz", arrays)
    receipt = recompute_pilot_receipt(tmp_path, CONTRACT)
    assert canonical_sha256(receipt["metrics"]) == receipt["metrics_sha256"]
