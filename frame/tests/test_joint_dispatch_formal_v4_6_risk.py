from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.joint_dispatch.formal_v4_6_risk import (
    RiskCapReceiptV46,
    fit_risk_caps_v46,
    load_risk_caps_v46,
    save_risk_caps_v46,
)


def make_cap_receipt(year: int = 2018) -> RiskCapReceiptV46:
    prediction = np.zeros((5, 4, 4), dtype=np.float64)
    target = np.arange(80, dtype=np.float64).reshape(5, 4, 4)
    timestamps = np.array([f"{year}-01-01"] * 5, dtype="datetime64[ns]")
    return fit_risk_caps_v46(
        prediction=prediction, target=target, timestamps=timestamps,
        split_role="early_stop", quantile=0.90,
        contract_sha256="a" * 64, parent_sha256="b" * 64,
    )


def test_caps_are_early_stop_residual_quantiles_and_exclude_gas() -> None:
    prediction = torch.zeros(5, 4, 4)
    target = torch.arange(80, dtype=torch.float32).reshape(5, 4, 4)
    receipt = fit_risk_caps_v46(
        prediction=prediction.numpy(), target=target.numpy(),
        timestamps=np.array(["2018-01-01"] * 5, dtype="datetime64[ns]"),
        split_role="early_stop", quantile=0.90,
        contract_sha256="a" * 64, parent_sha256="b" * 64,
    )
    expected = torch.quantile(target[..., :3].abs(), 0.90, dim=0)
    assert np.allclose(receipt.cap, expected.numpy())
    assert receipt.cap.shape == (4, 3)
    assert receipt.residual.shape == (5, 4, 3)
    assert receipt.split_role == "early_stop"
    assert receipt.years == (2018,)


def test_caps_reject_2019_or_2020_and_tampered_lineage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2015--2018"):
        make_cap_receipt(year=2019)
    receipt = make_cap_receipt(year=2018)
    save_risk_caps_v46(tmp_path, receipt)
    payload = json.loads((tmp_path / "RISK_CAPS.json").read_text(encoding="utf-8"))
    payload["cap"][0][0] += 1.0
    (tmp_path / "RISK_CAPS.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash|lineage|cap"):
        load_risk_caps_v46(tmp_path)
