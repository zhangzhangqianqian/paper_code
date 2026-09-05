"""Train-information-only risk-cap fitting and immutable receipts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .formal_v4_4_artifacts import canonical_sha256, write_json_once, write_npz_once


_ALLOWED_YEARS = (2015, 2016, 2017, 2018)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _years(timestamps: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(timestamps, dtype="datetime64[ns]")
    return tuple(sorted(set(values.astype("datetime64[Y]").astype(int) + 1970)))


def _lineage_payload(receipt: "RiskCapReceiptV46") -> dict[str, Any]:
    return {
        "schema": "formal-v4.6-risk-cap-v1",
        "cap_quantile": float(receipt.quantile),
        "split_role": receipt.split_role,
        "years": list(receipt.years),
        "contract_sha256": receipt.contract_sha256,
        "parent_sha256": receipt.parent_sha256,
        "arrays_sha256": receipt.arrays_sha256,
        "residual_sha256": receipt.residual_sha256,
        "timestamps_sha256": receipt.timestamps_sha256,
    }


@dataclass(frozen=True)
class RiskCapReceiptV46:
    cap: np.ndarray
    residual: np.ndarray
    timestamps: np.ndarray
    split_role: str
    quantile: float
    years: tuple[int, ...]
    contract_sha256: str
    parent_sha256: str
    arrays_sha256: str
    residual_sha256: str
    timestamps_sha256: str
    lineage_sha256: str

    def __post_init__(self) -> None:
        cap = np.asarray(self.cap, dtype=np.float64)
        residual = np.asarray(self.residual, dtype=np.float64)
        timestamps = np.asarray(self.timestamps, dtype="datetime64[ns]")
        if cap.shape != (4, 3) or residual.ndim != 3 or residual.shape[1:] != (4, 3):
            raise ValueError("risk caps must have shapes [4,3] and [N,4,3]")
        if timestamps.shape != (residual.shape[0],):
            raise ValueError("risk-cap timestamps must align with residuals")
        if not np.isfinite(cap).all() or not np.isfinite(residual).all() or (cap < 0.0).any() or (residual < 0.0).any():
            raise ValueError("risk caps and residuals must be finite and non-negative")
        if self.split_role != "early_stop" or float(self.quantile) != 0.90:
            raise ValueError("risk caps require early_stop and quantile 0.90")
        if tuple(int(year) for year in self.years) != _years(timestamps) or any(year not in _ALLOWED_YEARS for year in self.years):
            raise ValueError("risk-cap years must be a subset of 2015--2018")
        for name in ("contract_sha256", "parent_sha256", "arrays_sha256", "residual_sha256", "timestamps_sha256", "lineage_sha256"):
            if len(str(getattr(self, name))) != 64:
                raise ValueError(f"{name} must be a SHA-256 digest")
        object.__setattr__(self, "cap", cap)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "years", tuple(int(year) for year in self.years))


def fit_risk_caps_v46(
    prediction: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    split_role: str,
    quantile: float,
    contract_sha256: str,
    parent_sha256: str,
) -> RiskCapReceiptV46:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (4, 4):
        raise ValueError("P1 prediction and target must have shape [N,4,4]")
    if timestamps.shape != (prediction.shape[0],):
        raise ValueError("risk-cap timestamps must align with predictions")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("risk-cap inputs must be finite")
    if split_role != "early_stop":
        raise ValueError("risk caps require the purged early_stop split")
    if float(quantile) != 0.90:
        raise ValueError("risk caps require quantile 0.90")
    years = _years(timestamps)
    if not years or any(year not in _ALLOWED_YEARS for year in years):
        raise ValueError("risk-cap years must be a subset of 2015--2018")
    residual = np.abs(target[..., :3] - prediction[..., :3])
    cap = np.maximum(np.quantile(residual, 0.90, axis=0), 1.0e-6)
    arrays_sha256 = _array_sha256(np.asarray(cap, dtype="<f8"))
    residual_sha256 = _array_sha256(np.asarray(residual, dtype="<f8"))
    timestamps_sha256 = _array_sha256(np.asarray(timestamps, dtype="datetime64[ns]"))
    provisional = RiskCapReceiptV46(
        cap=cap, residual=residual, timestamps=timestamps, split_role="early_stop",
        quantile=0.90, years=years, contract_sha256=str(contract_sha256),
        parent_sha256=str(parent_sha256), arrays_sha256=arrays_sha256,
        residual_sha256=residual_sha256, timestamps_sha256=timestamps_sha256,
        lineage_sha256="0" * 64,
    )
    return RiskCapReceiptV46(
        cap=provisional.cap, residual=provisional.residual,
        timestamps=provisional.timestamps, split_role=provisional.split_role,
        quantile=provisional.quantile, years=provisional.years,
        contract_sha256=provisional.contract_sha256,
        parent_sha256=provisional.parent_sha256,
        arrays_sha256=provisional.arrays_sha256,
        residual_sha256=provisional.residual_sha256,
        timestamps_sha256=provisional.timestamps_sha256,
        lineage_sha256=canonical_sha256(_lineage_payload(provisional)),
    )


def _json_payload(receipt: RiskCapReceiptV46) -> dict[str, Any]:
    return {
        **_lineage_payload(receipt),
        "cap": receipt.cap.tolist(),
        "years": list(receipt.years),
        "lineage_sha256": receipt.lineage_sha256,
    }


def save_risk_caps_v46(root: str | Path, receipt: RiskCapReceiptV46) -> str:
    root = Path(root)
    json_path = root / "RISK_CAPS.json"
    npz_path = root / "RISK_CAPS.npz"
    if json_path.exists() or npz_path.exists():
        raise FileExistsError("risk-cap receipt already exists")
    write_npz_once(npz_path, {
        "cap": receipt.cap,
        "residual": receipt.residual,
        "timestamps": receipt.timestamps,
    })
    return write_json_once(json_path, _json_payload(receipt))


def load_risk_caps_v46(root: str | Path) -> RiskCapReceiptV46:
    root = Path(root)
    payload = json.loads((root / "RISK_CAPS.json").read_text(encoding="utf-8"))
    with np.load(root / "RISK_CAPS.npz", allow_pickle=False) as arrays:
        cap = np.asarray(arrays["cap"], dtype=np.float64)
        residual = np.asarray(arrays["residual"], dtype=np.float64)
        timestamps = np.asarray(arrays["timestamps"], dtype="datetime64[ns]")
    if not np.allclose(cap, np.asarray(payload.get("cap"), dtype=np.float64), rtol=0.0, atol=0.0):
        raise ValueError("risk-cap JSON and NPZ cap mismatch")
    receipt = RiskCapReceiptV46(
        cap=cap, residual=residual, timestamps=timestamps,
        split_role=str(payload["split_role"]), quantile=float(payload["cap_quantile"]),
        years=tuple(int(year) for year in payload["years"]),
        contract_sha256=str(payload["contract_sha256"]),
        parent_sha256=str(payload["parent_sha256"]),
        arrays_sha256=str(payload["arrays_sha256"]),
        residual_sha256=str(payload["residual_sha256"]),
        timestamps_sha256=str(payload["timestamps_sha256"]),
        lineage_sha256=str(payload["lineage_sha256"]),
    )
    if receipt.arrays_sha256 != _array_sha256(np.asarray(receipt.cap, dtype="<f8")):
        raise ValueError("risk-cap array hash mismatch")
    if receipt.residual_sha256 != _array_sha256(np.asarray(receipt.residual, dtype="<f8")):
        raise ValueError("risk residual hash mismatch")
    if receipt.timestamps_sha256 != _array_sha256(np.asarray(receipt.timestamps, dtype="datetime64[ns]")):
        raise ValueError("risk timestamp hash mismatch")
    if receipt.lineage_sha256 != canonical_sha256(_lineage_payload(receipt)):
        raise ValueError("risk-cap lineage hash mismatch")
    return receipt


__all__ = ["RiskCapReceiptV46", "fit_risk_caps_v46", "save_risk_caps_v46", "load_risk_caps_v46"]
