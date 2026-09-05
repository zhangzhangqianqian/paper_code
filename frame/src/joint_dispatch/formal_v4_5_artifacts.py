"""Immutable validation-selection artifacts and independent v4.5 audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .formal_v4_4_artifacts import canonical_sha256, write_json_once
from .formal_v4_5_contract import FormalV45Contract
from .formal_v4_5_training import StageReceiptV45


def _finite(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite(child)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        raise ValueError("v4.5 artifact contains a non-finite value")


def write_v45_stage_receipt(
    path: str | Path,
    receipt: StageReceiptV45,
    validation_history: Any,
    optimizer_groups: Any,
    *,
    accessed_years: list[int] | tuple[int, ...] = (2015, 2016, 2017, 2018),
) -> str:
    """Persist selection-critical state without serializing the live model."""

    history = [dict(row) for row in validation_history]
    payload = {
        "schema": "formal-v4.5-stage-selection-v1",
        "stage": receipt.stage,
        "mode": receipt.mode,
        "parent_sha256": receipt.parent_sha256,
        "selected_sha256": receipt.best_sha256,
        "terminal_sha256": receipt.terminal_sha256,
        "best_epoch": int(receipt.best_epoch),
        "epochs": int(receipt.epochs),
        "optimizer_steps": int(receipt.optimizer_steps),
        "forecast_optimizer_steps": int(receipt.forecast_optimizer_steps),
        "loss_history": list(receipt.loss_history),
        "validation_history": history,
        "selection_metric": float(receipt.selection_metric),
        "stopping_reason": receipt.stopping_reason,
        "gradient_norms": dict(receipt.gradient_norms),
        "optimizer_groups": optimizer_groups,
        "accessed_years": [int(value) for value in accessed_years],
    }
    _finite(payload)
    payload["payload_sha256"] = canonical_sha256(payload)
    return write_json_once(path, payload)


def audit_v45_selection(run_root: str | Path, contract: FormalV45Contract) -> dict[str, Any]:
    """Recompute best-epoch selection from the persisted validation history."""

    contract.validate()
    root = Path(run_root)
    path = root / "STAGE_SELECTION.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "formal-v4.5-stage-selection-v1":
        raise ValueError("v4.5 stage selection schema mismatch")
    accessed = tuple(int(value) for value in payload.get("accessed_years", ()))
    if any(value not in contract.train_years for value in accessed):
        raise ValueError("selection artifact accessed a non-training year")
    history = payload.get("validation_history")
    if not isinstance(history, list) or not history:
        raise ValueError("validation history is missing")
    eligible = [
        (index, float(row["metric"]))
        for index, row in enumerate(history)
        if bool(float(row.get("eligible", 0.0))) and np.isfinite(float(row["metric"]))
    ]
    if not eligible:
        raise ValueError("selection artifact has no eligible epoch")
    expected_epoch, expected_metric = min(eligible, key=lambda pair: pair[1])
    if int(payload.get("best_epoch", -1)) != expected_epoch or not np.isclose(float(payload.get("selection_metric")), expected_metric, rtol=0.0, atol=1.0e-12):
        raise ValueError("selection mismatch")
    return {
        "authorized_gate1": False,
        "best_epoch": expected_epoch,
        "selection_metric": expected_metric,
        "accessed_years": list(accessed),
        "payload_sha256": canonical_sha256(payload),
    }


__all__ = ["audit_v45_selection", "write_v45_stage_receipt"]
