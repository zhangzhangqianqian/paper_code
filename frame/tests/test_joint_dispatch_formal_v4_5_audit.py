from __future__ import annotations

import json
from pathlib import Path

import torch

from src.joint_dispatch.formal_v4_5_artifacts import audit_v45_selection, write_v45_stage_receipt
from src.joint_dispatch.formal_v4_5_contract import load_formal_v4_5_contract
from src.joint_dispatch.formal_v4_5_training import StageReceiptV45


def _receipt() -> StageReceiptV45:
    return StageReceiptV45(
        stage="J", mode="joint", parent_sha256="a" * 64,
        final_sha256="b" * 64, best_sha256="b" * 64, terminal_sha256="c" * 64,
        epochs=3, best_epoch=1, optimizer_steps=3, forecast_optimizer_steps=3,
        loss_history=(1.0, 0.8, 0.9),
        validation_history=({"metric": 0.5, "eligible": 1.0}, {"metric": 0.2, "eligible": 1.0}, {"metric": 0.4, "eligible": 0.0}),
        stopping_reason="early_stopped", selection_metric=0.2,
        gradient_norms={"decision_to_base": 1.0}, model=torch.nn.Linear(1, 1),
    )


def test_audit_reconstructs_best_epoch_and_rejects_tampering(tmp_path: Path) -> None:
    contract = load_formal_v4_5_contract("configs/joint_forecast_dispatch_formal_v4_5.json")
    receipt = _receipt()
    write_v45_stage_receipt(
        tmp_path / "STAGE_SELECTION.json", receipt, receipt.validation_history,
        [{"name": "base", "lr": 0.0002, "parameter_count": 1}],
    )
    result = audit_v45_selection(tmp_path, contract)
    assert result["authorized_gate1"] is False
    assert result["best_epoch"] == 1
    payload_path = tmp_path / "STAGE_SELECTION.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["selection_metric"] = 0.7
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        audit_v45_selection(tmp_path, contract)
    except ValueError as exc:
        assert "selection mismatch" in str(exc)
    else:
        raise AssertionError("tampered selection was accepted")
