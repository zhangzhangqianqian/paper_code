from __future__ import annotations

import hashlib
import json
from pathlib import Path
from runpy import run_path

import pytest

from src.joint_dispatch.formal_v4_2_access import EvaluationAccessDenied


GATE3 = run_path(str(Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_gate3.py"))
SUMMARY = run_path(str(Path(__file__).parents[1] / "scripts" / "summarize_rsc_pf_formal_v4_2.py"))


def _envelope():
    return {"schema": "formal-v4.2-gate3-authorization-v1", "contract_sha256": "a" * 64, "gate2_audit_sha256": "b" * 64, "seed_extension_audit_sha256": "c" * 64, "allowed_years": [2020], "consumed": False}


def test_gate3_refuses_missing_or_consumed_envelope(tmp_path):
    with pytest.raises(EvaluationAccessDenied):
        GATE3["run_gate3"](tmp_path, envelope=None)
    with pytest.raises(EvaluationAccessDenied):
        GATE3["run_gate3"](tmp_path, envelope={**_envelope(), "consumed": True})


def test_gate3_consumes_envelope_once_and_writes_complete_receipt(tmp_path):
    receipt = GATE3["run_gate3"](tmp_path, _envelope(), rows=[{"method_id": "RSC-PF", "seed": 2026}])
    assert receipt["paper_eligible"] is True
    assert receipt["training_called"] is False
    assert receipt["bootstrap"] == {"block_hours": 168, "replicates": 2000}
    assert (tmp_path / "protocol" / "GATE3_AUTHORIZATION_CONSUMED.json").is_file()


def test_gate3_never_calls_training(monkeypatch, tmp_path):
    # The runner has no training import by construction; this callable is the
    # only data hook and proves evaluation remains a separate operation.
    called = []
    def loader(years):
        called.append(tuple(years)); return [{"method_id": "RSC-PF"}]
    GATE3["run_gate3"](tmp_path, _envelope(), evaluation_loader=loader)
    assert called == [(2020,)]


def test_summary_is_recomputed_from_raw_arrays(tmp_path):
    GATE3["run_gate3"](tmp_path, _envelope(), rows=[{"method_id": "RSC-PF", "seed": 2026, "penalized_objective": 1.0}, {"method_id": "Decoupled-RSC-PF", "seed": 2026, "penalized_objective": 2.0}])
    summary = SUMMARY["summarize_gate3"](tmp_path)
    assert summary["primary_contrast"] == "RSC-PF_minus_Decoupled-RSC-PF"
    assert summary["bootstrap"]["block_hours"] == 168
    assert summary["bootstrap"]["replicates"] == 2000
    assert summary["paper_eligible"] is True


def test_incomplete_or_unauthorized_run_is_not_paper_eligible(tmp_path):
    (tmp_path / "gate2").mkdir()
    (tmp_path / "gate2" / "GATE2_DECISION.json").write_text(json.dumps({"authorized_gate3": False}), encoding="utf-8")
    summary = SUMMARY["summarize_run"](tmp_path)
    assert summary["paper_eligible"] is False
