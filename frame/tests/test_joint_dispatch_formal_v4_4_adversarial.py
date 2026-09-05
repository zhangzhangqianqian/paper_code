from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_rsc_pf_formal_v4_4_pilot import audit_run
from tests.test_joint_dispatch_formal_v4_4_pilot import CAPACITY, CONTRACT, REPORT, SELECTION, SOURCE, TRAIN, BENCHMARK, _authorized_gate0, _executor
from src.joint_dispatch.formal_v4_4_pilot import run_pilot_v44


def _valid_run(tmp_path: Path) -> Path:
    transition = _authorized_gate0(tmp_path)
    run_pilot_v44(contract_path=CONTRACT, gate0_transition=transition, source_manifest=SOURCE, base_train_data=TRAIN, base_selection_data=SELECTION, benchmark=BENCHMARK, capacity_receipt=CAPACITY, output_root=tmp_path, run_id="pilot", stage_executor=_executor)
    return tmp_path / "pilot"


def test_valid_run_passes_independent_audit(tmp_path: Path) -> None:
    run = _valid_run(tmp_path); result = audit_run(run, CONTRACT)
    assert result.authorized_gate1


def test_integrity_faults_deny_transition(tmp_path: Path) -> None:
    run = _valid_run(tmp_path); path = run / "pilot" / "PILOT_RECEIPT.json"; payload = json.loads(path.read_text(encoding="utf-8")); payload["joint"]["gradient_norms"]["decision_to_gate"] = 0.0; path.write_text(json.dumps(payload), encoding="utf-8")
    result = audit_run(run, CONTRACT)
    assert not result.authorized_gate1
    assert "gradient" in result.failures
