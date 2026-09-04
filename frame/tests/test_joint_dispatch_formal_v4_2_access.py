from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_2_access import (
    EvaluationAccessDenied,
    FormalV42AccessController,
    validate_gate3_envelope,
)


CONTRACT_HASH = "a" * 64


def _envelope() -> dict[str, object]:
    return {
        "schema": "formal-v4.2-gate3-authorization-v1",
        "contract_sha256": CONTRACT_HASH,
        "gate2_audit_sha256": "b" * 64,
        "seed_extension_audit_sha256": "c" * 64,
        "allowed_years": [2020],
        "consumed": False,
    }


def test_gate2_cannot_request_2020_and_denial_is_recorded(tmp_path: Path) -> None:
    controller = FormalV42AccessController(
        run_root=tmp_path,
        gate="gate2",
        contract_sha256=CONTRACT_HASH,
    )
    with pytest.raises(EvaluationAccessDenied, match="Gate 3"):
        controller.request_years("evaluation", [2020], caller="gate2")
    events = sorted((tmp_path / "access" / "events").glob("*.json"))
    assert len(events) == 1
    assert json.loads(events[0].read_text(encoding="utf-8"))["decision"] == "deny"


def test_gate3_requires_a_valid_unconsumed_envelope(tmp_path: Path) -> None:
    controller = FormalV42AccessController(
        run_root=tmp_path,
        gate="gate3",
        contract_sha256=CONTRACT_HASH,
    )
    with pytest.raises(EvaluationAccessDenied, match="authorization"):
        controller.request_years("evaluation", [2020], caller="gate3")


def test_gate3_consumes_authorization_once(tmp_path: Path) -> None:
    controller = FormalV42AccessController(
        run_root=tmp_path,
        gate="gate3",
        contract_sha256=CONTRACT_HASH,
        gate3_envelope=_envelope(),
    )
    assert controller.request_years("evaluation", [2020], caller="gate3") == (2020,)
    assert (tmp_path / "protocol" / "GATE3_AUTHORIZATION_CONSUMED.json").is_file()
    restarted = FormalV42AccessController(
        run_root=tmp_path,
        gate="gate3",
        contract_sha256=CONTRACT_HASH,
        gate3_envelope=_envelope(),
    )
    with pytest.raises(EvaluationAccessDenied, match="already consumed"):
        restarted.request_years("evaluation", [2020], caller="gate3-retry")


@pytest.mark.parametrize("gate", ["gate0", "pilot", "gate1", "gate2", "gate3"])
def test_2021_is_always_excluded(tmp_path: Path, gate: str) -> None:
    controller = FormalV42AccessController(
        run_root=tmp_path / gate,
        gate=gate,
        contract_sha256=CONTRACT_HASH,
        gate3_envelope=_envelope() if gate == "gate3" else None,
    )
    with pytest.raises(EvaluationAccessDenied, match="2021"):
        controller.request_years("evaluation", [2021], caller="test")


def test_train_and_selection_boundaries_are_exact(tmp_path: Path) -> None:
    controller = FormalV42AccessController(
        run_root=tmp_path,
        gate="gate1",
        contract_sha256=CONTRACT_HASH,
    )
    assert controller.request_years("train", [2015, 2016, 2017, 2018], caller="train") == (2015, 2016, 2017, 2018)
    assert controller.request_years("selection", [2019], caller="selection") == (2019,)
    with pytest.raises(EvaluationAccessDenied, match="split boundary"):
        controller.request_years("train", [2019], caller="bad")


def test_envelope_hash_and_year_are_validated() -> None:
    validate_gate3_envelope(_envelope(), CONTRACT_HASH)
    bad = dict(_envelope())
    bad["contract_sha256"] = "0" * 64
    with pytest.raises(EvaluationAccessDenied, match="contract hash"):
        validate_gate3_envelope(bad, CONTRACT_HASH)
