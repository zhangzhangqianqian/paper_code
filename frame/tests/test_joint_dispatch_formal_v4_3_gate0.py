from __future__ import annotations

from scripts.run_rsc_pf_formal_v4_3_gate0 import CHECKS, authorize_v43_pilot


def test_gate0_denies_pilot_when_any_check_fails() -> None:
    evidence = {"checks": {name: True for name in CHECKS}}
    evidence["checks"]["model_shape"] = False
    decision = authorize_v43_pilot(evidence)
    assert decision["authorized_pilot"] is False
    assert decision["evaluation_year_accessed"] is False


def test_gate0_authorizes_only_all_preflight_checks() -> None:
    evidence = {"checks": {name: True for name in CHECKS}}
    decision = authorize_v43_pilot(evidence)
    assert decision["authorized_pilot"] is True
    assert decision["paper_eligible"] is False
