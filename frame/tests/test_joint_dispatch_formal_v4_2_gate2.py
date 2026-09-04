from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np

from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract
from src.joint_dispatch.formal_v4_2_methods import registered_method_rows


GATE2 = run_path(str(Path(__file__).parents[1] / "scripts" / "run_rsc_pf_formal_v4_2_gate2.py"))
AUDIT = run_path(str(Path(__file__).parents[1] / "scripts" / "audit_rsc_pf_formal_v4_2_gate2.py"))


def _contract():
    return load_formal_v4_2_contract("configs/joint_forecast_dispatch_formal_v4_2.json")


def _valid_matrix():
    rows = {}
    for row in registered_method_rows(_contract(), gate="gate2"):
        key = (row.method_id, row.seed)
        rows[key] = {"method_id": row.method_id, "seed": row.seed, "status": "complete", "penalized_objective": 1.0 if row.method_id == "RSC-PF" else 2.0, "shortage_energy": 1.0, "balance_residual_max": 0.0, "capacity_violation_max": 0.0, "optimizer_calls": 0 if row.method_id in {"RSC-PF", "Decoupled-RSC-PF", "Direct-Policy"} else 1}
    return rows


def test_gate2_parser_exposes_no_training_budget_overrides():
    options = {action.dest for action in GATE2["build_gate2_parser"]()._actions}
    assert options.isdisjoint({"epochs", "patience", "learning_rate", "candidate_values", "validation_interval"})


def test_gate2_authorization_requires_every_registered_row():
    valid = _valid_matrix()
    missing = dict(valid); missing.pop(("Differentiable-LP", 2028))
    decision = GATE2["authorize_gate2"](missing, _contract())
    assert decision.authorized_gate3 is False
    assert "Differentiable-LP/2028" in decision.missing_rows


def test_gate2_requires_primary_direction_and_shortage_guardrail():
    valid = _valid_matrix()
    assert GATE2["authorize_gate2"](valid, _contract()).authorized_gate3 is True
    valid[("RSC-PF", 2026)]["shortage_energy"] = 1.051 * valid[("State-Conditioned-PTO", 2026)]["shortage_energy"]
    assert GATE2["authorize_gate2"](valid, _contract()).authorized_gate3 is False


def test_independent_audit_recomputes_stage_p_identity_and_extension(tmp_path):
    root = tmp_path / "run"; (root / "protocol").mkdir(parents=True)
    rows = _valid_matrix()
    rows[("State-Conditioned-PTO", 2026)]["stage_p_checkpoint_method"] = "State-Conditioned-PTO"
    (root / "gate2_rows.json").write_text(__import__("json").dumps({f"{key[0]}/{key[1]}": value for key, value in rows.items()}), encoding="utf-8")
    audit = AUDIT["audit_gate2"](root)
    assert audit.authorized_gate3 is True
    assert (root / "protocol" / "SEED_EXTENSION_AUTHORIZATION.json").is_file()


def test_failed_audit_never_authorizes_seed_extension(tmp_path):
    root = tmp_path / "run"; (root / "protocol").mkdir(parents=True)
    rows = _valid_matrix(); rows.pop(("RSC-PF", 2028))
    (root / "gate2_rows.json").write_text(__import__("json").dumps({f"{key[0]}/{key[1]}": value for key, value in rows.items()}), encoding="utf-8")
    audit = AUDIT["audit_gate2"](root)
    assert audit.authorized_gate3 is False
    assert not (root / "protocol" / "SEED_EXTENSION_AUTHORIZATION.json").exists()
