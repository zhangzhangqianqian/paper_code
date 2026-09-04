from __future__ import annotations

import json
from pathlib import Path
from runpy import run_path

import pytest

from src.joint_dispatch.formal_v4_2_access import EvaluationAccessDenied


EXT = run_path(str(Path(__file__).parents[1] / "scripts" / "extend_rsc_pf_formal_v4_2_seeds.py"))


def test_seed_extension_creates_only_missing_frozen_seeds(tmp_path: Path):
    receipt = EXT["run_seed_extension"]({"root": tmp_path, "authorization": {"allowed_seeds": [2029, 2030], "allow_evaluation_year": False}})
    assert receipt["trained_seeds"] == [2029, 2030]
    assert receipt["changed_frozen_configuration"] is False
    assert receipt["evaluation_year_accessed"] is False


def test_gate3_envelope_requires_all_five_checkpoint_seeds(tmp_path: Path):
    EXT["run_seed_extension"]({"root": tmp_path, "authorization": {"allowed_seeds": [2029, 2030], "allow_evaluation_year": False}})
    path = tmp_path / "checkpoints" / "RSC-PF" / "seed_2030.ckpt"; path.unlink()
    with pytest.raises(EvaluationAccessDenied):
        EXT["audit_extension_and_mint_gate3"](tmp_path)


def test_gate3_envelope_keeps_evaluation_year_locked(tmp_path: Path):
    EXT["run_seed_extension"]({"root": tmp_path, "authorization": {"allowed_seeds": [2029, 2030], "allow_evaluation_year": False}})
    audit = {"authorized_seed_extension": True, "all_five_seeds_ready": True, "evaluation_year_accessed": False}
    envelope = EXT["mint_gate3_envelope"]({"authorized_seed_extension": True}, audit, "a" * 64)
    assert envelope["allowed_years"] == [2020] and envelope["consumed"] is False
