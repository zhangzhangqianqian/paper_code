from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.audit_rsc_pf_matched_closed_loop_2019 import audit_manifest
from scripts.run_rsc_pf_matched_closed_loop_2019 import FRAME_ROOT, load_config, validate_config


CONFIG = FRAME_ROOT / "configs" / "rsc_pf_matched_closed_loop_2019.json"


def test_config_freezes_selection_and_refuses_evaluation_paths() -> None:
    config = load_config(CONFIG)
    assert config["selection_file"].endswith("selection_full.npz")
    assert config["selection_year"] == 2019
    assert config["evaluation_year"] == 2020
    assert config["allow_evaluation_access"] is False
    broken = dict(config)
    broken["selection_file"] = "reports/2020/selection_full.npz"
    with pytest.raises(ValueError, match="forbidden"):
        validate_config(broken)


def test_audit_rejects_wrong_itransformer_optimizer_role(tmp_path: Path) -> None:
    manifest = {
        "schema": "rsc-pf-matched-closed-loop-manifest-v1",
        "selection_year": 2019,
        "evaluation_year": 2020,
        "gate1_authorized": False,
        "formal_candidate": False,
        "test_set_accessed": False,
        "evaluation_year_accessed": False,
        "gate1_failure": {"criterion": "electricity_wape_ratio", "observed": 1.0353263112927777, "limit": 1.02},
        "smoke": True,
        "origins": 2,
        "rows": [
            {"method_id": "RSC-PF", "seed": 2026, "optimizer_role": "none at inference", "inference_lp_calls": 0},
            {"method_id": "iTransformer-PTO", "seed": 2026, "optimizer_role": "exact optimizer at inference", "inference_lp_calls": 2},
        ],
        "reference": {"method_id": "Perfect-Information-MPC", "reference_lp_calls": 2},
    }
    assert audit_manifest(manifest)["rows"] == 2
    broken = copy.deepcopy(manifest)
    broken["rows"][1]["optimizer_role"] = "none at inference"
    with pytest.raises(ValueError, match="optimizer role"):
        audit_manifest(broken)
