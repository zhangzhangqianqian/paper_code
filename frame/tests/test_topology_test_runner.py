from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from frame.src.topology_test_runner import (
    PHASE_A_SEEDS,
    PHASE_B_FORMAL_SEEDS,
    build_phase_b_run_plan,
    load_branch_freeze,
    validate_recorded_input_hashes,
    validate_phase_b_authorization,
)


def _freeze(branch: str) -> dict[str, object]:
    models = [
        {"model": "hard_share", "candidate_id": "H2"},
        {"model": "dynamic_symmetric", "candidate_id": "H1"},
        {"model": "mmoe_lite", "candidate_id": "external_fixed"},
        {"model": "ple_lite", "candidate_id": "ple_lite_fixed_v1"},
        {"model": "scheme2r", "candidate_id": "H4"},
    ]
    if branch == "core_conclusion_stable":
        models = [
            {"model": "scheme2r", "candidate_id": "H4"},
            {"model": "dynamic_symmetric", "candidate_id": "H1"},
        ]
    seeds = list(PHASE_B_FORMAL_SEEDS if branch != "core_conclusion_stable" else PHASE_A_SEEDS)
    return {
        "stage": "topology_protocol_pilot_task7",
        "branch": branch,
        "decision_rule_version": "topology_protocol_pilot_v1",
        "validation_year": 2020,
        "test_year_accessed": False,
        "git_revision": "test-revision",
        "phase_b_run_matrix": {
            "models": models,
            "seeds": seeds,
            "forecast_years": [2017, 2018, 2019, 2020, 2021],
            "training_years": [2017, 2018, 2019],
            "validation_year": 2020,
            "test_year": 2021,
        },
    }


class TopologyTestRunnerTests(unittest.TestCase):
    def test_run_plan_matches_frozen_matrix(self) -> None:
        plan = build_phase_b_run_plan(_freeze("core_prediction_changed"))
        self.assertEqual(len(plan), 25)
        self.assertEqual({row["model"] for row in plan}, {
            "hard_share", "dynamic_symmetric", "mmoe_lite", "ple_lite", "scheme2r"
        })
        self.assertEqual({row["seed"] for row in plan}, set(PHASE_B_FORMAL_SEEDS))
        stable = build_phase_b_run_plan(_freeze("core_conclusion_stable"))
        self.assertEqual(len(stable), 6)

    def test_invalid_branch_freeze_blocks_authorization(self) -> None:
        with self.assertRaises(ValueError):
            validate_phase_b_authorization({"branch": "pilot_invalid"})
        with self.assertRaises(ValueError):
            validate_phase_b_authorization({"branch": "core_conclusion_stable", "test_year_accessed": True})

    def test_missing_or_changed_freeze_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology_branch_freeze.json"
            with self.assertRaises(FileNotFoundError):
                load_branch_freeze(path)
            path.write_text(json.dumps(_freeze("core_conclusion_stable")), encoding="utf-8")
            loaded = load_branch_freeze(path)
            self.assertEqual(loaded["branch"], "core_conclusion_stable")
            path.write_text(json.dumps({**_freeze("core_conclusion_stable"), "test_year_accessed": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_branch_freeze(path)

    def test_test_windows_require_2021_only_after_authorization(self) -> None:
        from frame.src.topology_test_runner import validate_phase_b_years

        self.assertEqual(validate_phase_b_years((2017, 2018, 2019, 2020, 2021)), (2017, 2018, 2019, 2020, 2021))
        with self.assertRaises(ValueError):
            validate_phase_b_years((2017, 2018, 2019, 2020))

    def test_contract_and_audit_hash_mismatch_is_rejected(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.json"
            audit = root / "audit.json"
            contract.write_text("contract", encoding="utf-8")
            audit.write_text("audit", encoding="utf-8")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            freeze = _freeze("core_conclusion_stable")
            freeze["input_hashes"] = {
                "contract": digest(contract),
                "audit_files": {"audit.json": digest(audit)},
            }
            validate_recorded_input_hashes(freeze, contract_path=contract, audit_dir=root)
            audit.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_recorded_input_hashes(freeze, contract_path=contract, audit_dir=root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
