from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from frame.src.topology_test_runner import (
    PHASE_A_SEEDS,
    PHASE_B_FORMAL_SEEDS,
    build_phase_b_run_plan,
    load_branch_freeze,
    run_phase_b,
    _validate_phase_a_reuse_source,
    validate_recorded_input_hashes,
    validate_phase_b_authorization,
)
from frame.src.training import StandardizationStats


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

    def test_stable_branch_requires_phase_a_reuse_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            freeze_path = Path(directory) / "freeze.json"
            freeze_path.write_text(json.dumps(_freeze("core_conclusion_stable")), encoding="utf-8")
            with self.assertRaises(ValueError):
                run_phase_b(
                    data_dir=directory,
                    branch_freeze_path=freeze_path,
                    output_dir=Path(directory) / "out",
                )

    def test_phase_a_reuse_source_is_hash_and_protocol_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "scheme2r" / "H4" / "seed_2026"
            source.mkdir(parents=True)
            stats = StandardizationStats(
                load_mean=np.zeros(4, dtype=np.float32),
                load_scale=np.ones(4, dtype=np.float32),
                exog_mean=np.zeros(12, dtype=np.float32),
                exog_scale=np.ones(12, dtype=np.float32),
                exog_columns=tuple(f"x{i}" for i in range(12)),
                task_columns=("electricity", "cooling", "heating", "gas"),
            )
            stats.save(source / "normalization_stats.npz")
            (source / "best_model.pt").write_bytes(b"checkpoint")
            (source / "history.json").write_text(json.dumps({"history": []}), encoding="utf-8")
            import hashlib

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            manifest = {
                "stage": "topology_protocol_pilot_phase_a",
                "status": "passed",
                "model": "scheme2r",
                "candidate_id": "H4",
                "seed": 2026,
                "years_loaded": [2017, 2018, 2019, 2020],
                "test_set_accessed": False,
                "window": {"output_shape": [8781, 4, 4]},
                "git_revision": "test-revision",
                "artifact_sha256": {
                    "best_model.pt": digest(source / "best_model.pt"),
                    "normalization_stats.npz": digest(source / "normalization_stats.npz"),
                    "history.json": digest(source / "history.json"),
                },
            }
            (source / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded, reused_stats = _validate_phase_a_reuse_source(
                source,
                {"model": "scheme2r", "candidate_id": "H4", "seed": 2026},
                expected_git_revision="test-revision",
                expected_stats=stats,
            )
            self.assertEqual(loaded["stage"], "topology_protocol_pilot_phase_a")
            self.assertTrue(np.array_equal(reused_stats.load_scale, stats.load_scale))


if __name__ == "__main__":
    unittest.main(verbosity=2)
