import json
import unittest
from pathlib import Path

from scripts.run_stage7_7 import (
    EXPECTED_CANDIDATES,
    build_stage7_7_run_plan,
    validate_stage7_7_contract,
)


FRAME_ROOT = Path(__file__).resolve().parents[1]


class Stage77JointBaselinePlanTest(unittest.TestCase):
    def test_plan_has_exactly_thirty_unique_runs(self):
        plan = build_stage7_7_run_plan()

        self.assertEqual(len(plan), 30)
        self.assertEqual(len({row["run_id"] for row in plan}), 30)
        self.assertEqual(
            {row["model"] for row in plan},
            {"hard_share", "dynamic_symmetric", "ple-lite"},
        )
        self.assertEqual(
            {row["protocol"] for row in plan}, {"full", "small_sample"}
        )
        self.assertEqual(
            {row["seed"] for row in plan},
            {2026, 2027, 2028, 2029, 2030},
        )
        self.assertEqual(
            {(row["model"], row["candidate_id"]) for row in plan},
            set(EXPECTED_CANDIDATES.items()),
        )

    def test_non_frozen_seed_list_is_rejected(self):
        with self.assertRaises(ValueError):
            build_stage7_7_run_plan(seeds=(2026,))

    def test_contract_matches_run_matrix_and_forbids_test_selection(self):
        contract_path = (
            FRAME_ROOT / "configs" / "stage7_7_joint_baselines_contract.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))

        validate_stage7_7_contract(contract)

        self.assertEqual(contract["expected_formal_run_count"], 30)
        self.assertFalse(contract["test_policy"]["test_used_for_selection"])
        self.assertFalse(contract["input_output"]["future_exogenous_allowed"])

    def test_wrong_candidate_is_rejected(self):
        contract_path = (
            FRAME_ROOT / "configs" / "stage7_7_joint_baselines_contract.json"
        )
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["models"][0]["candidate_id"] = "H4"

        with self.assertRaises(ValueError):
            validate_stage7_7_contract(contract)


if __name__ == "__main__":
    unittest.main()
