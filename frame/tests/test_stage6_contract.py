import copy
import unittest

from src.stage6_contract import (
    load_stage6_selection_contract,
    validate_stage6_selection_contract,
)


class Stage6SelectionContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_stage6_selection_contract()

    def test_contract_freezes_validation_roles_and_candidates(self):
        self.assertEqual(self.contract.primary_protocol, "full")
        self.assertEqual(self.contract.secondary_protocol, "small_sample")
        self.assertEqual(
            self.contract.candidate_models,
            (
                "stl",
                "hard_share",
                "static_gate",
                "dynamic_symmetric",
                "dynamic_directed",
                "scheme2r",
            ),
        )
        self.assertEqual(len(self.contract.hyperparameter_candidates), 4)
        self.assertFalse(
            self.contract.raw["validation_policy"]["test_metrics_may_be_read"]
        )

    def test_contract_rejects_test_access(self):
        invalid = copy.deepcopy(dict(self.contract.raw))
        invalid["validation_policy"]["test_metrics_may_be_read"] = True
        with self.assertRaises(ValueError):
            validate_stage6_selection_contract(invalid)

    def test_contract_rejects_small_sample_as_primary_selection(self):
        invalid = copy.deepcopy(dict(self.contract.raw))
        invalid["validation_policy"]["primary_protocol"] = "small_sample"
        with self.assertRaises(ValueError):
            validate_stage6_selection_contract(invalid)

    def test_contract_rejects_unbounded_hyperparameter_search(self):
        invalid = copy.deepcopy(dict(self.contract.raw))
        invalid["finite_hyperparameter_candidates"] = invalid[
            "finite_hyperparameter_candidates"
        ][:3]
        with self.assertRaises(ValueError):
            validate_stage6_selection_contract(invalid)


if __name__ == "__main__":
    unittest.main()
