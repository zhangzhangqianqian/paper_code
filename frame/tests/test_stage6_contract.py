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
        self.assertEqual(self.contract.raw["contract_version"], "stage6.1-r1")
        self.assertEqual(self.contract.primary_protocol, "full")
        self.assertEqual(self.contract.secondary_protocol, "small_sample")
        self.assertEqual(
            self.contract.candidate_models,
            (
                "stl_matched",
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

    def test_protocol_training_policies_are_distinct_and_resolved(self):
        full = self.contract.training_policy("full")
        small = self.contract.training_policy("small_sample")
        self.assertEqual(
            (full["batch_size"], full["max_epochs"], full["early_stopping_patience"]),
            (256, 100, 12),
        )
        self.assertEqual(
            (small["batch_size"], small["max_epochs"], small["early_stopping_patience"]),
            (32, 200, 20),
        )
        self.assertNotEqual(full, small)

    def test_hyperparameter_effective_configurations_are_unique(self):
        candidates = self.contract.hyperparameter_candidates
        signatures = {
            (
                item["hidden_dim"],
                item["kernel_size"],
                tuple(item["dilations"]),
                item["scheme2r_rank"],
                item["dropout"],
                item["learning_rate"],
            )
            for item in candidates
        }
        self.assertEqual(len(signatures), len(candidates))
        h3 = next(item for item in candidates if item["candidate_id"] == "H3")
        self.assertEqual(h3["kernel_size"], 3)
        self.assertEqual(h3["dilations"], [1, 2, 4, 8])

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
