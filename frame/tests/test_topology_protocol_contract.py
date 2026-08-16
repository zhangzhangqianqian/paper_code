"""Tests for the frozen topology protocol contract."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from frame.src.topology_protocol_contract import (
    TopologyProtocolContractError,
    assert_phase_access,
    load_topology_contract,
    resolve_phase_b_matrix,
    validate_topology_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "topology_protocol_pilot_v1.json"


class TopologyProtocolContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_topology_contract(CONTRACT_PATH)

    def test_contract_is_valid(self) -> None:
        validate_topology_contract(self.contract)

    def test_two_protocol_dates_are_frozen(self) -> None:
        self.assertEqual(
            self.contract["protocols"]["cross_topology"]["train_start"],
            "2015-01-01 00:00:00",
        )
        self.assertEqual(
            self.contract["protocols"]["post_ge_regular_operation"]["train_start"],
            "2017-01-01 00:00:00",
        )
        self.assertEqual(
            self.contract["protocols"]["post_ge_regular_operation"]["test_end"],
            "2021-12-31 23:00:00",
        )

    def test_phase_a_seals_2021(self) -> None:
        assert_phase_access(
            self.contract,
            phase="A",
            requested_years=(2017, 2018, 2019, 2020),
            requested_splits=("train", "validation"),
        )
        with self.assertRaises(TopologyProtocolContractError):
            assert_phase_access(
                self.contract,
                phase="A",
                requested_years=(2017, 2018, 2019, 2020, 2021),
                requested_splits=("train", "validation"),
            )
        with self.assertRaises(TopologyProtocolContractError):
            assert_phase_access(
                self.contract,
                phase="A",
                requested_years=(2017, 2018, 2019, 2020),
                requested_splits=("train", "validation", "test"),
            )

    def test_phase_a_model_candidates_and_seeds(self) -> None:
        models = [
            (item["model"], item["candidate_id"])
            for item in self.contract["phase_a"]["models"]
        ]
        self.assertEqual(models, [("scheme2r", "H4"), ("dynamic_symmetric", "H1")])
        self.assertEqual(self.contract["phase_a"]["seeds"], [2026, 2027, 2028])

    def test_statistics_are_frozen(self) -> None:
        statistics = self.contract["statistics"]
        self.assertEqual(statistics["method"], "paired_circular_moving_block_bootstrap")
        self.assertEqual(statistics["block_length_origins"], 168)
        self.assertEqual(statistics["replicates"], 2000)
        self.assertEqual(statistics["multiple_testing"], "benjamini_hochberg")

    def test_phase_b_matrices_are_resolvable(self) -> None:
        for branch in (
            "core_prediction_changed",
            "gas_only_changed",
            "core_conclusion_stable",
        ):
            matrix = resolve_phase_b_matrix(self.contract, branch)
            self.assertIn("forecast_models", matrix)
            self.assertIn("scheduling_tracks", matrix)

    def test_invalid_contracts_are_rejected(self) -> None:
        altered = copy.deepcopy(self.contract)
        altered["phase_a"]["seeds"] = [2026, 2027]
        with self.assertRaises(TopologyProtocolContractError):
            validate_topology_contract(altered)

        altered = copy.deepcopy(self.contract)
        altered["window"]["future_exogenous_allowed"] = True
        with self.assertRaises(TopologyProtocolContractError):
            validate_topology_contract(altered)

        altered = copy.deepcopy(self.contract)
        altered["branch_rules"]["valid_branches"] = ["stable_topology"]
        with self.assertRaises(TopologyProtocolContractError):
            validate_topology_contract(altered)


if __name__ == "__main__":
    unittest.main()
