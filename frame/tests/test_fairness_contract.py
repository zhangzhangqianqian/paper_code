"""阶段5.1公平性契约测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fairness_contract import (  # noqa: E402
    BASELINE_NAMES,
    INPUT_MODES,
    load_fairness_contract,
    validate_contract,
)


class FairnessContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_fairness_contract()

    def test_contract_has_fixed_protocol_and_external_baselines(self):
        self.assertEqual(self.contract.lookback, 24)
        self.assertEqual(self.contract.horizon, 4)
        self.assertEqual(self.contract.task_count, 4)
        self.assertEqual(
            tuple(spec.name for spec in self.contract.baseline_specs), BASELINE_NAMES
        )
        self.assertEqual(INPUT_MODES, ("loads_only", "loads_and_exog"))

    def test_prediction_shape_contract(self):
        self.contract.validate_prediction_shape((8, 4, 4))
        with self.assertRaises(ValueError):
            self.contract.validate_prediction_shape((8, 3, 4))
        with self.assertRaises(ValueError):
            self.contract.validate_prediction_shape((8, 4))

    def test_baseline_input_declaration_is_explicit(self):
        self.contract.validate_baseline_input("DLinear", "loads_only")
        self.contract.validate_baseline_input("MMoE-lite", "loads_and_exog")
        with self.assertRaises(ValueError):
            self.contract.validate_baseline_input("SOFTS", "loads_and_exog")
        with self.assertRaises(ValueError):
            self.contract.validate_baseline_input(
                "DLinear", "loads_only", uses_future_exog=True
            )

    def test_contract_rejects_future_exogenous_variables(self):
        invalid = dict(self.contract.raw)
        invalid["input_output"] = dict(invalid["input_output"])
        invalid["input_output"]["future_exogenous_allowed"] = True
        with self.assertRaises(ValueError):
            validate_contract(invalid)

    def test_mmoe_lite_is_not_marked_as_full_shao_reimplementation(self):
        spec = self.contract.get_baseline("MMoE-lite")
        self.assertFalse(spec.exact_reimplementation)
        self.assertIn("omits the Frequency and STIM", spec.relation_to_shao)


if __name__ == "__main__":
    unittest.main(verbosity=2)
