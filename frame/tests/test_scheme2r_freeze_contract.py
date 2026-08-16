import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "scheme2r_v1_freeze.json"
sys.path.insert(0, str(ROOT / "src"))

from models import Scheme2RModel  # noqa: E402


class Scheme2RFreezeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_identity_and_candidate(self):
        self.assertEqual(self.contract["freeze_id"], "scheme2r-v1")
        self.assertEqual(
            self.contract["freeze_basis"]["stage6_candidate"], "H4"
        )

    def test_dataset_protocol(self):
        dataset = self.contract["dataset"]
        self.assertEqual(
            dataset["tasks"], ["electricity", "cooling", "heating", "gas"]
        )
        self.assertTrue(dataset["task_order_is_fixed"])
        self.assertEqual(dataset["lookback"], 24)
        self.assertEqual(dataset["horizon"], 4)
        self.assertEqual(dataset["exog_dim"], 12)
        self.assertEqual(dataset["output_shape"], ["B", 4, 4])

    def test_scheme2r_dimensions(self):
        encoder = self.contract["encoder"]
        self.assertEqual(encoder["task_specific_encoders"], 4)
        self.assertEqual(encoder["hidden_dim"], 32)
        self.assertEqual(encoder["kernel_size"], 5)
        self.assertEqual(encoder["dilations"], [1, 2, 4])
        self.assertEqual(encoder["representation_shape"], ["B", 4, 32])

        routing = self.contract["context_and_routing"]
        self.assertEqual(routing["state_dim"], 16)
        self.assertEqual(routing["task_role_embedding_dim"], 8)
        self.assertEqual(routing["forecast_step_embedding_dim"], 4)
        self.assertEqual(routing["shared_strength"], "rho[B,4,4]")
        self.assertEqual(routing["source_allocation"], "pi[B,4,4,4]")
        self.assertEqual(routing["final_gate"], "gates[B,4,4,4] = rho[...,target] * pi[...,target,source]")

    def test_prediction_and_claim_boundary(self):
        messages = self.contract["message_and_prediction"]
        self.assertEqual(messages["low_rank"], 8)
        self.assertEqual(messages["head_hidden_dim"], 16)
        self.assertEqual(messages["parameter_count"], 39834)
        claims = self.contract["claims_boundary"]
        self.assertIn("universal_negative_transfer_elimination", claims["not_supported"])
        self.assertIn("universal_superiority_over_all_joint_models", claims["not_supported"])

    def test_live_model_matches_frozen_shapes(self):
        dataset = self.contract["dataset"]
        encoder = self.contract["encoder"]
        routing = self.contract["context_and_routing"]
        messages = self.contract["message_and_prediction"]
        model = Scheme2RModel(
            exog_dim=dataset["exog_dim"],
            task_count=4,
            hidden_dim=encoder["hidden_dim"],
            lookback=dataset["lookback"],
            kernel_size=encoder["kernel_size"],
            dilations=tuple(encoder["dilations"]),
            dropout=encoder["dropout"],
            horizon=dataset["horizon"],
            state_dim=routing["state_dim"],
            state_hidden_dim=32,
            gate_hidden_dim=routing["gate_hidden_dim"],
            task_embedding_dim=routing["task_role_embedding_dim"],
            step_embedding_dim=routing["forecast_step_embedding_dim"],
            rank=messages["low_rank"],
            head_hidden_dim=messages["head_hidden_dim"],
        )
        model.eval()
        loads = torch.randn(2, 24, 4)
        exog = torch.randn(2, 24, 12)
        with torch.no_grad():
            prediction, details = model.forward_with_details(loads, exog)
        self.assertEqual(tuple(prediction.shape), (2, 4, 4))
        self.assertEqual(tuple(details["representations"].shape), (2, 4, 32))
        self.assertEqual(tuple(details["rho"].shape), (2, 4, 4))
        self.assertEqual(tuple(details["pi"].shape), (2, 4, 4, 4))
        self.assertEqual(tuple(details["gates"].shape), (2, 4, 4, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(float(details["gates"].diagonal(dim1=-2, dim2=-1).abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
