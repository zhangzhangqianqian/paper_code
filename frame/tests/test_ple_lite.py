import unittest

import torch

from src.external_models import PLELiteBaseline


class PLELiteBaselineTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.batch_size = 3
        self.lookback = 24
        self.horizon = 4
        self.task_count = 4
        self.exog_dim = 12
        self.loads = torch.randn(self.batch_size, self.lookback, self.task_count)
        self.exog = torch.randn(self.batch_size, self.lookback, self.exog_dim)

    def _model(self):
        return PLELiteBaseline(
            lookback=self.lookback,
            horizon=self.horizon,
            task_count=self.task_count,
            exog_dim=self.exog_dim,
            shared_expert_count=2,
            task_expert_count=1,
            expert_hidden_dim=32,
            representation_dim=32,
            head_hidden_dim=16,
            dropout=0.1,
        )

    def test_output_and_gate_shapes(self):
        model = self._model()

        prediction = model(self.loads, self.exog)
        gates = model.gate_weights(self.loads, self.exog)

        self.assertEqual(tuple(prediction.shape), (3, 4, 4))
        self.assertEqual(tuple(gates["layer1_task"].shape), (3, 4, 3))
        self.assertEqual(tuple(gates["layer1_shared"].shape), (3, 6))
        self.assertEqual(tuple(gates["layer2_task"].shape), (3, 4, 3))
        torch.testing.assert_close(
            gates["layer1_task"].sum(dim=-1), torch.ones(3, 4)
        )
        torch.testing.assert_close(
            gates["layer1_shared"].sum(dim=-1), torch.ones(3)
        )
        torch.testing.assert_close(
            gates["layer2_task"].sum(dim=-1), torch.ones(3, 4)
        )
        self.assertTrue(torch.isfinite(prediction).all())

    def test_task_gate_only_addresses_own_private_and_shared_experts(self):
        model = self._model()

        self.assertEqual(len(model.layer1.task_experts), self.task_count)
        self.assertEqual(len(model.layer2.task_experts), self.task_count)
        for layer in (model.layer1, model.layer2):
            self.assertTrue(
                all(len(experts) == 1 for experts in layer.task_experts)
            )
            self.assertTrue(
                all(gate.out_features == 3 for gate in layer.task_gates)
            )

    def test_requires_historical_exogenous_variables(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model(self.loads)

    def test_rejects_wrong_load_and_exogenous_shapes(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model(torch.randn(3, 23, 4), self.exog)
        with self.assertRaises(ValueError):
            model(self.loads, torch.randn(3, 24, 11))
        with self.assertRaises(ValueError):
            model(self.loads, torch.randn(2, 24, 12))

    def test_all_trainable_parameters_receive_finite_gradients(self):
        model = self._model()

        model(self.loads, self.exog).square().mean().backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        )


if __name__ == "__main__":
    unittest.main()
