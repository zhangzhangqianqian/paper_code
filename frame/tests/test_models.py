"""阶段3独立单任务DS-TCN测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    CausalDepthwiseSeparableConv1d,
    HardShareMTLModel,
    IndependentSTLModel,
    SingleTaskDSTCN,
    StaticDirectedMTLModel,
    count_trainable_parameters,
)


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(8)
        torch.manual_seed(2026)
        self.batch_size = 16
        self.lookback = 24
        self.task_count = 3
        self.exog_dim = 5
        self.horizon = 4
        self.loads = torch.randn(
            self.batch_size, self.lookback, self.task_count, dtype=torch.float32
        )
        self.exog = torch.randn(
            self.batch_size, self.lookback, self.exog_dim, dtype=torch.float32
        )

    def test_single_task_shape(self):
        model = SingleTaskDSTCN(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads[:, :, :1], self.exog)
        self.assertEqual(tuple(output.shape), (self.batch_size, self.horizon))

    def test_independent_model_shape(self):
        model = IndependentSTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        self.assertEqual(
            tuple(output.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        self.assertEqual(len(model.models), self.task_count)

    def test_heew_exogenous_dimension_forward(self):
        heew_exog_dim = 14
        model = IndependentSTLModel(exog_dim=heew_exog_dim, horizon=self.horizon)
        heew_exog = torch.randn(
            self.batch_size, self.lookback, heew_exog_dim, dtype=torch.float32
        )
        output = model(self.loads, heew_exog)
        self.assertEqual(
            tuple(output.shape),
            (self.batch_size, self.horizon, self.task_count),
        )

    def test_hard_share_model_shape_and_shared_encoder(self):
        model = HardShareMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        self.assertEqual(
            tuple(output.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        self.assertEqual(len(model.heads), self.task_count)

    def test_hard_share_model_has_one_encoder_parameter_set(self):
        model = HardShareMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        encoder_parameter_ids = {id(parameter) for parameter in model.encoder.parameters()}
        head_parameter_ids = {
            id(parameter) for head in model.heads for parameter in head.parameters()
        }
        self.assertTrue(encoder_parameter_ids.isdisjoint(head_parameter_ids))

    def test_static_directed_model_shape_and_gate_constraints(self):
        model = StaticDirectedMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        gates = model.get_gate_matrix()
        self.assertEqual(
            tuple(output.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        self.assertEqual(tuple(gates.shape), (self.task_count, self.task_count))
        self.assertTrue(torch.all(gates >= 0.0))
        self.assertTrue(torch.all(gates <= 1.0))
        torch.testing.assert_close(torch.diag(gates), torch.zeros(self.task_count))

    def test_static_directed_gate_is_asymmetric_and_zero_gate_degenerates(self):
        model = StaticDirectedMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        with torch.no_grad():
            model.gate_logits.fill_(-4.0)
            model.gate_logits[0, 1] = 4.0
        gates = model.get_gate_matrix()
        self.assertGreater(float(gates[0, 1].detach()), float(gates[1, 0].detach()))

        representations = model.encode_tasks(self.loads, self.exog)
        with torch.no_grad():
            model.gate_logits.fill_(-100.0)
        fused = model.fuse_representations(representations)
        torch.testing.assert_close(fused, representations)

    def test_static_directed_model_has_finite_gradients(self):
        model = StaticDirectedMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_task_models_do_not_share_parameter_objects(self):
        model = IndependentSTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        first_ids = {id(parameter) for parameter in model.models[0].parameters()}
        second_ids = {id(parameter) for parameter in model.models[1].parameters()}
        self.assertTrue(first_ids.isdisjoint(second_ids))

    def test_causal_conv_does_not_use_future_values(self):
        torch.manual_seed(2026)
        convolution = CausalDepthwiseSeparableConv1d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            dilation=2,
        ).eval()
        history = torch.randn(1, 1, 24)
        changed = history.clone()
        cutoff = 12
        changed[:, :, cutoff:] += 1000.0
        with torch.no_grad():
            original_output = convolution(history)
            changed_output = convolution(changed)
        # 改变cutoff之后的输入，不得影响cutoff之前的卷积输出。
        torch.testing.assert_close(
            original_output[:, :, :cutoff],
            changed_output[:, :, :cutoff],
        )

    def test_forward_backward_has_finite_gradients(self):
        model = IndependentSTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        loss = output.square().mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_small_batch_can_overfit(self):
        torch.manual_seed(2026)
        model = SingleTaskDSTCN(
            exog_dim=0,
            hidden_dim=16,
            dropout=0.0,
            horizon=4,
        )
        model.train()
        inputs = torch.randn(8, 24, 1)
        targets = torch.randn(8, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_function = nn.SmoothL1Loss()
        initial_loss = None
        final_loss = None
        for step in range(150):
            optimizer.zero_grad()
            prediction = model(inputs)
            loss = loss_function(prediction, targets)
            if initial_loss is None:
                initial_loss = float(loss.detach())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
        self.assertIsNotNone(initial_loss)
        self.assertIsNotNone(final_loss)
        self.assertLess(final_loss, initial_loss * 0.25)

    def test_parameter_count_is_positive(self):
        model = IndependentSTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        self.assertGreater(count_trainable_parameters(model), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
