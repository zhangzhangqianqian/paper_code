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
    DynamicDirectedMTLModel,
    DynamicSymmetricMTLModel,
    HardShareMTLModel,
    IndependentSTLModel,
    SingleTaskDSTCN,
    StateEncoder,
    StaticDirectedMTLModel,
    MODEL_NAMES,
    build_forecasting_model,
    count_trainable_parameters,
)
from src.external_models import (  # noqa: E402
    DLinearBaseline,
    MMoELiteBaseline,
    MovingAverageDecomposition,
    SOFTSBaseline,
    STARCore,
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

    def test_state_encoder_shape_and_finite_output(self):
        encoder = StateEncoder(
            input_dim=self.exog_dim,
            state_dim=16,
            hidden_dim=32,
        )
        output = encoder(self.exog)
        self.assertEqual(tuple(output.shape), (self.batch_size, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_state_encoder_accepts_optional_task_representations(self):
        encoder = StateEncoder(
            input_dim=self.exog_dim,
            state_dim=16,
            hidden_dim=32,
            task_repr_dim=8,
        )
        task_representations = torch.randn(self.batch_size, self.task_count, 8)
        output = encoder(self.exog, task_representations)
        self.assertEqual(tuple(output.shape), (self.batch_size, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_state_encoder_has_finite_gradients(self):
        encoder = StateEncoder(
            input_dim=self.exog_dim,
            state_dim=16,
            hidden_dim=32,
            task_repr_dim=8,
        )
        task_representations = torch.randn(self.batch_size, self.task_count, 8)
        output = encoder(self.exog, task_representations)
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in encoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

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

    def test_dynamic_symmetric_model_shape_and_gate_constraints(self):
        model = DynamicSymmetricMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        representations = model.encode_tasks(self.loads, self.exog)
        state = model.encode_state(self.exog, representations)
        gates = model.get_gate_matrix(state)
        self.assertEqual(
            tuple(output.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        self.assertEqual(
            tuple(gates.shape),
            (self.batch_size, self.task_count, self.task_count),
        )
        self.assertTrue(torch.all(gates >= 0.0))
        self.assertTrue(torch.all(gates <= 1.0))
        torch.testing.assert_close(gates, gates.transpose(1, 2))
        diagonal = torch.diagonal(gates, dim1=1, dim2=2)
        torch.testing.assert_close(diagonal, torch.zeros_like(diagonal))

    def test_dynamic_symmetric_zero_gate_degenerates(self):
        model = DynamicSymmetricMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        representations = model.encode_tasks(self.loads, self.exog)
        zero_gates = torch.zeros(
            self.batch_size, self.task_count, self.task_count
        )
        fused = model.fuse_representations(representations, zero_gates)
        torch.testing.assert_close(fused, representations)

    def test_dynamic_symmetric_model_has_finite_gradients(self):
        model = DynamicSymmetricMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        output.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_dynamic_directed_model_shape_and_gate_constraints(self):
        model = DynamicDirectedMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        output = model(self.loads, self.exog)
        representations = model.encode_tasks(self.loads, self.exog)
        state = model.encode_state(self.exog, representations)
        gates = model.get_gate_matrix(state, representations)
        self.assertEqual(
            tuple(output.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        self.assertEqual(
            tuple(gates.shape),
            (self.batch_size, self.task_count, self.task_count),
        )
        self.assertTrue(torch.all(gates >= 0.0))
        self.assertTrue(torch.all(gates <= 1.0))
        diagonal = torch.diagonal(gates, dim1=1, dim2=2)
        torch.testing.assert_close(diagonal, torch.zeros_like(diagonal))

    def test_dynamic_directed_gate_can_be_asymmetric(self):
        model = DynamicDirectedMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        model.eval()
        representations = model.encode_tasks(self.loads, self.exog)
        state = model.encode_state(self.exog, representations)
        first_layer = model.gate_network[0]
        final_layer = model.gate_network[-1]
        target_embedding_offset = model.state_dim + 2 * model.hidden_dim
        with torch.no_grad():
            model.task_embeddings.zero_()
            model.task_embeddings[0, 0] = 1.0
            model.task_embeddings[1, 0] = -1.0
            first_layer.weight.zero_()
            first_layer.bias.zero_()
            first_layer.weight[0, target_embedding_offset] = 1.0
            final_layer.weight.zero_()
            final_layer.bias.zero_()
            final_layer.weight[0, 0] = 1.0
        gates = model.get_gate_matrix(state, representations)
        self.assertFalse(torch.allclose(gates[:, 0, 1], gates[:, 1, 0]))

    def test_dynamic_directed_zero_gate_degenerates(self):
        model = DynamicDirectedMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
        representations = model.encode_tasks(self.loads, self.exog)
        zero_gates = torch.zeros(
            self.batch_size, self.task_count, self.task_count
        )
        fused = model.fuse_representations(representations, zero_gates)
        torch.testing.assert_close(fused, representations)

    def test_dynamic_directed_model_has_finite_gradients(self):
        model = DynamicDirectedMTLModel(exog_dim=self.exog_dim, horizon=self.horizon)
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

    def test_unified_model_factory_has_common_interface(self):
        expected_shape = (self.batch_size, self.horizon, self.task_count)
        self.assertEqual(
            set(MODEL_NAMES),
            {
                "stl",
                "hard_share",
                "static_gate",
                "dynamic_symmetric",
                "dynamic_directed",
            },
        )
        for model_name in MODEL_NAMES:
            model = build_forecasting_model(
                model_name,
                exog_dim=self.exog_dim,
                horizon=self.horizon,
            )
            prediction = model(self.loads, self.exog)
            self.assertEqual(tuple(prediction.shape), expected_shape)

    def test_dlinear_baseline_shape_and_loads_only_contract(self):
        model = DLinearBaseline(lookback=self.lookback, horizon=self.horizon)
        prediction = model(self.loads)
        self.assertEqual(
            tuple(prediction.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        with self.assertRaises(ValueError):
            model(self.loads, self.exog)

    def test_dlinear_baseline_has_finite_gradients(self):
        model = DLinearBaseline(lookback=self.lookback, horizon=self.horizon)
        prediction = model(self.loads)
        prediction.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_moving_average_decomposition_preserves_window_shape(self):
        decomposition = MovingAverageDecomposition(kernel_size=5)
        seasonal, trend = decomposition(self.loads)
        self.assertEqual(tuple(seasonal.shape), tuple(self.loads.shape))
        self.assertEqual(tuple(trend.shape), tuple(self.loads.shape))
        torch.testing.assert_close(seasonal + trend, self.loads)

    def test_mmoe_lite_shape_and_gate_contract(self):
        model = MMoELiteBaseline(
            lookback=self.lookback,
            horizon=self.horizon,
            task_count=self.task_count,
            exog_dim=self.exog_dim,
        )
        prediction = model(self.loads, self.exog)
        gate_weights = model.gate_weights(self.loads, self.exog)
        self.assertEqual(
            tuple(prediction.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        self.assertEqual(
            tuple(gate_weights.shape),
            (self.batch_size, self.task_count, 4),
        )
        torch.testing.assert_close(
            gate_weights.sum(dim=-1),
            torch.ones(self.batch_size, self.task_count),
        )
        self.assertTrue(torch.all(gate_weights >= 0.0))

    def test_mmoe_lite_requires_historical_exog(self):
        model = MMoELiteBaseline(
            lookback=self.lookback,
            horizon=self.horizon,
            task_count=self.task_count,
            exog_dim=self.exog_dim,
        )
        with self.assertRaises(ValueError):
            model(self.loads)

    def test_mmoe_lite_has_finite_gradients(self):
        model = MMoELiteBaseline(
            lookback=self.lookback,
            horizon=self.horizon,
            task_count=self.task_count,
            exog_dim=self.exog_dim,
        )
        prediction = model(self.loads, self.exog)
        prediction.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_star_core_shape_and_finite_output(self):
        core = STARCore(d_series=32, d_core=16, stochastic_pooling=False)
        values = torch.randn(self.batch_size, self.task_count, 32)
        output = core(values)
        self.assertEqual(tuple(output.shape), tuple(values.shape))
        self.assertTrue(torch.isfinite(output).all())

    def test_softs_baseline_shape_and_loads_only_contract(self):
        model = SOFTSBaseline(
            lookback=self.lookback,
            horizon=self.horizon,
            task_count=self.task_count,
            d_model=16,
            d_core=8,
            d_ff=32,
            e_layers=1,
            dropout=0.0,
            stochastic_pooling=False,
        )
        prediction = model(self.loads)
        self.assertEqual(
            tuple(prediction.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        with self.assertRaises(ValueError):
            model(self.loads, self.exog)

    def test_softs_baseline_has_finite_gradients(self):
        model = SOFTSBaseline(
            lookback=self.lookback,
            horizon=self.horizon,
            task_count=self.task_count,
            d_model=16,
            d_core=8,
            d_ff=32,
            e_layers=1,
            dropout=0.0,
            stochastic_pooling=False,
        )
        prediction = model(self.loads)
        prediction.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main(verbosity=2)
