"""方案 2-R 阶段 B：全窗口 DS-TCN 编码器测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (  # noqa: E402
    EnhancedStateEncoder,
    ForecastStepEmbedding,
    FullWindowDSTCNEncoder,
    LowRankDirectedMessageProjector,
    Scheme2RModel,
    StableResidualTaskFusion,
    TaskRoleEmbeddings,
    TaskStepForecastHead,
    TwoLevelDirectedRouter,
    build_forecasting_model,
)
from src.training import (  # noqa: E402
    TrainerConfig,
    evaluate_model,
    fit_model,
    make_dataloader,
)


class FullWindowDSTCNEncoderTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.batch_size = 8
        self.lookback = 24
        self.exog_dim = 5
        self.encoder = FullWindowDSTCNEncoder(
            exog_dim=self.exog_dim,
            hidden_dim=32,
            lookback=self.lookback,
        )
        self.loads = torch.randn(self.batch_size, self.lookback, 1)
        self.exog = torch.randn(
            self.batch_size,
            self.lookback,
            self.exog_dim,
        )

    def test_shape_and_receptive_field(self) -> None:
        output = self.encoder(self.loads, self.exog)
        self.assertEqual(tuple(output.shape), (self.batch_size, 32))
        self.assertEqual(self.encoder.receptive_field, 29)

    def test_rejects_wrong_history_length(self) -> None:
        wrong_loads = self.loads[:, :-1, :]
        with self.assertRaises(ValueError):
            self.encoder(wrong_loads, self.exog[:, :-1, :])

    def test_internal_sequence_is_causal(self) -> None:
        self.encoder.eval()
        inputs = torch.cat((self.loads, self.exog), dim=-1)
        changed_inputs = inputs.clone()
        changed_inputs[:, 16:, :] += 100.0

        with torch.no_grad():
            encoded = self.encoder.encoder(inputs.transpose(1, 2))
            changed = self.encoder.encoder(changed_inputs.transpose(1, 2))

        prefix = 16
        self.assertTrue(torch.allclose(encoded[:, :, :prefix], changed[:, :, :prefix]))

    def test_finite_gradients(self) -> None:
        output = self.encoder(self.loads, self.exog)
        loss = output.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        gradients = [
            parameter.grad
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))


class Scheme2RContextTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.batch_size = 8
        self.lookback = 24
        self.exog_dim = 12
        self.state_history = torch.randn(
            self.batch_size,
            self.lookback,
            self.exog_dim,
        )

    def test_enhanced_state_encoder_shape_and_gradients(self) -> None:
        encoder = EnhancedStateEncoder(
            input_dim=self.exog_dim,
            state_dim=16,
            hidden_dim=32,
        )
        state = encoder(self.state_history)
        self.assertEqual(tuple(state.shape), (self.batch_size, 16))
        loss = state.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in encoder.parameters()
                if parameter.requires_grad
            )
        )

    def test_forecast_step_embedding_shape_and_gradients(self) -> None:
        embedding = ForecastStepEmbedding(horizon=4, embedding_dim=4)
        output = embedding()
        self.assertEqual(tuple(output.shape), (4, 4))
        self.assertTrue(embedding.embedding.weight.requires_grad)
        output.square().sum().backward()
        self.assertIsNotNone(embedding.embedding.weight.grad)

    def test_task_role_embeddings_are_independent(self) -> None:
        embeddings = TaskRoleEmbeddings(task_count=4, embedding_dim=8)
        target, source = embeddings()
        self.assertEqual(tuple(target.shape), (4, 8))
        self.assertEqual(tuple(source.shape), (4, 8))
        self.assertIsNot(target, source)
        self.assertIsNot(embeddings.target.weight, embeddings.source.weight)
        (target.square().sum() + source.square().sum()).backward()
        self.assertIsNotNone(embeddings.target.weight.grad)
        self.assertIsNotNone(embeddings.source.weight.grad)


class TwoLevelDirectedRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.batch_size = 6
        self.task_count = 4
        self.horizon = 4
        self.state = torch.randn(self.batch_size, 16)
        self.task_representations = torch.randn(
            self.batch_size,
            self.task_count,
            32,
        )
        self.step_embeddings = ForecastStepEmbedding(
            horizon=self.horizon,
            embedding_dim=4,
        )
        self.role_embeddings = TaskRoleEmbeddings(
            task_count=self.task_count,
            embedding_dim=8,
        )
        self.router = TwoLevelDirectedRouter(
            state_dim=16,
            task_repr_dim=32,
            task_count=self.task_count,
            task_embedding_dim=8,
            step_embedding_dim=4,
            gate_hidden_dim=16,
        )

    def _forward(self):
        target, source = self.role_embeddings()
        return self.router(
            self.state,
            self.task_representations,
            self.step_embeddings(),
            target,
            source,
        )

    def test_shapes_and_two_level_normalization(self) -> None:
        rho, pi, gates = self._forward()
        self.assertEqual(tuple(rho.shape), (self.batch_size, 4, 4))
        self.assertEqual(tuple(pi.shape), (self.batch_size, 4, 4, 4))
        self.assertEqual(tuple(gates.shape), (self.batch_size, 4, 4, 4))

        diagonal = torch.diagonal(pi, dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))
        source_sum = pi.sum(dim=-1)
        gate_sum = gates.sum(dim=-1)
        self.assertTrue(torch.allclose(source_sum, torch.ones_like(source_sum)))
        self.assertTrue(torch.allclose(gate_sum, rho, atol=1e-6))
        self.assertTrue(torch.all((rho >= 0.0) & (rho <= 1.0)).item())

    def test_direction_and_forecast_step_are_not_forced_symmetric(self) -> None:
        _, _, gates = self._forward()
        self.assertFalse(torch.allclose(gates[..., 0, 1], gates[..., 1, 0]))
        self.assertFalse(torch.allclose(gates[:, 0], gates[:, 1]))

    def test_finite_gradients(self) -> None:
        rho, pi, gates = self._forward()
        loss = rho.square().mean() + pi.square().mean() + gates.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        parameters = list(self.router.parameters())
        parameters.extend(self.step_embeddings.parameters())
        parameters.extend(self.role_embeddings.parameters())
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in parameters
                if parameter.requires_grad
            )
        )


class Scheme2RMessageAndHeadTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.batch_size = 6
        self.task_count = 4
        self.horizon = 4
        self.hidden_dim = 32
        self.task_representations = torch.randn(
            self.batch_size,
            self.task_count,
            self.hidden_dim,
        )
        self.rho = torch.rand(self.batch_size, self.horizon, self.task_count)
        self.pi = torch.softmax(
            torch.randn(
                self.batch_size,
                self.horizon,
                self.task_count,
                self.task_count,
            ),
            dim=-1,
        )
        diagonal = torch.eye(self.task_count, dtype=torch.bool)[None, None, :, :]
        self.pi = self.pi.masked_fill(diagonal, 0.0)
        self.pi = self.pi / self.pi.sum(dim=-1, keepdim=True)

    def test_low_rank_projector_shape_and_zero_diagonal(self) -> None:
        projector = LowRankDirectedMessageProjector(
            task_count=self.task_count,
            task_repr_dim=self.hidden_dim,
            rank=8,
        )
        messages = projector(self.task_representations)
        self.assertEqual(
            tuple(messages.shape),
            (self.batch_size, self.task_count, self.task_count, self.hidden_dim),
        )
        diagonal = torch.diagonal(messages, dim1=1, dim2=2)
        self.assertTrue(torch.allclose(diagonal, torch.zeros_like(diagonal)))

    def test_stable_fusion_zero_rho_is_single_task_path(self) -> None:
        projector = LowRankDirectedMessageProjector(
            task_count=self.task_count,
            task_repr_dim=self.hidden_dim,
            rank=8,
        )
        pair_messages = projector(self.task_representations)
        fusion = StableResidualTaskFusion(
            task_count=self.task_count,
            task_repr_dim=self.hidden_dim,
        )
        zero_rho = torch.zeros_like(self.rho)
        fused = fusion(
            self.task_representations,
            zero_rho,
            self.pi,
            pair_messages,
        )
        expected = self.task_representations[:, None, :, :].expand_as(fused)
        self.assertTrue(torch.equal(fused, expected))

    def test_task_step_head_shape_and_gradients(self) -> None:
        head = TaskStepForecastHead(
            task_count=self.task_count,
            horizon=self.horizon,
            hidden_dim=self.hidden_dim,
            head_hidden_dim=16,
        )
        fused = torch.randn(
            self.batch_size,
            self.horizon,
            self.task_count,
            self.hidden_dim,
        )
        prediction = head(fused)
        self.assertEqual(
            tuple(prediction.shape),
            (self.batch_size, self.horizon, self.task_count),
        )
        loss = prediction.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in head.parameters()
                if parameter.requires_grad
            )
        )

    def test_message_fusion_has_finite_gradients(self) -> None:
        projector = LowRankDirectedMessageProjector(
            task_count=self.task_count,
            task_repr_dim=self.hidden_dim,
            rank=8,
        )
        fusion = StableResidualTaskFusion(
            task_count=self.task_count,
            task_repr_dim=self.hidden_dim,
        )
        pair_messages = projector(self.task_representations)
        fused = fusion(
            self.task_representations,
            self.rho,
            self.pi,
            pair_messages,
        )
        loss = fused.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        parameters = list(projector.parameters()) + list(fusion.parameters())
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in parameters
                if parameter.requires_grad
            )
        )


class Scheme2RModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        self.batch_size = 4
        self.lookback = 24
        self.task_count = 4
        self.exog_dim = 12
        self.model = Scheme2RModel(
            exog_dim=self.exog_dim,
            task_count=self.task_count,
            hidden_dim=32,
            lookback=self.lookback,
            horizon=4,
        )
        self.loads = torch.randn(
            self.batch_size,
            self.lookback,
            self.task_count,
        )
        self.exog = torch.randn(
            self.batch_size,
            self.lookback,
            self.exog_dim,
        )

    def test_scheme2r_forward_and_detail_shapes(self) -> None:
        prediction, details = self.model.forward_with_details(
            self.loads,
            self.exog,
        )
        self.assertEqual(
            tuple(prediction.shape),
            (self.batch_size, 4, self.task_count),
        )
        self.assertEqual(tuple(details["representations"].shape), (4, 4, 32))
        self.assertEqual(tuple(details["state"].shape), (4, 16))
        self.assertEqual(tuple(details["rho"].shape), (4, 4, 4))
        self.assertEqual(tuple(details["pi"].shape), (4, 4, 4, 4))
        self.assertEqual(tuple(details["gates"].shape), (4, 4, 4, 4))
        self.assertEqual(
            tuple(details["fused_representations"].shape),
            (4, 4, 4, 32),
        )

    def test_scheme2r_finite_gradients(self) -> None:
        prediction = self.model(self.loads, self.exog)
        loss = prediction.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(
            all(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all().item()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        )

    def test_scheme2r_factory_and_lookback_contract(self) -> None:
        model = build_forecasting_model(
            "scheme2r",
            exog_dim=self.exog_dim,
            task_count=self.task_count,
            lookback=self.lookback,
            horizon=4,
        )
        self.assertIsInstance(model, Scheme2RModel)
        with self.assertRaises(ValueError):
            model(self.loads[:, :-1, :], self.exog[:, :-1, :])

    def test_scheme2r_training_checkpoint_and_evaluation_interface(self) -> None:
        sample_count = 12
        windows = {
            "loads": np.random.default_rng(2026).normal(
                size=(sample_count, self.lookback, self.task_count)
            ).astype(np.float32),
            "exog": np.random.default_rng(2027).normal(
                size=(sample_count, self.lookback, self.exog_dim)
            ).astype(np.float32),
            "target": np.random.default_rng(2028).normal(
                size=(sample_count, 4, self.task_count)
            ).astype(np.float32),
            "target_times": np.arange(sample_count),
        }
        loader = make_dataloader(windows, batch_size=4, shuffle=False)
        config = TrainerConfig(
            max_epochs=2,
            early_stopping_patience=1,
            torch_threads=1,
            seed=2026,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best_model.pt"
            history = fit_model(
                self.model,
                loader,
                loader,
                config,
                checkpoint,
            )
            self.assertTrue(checkpoint.exists())
            self.assertGreaterEqual(len(history), 1)
            loss, prediction, target = evaluate_model(self.model, loader, "cpu")
            self.assertTrue(np.isfinite(loss))
            self.assertEqual(
                prediction.shape,
                (sample_count, 4, self.task_count),
            )
            self.assertEqual(target.shape, prediction.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
