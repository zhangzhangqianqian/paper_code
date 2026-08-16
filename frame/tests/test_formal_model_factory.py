"""Parity and shape tests for the frozen formal model factory."""

from __future__ import annotations

import unittest

import torch

from frame.src.external_models import MMoELiteBaseline, PLELiteBaseline
from frame.src.formal_model_factory import (
    FORMAL_MODEL_NAMES,
    build_formal_forecasting_model,
)
from frame.src.models import (
    HardShareMTLModel,
    Scheme2RModel,
    DynamicSymmetricMTLModel,
    count_trainable_parameters,
)


COMMON = {
    "hidden_dim": 32,
    "kernel_size": 5,
    "dilations": [1, 2, 4],
    "dropout": 0.1,
    "prediction_head_hidden_dim": 16,
}


def hard_hyperparameters() -> dict[str, object]:
    return dict(COMMON)


def dynamic_hyperparameters() -> dict[str, object]:
    values = dict(COMMON)
    values["hidden_dim"] = 16
    values["dropout"] = 0.0
    return values


def scheme_hyperparameters() -> dict[str, object]:
    values = dict(COMMON)
    values.update(
        {
            "scheme2r_kernel_size": 5,
            "scheme2r_dilations": [1, 2, 4],
            "scheme2r_rank": 8,
            "scheme2r_gate_hidden_dim": 16,
            "scheme2r_step_embedding_dim": 4,
        }
    )
    return values


class FormalModelFactoryTests(unittest.TestCase):
    def test_formal_model_names_are_frozen(self) -> None:
        self.assertEqual(
            FORMAL_MODEL_NAMES,
            ("hard_share", "dynamic_symmetric", "mmoe_lite", "ple_lite", "scheme2r"),
        )

    def test_all_five_models_have_expected_output_shape(self) -> None:
        configurations = {
            "hard_share": hard_hyperparameters(),
            "dynamic_symmetric": dynamic_hyperparameters(),
            "mmoe_lite": {
                "expert_count": 4,
                "expert_hidden_dim": 32,
                "representation_dim": 32,
                "prediction_head_hidden_dim": 16,
                "dropout": 0.1,
            },
            "ple_lite": {
                "shared_expert_count": 2,
                "task_expert_count": 1,
                "expert_hidden_dim": 32,
                "representation_dim": 32,
                "prediction_head_hidden_dim": 16,
                "dropout": 0.1,
            },
            "scheme2r": scheme_hyperparameters(),
        }
        loads = torch.randn(2, 24, 4)
        exog = torch.randn(2, 24, 12)
        for model_name, hyperparameters in configurations.items():
            model = build_formal_forecasting_model(
                model_name,
                hyperparameters,
                exog_dim=12,
            )
            output = model(loads, exog)
            self.assertEqual(tuple(output.shape), (2, 4, 4), model_name)
            self.assertGreater(count_trainable_parameters(model), 0)

    def test_factory_matches_direct_constructors_for_frozen_models(self) -> None:
        direct = {
            "hard_share": lambda: HardShareMTLModel(
                exog_dim=12,
                task_count=4,
                hidden_dim=32,
                lookback=24,
                kernel_size=5,
                dilations=(1, 2, 4),
                dropout=0.1,
                horizon=4,
                head_hidden_dim=16,
            ),
            "dynamic_symmetric": lambda: DynamicSymmetricMTLModel(
                exog_dim=12,
                task_count=4,
                hidden_dim=16,
                lookback=24,
                kernel_size=5,
                dilations=(1, 2, 4),
                dropout=0.0,
                horizon=4,
                head_hidden_dim=16,
            ),
            "scheme2r": lambda: Scheme2RModel(
                exog_dim=12,
                task_count=4,
                hidden_dim=32,
                lookback=24,
                kernel_size=5,
                dilations=(1, 2, 4),
                dropout=0.1,
                horizon=4,
                rank=8,
                gate_hidden_dim=16,
                step_embedding_dim=4,
                head_hidden_dim=16,
            ),
            "mmoe_lite": lambda: MMoELiteBaseline(
                lookback=24,
                horizon=4,
                task_count=4,
                exog_dim=12,
                expert_count=4,
                expert_hidden_dim=32,
                representation_dim=32,
                head_hidden_dim=16,
                dropout=0.1,
            ),
            "ple_lite": lambda: PLELiteBaseline(
                lookback=24,
                horizon=4,
                task_count=4,
                exog_dim=12,
                shared_expert_count=2,
                task_expert_count=1,
                expert_hidden_dim=32,
                representation_dim=32,
                head_hidden_dim=16,
                dropout=0.1,
            ),
        }
        hyperparameters = {
            "hard_share": hard_hyperparameters(),
            "dynamic_symmetric": dynamic_hyperparameters(),
            "scheme2r": scheme_hyperparameters(),
            "mmoe_lite": {
                "expert_count": 4,
                "expert_hidden_dim": 32,
                "representation_dim": 32,
                "prediction_head_hidden_dim": 16,
                "dropout": 0.1,
            },
            "ple_lite": {
                "shared_expert_count": 2,
                "task_expert_count": 1,
                "expert_hidden_dim": 32,
                "representation_dim": 32,
                "prediction_head_hidden_dim": 16,
                "dropout": 0.1,
            },
        }
        for model_name, constructor in direct.items():
            torch.manual_seed(123)
            expected = constructor()
            torch.manual_seed(123)
            actual = build_formal_forecasting_model(
                model_name,
                hyperparameters[model_name],
                exog_dim=12,
            )
            self.assertEqual(
                set(expected.state_dict()), set(actual.state_dict()), model_name
            )
            self.assertEqual(
                count_trainable_parameters(expected),
                count_trainable_parameters(actual),
                model_name,
            )
            for key, tensor in expected.state_dict().items():
                self.assertEqual(tuple(tensor.shape), tuple(actual.state_dict()[key].shape))

    def test_external_aliases_are_accepted(self) -> None:
        hyperparameters = {
            "shared_expert_count": 2,
            "task_expert_count": 1,
            "expert_hidden_dim": 32,
            "representation_dim": 32,
            "prediction_head_hidden_dim": 16,
            "dropout": 0.1,
        }
        self.assertIsInstance(
            build_formal_forecasting_model("ple-lite", hyperparameters, exog_dim=12),
            PLELiteBaseline,
        )

    def test_missing_frozen_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_formal_forecasting_model(
                "scheme2r", {"hidden_dim": 32}, exog_dim=12
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
