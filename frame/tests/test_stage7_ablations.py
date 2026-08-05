"""Stage 7.1 interface tests for the recursive A0--A4 ablations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import count_trainable_parameters  # noqa: E402
from src.stage7_ablations import (  # noqa: E402
    ABLATION_NAMES,
    ABLATION_SPECS,
    build_stage7_ablation_model,
)


class Stage7AblationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        self.batch = 2
        self.lookback = 24
        self.task_count = 4
        self.exog_dim = 12
        self.horizon = 4
        self.loads = torch.randn(self.batch, self.lookback, self.task_count)
        self.exog = torch.randn(self.batch, self.lookback, self.exog_dim)

    def test_recursive_names_and_specs_are_fixed(self) -> None:
        self.assertEqual(ABLATION_NAMES, ("A0", "A1", "A2", "A3", "A4"))
        self.assertEqual(tuple(item.ablation_id for item in ABLATION_SPECS), ABLATION_NAMES)
        self.assertTrue(all(item.description for item in ABLATION_SPECS))

    def test_all_ablations_have_common_io_and_finite_output(self) -> None:
        parameter_counts = []
        for ablation_id in ABLATION_NAMES:
            torch.manual_seed(2026)
            model = build_stage7_ablation_model(
                ablation_id,
                exog_dim=self.exog_dim,
                task_count=self.task_count,
                lookback=self.lookback,
                horizon=self.horizon,
            )
            output, details = model.forward_with_details(self.loads, self.exog)
            self.assertEqual(tuple(output.shape), (self.batch, self.horizon, self.task_count))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(model.ablation_id, ablation_id)
            self.assertTrue(model.ablation_description)
            self.assertIn("gates", details)
            self.assertTrue(torch.isfinite(details["gates"]).all())
            parameter_counts.append(count_trainable_parameters(model))
        self.assertTrue(all(value > 0 for value in parameter_counts))
        self.assertGreater(len(set(parameter_counts)), 1)

    def test_gate_shapes_and_diagonals(self) -> None:
        for ablation_id in ABLATION_NAMES:
            model = build_stage7_ablation_model(
                ablation_id,
                exog_dim=self.exog_dim,
                task_count=self.task_count,
                lookback=self.lookback,
                horizon=self.horizon,
            )
            _, details = model.forward_with_details(self.loads, self.exog)
            gates = details["gates"]
            if ablation_id in ("A0", "A1"):
                self.assertEqual(tuple(gates.shape), (self.batch, 4, 4))
                diagonal = torch.diagonal(gates, dim1=1, dim2=2)
            else:
                self.assertEqual(tuple(gates.shape), (self.batch, 4, 4, 4))
                diagonal = torch.diagonal(gates, dim1=2, dim2=3)
            self.assertTrue(torch.all(gates >= 0.0))
            self.assertTrue(torch.all(gates <= 1.0))
            torch.testing.assert_close(diagonal, torch.zeros_like(diagonal))

    def test_a3_source_probabilities_are_normalized(self) -> None:
        model = build_stage7_ablation_model(
            "A3",
            exog_dim=self.exog_dim,
            task_count=self.task_count,
            lookback=self.lookback,
            horizon=self.horizon,
        )
        _, details = model.forward_with_details(self.loads, self.exog)
        pi = details["pi"]
        diagonal = torch.eye(self.task_count, dtype=torch.bool)[None, None]
        self.assertTrue(torch.allclose(pi.masked_select(diagonal), torch.zeros(1)))
        source_sums = pi.sum(dim=-1)
        torch.testing.assert_close(source_sums, torch.ones_like(source_sums))

    def test_a4_is_the_only_low_rank_stable_scheme2r_endpoint(self) -> None:
        a3 = build_stage7_ablation_model(
            "A3", exog_dim=self.exog_dim, lookback=self.lookback
        )
        a4 = build_stage7_ablation_model(
            "A4", exog_dim=self.exog_dim, lookback=self.lookback
        )
        self.assertTrue(hasattr(a3, "message_projections"))
        self.assertTrue(hasattr(a4, "message_projector"))
        self.assertTrue(hasattr(a4, "fusion"))
        self.assertTrue(hasattr(a4, "head"))


if __name__ == "__main__":
    unittest.main()

