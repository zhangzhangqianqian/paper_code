"""Tests for the structure-matched Stage 7-R STL reference."""

from __future__ import annotations

import unittest

import torch

from src.stage7_reference import MatchedSingleTaskScheme2RModel


class Stage7ReferenceTests(unittest.TestCase):
    def _model(self, task_index: int = 0) -> MatchedSingleTaskScheme2RModel:
        return MatchedSingleTaskScheme2RModel(
            exog_dim=3,
            task_index=task_index,
            task_count=4,
            hidden_dim=8,
            lookback=24,
            kernel_size=5,
            dilations=(1, 2, 4),
            dropout=0.0,
            horizon=4,
            head_hidden_dim=4,
        )

    def test_output_shape(self):
        model = self._model(task_index=2)
        prediction = model(torch.randn(5, 24, 4), torch.randn(5, 24, 3))
        self.assertEqual(tuple(prediction.shape), (5, 4, 1))

    def test_other_task_loads_cannot_change_prediction(self):
        model = self._model(task_index=1).eval()
        loads = torch.randn(3, 24, 4)
        exog = torch.randn(3, 24, 3)
        changed = loads.clone()
        changed[:, :, 0] += 100.0
        changed[:, :, 2] -= 100.0
        changed[:, :, 3] *= -5.0
        with torch.no_grad():
            original_prediction = model(loads, exog)
            changed_prediction = model(changed, exog)
        torch.testing.assert_close(original_prediction, changed_prediction)


if __name__ == "__main__":
    unittest.main()
