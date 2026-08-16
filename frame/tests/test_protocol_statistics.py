"""Tests for paired circular moving-block protocol statistics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from frame.src.protocol_statistics import (
    benjamini_hochberg,
    circular_moving_block_indices,
    difference_in_differences_bootstrap,
    paired_protocol_bootstrap,
)


class _FixedStartRng:
    def __init__(self, start: int) -> None:
        self.start = int(start)

    def integers(self, low: int, high: int, *, size: tuple[int, int]) -> np.ndarray:
        del low, high
        return np.full(size, self.start, dtype=np.int64)


class ProtocolStatisticsTests(unittest.TestCase):
    def test_identical_arrays_have_zero_effect_and_interval_contains_zero(self) -> None:
        target = np.ones((12, 2, 2), dtype=float)
        prediction = target.copy()
        result = paired_protocol_bootstrap(
            prediction,
            prediction,
            target,
            metric="MAE",
            block_length=4,
            replicates=50,
            seed=2026,
        )
        self.assertEqual(result.estimate, 0.0)
        self.assertEqual(result.ci_low, 0.0)
        self.assertEqual(result.ci_high, 0.0)
        self.assertEqual(result.p_value, 1.0)

    def test_constant_improvement_has_negative_effect_and_excludes_zero(self) -> None:
        target = np.zeros((12, 2, 2), dtype=float)
        cross = np.ones_like(target)
        post = np.full_like(target, 0.25)
        result = paired_protocol_bootstrap(
            cross,
            post,
            target,
            metric="MAE",
            block_length=4,
            replicates=100,
            seed=7,
        )
        self.assertAlmostEqual(result.estimate, -0.75)
        self.assertLess(result.ci_high, 0.0)
        self.assertLess(result.p_value, 0.05)

    def test_circular_blocks_wrap_at_the_end(self) -> None:
        indices = circular_moving_block_indices(
            5, 3, 1, _FixedStartRng(4)  # type: ignore[arg-type]
        )
        np.testing.assert_array_equal(indices, np.array([[4, 0, 1, 4, 0]]))

    def test_same_resampled_indices_are_used_for_all_protocols(self) -> None:
        target = np.arange(24, dtype=float).reshape(6, 2, 2)
        cross = target + 1.0
        post = target + 2.0
        fixed = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
        with patch(
            "frame.src.protocol_statistics.circular_moving_block_indices",
            return_value=fixed,
        ) as sampler:
            result = paired_protocol_bootstrap(
                cross,
                post,
                target,
                metric="MAE",
                block_length=2,
                replicates=1,
                seed=1,
            )
        sampler.assert_called_once()
        self.assertAlmostEqual(result.samples[0], 1.0)

    def test_fixed_seed_is_byte_identical(self) -> None:
        target = np.arange(48, dtype=float).reshape(12, 2, 2)
        cross = target + 1.0
        post = target + 0.5
        first = paired_protocol_bootstrap(
            cross, post, target, metric="RMSE", block_length=4, replicates=30, seed=9
        )
        second = paired_protocol_bootstrap(
            cross, post, target, metric="RMSE", block_length=4, replicates=30, seed=9
        )
        self.assertEqual(first.samples.tobytes(), second.samples.tobytes())
        self.assertEqual(first.estimate, second.estimate)
        self.assertEqual(first.ci_low, second.ci_low)
        self.assertEqual(first.ci_high, second.ci_high)
        self.assertEqual(first.p_value, second.p_value)

    def test_sample_count_smaller_than_block_length_is_rejected(self) -> None:
        target = np.ones((3, 2, 2), dtype=float)
        with self.assertRaises(ValueError):
            paired_protocol_bootstrap(
                target,
                target,
                target,
                metric="MAE",
                block_length=4,
                replicates=10,
                seed=1,
            )

    def test_bh_adjustment_is_bounded_and_monotone_in_sorted_order(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.20, 0.001])
        self.assertTrue(np.all((adjusted >= 0.0) & (adjusted <= 1.0)))
        ranked = adjusted[np.argsort([0.01, 0.04, 0.20, 0.001])]
        self.assertTrue(np.all(np.diff(ranked) >= -1e-12))

    def test_task_and_horizon_axes_are_not_flattened_into_time(self) -> None:
        target = np.zeros((8, 2, 2), dtype=float)
        cross = np.zeros_like(target)
        post = np.zeros_like(target)
        post[:, 0, 0] = 1.0
        result = paired_protocol_bootstrap(
            cross,
            post,
            target,
            metric="MAE",
            block_length=4,
            replicates=20,
            seed=5,
        )
        self.assertAlmostEqual(result.estimate, 0.25)
        self.assertEqual(result.n_origins, 8)

    def test_difference_in_differences_uses_four_aligned_streams(self) -> None:
        target = np.zeros((12, 2, 2), dtype=float)
        predictions = {
            ("cross_topology", "scheme2r"): np.full_like(target, 2.0),
            ("cross_topology", "dynamic_symmetric"): np.full_like(target, 1.0),
            ("post_ge_regular_operation", "scheme2r"): np.full_like(target, 1.5),
            ("post_ge_regular_operation", "dynamic_symmetric"): np.full_like(target, 1.0),
        }
        result = difference_in_differences_bootstrap(
            predictions,
            target,
            metric="MAE",
            block_length=4,
            replicates=40,
            seed=11,
        )
        self.assertAlmostEqual(result.estimate, -0.5)
        self.assertLess(result.ci_high, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
