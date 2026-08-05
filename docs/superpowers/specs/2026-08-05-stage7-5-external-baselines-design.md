# Stage 7.5 external-baseline formal runner

## Goal

Create a formal Stage 7.5 runner that compares the frozen Scheme2R experiment
with five external baselines on the same Kitakyushu protocols, without sample
caps or smoke-training shortcuts.

## Frozen run matrix

- Deterministic baselines: Persistence and Seasonal Naive, one run per
  protocol because they have no trainable randomness.
- Learned baselines: DLinear, MMoE-lite and SOFTS, each run under `full` and
  `small_sample` with seeds 2026--2030.
- Total: 2 x 2 + 3 x 2 x 5 = 34 runs.

All runs use the Stage 7 contract: Kitakyushu 2015--2021, 24-to-4 windows,
the frozen train/validation/test splits, no future exogenous variables, batch
size 256, at most 100 epochs, validation early stopping patience 12, eight
CPU threads, and no sample limits.

## Implementation

`frame/scripts/run_stage7_5.py` orchestrates the run matrix. It reuses the
tested Stage 5 external-model trainer for learned baselines and computes the
two deterministic baselines directly from the frozen test windows. Every run
gets its own directory and manifest; the root directory receives run-level and
mean/std summary CSV files plus a completion manifest.

## Acceptance

The script must support `--dry-run`, reject an existing completed output unless
`--force` is supplied, print one progress record per run, preserve failures in
`error.json`, and pass a unit test for the 34-run plan. Formal training is not
started as part of implementation.
