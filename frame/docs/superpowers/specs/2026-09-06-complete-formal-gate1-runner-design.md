# Complete Formal Gate 1 Runner Design

## Goal

Add a real Gate 1 executor for `rsc-pf-complete-formal-v1` that trains all
35 stochastic rows, evaluates all 37 rows on the complete 2019 rolling
selection chronology, and emits auditable row receipts that can authorize
Gate 2 only when every row is real and complete.

## Boundary and invariants

- Training data are exactly 2015--2018; normalization is fit on training
  windows only.
- Gate 1 selection data are exactly 2019 and contain 8,709 chronological
  origins.  2020 and 2021 are never loaded by the runner.
- The frozen roster is the nine-method, 37-row matrix in
  `CompleteFormalContract.expected_rows("gate1")`.
- Stochastic methods use seeds 2026--2030. Seasonal-Naive-PTO and
  Perfect-Information-MPC are deterministic rows with a null seed.
- Existing v4.2 training and rollout kernels are adapters only; every output
  is relabeled and lineage-stamped with the complete-v1 contract hash.
- A smoke run may validate wiring but is explicitly non-paper and can never
  authorize Gate 2. The full runner is the only path that can write a
  `paper_result=true` transition.

## Architecture

`complete_formal_gate1.py` owns data-boundary validation, the explicit
legacy-kernel adapter, method-matrix training, chronological selection
evaluation, and row receipt creation. The CLI script performs fail-closed
Gate 0 authorization and writes one immutable run directory. Existing
`complete_formal_execution.audit_rows` remains the independent row-accounting
auditor rather than a training implementation.

## Outputs

```
<output>/<run-id>/gate1/
  TRAIN_WINDOWS.npz
  SELECTION_WINDOWS.npz
  NORMALIZATION.json
  DATA_LINEAGE.json
  rows/<method>/<seed>/CHECKPOINT.pt        # stochastic rows
  rows/<method>/<seed>/ROLLOUT.npz
  rows/<method>/<seed>/METRICS.json
  rows/<method>/<seed>/ROW_RECEIPT.json
  ROWS.json
  GATE1_EVIDENCE.json
  GATE1_AUDIT.json
protocol/GATE1_TRANSITION.json
```

The transition is authorized only when 37 row receipts, checkpoints for all
stochastic rows, full 8,709-origin rollouts, finite metrics, chronology,
physical-feasibility checks, expected LP-call counts, and the independent
complete-formal audit all pass.
