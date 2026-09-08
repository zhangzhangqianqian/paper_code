# Complete Formal Gate1 Recovery Repair Report

## Scope

This repair addresses the failed recovery run
`reports/rsc_pf_complete_formal/complete_formal_gate1_recovered_20260907_a`.
The run stopped before Official iTransformer-PTO because its source root was
resolved from the data-run parent instead of from the current formal frame.
No long formal experiment was launched while applying this repair.

## Changes

1. The official iTransformer receipt is now resolved against the current
   `src/joint_dispatch` checkout, with `frame/` prefixes normalized and
   repository-relative and `..` paths rejected. The pinned Git commit,
   imported-file hashes, and license hash are verified before search or matrix
   training; the resolved path is recorded in
   `ITRANSFORMER_ADAPTER_RECEIPT.json`.
2. Recovery inspection now validates final seed-2026 training checkpoints for
   RSC-PF, Decoupled-RSC-PF, State-Conditioned-PTO, Direct-Policy, and
   Scheme2R-PTO. Validation checks the immutable receipt, checkpoint hash,
   checkpoint schema/epoch, train-only lineage, and test-set access flag.
3. Valid final checkpoints are copied into the new run and restored with the
   original model architecture and optimizer parameter groups. Restoration
   loads parameters and optimizer state but never calls an optimizer update.
4. The matrix trainer now reuses a complete RSC family atomically and falls
   back to normal training for missing or invalid rows. Every action is
   recorded in `GATE1_MATRIX_RECOVERY.json`; the existing failed source run is
   never overwritten.
5. Recovery manifests now include final-row reuse evidence and the matrix
   receipt hash. The Gate1 evidence remains fail-closed: no evidence or
   transition is written after a preflight/training exception.

## Verification

The following checks passed:

```text
19 passed: complete Gate1 recovery, source preflight, and iTransformer tests
short smoke continuation: 37 rows, synthetic=true, paper_result=false,
authorized_gate2=false, evaluation_year_accessed=false
git diff --check passed
```

The five existing final seed-2026 checkpoints from the recovered partial run
were read-only validated and restored successfully, retaining their original
SHA-256 identities. The old failed run and its source artifacts were not
modified.

## Operator boundary

The repair is code-complete and test-verified. A new full Gate1 continuation is
still an operator-started multi-hour experiment. It should be launched only
after reviewing the repair receipts; this task intentionally did not start it.
