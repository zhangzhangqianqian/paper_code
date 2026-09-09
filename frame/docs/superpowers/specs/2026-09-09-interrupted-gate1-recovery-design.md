# Interrupted Gate 1 Recovery Design

## Goal

Convert a Gate 1 run that was stopped by a system restart into an immutable,
auditable recovery source, without changing any completed checkpoints or
allowing evaluation-year data to enter the recovery lineage.

## Design

Add a small sealing utility that performs read-only checks on an interrupted
run, then writes the existing Gate 1 failure-receipt schema with an explicit
`interrupted=true` marker. The utility refuses runs that already have evidence,
a Gate 1 transition, a failure receipt, or any JSON artifact reporting
evaluation/test-set access. The normal recovery inspector then validates the
sealed run and reuses only complete, hash-checked rows.

The original interrupted directory remains immutable except for the one
missing failure receipt. A new run id is used for continuation; no output is
overwritten.

## Acceptance checks

- The seal operation is idempotent and refuses conflicting existing receipts.
- The sealed `_d` run is accepted by `inspect_recovery_source`.
- Existing completed rows remain byte-identical.
- A new continuation run can reuse those rows and train only missing rows.
- No evaluation-year data is opened or authorized by sealing.
