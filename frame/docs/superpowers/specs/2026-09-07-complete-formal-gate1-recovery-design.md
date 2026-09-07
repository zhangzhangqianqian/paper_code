# Complete Formal Gate 1 Recovery Design

## Goal

Resume the failed `complete_formal_gate1_20260906_b` run without retraining
verified work, while preserving an immutable and independently auditable record
of the original failure and every reused artifact.

The recovery must reuse all valid completed RSC-PF search trials and the valid
trained Differentiable-LP `1e-5` checkpoint. It must rerun only the incomplete
evaluation for that Differentiable-LP trial, then train the three remaining
Differentiable-LP search trials and continue the normal Gate 1 matrix.

## Non-negotiable invariants

- The source run is read-only. Recovery never edits, deletes, renames, or adds
  files below `complete_formal_gate1_20260906_b`.
- Recovery writes to a new run ID and refuses to overwrite any existing run.
- Reuse is allowed only after contract, Gate 0 transition, source data,
  normalization, method, seed, epoch count, checkpoint, and receipt hashes pass.
- A missing, malformed, or mismatched artifact is not silently accepted. Only
  the affected trial is retrained from the frozen protocol.
- Recovery does not load 2020 or 2021 data. Search and Gate 1 evaluation remain
  restricted to the frozen 2019 selection chronology.
- Reused search rows remain `paper_result=false`; final Gate 1 rows become paper
  results only after the existing 37-row audit passes.
- A recovered run can authorize Gate 2 only through the same independent audit
  and transition checks as a clean full run.

## Chosen architecture

### Recovery entry point

Extend the real Gate 1 command with an optional `--resume-from` argument. The
argument names a prior complete-formal Gate 1 run directory. A recovery run
still requires a new `--run-id`; the current refusal to overwrite an existing
run remains unchanged.

`Gate1RunConfig` carries the optional recovery source. `run_complete_gate1`
validates the source before starting search and writes a recovery manifest into
the new Gate 1 directory.

### Artifact validator

Add a narrow recovery validator for search trials. For each expected candidate,
it checks:

1. the recovery source belongs to the same complete-formal contract and Gate 0
   transition;
2. copied `TRAIN_WINDOWS.npz`, `SELECTION_WINDOWS.npz`, and
   `NORMALIZATION.json` match the current source run byte-for-byte;
3. the training receipt identifies the expected method, seed 2026, and 30
   epochs;
4. the checkpoint exists and its SHA-256 equals the training receipt;
5. a completed candidate receipt, when present, contains 8,709 chronological
   origins, finite metrics, physical feasibility, and hashes matching its
   checkpoint, rollout, and metrics files.

The validator returns one of three states per candidate:

- `reusable-complete`: training and evaluation are both valid;
- `reusable-checkpoint`: training is valid but evaluation is absent or
  incomplete;
- `retrain-required`: training evidence is absent or invalid.

### Immutable materialization

Validated files are copied, not moved or hard-linked, into the new run. The
recovery run therefore remains independent of later changes to the old run.
The current recovery source is about 77 MB, so copying is preferable to shared
filesystem links.

For the first Differentiable-LP trial, the old checkpoint and training receipt
were written one directory above the canonical row directory. Recovery accepts
this exact legacy layout only when both hashes and receipt fields validate, then
copies the files into:

`search/difflp_lr_1e-05/rows/Differentiable-LP/2026/`

This is recorded as a canonicalization operation; the source files are not
altered.

### Search continuation

`select_gate1_hyperparameters` processes candidates independently:

- For `reusable-complete`, it reads the verified candidate score and eligibility
  from the copied complete receipt and skips training and evaluation.
- For `reusable-checkpoint`, it reconstructs the appropriate model, loads the
  verified checkpoint, reruns only the 2019 evaluation, and writes a fresh
  complete candidate receipt.
- For `retrain-required`, it runs the frozen 30-epoch training and evaluation.

The recovered RSC-PF multipliers `1`, `1.5`, `2`, and `3` should all be
`reusable-complete`. Differentiable-LP learning rate `1e-5` should be
`reusable-checkpoint`. Learning rates `3e-5`, `1e-4`, and `3e-4` are absent and
must be trained.

After every candidate is complete, the existing finite/physical-feasibility
selection rule writes `GATE1_SEARCH.json`. The final 37-row matrix then follows
the unchanged Gate 1 protocol using the selected values.

## Recovery manifest

Write `gate1/RECOVERY_MANIFEST.json` before continuation and finalize it after
search. It contains:

- source and destination run IDs and absolute paths;
- source failure receipt hash;
- contract, Gate 0 transition, train, selection, and normalization hashes;
- one entry per search candidate with validation state, copied file hashes,
  action taken, and reason;
- aggregate reused-training time and wall-clock time avoided;
- `source_modified=false`, `evaluation_year_accessed=false`, and
  `test_set_accessed=false`;
- final manifest status and SHA-256.

The manifest is evidence of provenance, not authorization by itself.

## Failure handling

- Source-level mismatch: abort recovery before copying any candidate and write
  `GATE1_FAILURE.json`.
- Candidate checkpoint mismatch: mark only that candidate `retrain-required`.
- Complete receipt or rollout mismatch: retain the verified checkpoint if
  possible and rerun evaluation; otherwise retrain that candidate.
- Failure during continuation: write the normal failure receipt plus the current
  recovery manifest. A later recovery may use this newer run after validation.
- Existing destination run: refuse immediately; never resume by mutating it.

## Tests

Focused tests must cover:

- rejection of a contract, Gate 0, data, normalization, seed, epoch, or
  checkpoint-hash mismatch;
- recognition of all four completed RSC-PF candidates;
- recognition and canonicalization of the misplaced Differentiable-LP `1e-5`
  checkpoint without retraining;
- evaluation-only continuation from a reconstructed Differentiable-LP model;
- retraining of only absent/invalid candidates;
- immutable source behavior;
- recovery manifest completeness;
- smoke and recovered search rows remaining non-paper results;
- unchanged fail-closed Gate 2 authorization.

## Success criteria

The recovery implementation is ready for the long run when focused regression
tests and static checks pass, a bounded fixture proves that completed trials are
not retrained, and a dry-run audit classifies the real failed run as four
`reusable-complete` RSC-PF candidates plus one `reusable-checkpoint`
Differentiable-LP candidate. The long experiment is not considered complete
until the final transition reports `authorized_gate2=true` and the independent
audit reports `pass`.
