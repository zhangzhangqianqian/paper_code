# Complete-formal execution hardening design

## Scope

Harden the existing complete-formal protocol without changing the RSC-PF
architecture, frozen method roster, data split, or objective definitions.

## Design

1. Synthetic/tiny row matrices remain available for unit tests, but they can
   never create an authorized Gate 1 transition or unlock Gate 2. A transition
   that unlocks the sealed evaluation must explicitly identify non-synthetic,
   paper-result artifacts and a passing audit.
2. The complete-formal provider result will use the same planned-step type as
   the chronological closed-loop evaluator. Decision-only methods will provide
   a finite placeholder forecast for the shared transport interface; the method
   contract continues to mark their forecast metrics as not applicable.
3. Existing real training and rollout implementations remain the source of
   scientific computation. The complete-formal orchestration layer must pass
   real checkpoints and rollout receipts into the gates; metadata-only rows are
   test fixtures, not experiment evidence.
4. Gate 0 uses a fixed 20 GiB post-checkpoint disk reserve rather than a
   percentage of the drive capacity. The reserve is reported in both bytes and
   GiB; the old percentage is retained only as an informational diagnostic.
5. The complete contract binds a compact, training-only Standard-IES capacity
   artifact and its rules/ledger hashes. Gate 0 runs its exact LP smoke with
   those bound values, rather than with an unrelated unit-test fixture.
6. The command-line Gate 1 entry point requires an explicit real-row receipt;
   synthetic rows remain available only through the test API and cannot be
   mistaken for a formal experiment invocation.

## Acceptance criteria

- A synthetic Gate 1 transition has `authorized_gate2=false` and is rejected by
  sealed evaluation authorization.
- A complete-formal deployable provider returns a planned step accepted by the
  chronological evaluator's type contract.
- Existing complete-formal and matched closed-loop regression tests pass.
- No 2019/2020 arrays are loaded by the hardening tests.
- A drive with at least 20 GiB free after the estimated checkpoint footprint
  passes the disk resource check regardless of total drive capacity.
- The Gate 0 receipt includes a passing training-only capacity binding and the
  exact LP smoke reports the bound Standard-IES parameters.
- Invoking the formal Gate 1 command without a real row receipt is rejected.
