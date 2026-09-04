# RSC-PF Formal-v4.2 Gate 0 Self-Contained Evidence Repair Design

## Status

Approved direction: Scheme A, a self-contained formal-v4.2 Gate 0 evidence producer and validator.

## Problem

The formal-v4.2 Gate 0 core accepts four prerequisite receipt paths, but its command-line entry point neither creates nor accepts those paths. Consequently, every normal command invocation supplies `None` for the iTransformer, differentiable-LP, capacity, and source-manifest receipts. The run therefore writes `authorized_pilot=false` even when the corresponding runtime operations succeed.

The diagnostic run `formal_v4_2_20260904_a` must remain immutable and must not authorize later gates. It records an entry-point defect, not a failed model or environment.

## Decision

Gate 0 will become a single fail-closed orchestration boundary. One command will create a fresh run root, produce the four formal-v4.2 prerequisite receipts from current frozen inputs, validate their contents and hashes, run the registered real-operation probes, and issue authorization only if every check passes.

Old formal-v4.1 receipts may be inspected for implementation compatibility, but they may not be copied, referenced, or counted as formal-v4.2 evidence.

## Alternatives Considered

### A. Self-contained producer and validator — selected

The Gate 0 command produces and validates every prerequisite under the new run root. This minimizes manual steps, keeps all evidence under one lineage, and prevents omitted CLI arguments from silently denying or incorrectly authorizing the run.

### B. Add four receipt-path CLI arguments

This is a smaller patch, but it leaves receipt production outside the formal workflow and makes future runs dependent on manual path selection. It also makes it easier to pass stale or incompatible receipts.

### C. Reuse formal-v4.1 receipts

Rejected. The receipts belong to a different protocol and some v4.1 runs were already classified as diagnostic-only. Reuse would break the formal-v4.2 lineage claim.

## Architecture

The repaired flow has two layers:

1. **Evidence producers** create immutable, content-addressed formal-v4.2 receipts.
2. **Gate validator** reloads each receipt from disk, validates scientific boundaries and hashes, performs the real runtime probes, and decides whether the pilot is authorized.

The command sequence is:

```text
clean committed v4.2 source closure
        |
        +--> SOURCE_MANIFEST.json
        +--> ITRANSFORMER_SOURCE_RECEIPT.json
        +--> DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json
        +--> CAPACITY_FREEZE.json (2015-2018 only)
        |
        +--> real RSC forward/backward probe
        +--> real HiGHS LP probe
        +--> real CVXPYlayers forward/backward probe
        |
        +--> GATE0_EVIDENCE.json
               |
               +-- pass --> CURRENT_GATE.json, next_gate=pilot
               +-- fail --> GATE0_FAILURE.json, nonzero exit
```

## Receipt Contracts

Every receipt carries:

- `schema` identifying a formal-v4.2 receipt;
- `run_id`;
- `contract_sha256`;
- `source_manifest_sha256` where the source manifest already exists;
- producer identity and content hashes;
- `evaluation_year_accessed=false`;
- an immutable status field.

### Source manifest

The producer reads `configs/formal_v4_source_closure_v4_2.txt`, requires every entry to exist, be tracked, and be clean at the current Git `HEAD`, and hashes the exact file bytes. The manifest is produced only after the repair commit is complete.

### Official iTransformer receipt

The producer requires the local source root to be a readable Git checkout at the frozen THUML commit. It hashes the imported Python files and license, records `official_backbone_adaptation`, and validates the receipt by re-reading the files. Merely finding the directory is insufficient.

### Differentiable-LP environment receipt

The producer verifies the frozen dependency lock, imports CVXPY, CVXPYlayers, diffcp, ECOS, NumPy, and PyTorch from the configured isolated interpreter, and executes a finite DPP-compliant forward/backward layer probe. It records interpreter identity, package versions, lock hash, solver identity, finite primal output, and a finite nonzero gradient.

### Capacity receipt

The producer uses only 2015-2018 training data and the frozen candidate multipliers and thresholds from the v4.2 contract. It performs both the registered stratified diagnostic and chronological certification, records all candidate rows, and freezes a multiplier only when both checks pass. Selection-year data may be materialized for later gates but may not influence the capacity choice. Evaluation-year data is never opened.

## Command Interface

The formal command accepts stable source inputs rather than four receipt paths:

```text
--contract
--output-root
--run-id
--data-dir
--diffopt-python
--itransformer-source
```

Defaults come from the frozen v4.2 contract where available. The four receipts are always written beneath the new run root and then passed internally to the validator. The command refuses an existing run root.

## Validation and Authorization

The validator must parse and content-validate every receipt. File existence alone never counts as a pass. It verifies matching run id, contract hash, source-manifest hash, expected years, official iTransformer commit, frozen dependency lock, differentiable gradient, capacity thresholds, real-operation identities, finite timings, and absence of 2020/2021 access.

`authorized_pilot=true` is written only when all mandatory checks pass. A successful run also writes an immutable `protocol/CURRENT_GATE.json` containing the Gate 0 evidence hash and `next_gate: pilot`.

## Failure Handling

Any producer or validator exception results in an immutable `gate0/GATE0_FAILURE.json` with the failed stage, exception class, message, and known lineage. The process exits nonzero and never imports or starts the pilot or Gate 1 runner.

The incomplete run root is retained for audit. A repaired retry must use a new run id. The existing `formal_v4_2_20260904_a` root remains a diagnostic artifact and will not be modified.

## Testing

Tests will cover:

- normal CLI invocation supplies real internally generated receipts;
- missing, malformed, stale, cross-run, or hash-mismatched receipts fail closed;
- dirty or untracked source-closure entries fail before authorization;
- the iTransformer commit and file hashes are revalidated;
- the DiffLP probe produces a finite nonzero gradient in the isolated interpreter;
- capacity fitting uses only 2015-2018 and never reads 2020/2021;
- failure writes `GATE0_FAILURE.json` and returns nonzero;
- success writes `authorized_pilot=true` and `CURRENT_GATE.json`;
- no pilot or Gate 1 code is invoked by Gate 0;
- existing formal-v4.2 and differentiable-LP regression suites remain green.

## Operational Sequence

After implementation and tests are committed:

1. verify the complete source closure against the new `HEAD`;
2. generate a new source manifest;
3. run Gate 0 with a fresh run id, initially `formal_v4_2_20260904_b` if unused;
4. inspect the immutable evidence and failure/authorization marker;
5. stop after Gate 0 and report the result; do not automatically launch the pilot as part of this repair.

