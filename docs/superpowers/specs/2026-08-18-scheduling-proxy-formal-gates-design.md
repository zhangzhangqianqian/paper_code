# Scheduling Proxy V2 Formal Gate 4/5 Design

## 1. Purpose

This design adds a strict formal evaluation boundary for the feasible neural scheduling proxy v2. It separates validation acceptance from the single locked test evaluation so that test data cannot influence model, checkpoint, threshold, or architecture selection.

The workflow is:

```text
prepare formal data and train
        -> validate against Gate 4
        -> write an immutable freeze record and stop
        -> user issues a separate test command
        -> verify the freeze record
        -> evaluate the test split once (Gate 5)
```

The existing smoke workflow remains validation-only. The v1 pipeline and all existing v1/v2 artifacts remain readable and are not overwritten.

## 2. Scope

Included:

- a machine-readable Gate 4/5 contract;
- a dedicated formal runner with separate `prepare`, `validate`, and `test` commands;
- explicit validation-split evaluation support in the shared pipeline;
- validation acceptance checks and a cryptographically bound freeze record;
- a fail-closed, one-time test command;
- formal manifests, audits, and regression tests.

Excluded:

- changes to Scheme2R or forecasting checkpoints;
- changes to the v2 scheduler architecture, decoder, loss weights, or training hyperparameters;
- formal training execution as part of implementation;
- automatic test evaluation after validation;
- manuscript, figure, or published-result changes;
- deletion or reinterpretation of v1 artifacts.

## 3. Fixed inputs and thresholds

The runner consumes:

- benchmark: `D:/Paper/standard_ies_benchmark_v1.yaml`;
- scheduler contract: `frame/configs/scheduling_proxy_contract_v2.json`;
- formal gate contract: `frame/configs/scheduling_proxy_formal_gate_v2.json`;
- caller-supplied new output directory.

Formal split sizes and seeds remain those in the v2 scheduler contract:

| Split | Seed | Scenarios |
|---|---:|---:|
| train | 2026 | 8192 |
| validation | 2027 | 2048 |
| test | 2028 | 2048 |

Gate 4 passes only when all conditions hold:

- validation raw feasible rate equals `1.0`;
- maximum balance, conversion, SOC, renewable-split, bound, ramp, capacity, and charge/discharge-overlap residual is at most `1e-6`;
- mean and maximum electricity, cooling, and heating slack are each at most `1e-6`;
- fallback rate equals `0.0`;
- inference LP calls equal `0`;
- validation relative absolute objective gap is at most `0.05`;
- validation relative cost regret is at most `0.05`;
- validation relative carbon error is at most `0.10`;
- combined model-plus-decoder CPU median latency is at most `10.0 ms` for one batch containing all 2048 validation scenarios, after two warm-up runs and over five measured repeats.

Thresholds are read from the formal gate contract and copied verbatim into every acceptance and test manifest. The test command cannot change them.

## 4. Commands and state transitions

The dedicated entry point is:

```text
frame/scripts/run_scheduling_proxy_formal.py
```

It accepts exactly one stage per invocation.

### 4.1 `prepare`

`prepare`:

1. requires an empty or nonexistent output directory;
2. loads and validates the benchmark, v2 scheduler contract, and formal gate contract;
3. generates the deterministic formal train, validation, and sealed test artifacts;
4. records hashes of all three split files and the dataset manifest;
5. trains only with train/validation data and selects the best checkpoint only by validation loss;
6. writes `formal_prepare_manifest.json` atomically;
7. does not load the test split for evaluation and creates no test metrics or predictions.

Generating and hashing the deterministic test artifact is not test evaluation. No model output, metric, acceptance decision, or human-readable test summary is produced during `prepare`.

### 4.2 `validate`

`validate`:

1. requires a complete `prepare` bundle;
2. verifies benchmark, contracts, dataset, normalization statistics, checkpoint, and training-history hashes;
3. evaluates only `validation.npz`;
4. writes `metrics_validation.json`, `predictions_validation.npz`, and validation constraint diagnostics;
5. computes every Gate 4 condition without rounding;
6. writes `formal_validation_acceptance.json` with one row per condition, measured value, comparator, threshold, and pass/fail result;
7. if every condition passes, writes `formal_freeze.json` atomically and exits successfully;
8. if any condition fails, does not write a freeze record, writes the failed acceptance report, and exits with a nonzero status.

The freeze record binds the exact benchmark, both contracts, all dataset splits, normalization statistics, checkpoint, history, validation metrics, validation predictions, diagnostics, thresholds, Python/Torch versions, CPU identity, and timestamp by SHA-256 or literal metadata.

Running `validate` again is allowed only when no test receipt exists and the inputs are byte-identical to the prepare manifest. It recomputes validation evidence but cannot silently replace a freeze record with different content.

### 4.3 `test`

`test` is a separate manual command. It never runs automatically after `validate`.

Before reading `test.npz`, it requires:

- a passing validation acceptance report;
- an existing freeze record;
- exact hash agreement for every item bound by the freeze record;
- absence of test metrics, predictions, diagnostics, manifest, receipt, or in-progress marker;
- the v2 model family, decoder schema, zero-fallback flag, and zero-LP-inference provenance.

The command then:

1. creates an exclusive `formal_test_in_progress.json` marker;
2. evaluates the sealed test split once;
3. writes test metrics, predictions, diagnostics, and `formal_test_manifest.json` through temporary files followed by atomic replacement;
4. writes `formal_test_receipt.json` containing hashes of all test outputs and the freeze record;
5. removes the in-progress marker only after every artifact and audit passes.

If a test receipt or any completed test artifact already exists, the command fails closed. There is no `--force`, overwrite, or automatic retry option. An interrupted run leaves its marker and temporary artifacts for explicit forensic recovery; it must not be treated as a completed test.

## 5. Shared evaluation interface

The shared evaluation function gains an explicit split argument:

```python
evaluate_from_dataset(
    contract: ProxyContract,
    benchmark_path: Path,
    output_dir: Path,
    *,
    evaluation_split: Literal["validation", "test"],
    smoke: bool = False,
) -> dict[str, Any]
```

Callers must name the split. Existing CLI behavior remains compatible:

- v1/v2 `--mode evaluate` continues to mean test evaluation;
- v2 `--mode smoke` continues to mean validation-only smoke evaluation;
- the formal runner calls the function explicitly for validation or test.

Metrics and prediction artifacts retain checkpoint SHA-256, contract hashes, split identity, model family, decoder metadata, and zero-LP provenance.

## 6. Acceptance computation

Acceptance logic is a pure function so it can be tested independently:

```python
evaluate_gate4_acceptance(
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    gate_contract: Mapping[str, Any],
) -> GateDecision
```

`GateDecision` contains:

- overall status: `passed` or `failed`;
- ordered condition results;
- raw measured values and thresholds;
- no rounded values used in comparisons;
- a stable schema version.

Slack maxima are computed from physical prediction arrays, not inferred from mean metrics. Latency must report batch size, warm-up runs, repeats, median, and p95; only the frozen median threshold determines acceptance.

NaN, infinity, missing keys, wrong shapes, wrong split identity, or malformed provenance is an automatic failure.

## 7. Leakage and tamper protection

The following are hard errors:

- validation artifacts claiming the test split;
- test artifacts existing before a valid freeze record;
- training history claiming test-based selection;
- mismatch between stored predictions and recomputed metrics;
- mismatch between checkpoint SHA-256 and prediction/metric provenance;
- changed threshold, contract, benchmark, dataset, normalization, history, checkpoint, or validation evidence after freezing;
- v2 inference importing/calling the LP solver or using fallback;
- a second completed test invocation.

The formal runner never accepts caller-supplied metric values. It derives acceptance from audited artifacts generated in the same invocation.

## 8. Error handling and recovery

- All manifests and receipts are written to temporary files and atomically renamed.
- `prepare` refuses nonempty output directories to prevent mixing runs.
- `validate` preserves a failed report for diagnosis but does not create a freeze record.
- `test` refuses any existing completed test artifact.
- A stale in-progress marker is not automatically removed. Recovery requires a separate future procedure that verifies no completed output was consumed; this design intentionally does not implement an unsafe automatic resume.
- No command deletes v1, smoke, or prior formal directories.

## 9. Test strategy

Automated tests must cover:

1. explicit validation and test split routing;
2. `prepare` produces no test prediction/metric output;
3. every Gate 4 threshold passes at its boundary and fails immediately above it;
4. missing/nonfinite metrics fail closed;
5. slack maxima are computed from predictions;
6. no freeze record when any condition fails;
7. valid freeze record contains all required hashes;
8. changed checkpoint, contract, threshold, dataset, history, or validation artifact blocks test;
9. absent or failed acceptance blocks test before test data is loaded;
10. test command is separate and creates one receipt;
11. second test invocation is rejected;
12. interrupted-test marker blocks re-entry;
13. v2 formal inference remains isolated from SciPy/LP and has zero fallback;
14. v1 and smoke regression tests remain green.

A formal dry-run fixture uses tiny synthetic split sizes and a deliberately configurable acceptance fixture. It verifies orchestration only and is never reported as a paper result.

## 10. Completion criteria

Implementation is complete when:

- the dedicated runner and gate contract exist;
- all new focused tests pass;
- the complete `frame/tests` suite passes;
- a tiny validation-pass dry run creates a freeze record without test outputs;
- a separate tiny test dry run creates exactly one audited test receipt;
- a second test call is rejected;
- no formal full-size training is run during implementation;
- no unrelated user files are staged or modified.

