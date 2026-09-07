# Complete Formal Gate 1 Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the failed 15-hour Gate 1 run from verified search artifacts without modifying the source run or retraining completed candidates.

**Architecture:** Add a focused recovery module that classifies each search candidate as fully reusable, checkpoint-reusable, or requiring retraining. The real Gate 1 runner accepts an optional read-only recovery source, copies verified evidence into a new immutable run, continues only missing work, and records every decision in `RECOVERY_MANIFEST.json` before the unchanged 37-row audit can authorize Gate 2.

**Tech Stack:** Python 3, pathlib, dataclasses, JSON, SHA-256, shutil, PyTorch checkpoint loading, pytest.

## Global Constraints

- The source run is read-only; no file below `complete_formal_gate1_20260906_b` may change.
- The destination must use a new run ID and must not already exist.
- Training and selection boundaries remain 2015--2018 and 2019; 2020 and 2021 remain inaccessible.
- Reuse requires matching contract, Gate 0, data, normalization, method, seed, 30-epoch receipt, and checkpoint hashes.
- Invalid evidence triggers recomputation of only the affected candidate.
- Search candidates remain `paper_result=false` and cannot independently authorize Gate 2.
- Gate 2 authorization still requires the existing complete 37-row audit.
- The existing failed source directory contains about 77 MB, so validated evidence is copied rather than moved or hard-linked.

---

### Task 1: Define recovery inspection types and source-level validation

**Files:**
- Create: `src/joint_dispatch/complete_formal_gate1_recovery.py`
- Modify: `src/joint_dispatch/complete_formal_gate1.py:39-55`
- Test: `tests/test_complete_formal_gate1_recovery.py`

**Interfaces:**
- Consumes: `CompleteFormalContract`, `Gate1DataBundle`, current Gate 0 transition path, and an optional recovery run path.
- Produces: `RecoveryCandidateKey`, `RecoveryCandidateEvidence`, `Gate1RecoveryInspection`, and `inspect_recovery_source(...)`.

- [ ] **Step 1: Write failing tests for exact candidate keys and source-level rejection**

```python
def test_expected_recovery_candidates_match_frozen_search_grid():
    contract = CompleteFormalContract.from_path(CONTRACT_PATH)
    keys = expected_recovery_candidates(contract)
    assert [(k.family, k.value) for k in keys] == [
        ("RSC-PF", 1.0), ("RSC-PF", 1.5),
        ("RSC-PF", 2.0), ("RSC-PF", 3.0),
        ("Differentiable-LP", 1e-5),
        ("Differentiable-LP", 3e-5),
        ("Differentiable-LP", 1e-4),
        ("Differentiable-LP", 3e-4),
    ]


def test_recovery_rejects_contract_mismatch(tmp_path, gate1_data, contract):
    source = copy_recovery_fixture(tmp_path)
    payload = json.loads((source / "gate1" / "DATA_LINEAGE.json").read_text())
    payload["contract_sha256"] = "0" * 64
    (source / "gate1" / "DATA_LINEAGE.json").write_text(json.dumps(payload))
    with pytest.raises(PermissionError, match="contract"):
        inspect_recovery_source(source, contract, gate1_data, GATE0_TRANSITION)
```

- [ ] **Step 2: Run the tests and confirm the recovery module is absent**

Run: `pytest tests/test_complete_formal_gate1_recovery.py -q`

Expected: FAIL during import because the recovery types and inspection function do not exist.

- [ ] **Step 3: Add immutable recovery types and candidate enumeration**

```python
@dataclass(frozen=True)
class RecoveryCandidateKey:
    family: str
    value: float
    method_id: str
    seed: int = 2026


@dataclass(frozen=True)
class RecoveryCandidateEvidence:
    key: RecoveryCandidateKey
    state: Literal["reusable-complete", "reusable-checkpoint", "retrain-required"]
    source_trial_root: Path
    checkpoint_path: Path | None
    training_receipt_path: Path | None
    complete_receipt_path: Path | None
    reason: str
    file_sha256: Mapping[str, str]
    runtime_seconds_reused: float


@dataclass(frozen=True)
class Gate1RecoveryInspection:
    source_root: Path
    failure_receipt_sha256: str
    candidates: tuple[RecoveryCandidateEvidence, ...]
```

`expected_recovery_candidates` must derive all eight values from
`contract.payload["training"]["search"]`; it must not duplicate the numeric
grid in production code.

- [ ] **Step 4: Validate the source lineage before candidate inspection**

```python
def inspect_recovery_source(
    source_root: Path,
    contract: CompleteFormalContract,
    data: Gate1DataBundle,
    gate0_transition_path: Path,
) -> Gate1RecoveryInspection:
    gate1 = source_root.resolve() / "gate1"
    lineage = load_json_object(gate1 / "DATA_LINEAGE.json")
    expected = {
        "contract_sha256": contract.contract_sha256,
        "gate0_transition_sha256": sha256_file(gate0_transition_path),
        "train_windows_sha256": sha256_file(data.source_run_root / "gate1" / "TRAIN_WINDOWS.npz"),
        "selection_windows_sha256": sha256_file(data.source_run_root / "gate1" / "SELECTION_WINDOWS.npz"),
        "normalization_sha256": sha256_file(data.source_run_root / "gate1" / "NORMALIZATION.json"),
    }
    for field, value in expected.items():
        if lineage.get(field) != value:
            raise PermissionError(f"recovery source {field} mismatch")
    if lineage.get("evaluation_year_accessed") is not False:
        raise PermissionError("recovery source accessed the evaluation year")
    return inspect_all_candidates(gate1, contract)
```

- [ ] **Step 5: Add the optional recovery source to `Gate1RunConfig`**

```python
@dataclass(frozen=True)
class Gate1RunConfig:
    contract_path: Path
    gate0_transition_path: Path
    source_run_root: Path
    output_root: Path
    run_id: str
    smoke: bool = False
    resume_from: Path | None = None
```

The new field is last and defaults to `None`, preserving every existing caller.

- [ ] **Step 6: Run focused tests and commit the source validator**

Run: `pytest tests/test_complete_formal_gate1_recovery.py -q`

Expected: PASS for enumeration and source-level rejection.

```powershell
git add src/joint_dispatch/complete_formal_gate1_recovery.py src/joint_dispatch/complete_formal_gate1.py tests/test_complete_formal_gate1_recovery.py
git commit -m "feat: validate Gate1 recovery sources"
```

### Task 2: Classify and materialize candidate evidence

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1_recovery.py`
- Modify: `tests/test_complete_formal_gate1_recovery.py`

**Interfaces:**
- Consumes: source-level validation from Task 1 and the frozen search grid.
- Produces: `inspect_candidate(...)` and `materialize_candidate(...)`.

- [ ] **Step 1: Write failing tests for the three recovery states**

```python
def test_complete_rsc_candidate_is_reusable(recovery_fixture, contract, gate1_data):
    evidence = inspect_candidate(
        recovery_fixture / "gate1",
        RecoveryCandidateKey("RSC-PF", 1.0, "RSC-PF"),
        contract,
        gate1_data,
    )
    assert evidence.state == "reusable-complete"


def test_misplaced_difflp_checkpoint_is_reusable(recovery_fixture, contract, gate1_data):
    evidence = inspect_candidate(
        recovery_fixture / "gate1",
        RecoveryCandidateKey("Differentiable-LP", 1e-5, "Differentiable-LP"),
        contract,
        gate1_data,
    )
    assert evidence.state == "reusable-checkpoint"
    assert evidence.checkpoint_path.name == "CHECKPOINT.pt"


def test_absent_candidate_requires_training(recovery_fixture, contract, gate1_data):
    evidence = inspect_candidate(
        recovery_fixture / "gate1",
        RecoveryCandidateKey("Differentiable-LP", 3e-5, "Differentiable-LP"),
        contract,
        gate1_data,
    )
    assert evidence.state == "retrain-required"
```

- [ ] **Step 2: Run the new tests and verify the classifier is absent**

Run: `pytest tests/test_complete_formal_gate1_recovery.py -q`

Expected: FAIL because `inspect_candidate` is not defined.

- [ ] **Step 3: Implement canonical and legacy path resolution**

```python
def candidate_paths(gate1_root: Path, key: RecoveryCandidateKey) -> CandidatePaths:
    label = f"rsc_multiplier_{key.value:g}" if key.family == "RSC-PF" else f"difflp_lr_{key.value:g}"
    trial = gate1_root / "search" / label
    canonical = trial / "rows" / key.method_id / str(key.seed)
    legacy = trial / "rows" if key.family == "Differentiable-LP" else canonical
    return CandidatePaths(trial, canonical, legacy)
```

For Differentiable-LP, accept the legacy checkpoint only for the known
one-level-high layout; do not recursively choose an arbitrary checkpoint.

- [ ] **Step 4: Implement receipt and checkpoint validation**

```python
def validate_training_evidence(paths, key, contract, data):
    receipt_path = paths.canonical / "TRAINING_RECEIPT.json"
    checkpoint_path = paths.canonical / "CHECKPOINT.pt"
    if not receipt_path.is_file() and key.family == "Differentiable-LP":
        receipt_path = paths.legacy / "TRAINING_RECEIPT.json"
        checkpoint_path = paths.legacy / "CHECKPOINT.pt"
    if not receipt_path.is_file() or not checkpoint_path.is_file():
        return None, "training artifact absent"
    receipt = load_json_object(receipt_path)
    required = {
        "method_id": key.method_id,
        "seed": key.seed,
        "epochs": 30,
        "test_set_accessed": False,
    }
    if any(receipt.get(name) != value for name, value in required.items()):
        return None, "training receipt field mismatch"
    if receipt.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        return None, "checkpoint hash mismatch"
    validate_checkpoint_lineage(checkpoint_path, contract, data)
    return ValidatedTraining(receipt_path, checkpoint_path, receipt), "validated"
```

`validate_checkpoint_lineage` must inspect the PyTorch payload on CPU and
require the current contract, source manifest, train manifest, and
normalization hashes. It must also require checkpoint epoch `29`.

- [ ] **Step 5: Validate completed candidate receipts and their files**

Require `status=complete`, `complete=true`, `synthetic=false`,
`paper_result=false`, `origin_count=8709`, `chronological=true`,
`finite=true`, `physical_feasible=true`, and matching checkpoint, metrics, and
rollout hashes. Invalid evaluation evidence downgrades a valid checkpoint to
`reusable-checkpoint`; it does not force retraining.

- [ ] **Step 6: Copy validated files into canonical destination paths**

```python
def materialize_candidate(evidence: RecoveryCandidateEvidence, destination_trial: Path) -> Path:
    destination_row = destination_trial / "rows" / evidence.key.method_id / str(evidence.key.seed)
    destination_row.mkdir(parents=True, exist_ok=False)
    for source in files_for_state(evidence):
        target = destination_row / source.name
        shutil.copy2(source, target)
        if sha256_file(target) != evidence.file_sha256[source.name]:
            raise IOError(f"copied recovery artifact hash mismatch: {source.name}")
    return destination_row
```

For a complete RSC candidate, copy the canonical row files plus any required
shared Stage-S evidence. For checkpoint-only Differentiable-LP, copy only the
validated checkpoint and training receipt; old partial rollout and metrics are
not treated as authoritative.

- [ ] **Step 7: Prove the recovery source remains byte-identical**

The test records a canonical hash over every source-relative file path and
file SHA-256 before and after materialization and asserts equality.

Run: `pytest tests/test_complete_formal_gate1_recovery.py -q`

Expected: PASS for all three states, canonicalization, copied hashes, and source immutability.

```powershell
git add src/joint_dispatch/complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_recovery.py
git commit -m "feat: classify and copy Gate1 recovery candidates"
```

### Task 3: Restore a trained Differentiable-LP model without retraining

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1_recovery.py`
- Modify: `tests/test_complete_formal_gate1_recovery.py`

**Interfaces:**
- Consumes: canonical checkpoint and training receipt created by Task 2.
- Produces: `restore_differentiable_lp_artifact(...) -> TrainedMethodArtifact`.

- [ ] **Step 1: Write a failing restoration test with a saved Scheme2R checkpoint**

```python
def test_restore_difflp_reproduces_checkpoint_parameters(
    canonical_difflp_row, gate1_data, contract
):
    artifact = restore_differentiable_lp_artifact(
        canonical_difflp_row, gate1_data, contract
    )
    payload = torch.load(canonical_difflp_row / "CHECKPOINT.pt", map_location="cpu", weights_only=False)
    assert artifact.method_id == "Differentiable-LP"
    assert artifact.seed == 2026
    for name, tensor in artifact.model.state_dict().items():
        assert torch.equal(tensor.cpu(), payload["model"][name].cpu())
```

- [ ] **Step 2: Run the test and verify restoration is not implemented**

Run: `pytest tests/test_complete_formal_gate1_recovery.py::test_restore_difflp_reproduces_checkpoint_parameters -q`

Expected: FAIL because the restoration function is absent.

- [ ] **Step 3: Reconstruct and load the model with exact lineage checks**

```python
def restore_differentiable_lp_artifact(row_dir, data, contract):
    receipt = load_json_object(row_dir / "TRAINING_RECEIPT.json")
    model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=StageBudgetV42().weight_decay,
    )
    expected_lineage = checkpoint_lineage(contract, data)
    checkpoint = load_training_checkpoint(
        row_dir / "CHECKPOINT.pt",
        model=model,
        optimizer=optimizer,
        expected_lineage=expected_lineage,
    )
    return TrainedMethodArtifact(
        method_id="Differentiable-LP",
        seed=2026,
        model=model,
        checkpoint_path=checkpoint.path,
        checkpoint_sha256=checkpoint.model_sha256,
        training_receipt=receipt,
        decision_forecaster_gradient_norm=float(receipt["decision_forecaster_gradient_norm"]),
    )
```

The original Differentiable-LP search used the `StageBudgetV42` default
`weight_decay=1e-4`; the restoration constructor must use that exact value.
The test must assert the optimizer state loaded from the checkpoint replaces
the fresh optimizer state without executing an update.

- [ ] **Step 4: Verify that restoration performs zero optimizer steps**

Monkeypatch `torch.optim.AdamW.step` to raise if called during restoration;
the restoration test must still pass.

- [ ] **Step 5: Run focused tests and commit restoration**

Run: `pytest tests/test_complete_formal_gate1_recovery.py -q`

Expected: PASS, including byte-identical restored parameters and zero training steps.

```powershell
git add src/joint_dispatch/complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_recovery.py
git commit -m "feat: restore Differentiable-LP search checkpoints"
```

### Task 4: Make hyperparameter search recovery-aware

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1.py:278-355`
- Modify: `tests/test_complete_formal_gate1_recovery.py`

**Interfaces:**
- Consumes: `Gate1RecoveryInspection` and materialized/restored candidates.
- Produces: recovery-aware `select_gate1_hyperparameters(..., recovery=None)` with the existing `GATE1_SEARCH.json` schema plus candidate action fields.

- [ ] **Step 1: Write a failing orchestration test that forbids completed retraining**

```python
def test_search_reuses_completed_trials_and_only_trains_missing(
    recovery_inspection, gate1_data, contract, tmp_path, monkeypatch
):
    trained = []
    monkeypatch.setattr(gate1, "train_rsc_family", lambda *args, **kwargs: pytest.fail("RSC trial retrained"))
    monkeypatch.setattr(gate1, "train_differentiable_lp", record_missing_trials(trained))
    result = select_gate1_hyperparameters(
        gate1_data, contract, tmp_path, recovery=recovery_inspection
    )
    assert trained == [3e-5, 1e-4, 3e-4]
    assert result["recovery_used"] is True
```

- [ ] **Step 2: Run the test and confirm the search has no recovery parameter**

Run: `pytest tests/test_complete_formal_gate1_recovery.py::test_search_reuses_completed_trials_and_only_trains_missing -q`

Expected: FAIL because `select_gate1_hyperparameters` does not accept `recovery`.

- [ ] **Step 3: Add candidate-by-candidate dispatch**

```python
def _recover_or_run_candidate(key, recovery, trial_root, train, evaluate):
    evidence = recovery.by_key.get(key) if recovery is not None else None
    if evidence is not None and evidence.state == "reusable-complete":
        row_dir = materialize_candidate(evidence, trial_root)
        return load_json_object(row_dir / "COMPLETE_GATE1_ROW_RECEIPT.json"), "reused-complete"
    if evidence is not None and evidence.state == "reusable-checkpoint":
        row_dir = materialize_candidate(evidence, trial_root)
        artifact = restore_differentiable_lp_artifact(row_dir, data, contract)
        return evaluate(artifact), "reused-training-reran-evaluation"
    artifact = train()
    return evaluate(artifact), "trained"
```

Use the existing score and eligibility calculation after this dispatch so
recovered and newly trained candidates follow exactly the same selection rule.

- [ ] **Step 4: Ensure the repaired Differentiable-LP output path is canonical**

Both recovered and newly trained Differentiable-LP candidates must use:

```python
trial_root / "rows" / "Differentiable-LP" / "2026"
```

The evaluator continues receiving `trial_root`, so it resolves the same row
directory. Add an assertion test for the exact path.

- [ ] **Step 5: Record action and provenance in every search trial entry**

Each entry in `GATE1_SEARCH.json` gains `action`, `source_run_id`,
`training_reused`, `evaluation_reused`, and `artifact_hashes`. Existing fields
`family`, `value`, `score`, and `eligible` remain unchanged.

- [ ] **Step 6: Run search recovery tests and commit integration**

Run: `pytest tests/test_complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_real.py -q`

Expected: PASS; completed RSC trials are never sent to a trainer and the `1e-5` Differentiable-LP trial is never optimized again.

```powershell
git add src/joint_dispatch/complete_formal_gate1.py tests/test_complete_formal_gate1_recovery.py
git commit -m "feat: resume complete-formal Gate1 search"
```

### Task 5: Add CLI recovery and an auditable recovery manifest

**Files:**
- Modify: `scripts/run_rsc_pf_complete_formal_gate1_real.py`
- Modify: `src/joint_dispatch/complete_formal_gate1.py:498-560`
- Modify: `src/joint_dispatch/complete_formal_gate1_recovery.py`
- Test: `tests/test_complete_formal_gate1_recovery.py`

**Interfaces:**
- Consumes: `--resume-from <prior-run-directory>`.
- Produces: `gate1/RECOVERY_MANIFEST.json` and unchanged final Gate 1 transition semantics.

- [ ] **Step 1: Write failing CLI parsing and manifest tests**

```python
def test_cli_accepts_explicit_recovery_source():
    args = build_parser().parse_args([
        "--gate0-transition", str(GATE0_TRANSITION),
        "--source-run", str(SOURCE_RUN),
        "--run-id", "recovered-run",
        "--resume-from", str(FAILED_RUN),
    ])
    assert args.resume_from.resolve() == FAILED_RUN.resolve()


def test_recovery_manifest_reports_saved_training_time(recovered_smoke_run):
    manifest = json.loads((recovered_smoke_run / "gate1" / "RECOVERY_MANIFEST.json").read_text())
    assert manifest["source_modified"] is False
    assert manifest["reused_candidate_count"] == 5
    assert manifest["reused_training_runtime_seconds"] > 0
    assert manifest["evaluation_year_accessed"] is False
```

- [ ] **Step 2: Run the tests and verify CLI recovery is unavailable**

Run: `pytest tests/test_complete_formal_gate1_recovery.py -q`

Expected: FAIL because `--resume-from` and the manifest do not exist.

- [ ] **Step 3: Add the CLI argument and resolve it before configuration**

```python
parser.add_argument(
    "--resume-from",
    type=Path,
    help="read-only prior Gate1 run whose verified search artifacts may be reused",
)

config = Gate1RunConfig(
    contract_path=args.contract.resolve(),
    gate0_transition_path=args.gate0_transition.resolve(),
    source_run_root=args.source_run.resolve(),
    output_root=args.output_root.resolve(),
    run_id=str(args.run_id),
    smoke=bool(args.smoke),
    resume_from=None if args.resume_from is None else args.resume_from.resolve(),
)
```

- [ ] **Step 4: Inspect recovery after current data validation and before search**

Inside `run_complete_gate1`, first validate Gate 0 and load current data. If
`resume_from` is set, call `inspect_recovery_source`, write the initial manifest,
and pass the inspection to `select_gate1_hyperparameters`. Finalize the
manifest after search with actions, copied hashes, and avoided runtime.

- [ ] **Step 5: Keep failure and authorization fail-closed**

On any exception, `GATE1_FAILURE.json` must include `resume_from`, the current
recovery-manifest hash when available, and `authorized_gate2=false`. The normal
37-row audit remains the sole place that can set `authorized_gate2=true`.

- [ ] **Step 6: Run CLI and manifest tests and commit**

Run: `pytest tests/test_complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_real.py -q`

Expected: PASS for parsing, immutable source, manifest fields, and fail-closed authorization.

```powershell
git add scripts/run_rsc_pf_complete_formal_gate1_real.py src/joint_dispatch/complete_formal_gate1.py src/joint_dispatch/complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_recovery.py
git commit -m "feat: expose auditable Gate1 recovery"
```

### Task 6: Audit the real failed run and prepare the continuation command

**Files:**
- Modify: `docs/superpowers/reports/2026-09-06-complete-formal-gate1-implementation.md`
- Test: `tests/test_complete_formal_gate1_recovery.py`

**Interfaces:**
- Consumes: real failed run `complete_formal_gate1_20260906_b` read-only.
- Produces: verified recovery classification, regression evidence, and the operator command for a new run.

- [ ] **Step 1: Record a whole-tree hash inventory of the failed run**

Use `inspect_recovery_source` to calculate a deterministic digest from sorted
relative paths and SHA-256 values. Save the digest only in the new recovery
manifest and implementation report; do not write into the failed run.

- [ ] **Step 2: Run the real read-only recovery preflight**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_complete_formal_gate1_recovery.py -q
```

Expected real classification:

```text
RSC-PF/1       reusable-complete
RSC-PF/1.5     reusable-complete
RSC-PF/2       reusable-complete
RSC-PF/3       reusable-complete
Differentiable-LP/1e-5 reusable-checkpoint
Differentiable-LP/3e-5 retrain-required
Differentiable-LP/1e-4 retrain-required
Differentiable-LP/3e-4 retrain-required
```

- [ ] **Step 3: Run all relevant regression checks**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m py_compile src\joint_dispatch\complete_formal_gate1.py src\joint_dispatch\complete_formal_gate1_recovery.py scripts\run_rsc_pf_complete_formal_gate1_real.py
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_complete_formal_gate1_recovery.py tests\test_complete_formal_gate1_real.py tests\test_rsc_pf_complete_formal_gate0.py tests\test_rsc_pf_complete_formal_contract.py tests\test_rsc_pf_complete_formal_execution.py tests\test_rsc_pf_complete_formal_providers.py tests\test_rsc_pf_complete_formal_training.py tests\test_joint_dispatch_formal_v4_2_gate2_training.py tests\test_joint_dispatch_formal_v4_6_contract.py tests\test_joint_dispatch_formal_v4_6_audit.py tests\test_joint_dispatch_formal_v4_6_training.py -q
git diff --check
```

Expected: all focused tests pass, compilation succeeds, and no whitespace errors are reported. Line-ending notices are informational.

- [ ] **Step 4: Verify the failed source tree is unchanged**

Recompute its whole-tree digest and require exact equality with Step 1.

- [ ] **Step 5: Record the verified continuation command**

```powershell
cd 'D:\Paper\github_work\paper-code-formal-v42-gate0\frame'
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\run_rsc_pf_complete_formal_gate1_real.py --contract configs\rsc_pf_complete_formal_v1.json --gate0-transition reports\rsc_pf_complete_formal\complete_formal_gate0_20260906_i\gate0\GATE0_TRANSITION.json --source-run reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j --output-root reports\rsc_pf_complete_formal --run-id complete_formal_gate1_recovered_20260907_a --resume-from reports\rsc_pf_complete_formal\complete_formal_gate1_20260906_b
```

Do not start this long continuation until the real preflight classification,
focused regression suite, and source immutability check all pass.

- [ ] **Step 6: Commit the verified implementation report**

```powershell
git add docs/superpowers/reports/2026-09-06-complete-formal-gate1-implementation.md
git commit -m "docs: record Gate1 recovery verification"
```
