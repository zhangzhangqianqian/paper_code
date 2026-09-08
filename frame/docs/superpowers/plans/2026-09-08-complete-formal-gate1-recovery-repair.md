# Complete Formal Gate1 Recovery Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a failed complete-v1 Gate1 run safely resumable by resolving and validating the official iTransformer source before any long training, reusing validated completed search/final training artifacts, and continuing only the missing rows without changing the frozen protocol.

**Architecture:** Keep the failed run immutable. Add a deterministic repository-relative iTransformer resolver anchored to the current formal frame and run it as a preflight before search/matrix training. Extend the existing recovery inspection to classify the five completed final seed-2026 training rows as reusable checkpoints, restore them into the new run with lineage and hash checks, and let the existing trainer handle missing methods/seeds. The orchestrator will record every reuse/training action and fail closed before writing any formal result if provenance or source validation fails.

**Tech Stack:** Python 3.11+, PyTorch checkpoint serialization, NumPy, pytest, existing complete-v1 Gate1 contract and v4.2 training/evaluation kernels.

## Global Constraints

- Do not modify or delete the failed source run; all recovery writes go to a new run directory.
- Preserve the frozen complete-v1 roster, 2015–2018 training split, 2019 selection/evaluation split, 37 Gate1 rows, and existing checkpoint/source hashes.
- Official iTransformer must remain the pinned upstream commit and verified imported-file hash set; a missing or ambiguous source path must stop before training.
- No 2020 or 2021 data may be opened by Gate1, and no smoke/synthetic artifact may authorize Gate2.
- Do not start the long formal Gate1 experiment during implementation; use unit, integration, and short smoke tests only.

---

### Task 1: Deterministic iTransformer source resolution and preflight

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1.py` (source receipt loading and matrix setup)
- Modify: `src/joint_dispatch/formal_v4_itransformer.py` only if a small path-validation helper is needed
- Test: `tests/test_complete_formal_gate1_recovery.py`
- Test: `tests/test_joint_dispatch_formal_v4_itransformer.py`

**Interfaces:**
- Add `resolve_verified_itransformer_source(data: Gate1DataBundle, output_dir: Path) -> tuple[dict[str, Any], Path, Path]` beside `_itransformer_receipt`.
- The function reads the source receipt, resolves repository-relative `source_root` against `Path(__file__).resolve().parents[2]` (the current formal frame), rejects absolute/`..` escape paths, calls `verify_itransformer_source_files`, and writes the immutable adapter receipt only after validation.
- `train_gate1_matrix` and `select_gate1_hyperparameters` consume this verified tuple; neither function performs ad-hoc path concatenation.

- [ ] **Step 1: Write failing tests for an invalid source root and a valid linked-worktree source.**

```python
def test_itransformer_preflight_rejects_missing_root(monkeypatch, data, tmp_path):
    write_source_receipt(data.source_run_root, source_root="frame/third_party/missing")
    with pytest.raises(FileNotFoundError, match="iTransformer source root"):
        resolve_verified_itransformer_source(data, tmp_path)

def test_itransformer_preflight_resolves_current_frame_root(data, tmp_path):
    source, receipt, diff = resolve_verified_itransformer_source(data, tmp_path)
    assert Path(source["source_root"]).is_absolute() is False
    assert receipt.is_file()
    assert diff.is_file()
```

- [ ] **Step 2: Run the two new tests and verify the missing-root test fails because the resolver is not defined.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py::test_itransformer_preflight_rejects_missing_root tests/test_complete_formal_gate1_recovery.py::test_itransformer_preflight_resolves_current_frame_root -q`

Expected: FAIL during collection with an import/attribute error for `resolve_verified_itransformer_source`.

- [ ] **Step 3: Implement the resolver and call it before any Gate1 search or matrix training.**

```python
def resolve_verified_itransformer_source(data, output_dir):
    source_receipt_path = data.source_run_root / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json"
    diff_receipt_path = data.source_run_root / "protocol" / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json"
    source = _json(source_receipt_path)
    relative = Path(str(source.get("source_root", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise PermissionError("iTransformer source_root must be repository-relative")
    frame_root = Path(__file__).resolve().parents[2]
    source_root = frame_root / (Path(*relative.parts[1:]) if relative.parts and relative.parts[0].lower() == "frame" else relative)
    if not source_root.is_dir():
        raise FileNotFoundError(f"iTransformer source root is not a readable Git checkout: {source_root}")
    verify_itransformer_source_files(source_root, source, require_license=True)
    receipt = output_dir / "ITRANSFORMER_ADAPTER_RECEIPT.json"
    fields = ("source_root", "repository", "commit", "backbone_class", "imported_file_hashes", "license_file", "license_sha256", "reproduction_level", "verified")
    write_once_json(receipt, {"schema_version": "formal-v4.1-itransformer-source-v1", **{name: source[name] for name in fields if name in source}, "resolved_source_root": str(source_root)})
    return source, receipt, diff_receipt_path
```

Use the existing frozen receipt fields rather than changing the upstream commit or hash values. Replace both `_itransformer_receipt` call sites and remove the duplicated ad-hoc resolution in `train_gate1_matrix`.

- [ ] **Step 4: Run the focused source and Gate1 recovery tests.**

Run: `pytest tests/test_joint_dispatch_formal_v4_itransformer.py tests/test_complete_formal_gate1_recovery.py -q`

Expected: all focused tests pass, including verification that the source checkout is not modified.

- [ ] **Step 5: Commit the preflight change.**

```bash
git add src/joint_dispatch/complete_formal_gate1.py tests/test_complete_formal_gate1_recovery.py tests/test_joint_dispatch_formal_v4_itransformer.py
git commit -m "fix: preflight official itransformer source before gate1"
```

### Task 2: Inspect and restore completed final matrix checkpoints

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1_recovery.py`
- Modify: `src/joint_dispatch/complete_formal_gate1.py`
- Test: `tests/test_complete_formal_gate1_recovery.py`

**Interfaces:**
- Add `FinalRowKey(method_id: str, seed: int)` and `FinalRowEvidence` with state, source row, checkpoint, receipt, hashes, and runtime.
- Add `inspect_final_rows(source_root: Path, contract: CompleteFormalContract, data: Gate1DataBundle) -> tuple[FinalRowEvidence, ...]` for the five stochastic methods already trained in a partial run.
- Add `restore_final_artifact(evidence: FinalRowEvidence, data: Gate1DataBundle, contract: CompleteFormalContract, output_row: Path, source_info: Mapping[str, Any]) -> TrainedMethodArtifact` for RSC-PF, Decoupled-RSC-PF, State-Conditioned-PTO, Direct-Policy, and Scheme2R-PTO. It must instantiate the same model class, create an optimizer with the contract budget, call `load_training_checkpoint` with exact lineage, and copy the immutable checkpoint/receipt to the destination before returning the in-memory artifact.
- Extend `Gate1RecoveryInspection` with `final_rows` and expose `final_by_key`.

- [ ] **Step 1: Write failing tests for seed-2026 final-row reuse and tampered checkpoint rejection.**

```python
def test_inspect_final_rows_marks_existing_seed_2026_rows_reusable(partial_run, contract, data):
    rows = inspect_final_rows(partial_run, contract, data)
    expected = {FinalRowKey(name, 2026) for name in ("RSC-PF", "Decoupled-RSC-PF", "State-Conditioned-PTO", "Direct-Policy", "Scheme2R-PTO")}
    assert {row.key for row in rows} == expected
    assert all(row.state == "reusable-checkpoint" for row in rows)

def test_inspect_final_rows_rejects_checkpoint_hash_change(partial_run, contract, data):
    checkpoint = partial_run / "gate1" / "rows" / "RSC-PF" / "2026" / "CHECKPOINT.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    evidence = inspect_final_rows(partial_run, contract, data)
    assert next(row for row in evidence if row.key.method_id == "RSC-PF").state == "retrain-required"
```

- [ ] **Step 2: Run the new tests and verify they fail before the new evidence types/functions exist.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py -k "final_rows" -q`

Expected: FAIL with missing `inspect_final_rows`/`FinalRowKey`.

- [ ] **Step 3: Implement exact receipt, checkpoint, lineage, and model-schema validation.**

Use `TRAINING_RECEIPT.json` fields (`method_id`, `seed`, `epochs`, `checkpoint_sha256`, `test_set_accessed`) and `load_training_checkpoint`; expected lineage is the existing `_expected_checkpoint_lineage`. Validate that the method-specific receipt marks `forecast_loss_applicable` correctly, that epochs equal the frozen complete budget, and that no destination path is overwritten. Use the existing constructors (`build_rsc_model`, `build_direct_policy_model`, `Scheme2RModel`) and the existing registered artifact adapter to preserve inference semantics.

- [ ] **Step 4: Run the recovery tests and verify source immutability.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py -q`

Expected: all recovery tests pass and source-run hashes remain unchanged.

- [ ] **Step 5: Commit final-row recovery support.**

```bash
git add src/joint_dispatch/complete_formal_gate1_recovery.py src/joint_dispatch/complete_formal_gate1.py tests/test_complete_formal_gate1_recovery.py
git commit -m "feat: reuse validated gate1 matrix checkpoints"
```

### Task 3: Make the Gate1 matrix continuation-aware

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1.py`
- Modify: `src/joint_dispatch/complete_formal_gate1_recovery.py`
- Test: `tests/test_complete_formal_gate1_recovery.py`
- Test: `tests/test_complete_formal_gate1_real.py`

**Interfaces:**
- Change `train_gate1_matrix(..., recovery: Gate1RecoveryInspection | None = None) -> Mapping[MethodSeedKey, TrainedMethodArtifact | None]`.
- For every method/seed, reuse a validated final checkpoint when `final_by_key` says `reusable-checkpoint`; otherwise train exactly once with the existing function and budget.
- Keep Official iTransformer and Differentiable-LP on the verified source path; if no valid checkpoint exists they are trained normally. Deterministic rows remain evaluation-only.
- Add recovery actions to the return metadata or a separate `GATE1_MATRIX_RECOVERY.json` receipt so the final evidence can report reused versus newly trained rows.

- [ ] **Step 1: Write a failing integration test that monkeypatches trainers and asserts completed seed-2026 rows are not retrained.**

```python
def test_matrix_reuses_valid_final_rows(monkeypatch, recovery, data, contract, tmp_path):
    calls = []
    monkeypatch.setattr(gate1, "train_rsc_family", lambda *a, **k: calls.append("rsc") or pytest.fail("RSC should be restored"))
    artifacts = train_gate1_matrix(data, contract, tmp_path, selected_hyperparameters=SEARCH, recovery=recovery)
    reusable_methods = ("RSC-PF", "Decoupled-RSC-PF", "State-Conditioned-PTO", "Direct-Policy", "Scheme2R-PTO")
    assert all(artifacts[MethodSeedKey(name, 2026)] is not None for name in reusable_methods)
    assert "rsc" not in calls
```

- [ ] **Step 2: Run the integration test and verify it fails because `train_gate1_matrix` has no recovery parameter.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py::test_matrix_reuses_valid_final_rows -q`

Expected: FAIL with an unexpected keyword argument or a retraining assertion.

- [ ] **Step 3: Implement per-row reuse and retraining fallback.**

Before entering the seed loop, obtain the verified iTransformer tuple from Task 1. For each seed, restore reusable family artifacts into their canonical destination and load the shared Stage-S/teacher evidence when required. For missing rows, retain the current training order so teacher data are generated once per seed and shared by Direct-Policy. Record `"action": "reused-checkpoint"` or `"action": "trained"` for every row and refuse a partial family reuse that would break the RSC shared parent lineage.

- [ ] **Step 4: Run focused Gate1 tests and a tiny smoke invocation.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_real.py -q`

Expected: focused tests pass; smoke output remains `authorized_gate2=false` and contains no formal paper result.

- [ ] **Step 5: Commit continuation-aware matrix training.**

```bash
git add src/joint_dispatch/complete_formal_gate1.py src/joint_dispatch/complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_real.py
git commit -m "fix: continue gate1 matrix without retraining valid rows"
```

### Task 4: Orchestrator receipts and fail-closed recovery behavior

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1.py`
- Modify: `src/joint_dispatch/complete_formal_gate1_recovery.py`
- Test: `tests/test_complete_formal_gate1_recovery.py`
- Test: `tests/test_complete_formal_gate1_contract.py`

**Interfaces:**
- `run_complete_gate1` calls the source preflight immediately after data lineage validation and before search/matrix training.
- When `resume_from` is set, it inspects both search candidates and final rows, writes `RECOVERY_PLAN.json`, and writes `RECOVERY_MANIFEST.json` containing candidate/final-row actions, hashes, runtimes, and `source_modified=false`.
- Any preflight, lineage, checkpoint, source hash, or output collision failure writes `GATE1_FAILURE.json` and never writes `GATE1_TRANSITION.json` or `GATE1_EVIDENCE.json`.

- [ ] **Step 1: Add a test proving an invalid iTransformer source stops before a trainer is called.**

```python
def test_gate1_fails_before_training_on_invalid_official_source(monkeypatch, valid_config, tmp_path):
    monkeypatch.setattr(gate1, "train_rsc_family", lambda *a, **k: pytest.fail("training must not start"))
    with pytest.raises((FileNotFoundError, PermissionError, ValueError)):
        run_complete_gate1(valid_config_with_bad_source, smoke=False)
    failure = json.loads((tmp_path / "gate1" / "GATE1_FAILURE.json").read_text())
    assert "iTransformer" in failure["reason"]
```

- [ ] **Step 2: Run the failure test and verify it fails until the preflight is placed before training.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py::test_gate1_fails_before_training_on_invalid_official_source -q`

Expected: FAIL because the current runner reaches a trainer before resolving the official source.

- [ ] **Step 3: Wire the preflight, final-row inspection, manifest, and matrix recovery into `run_complete_gate1`.**

Keep all existing complete-v1 hashes and row-count checks. Add `matrix_recovery` metadata to the evidence payload and include it in the recovery manifest hash. Do not convert a recovered or smoke run into an authorized result unless `audit_complete_formal` passes over all 37 newly materialized/evaluated rows.

- [ ] **Step 4: Run the complete focused regression suite.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_real.py tests/test_complete_formal_gate1_contract.py -q`

Expected: all focused tests pass, with no formal Gate1 launch.

- [ ] **Step 5: Commit the fail-closed orchestrator repair.**

```bash
git add src/joint_dispatch/complete_formal_gate1.py src/joint_dispatch/complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_real.py tests/test_complete_formal_gate1_contract.py
git commit -m "fix: make complete gate1 recovery fail closed"
```

### Task 5: Documentation and final verification

**Files:**
- Modify: `docs/superpowers/reports/2026-09-06-complete-formal-gate1-implementation.md`
- Create: `docs/superpowers/reports/2026-09-08-complete-formal-gate1-recovery-repair.md`

- [ ] **Step 1: Run all Gate1/recovery and iTransformer tests plus static checks.**

Run: `pytest tests/test_complete_formal_gate1_recovery.py tests/test_complete_formal_gate1_real.py tests/test_complete_formal_gate1_contract.py tests/test_joint_dispatch_formal_v4_itransformer.py -q`

Expected: all tests pass; `git diff --check` exits successfully.

- [ ] **Step 2: Run a short non-authorizing smoke continuation with the repaired CLI.**

Run: `& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\run_rsc_pf_complete_formal_gate1_real.py --contract configs\rsc_pf_complete_formal_v1.json --gate0-transition reports\rsc_pf_complete_formal\complete_formal_gate0_20260906_i\gate0\GATE0_TRANSITION.json --source-run reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j --output-root reports\rsc_pf_complete_formal --run-id complete_formal_gate1_repair_smoke_20260908 --resume-from reports\rsc_pf_complete_formal\complete_formal_gate1_recovered_20260907_a --smoke`

Expected: 37 smoke rows, `authorized_gate2=false`, `synthetic=true`, no Gate2 transition.

- [ ] **Step 3: Write the repair report.**

Record the original failure, deterministic source resolution, rows reused, tests run, and the explicit statement that no long formal experiment was launched by this repair.

- [ ] **Step 4: Commit the documentation and verification report.**

```bash
git add docs/superpowers/reports/2026-09-06-complete-formal-gate1-implementation.md docs/superpowers/reports/2026-09-08-complete-formal-gate1-recovery-repair.md
git commit -m "docs: record complete gate1 recovery repair"
```
