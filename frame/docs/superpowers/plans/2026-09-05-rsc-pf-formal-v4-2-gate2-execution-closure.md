# RSC-PF Formal-v4.2 Gate 2 Execution Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert formal-v4.2 Gate 2 from a JSON-only decision checker into a complete, checkpoint-backed 23-row training, rolling-evaluation, and independent-audit pipeline, then run the immutable Gate 0 → Pilot → Gate 1 → Gate 2 chain and stop.

**Architecture:** Keep the existing formal-v4.2 scientific protocol and model identities. Add a frozen execution budget, focused Gate 2 manifest/artifact services, family-specific trainers, a common chronological evaluator, and a fail-closed orchestrator. Gate 2 consumes the Gate 1 freeze, trains on the complete eligible 2015--2018 chronology, evaluates the complete eligible 2019 chronology, and never accesses 2020.

**Tech Stack:** Python 3.9, PyTorch 2.8, NumPy 2.0, SciPy 1.13/HiGHS, CVXPY/CVXPYlayers, official THUML iTransformer source, pytest, JSON/NPZ/PT/SHA-256 receipts, PowerShell.

## Global Constraints

- The protocol name remains `formal-v4.2`; no model architecture, data-year boundary, objective, physical decoder, method identity, or Gate 3 rule changes.
- Train and normalization years are exactly `2015, 2016, 2017, 2018`; calibration and Gate 2 selection year is exactly `2019`; `2020` is inaccessible before Gate 2 authorization; `2021` is excluded.
- Formal training uses every eligible causal 2015--2018 window in one immutable shared manifest.
- Gate 1 uses exactly 1,000 representative 2019 calibration origins; Gate 2 evaluates every eligible chronological 2019 rolling origin.
- Effective batch size is 64. Differentiable-LP may use deterministic gradient-accumulated micro-batches but must preserve the effective batch and complete sample exposures.
- Stage P/S/J maxima are 30 epochs, minimum Stage J is 18 epochs, patience is 5, and validation occurs every epoch.
- Gate 2 stochastic seeds are `2026, 2027, 2028`; deterministic references execute once.
- Gate 2 contains exactly 23 rows: seven stochastic methods × three seeds plus two deterministic references.
- Every neural row requires a trained checkpoint; random initialization may never produce a row receipt.
- Official iTransformer is a disclosed adaptation of the frozen upstream backbone commit, not an exact reproduction of the source paper's full experiment.
- Existing run roots are immutable. Resume is permitted only for hash-identical completed method/seed rows.
- Any Gate or audit failure stops execution. A successful Gate 2 also stops execution; Gate 3 and 2020 remain untouched.
- Work only in `D:\Paper\github_work\paper-code-formal-v42-gate0`; do not modify or clean the user's dirty primary checkout.
- Use `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe` for tests and execution.

---

## File Structure Map

- Modify `frame/configs/joint_forecast_dispatch_formal_v4_2.json`: final Gate 2 and DiffLP budgets.
- Modify `frame/configs/formal_v4_source_closure_v4_2.txt`: final controlled-source set.
- Modify `frame/src/joint_dispatch/formal_v4_2_contract.py`: strict budget validation.
- Create `frame/src/joint_dispatch/formal_v4_2_gate2_artifacts.py`: immutable row artifacts and resume validation.
- Create `frame/src/joint_dispatch/formal_v4_2_gate2_training.py`: full-window data and all trainable method executors.
- Create `frame/src/joint_dispatch/formal_v4_2_gate2_execution.py`: common row execution and 23-row orchestration.
- Modify `frame/src/joint_dispatch/formal_v4_2_methods.py`: checkpoint-backed method construction.
- Modify `frame/src/joint_dispatch/formal_v4_2_metrics.py`: row metric serialization and not-applicable forecasts.
- Modify `frame/src/joint_dispatch/formal_v4_2_gate0.py`: project the complete frozen Gate 2 workload rather than isolated operation counts.
- Modify `frame/scripts/run_rsc_pf_formal_v4_2_gate2.py`: real execution CLI.
- Modify `frame/scripts/audit_rsc_pf_formal_v4_2_gate2.py`: independent artifact recomputation.
- Create `frame/scripts/smoke_rsc_pf_formal_v4_2_gate2.py`: bounded all-family smoke run.
- Add focused tests for contract, artifacts, training, execution, smoke, and audit.

---

### Task 1: Anchor the final branch and integrate real Gate 1

**Files:**
- Integrate commit: `55bf597 feat: execute representative formal v4.2 gate1`
- Verify: `frame/scripts/run_rsc_pf_formal_v4_2_gate1.py`
- Verify: `frame/tests/test_joint_dispatch_formal_v4_2_gate1.py`

**Interfaces:**
- Consumes: the reviewed spec and this plan based on `0eb3db8`.
- Produces: branch `formal-v42-gate2-final` with the real Gate 1 runner.

- [ ] **Step 1: Create the named branch**

```powershell
git switch -c formal-v42-gate2-final
```

- [ ] **Step 2: Integrate Gate 1 and run its focused tests**

```powershell
git cherry-pick 55bf597
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_contract.py frame/tests/test_joint_dispatch_formal_v4_2_gate1.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task1
```

Expected: tests pass and no run artifact is created.

---

### Task 2: Freeze the complete Gate 2 budget

**Files:**
- Modify: `frame/configs/joint_forecast_dispatch_formal_v4_2.json`
- Modify: `frame/src/joint_dispatch/formal_v4_2_contract.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_contract.py`

**Interfaces:**
- Produces: `FormalV42Contract.gate2_budget -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing tests**

```python
def test_gate2_budget_is_complete_and_frozen():
    budget = load_formal_v4_2_contract(CONTRACT).gate2_budget
    assert budget["train_scope"] == "all_eligible_2015_2018"
    assert budget["evaluation_scope"] == "all_eligible_2019_chronology"
    assert budget["effective_batch_size"] == 64
    assert budget["max_epochs"] == {"P": 30, "S": 30, "J": 30}
    assert budget["minimum_stage_j_epochs"] == 18
    assert budget["patience"] == 5
    assert budget["validation_interval"] == 1
    assert budget["diff_lp"]["preserve_effective_batch"] is True
    assert budget["resource_envelope_hours"] == 24.0
```

- [ ] **Step 2: Verify the tests fail, then add the exact payload and validator**

```json
"gate2_budget": {
  "train_scope": "all_eligible_2015_2018",
  "calibration_origin_count": 1000,
  "evaluation_scope": "all_eligible_2019_chronology",
  "effective_batch_size": 64,
  "max_epochs": {"P": 30, "S": 30, "J": 30},
  "minimum_stage_j_epochs": 18,
  "patience": 5,
  "validation_interval": 1,
  "diff_lp": {"allow_micro_batch": true, "preserve_effective_batch": true},
  "resource_envelope_hours": 24.0
}
```

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_contract.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task2
```

- [ ] **Step 3: Commit**

```powershell
git add frame/configs/joint_forecast_dispatch_formal_v4_2.json frame/src/joint_dispatch/formal_v4_2_contract.py frame/tests/test_joint_dispatch_formal_v4_2_contract.py
git commit -m "feat: freeze formal v4.2 gate2 budget"
```

---

### Task 3: Require trained checkpoints and immutable row artifacts

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_gate2_artifacts.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate2_artifacts.py`
- Modify: `frame/src/joint_dispatch/formal_v4_2_methods.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_methods.py`

**Interfaces:**
- Produces: `Gate2RowKey(method_id: str, seed: int | None)`.
- Produces: `write_gate2_rollout(path, rollout, lineage) -> str`.
- Produces: `write_gate2_row_receipt(row_dir, payload) -> str`.
- Produces: `validate_gate2_row(row_dir, expected) -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing tests**

```python
def test_stochastic_neural_method_rejects_missing_checkpoint():
    with pytest.raises(ValueError, match="trained checkpoint"):
        build_v42_method("RSC-PF", seed=2026, parameters=PARAMETERS)

def test_row_receipt_requires_persisted_checkpoint_and_rollout(tmp_path):
    row_dir = tmp_path / "RSC-PF" / "2026"
    row_dir.mkdir(parents=True)
    with pytest.raises(LineageError):
        write_gate2_row_receipt(row_dir, COMPLETE_PAYLOAD)
```

- [ ] **Step 2: Implement required hashes and deterministic not-applicable values**

```python
REQUIRED_ROW_HASHES = (
    "contract_sha256", "source_manifest_sha256", "train_manifest_sha256",
    "calibration_manifest_sha256", "evaluation_manifest_sha256",
    "normalization_sha256", "checkpoint_sha256", "rollout_sha256",
    "metrics_sha256", "training_receipt_sha256",
)
```

All stochastic rows require 64-character hashes and `optimizer_steps > 0`. Deterministic rows use `"not_applicable"` for checkpoint and training hashes.

- [ ] **Step 3: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_gate2_artifacts.py frame/tests/test_joint_dispatch_formal_v4_2_methods.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task3
git add frame/src/joint_dispatch/formal_v4_2_gate2_artifacts.py frame/src/joint_dispatch/formal_v4_2_methods.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_artifacts.py frame/tests/test_joint_dispatch_formal_v4_2_methods.py
git commit -m "feat: require trained formal v4.2 gate2 rows"
```

---

### Task 4: Build full-window manifests and effective batches

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_gate2_training.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py`

**Interfaces:**
- Produces: `Gate2DataBundle(train, calibration, evaluation, normalization, hashes)`.
- Produces: `load_gate2_data(run_root, contract) -> Gate2DataBundle`.
- Produces: `iter_gate2_batches(split, normalization, effective_batch_size=64, micro_batch_size=None)`.

- [ ] **Step 1: Write failing scope tests**

```python
def test_gate2_uses_all_eligible_train_windows(run_root):
    bundle = load_gate2_data(run_root, CONTRACT)
    assert set(np.unique(bundle.train.years)) == {2015, 2016, 2017, 2018}
    assert len(bundle.train) == count_eligible_origins(2015, 2016, 2017, 2018)

def test_gate2_evaluates_complete_2019_chronology(run_root):
    bundle = load_gate2_data(run_root, CONTRACT)
    assert set(np.unique(bundle.evaluation.years)) == {2019}
    assert np.all(np.diff(bundle.evaluation.origin_indices) == 1)
```

- [ ] **Step 2: Implement exact-year/hash checks and deterministic micro-batch accumulation metadata**

`load_gate2_data` must validate the Gate 1 train windows, selection windows, normalization, and origin manifest. It derives full 2019 ordering without subsampling.

- [ ] **Step 3: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task4
git add frame/src/joint_dispatch/formal_v4_2_gate2_training.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py
git commit -m "feat: freeze gate2 data manifests and batches"
```

---

### Task 5: Implement core and PTO training families

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_2_gate2_training.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py`

**Interfaces:**
- Produces: `train_rsc_family(seed, data, freeze, parameters, output_dir)`.
- Produces: `train_direct_policy(seed, data, freeze, parameters, output_dir)`.
- Produces: `train_scheme2r_pto(seed, data, freeze, output_dir)`.
- Produces: `train_official_itransformer_pto(seed, data, freeze, source_receipt, output_dir)`.

- [ ] **Step 1: Write failing identity tests**

```python
def test_rsc_pair_has_identical_stage_s_parent(tiny_bundle, tmp_path):
    rows = train_rsc_family(2026, tiny_bundle, FREEZE, PARAMETERS, tmp_path)
    assert rows["RSC-PF"].stage_s_parent_sha256 == rows["Decoupled-RSC-PF"].stage_s_parent_sha256
    assert rows["RSC-PF"].decision_forecaster_gradient_norm > 0.0
    assert rows["Decoupled-RSC-PF"].decision_forecaster_gradient_norm == 0.0

def test_direct_policy_has_no_forecast_training(tiny_bundle, tmp_path):
    row = train_direct_policy(2026, tiny_bundle, FREEZE, PARAMETERS, tmp_path)
    assert row.training_receipt["forecast_loss_applicable"] is False
    assert row.training_receipt["optimizer_steps"] > 0

def test_itransformer_records_upstream_adaptation(tiny_bundle, receipt, tmp_path):
    row = train_official_itransformer_pto(2026, tiny_bundle, FREEZE, receipt, tmp_path)
    assert row.training_receipt["upstream_commit"] == "c2426e68ca13f74aaec08045c5c724d8ad328124"
    assert row.training_receipt["method_label"] == "official_backbone_adaptation"
```

- [ ] **Step 2: Implement dedicated trainers**

RSC and Decoupled share byte-identical Stage S ancestry. State-Conditioned-PTO references the same seed's Stage P checkpoint. Direct-Policy never calls Stage P and optimizes only imitation/decision/constraint losses. Scheme2R and iTransformer train distinct registered forecasters and each use the canonical exact LP at evaluation.

- [ ] **Step 3: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task5
git add frame/src/joint_dispatch/formal_v4_2_gate2_training.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py
git commit -m "feat: train formal v4.2 core and pto methods"
```

---

### Task 6: Implement Differentiable-LP training and solver accounting

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_2_gate2_training.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py`

**Interfaces:**
- Produces: `train_differentiable_lp(seed, data, freeze, diffopt_receipt, parameters, output_dir)`.

- [ ] **Step 1: Write the failing gradient/exposure test**

```python
def test_diff_lp_training_is_real_and_complete(tiny_bundle, receipt, tmp_path):
    row = train_differentiable_lp(2026, tiny_bundle, FREEZE, receipt, PARAMETERS, tmp_path)
    evidence = row.training_receipt
    assert evidence["gradient_norm"] > 0.0
    assert evidence["optimizer_steps"] > 0
    assert evidence["sample_exposures"] == evidence["expected_sample_exposures"]
    assert evidence["failed_solves"] == 0
    assert evidence["effective_batch_size"] == 64
    assert evidence["training_solver_calls"] > 0
```

- [ ] **Step 2: Implement gradient-accumulated micro-batches**

Divide each micro-batch loss by the number of parts, call backward for every part, verify finite accumulated gradients, clip once, and step once per effective batch. Record micro-batch size and actual layer calls.

- [ ] **Step 3: Run isolated DiffLP tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py -k diff_lp -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task6
git add frame/src/joint_dispatch/formal_v4_2_gate2_training.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_training.py
git commit -m "feat: train formal v4.2 differentiable lp"
```

---

### Task 7: Persist and evaluate each row through the common rolling evaluator

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_2_gate2_execution.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate2_execution.py`
- Modify: `frame/src/joint_dispatch/formal_v4_2_metrics.py`

**Interfaces:**
- Produces: `execute_gate2_row(key, trained, data, parameters, row_dir, lineage) -> Mapping[str, Any]`.
- Reuses: `evaluate_chronological_v42` and `compute_v42_metrics`.

- [ ] **Step 1: Write failing persisted-evidence tests**

```python
def test_row_is_computed_from_reopened_rollout(tiny_method, tiny_bundle, tmp_path):
    row = execute_gate2_row(KEY, tiny_method, tiny_bundle, PARAMETERS, tmp_path, LINEAGE)
    saved = np.load(tmp_path / "ROLLOUT.npz")
    assert row["settled_hours"] == len(saved["settled_dispatch"])
    assert row["metrics_sha256"] == sha256_file(tmp_path / "METRICS.json")

def test_direct_policy_forecast_is_not_applicable(direct_key, direct_method, tiny_bundle, parameters, tmp_path, lineage):
    row = execute_gate2_row(direct_key, direct_method, tiny_bundle, parameters, tmp_path, lineage)
    assert row["forecast_metrics_applicable"] is False
    assert row["forecast_metrics"] is None
```

- [ ] **Step 2: Implement rollout persistence and metric recomputation**

Persist timestamps, origins, forecasts where applicable, first actions, settled dispatch, objective components, shortage, residual families, state hashes, runtime, and optimizer calls. Reopen `ROLLOUT.npz` before writing `METRICS.json` and `ROW_RECEIPT.json`.

- [ ] **Step 3: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_gate2_execution.py frame/tests/test_joint_dispatch_formal_v4_2_metrics.py frame/tests/test_joint_dispatch_formal_v4_2_rollout.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task7
git add frame/src/joint_dispatch/formal_v4_2_gate2_execution.py frame/src/joint_dispatch/formal_v4_2_metrics.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_execution.py
git commit -m "feat: persist formal v4.2 gate2 rolling rows"
```

---

### Task 8: Turn Gate 2 into the fail-closed 23-row orchestrator

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_2_gate2_execution.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_2_gate2.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_gate2.py`

**Interfaces:**
- Produces: `run_gate2(contract_path, output_root, run_id) -> Gate2DecisionV42`.
- Retains: `authorize_gate2(rows, contract) -> Gate2DecisionV42` as a pure decision function.

- [ ] **Step 1: Write failing orchestration tests**

```python
def test_runner_executes_exact_registered_matrix(fake_executor, authorized_gate1_root):
    decision = run_gate2(CONTRACT, authorized_gate1_root.parent, RUN_ID)
    assert fake_executor.keys == list(registered_method_rows(CONTRACT_OBJECT, gate="gate2"))
    assert decision.row_count == 23

def test_cli_exposes_no_budget_overrides():
    options = {action.dest for action in build_gate2_parser()._actions}
    assert options.isdisjoint({"epochs", "batch_size", "train_windows", "patience", "learning_rate"})

def test_online_solver_counts_match_settled_origins(valid_rows):
    for row in valid_rows:
        if row["method_id"] in {"Scheme2R-PTO", "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP", "Seasonal-Naive-PTO", "Perfect-Information-MPC"}:
            assert row["optimizer_calls"] == row["settled_hours"]
        else:
            assert row["optimizer_calls"] == 0
```

- [ ] **Step 2: Implement transition validation, deterministic row order, resume, failure receipts, assembly, and decision**

The CLI accepts only `--contract`, `--output-root`, and `--run-id`. It assembles `gate2_rows.json` only from validated persisted row receipts.

- [ ] **Step 3: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_gate2.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_execution.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task8
git add frame/src/joint_dispatch/formal_v4_2_gate2_execution.py frame/scripts/run_rsc_pf_formal_v4_2_gate2.py frame/tests/test_joint_dispatch_formal_v4_2_gate2.py
git commit -m "feat: execute complete formal v4.2 gate2 matrix"
```

---

### Task 9: Independently audit Gate 2 artifacts

**Files:**
- Modify: `frame/scripts/audit_rsc_pf_formal_v4_2_gate2.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate2_audit.py`

**Interfaces:**
- Produces: `audit_gate2(run_root: Path) -> Gate2AuditV42`.

- [ ] **Step 1: Write failing tamper tests**

```python
def test_audit_rejects_metric_not_supported_by_rollout(complete_gate2_root):
    mutate_json(METRICS, "penalized_objective", -1.0)
    assert audit_gate2(complete_gate2_root).authorized_gate3 is False

def test_audit_rejects_wrong_stage_s_ancestor(complete_gate2_root):
    mutate_json(ROW_RECEIPT, "stage_s_parent_sha256", "0" * 64)
    assert audit_gate2(complete_gate2_root).authorized_gate3 is False
```

- [ ] **Step 2: Recompute matrix completeness, hashes, metrics, gradient boundaries, optimizer calls, and access receipts**

The audit reads persisted files directly. Only agreement between `GATE2_DECISION.json` and `GATE2_AUDIT.json` may create `GATE2_TRANSITION.json`.

- [ ] **Step 3: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_gate2_audit.py frame/tests/test_joint_dispatch_formal_v4_2_gate2.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_task9
git add frame/scripts/audit_rsc_pf_formal_v4_2_gate2.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_audit.py
git commit -m "feat: independently audit formal v4.2 gate2"
```

---

### Task 10: Run all-family smoke and freeze controlled sources

**Files:**
- Create: `frame/scripts/smoke_rsc_pf_formal_v4_2_gate2.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_2_gate2_smoke.py`
- Modify: `frame/configs/formal_v4_source_closure_v4_2.txt`
- Modify: `frame/src/joint_dispatch/formal_v4_2_gate0.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_2_gate0.py`

**Interfaces:**
- Produces: `run_gate2_smoke(output_root) -> Mapping[str, Any]` with `paper_eligible=false`.

- [ ] **Step 1: Implement and test a fixed 16-window, two-epoch, one-seed, eight-origin smoke**

```python
def test_smoke_runs_every_family_without_formal_transition(tmp_path):
    receipt = run_gate2_smoke(tmp_path)
    assert receipt["paper_eligible"] is False
    assert receipt["families"] == ["rsc", "direct", "pto", "itransformer", "diff_lp", "deterministic"]
    assert all(item["finite"] for item in receipt["results"])
    assert not (tmp_path / "GATE2_TRANSITION.json").exists()
```

Add a Gate 0 resource-projection test that calculates training updates and rolling solves from the frozen full manifests:

```python
def test_gate0_projects_complete_gate2_workload(full_manifest_counts):
    projection = project_gate2_resources(CONTRACT, BENCHMARKS, full_manifest_counts)
    assert projection["training_sample_exposures"] == full_manifest_counts.train_windows * 30 * projection["trained_stage_count"]
    assert projection["rolling_optimizer_calls"] == full_manifest_counts.evaluation_origins * (4 * 3 + 2)
    assert projection["authorized"] == (projection["total_hours"] <= 24.0 and projection["disk_margin"] >= 0.20)
```

- [ ] **Step 2: Run smoke and the complete formal-v4.2 suite**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_joint_dispatch_formal_v4_2_*.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_full
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/smoke_rsc_pf_formal_v4_2_gate2.py --output-root D:\Paper\formal_v42_gate2_smoke
```

- [ ] **Step 3: Run affected regression tests and freeze source closure**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest frame/tests/test_dispatch_lp.py frame/tests/test_scheduling_proxy_decoder.py frame/tests/test_joint_dispatch_model.py frame/tests/test_scheme2r.py -q -p no:cacheprovider --basetemp D:\Paper\pytest_tmp_v42_gate2_regression
git diff --check
git add frame/scripts/smoke_rsc_pf_formal_v4_2_gate2.py frame/tests/test_joint_dispatch_formal_v4_2_gate2_smoke.py frame/configs/formal_v4_source_closure_v4_2.txt frame/src/joint_dispatch/formal_v4_2_gate0.py frame/tests/test_joint_dispatch_formal_v4_2_gate0.py
git commit -m "chore: freeze final formal v4.2 gate2 sources"
```

---

### Task 11: Execute the final Gate chain and stop at Gate 2

**Files:**
- Generate only under `frame/reports/joint_forecast_dispatch_formal_v4_2/<new-run-id>/`.

**Interfaces:**
- Produces: a continuous Gate 0 → Pilot → Gate 1 → Gate 2 lineage or the first immutable failure receipt.

- [ ] **Step 1: Choose a fresh run ID and verify it does not exist**

```powershell
$RunId = 'formal_v4_2_20260905_g'
Test-Path "frame\reports\joint_forecast_dispatch_formal_v4_2\$RunId"
```

Expected: `False`. Otherwise increment the suffix before running anything.

- [ ] **Step 2: Run Gate 0 and stop unless `authorized_pilot=true`**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_gate0.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
```

- [ ] **Step 3: Run Pilot and stop unless `authorized_gate1=true`**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_pilot.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
```

- [ ] **Step 4: Run Gate 1 and stop unless `authorized_gate2=true`**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_gate1.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
```

- [ ] **Step 5: Run Gate 2 and its independent audit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_gate2.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/audit_rsc_pf_formal_v4_2_gate2.py --run-root "frame/reports/joint_forecast_dispatch_formal_v4_2/$RunId"
```

- [ ] **Step 6: Stop and report**

Report the first failed Gate and its receipt, or report the Gate 2 decision, independent-audit agreement, 23-row completeness, runtime, and primary RSC-PF versus Decoupled-RSC-PF result. Do not invoke Gate 3, read 2020, run ablations, or edit the manuscript.

---

## Plan Self-Review Record

- Specification coverage: every reviewed requirement maps to Tasks 2--11.
- Completeness scan: every implementation step names concrete files, interfaces, assertions, commands, and failure behavior.
- Type consistency: `Gate2DataBundle`, `Gate2RowKey`, training functions, `execute_gate2_row`, `run_gate2`, and `audit_gate2` are introduced before downstream use.
- Scope: execution ends at the independent Gate 2 audit.
- Execution mode: inline in the current task, as explicitly requested; no subagents are used.
