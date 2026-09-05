# RSC-PF Formal-v4.4 Real Pilot Executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the frozen formal-v4.4 Pilot from an orchestration shell into a real, resumable, independently auditable P0/P1/S/J execution over causal device trajectories, same-information LP labels, and the complete 2019 chronology.

**Architecture:** Add version-isolated v4.4 adapters for provenance, materialized windows, LP teacher labels, rolling evaluation, and artifact resumption. Keep the existing v4.4 model, loss, training stages, metrics, and authorization criteria unchanged; connect them through one production executor and rerun Gate 0 against a new source manifest after implementation.

**Tech Stack:** Python 3.11, PyTorch, NumPy, PyYAML, existing standard-IES simulation and LP utilities, pytest, PowerShell, SHA-256 lineage receipts.

## Global Constraints

- Train years are exactly `2015--2018`; Pilot evaluation is exactly `2019`; `2020` is sealed until Gate 2.
- Pilot training size is exactly `4096` windows with seed `2026`, batch size `32`, and purge `28` hours.
- The frozen v4.4 architecture remains `P0 → P1 → S → Joint/Decoupled J`; no binary future decisions are added.
- Gas remains the supervised station-side auxiliary prior, not a rigid terminal gas-demand balance.
- The final dispatch remains 15 continuous controls decoded to 21 physical dispatch values.
- Pilot results are non-paper engineering evidence; the Pilot never launches Gate 1 automatically.
- Existing v4.2/v4.3 receipts are read-only diagnostics and are never copied into v4.4 receipts.
- The existing untracked `third_party/iTransformer_source/` directory is not modified or staged.
- Every persisted artifact is immutable; reruns use a new run ID unless all lineage hashes match an unfinished stage exactly.

---

### Task 1: Bind a fresh v4.4 source manifest and Gate-0 lineage

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_provenance.py`
- Create: `scripts/build_rsc_pf_formal_v4_4_source_manifest.py`
- Modify: `scripts/run_rsc_pf_formal_v4_4_gate0.py`
- Modify: `src/joint_dispatch/formal_v4_4_gate0.py`
- Create: `tests/test_joint_dispatch_formal_v4_4_provenance.py`
- Modify: `tests/test_joint_dispatch_formal_v4_4_gate0.py`

**Interfaces:**
- Produce `build_source_manifest(*, run_id: str, contract_sha256: str, repo_root: Path, required_paths: Sequence[Path]) -> dict[str, Any]`.
- Produce `write_source_manifest(path: Path, manifest: Mapping[str, Any]) -> str`, returning the file SHA-256.
- Expose `scripts/build_rsc_pf_formal_v4_4_source_manifest.py` with required arguments `--contract`, `--repo-root`, `--run-id`, and `--output`; it must call the two provenance functions and print the output path and digest.
- Require the manifest to contain `schema="formal-v4.4-source-manifest-v1"`, current Git commit, clean/dirty status, contract hash, implementation path hashes, and `evaluation_year_accessed=false`.
- Record `third_party/iTransformer_source/` as an explicit pre-existing allowlisted untracked path; do not stage, hash as v4.4 implementation, delete, or modify it.
- Make Gate 0 reject a manifest whose commit or implementation hashes do not describe the current checkout.
- Make Gate 0 copy the externally staged manifest into `<run>/protocol/SOURCE_MANIFEST.json` after creating the new run directory; the source-manifest staging directory must remain outside the run directory so Gate 0 can create the run atomically.
- Add a measured `executor_entrypoint` check that imports the production executor and runs its non-training contract/data-shape preflight on a deterministic two-window fixture.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_v44_manifest_records_current_commit_and_executor_hash(tmp_path, repo_root):
    manifest = build_source_manifest(
        run_id="v44-provenance-test",
        contract_sha256="a" * 64,
        repo_root=repo_root,
        required_paths=[repo_root / "src" / "joint_dispatch" / "formal_v4_4_pilot_executor.py"],
    )
    assert manifest["schema"] == "formal-v4.4-source-manifest-v1"
    assert manifest["git_commit"]
    assert manifest["files"]["src/joint_dispatch/formal_v4_4_pilot_executor.py"]
    assert manifest["evaluation_year_accessed"] is False
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_provenance.py -q
```

Expected: FAIL because the v4.4 provenance module does not exist.

- [ ] **Step 3: Implement manifest generation and strict Gate-0 validation**

Hash the exact v4.4 contract, executor, materializer, teacher, rollout,
training, model, loss, metrics, pilot, and audit modules. Use the current Git
commit and reject a dirty checkout unless the manifest explicitly records the
dirty state and the Gate-0 command is run from that same immutable state.

- [ ] **Step 4: Run the focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_4_provenance.py scripts/build_rsc_pf_formal_v4_4_source_manifest.py scripts/run_rsc_pf_formal_v4_4_gate0.py src/joint_dispatch/formal_v4_4_gate0.py tests/test_joint_dispatch_formal_v4_4_provenance.py tests/test_joint_dispatch_formal_v4_4_gate0.py
git commit -m "feat: bind formal v4.4 source lineage"
```

### Task 2: Add causal v4.4 materialization and immutable resume state

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_pilot_materializer.py`
- Modify: `src/joint_dispatch/formal_v4_4_pilot_data.py`
- Modify: `src/joint_dispatch/formal_v4_4_artifacts.py`
- Create: `tests/test_joint_dispatch_formal_v4_4_pilot_materializer.py`
- Modify: `tests/test_joint_dispatch_formal_v4_4_artifacts.py`

**Interfaces:**
- Produce `load_v44_base_series(train_data: Path, selection_data: Path) -> tuple[FormalV4BaseSeries, FormalV4BaseSeries]`.
- Produce `materialize_v44_pilot_data(*, train_data: Path, selection_data: Path, benchmark: Path, capacity_receipt: Path, split: Mapping[str, np.ndarray], artifact_root: Path, contract: FormalV44Contract) -> MaterializedV44PilotData`.
- Produce `save_or_load_materialized_v44(...)` that accepts a lineage key and rejects mismatched cached arrays.
- Define `MaterializedV44PilotData` with `.train`, `.early_stop`, `.selection_full`, and `.selection_stress` window collections; each collection exposes `load_history`, `exog_history`, `device_history`, `activity_history`, `scheduler_context`, `previous_chp`, `initial_soc`, `forecast_target`, `renewable_forecast`, `renewable_realized`, and `timestamps`.
- Return 24-hour histories, 17 continuous device-history features, six activity/status features, renewable forecasts and realizations, scheduler context, initial SOC, previous CHP output, four-hour targets, timestamps, and split role.

- [ ] **Step 1: Write failing causal-boundary and resume tests**

```python
def test_materializer_keeps_device_history_and_split_roles(synthetic_sources, tmp_path):
    result = materialize_v44_pilot_data(**synthetic_sources, artifact_root=tmp_path)
    assert result.train.load_history.shape[1:] == (24, 4)
    assert result.train.device_history.shape[1:] == (24, 17)
    assert result.train.activity_history.shape[1:] == (24, 6)
    assert set(result.selection.timestamps.astype("datetime64[Y]").astype(int) + 1970) == {2019}

def test_materializer_rejects_mismatched_resume_hash(synthetic_sources, tmp_path):
    materialize_v44_pilot_data(**synthetic_sources, artifact_root=tmp_path)
    with pytest.raises(ValueError, match="lineage"):
        materialize_v44_pilot_data(**{**synthetic_sources, "capacity_receipt": synthetic_sources["other_capacity"]}, artifact_root=tmp_path)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_pilot_materializer.py -q
```

Expected: FAIL because the production materializer and lineage cache do not exist.

- [ ] **Step 3: Implement the materializer**

Reuse only version-independent functions from `formal_v4_history.py` and
`formal_v4_data.py`. Generate settled standard-IES device trajectories from
the explicit train and selection archives, materialize causal windows, apply
the frozen v4.4 split indices, and persist arrays atomically. Fit no statistic
from the 2019 selection arrays.

- [ ] **Step 4: Implement atomic cache validation**

Persist an immutable metadata sidecar containing contract, source, capacity,
benchmark, split, trajectory, and implementation hashes. Reuse a cache only if
all hashes match byte-for-byte; otherwise raise and require a new run ID.

- [ ] **Step 5: Run the focused tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src/joint_dispatch/formal_v4_4_pilot_materializer.py src/joint_dispatch/formal_v4_4_pilot_data.py src/joint_dispatch/formal_v4_4_artifacts.py tests/test_joint_dispatch_formal_v4_4_pilot_materializer.py tests/test_joint_dispatch_formal_v4_4_artifacts.py
git commit -m "feat: materialize causal formal v4.4 pilot windows"
```

### Task 3: Add a version-isolated same-information LP teacher

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_teacher.py`
- Create: `tests/test_joint_dispatch_formal_v4_4_teacher.py`

**Interfaces:**
- Produce `build_same_information_teacher_v44(*, model: nn.Module, materialized: MaterializedV44PilotData, indices: np.ndarray, benchmark: Path, capacity_receipt: Path, artifact_root: Path, contract: FormalV44Contract, seed: int) -> TeacherReceiptV44`.
- Produce `TeacherReceiptV44.dispatch` with shape `[N, 4, 21]` and `TeacherReceiptV44.objective`, `shortage`, `timestamps`, `state_hashes`, and `lineage` fields.

- [ ] **Step 1: Write failing information-firewall tests**

```python
def test_teacher_uses_forecasts_not_realized_future_targets(teacher_fixture):
    first = build_same_information_teacher_v44(**teacher_fixture)
    changed = teacher_fixture["materialized"].with_future_targets_added(1000.0)
    second = build_same_information_teacher_v44(**{**teacher_fixture, "materialized": changed})
    np.testing.assert_allclose(first.dispatch, second.dispatch)

def test_teacher_rejects_failed_or_nonfinite_lp_row(teacher_fixture, monkeypatch):
    monkeypatch.setattr("src.joint_dispatch.formal_v4_4_teacher.solve_dispatch_lp", lambda **_: None)
    with pytest.raises(RuntimeError, match="LP"):
        build_same_information_teacher_v44(**teacher_fixture)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_teacher.py -q
```

Expected: FAIL because the v4.4 teacher adapter does not exist.

- [ ] **Step 3: Implement teacher generation**

Use P1 physical forecasts, renewable forecasts, prices/carbon context, initial
SOC, previous CHP output, benchmark parameters, and capacity receipt. Do not
pass realized future demand. Persist each row's solver status and input hash
before accepting its dispatch label.

- [ ] **Step 4: Run the focused tests**

Expected: PASS, including the future-target mutation test.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_4_teacher.py tests/test_joint_dispatch_formal_v4_4_teacher.py
git commit -m "feat: add formal v4.4 same-information teacher"
```

### Task 4: Implement matched v4.4 stage orchestration

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_pilot_executor.py`
- Modify: `src/joint_dispatch/formal_v4_4_training.py`
- Create: `tests/test_joint_dispatch_formal_v4_4_pilot_executor.py`

**Interfaces:**
- Produce `build_v44_model(materialized: MaterializedV44PilotData, contract: FormalV44Contract, prior: ThermalPriorV44, parameters: Mapping[str, Any]) -> ResidualGatedRSCPFModel`.
- Produce `build_continuous_control_v44(p0_model: ResidualGatedRSCPFModel) -> nn.Module` that reuses the identical P0 base state but emits direct continuous cooling/heating forecasts without the learned regime gate or thermal residual heads.
- Produce `execute_training_stages_v44(*, materialized, teacher, contract, artifact_root, seed) -> TrainingBundleV44`.
- `TrainingBundleV44` must expose P0, P1, continuous-control, S, J-joint, and J-decoupled receipts and models with parent hashes.

- [ ] **Step 1: Write failing stage-order and matched-parent tests**

```python
def test_training_executor_runs_frozen_stage_order(executor_fixture):
    bundle = execute_training_stages_v44(**executor_fixture)
    assert bundle.stage_order == ("P0", "P1", "P1-control", "S", "J-joint", "J-decoupled")
    assert bundle.joint.parent_sha256 == bundle.decoupled.parent_sha256
    assert bundle.joint.gradient_norms["decision_to_base"] > 0.0
    assert bundle.decoupled.gradient_norms["decision_to_base"] <= 1e-12
```

- [ ] **Step 2: Run the focused test and verify it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_pilot_executor.py -q
```

Expected: FAIL because the production executor and bundle do not exist.

- [ ] **Step 3: Implement model construction and stage orchestration**

Construct the v4.4 model from train-only normalization and transition priors.
Run existing `run_stage_p0_v44`, `run_stage_p1_v44`,
`run_stage_s_v44`, and `run_stage_j_pair_v44` with the frozen budget. Build the
continuous control through `build_continuous_control_v44` and train it with
the same P0-derived forecast optimizer-step count; do not let its regime gate
or residual magnitude heads become a hidden second mechanism. Save each stage
before starting the next one and make resume load only a hash-matched parent.

- [ ] **Step 4: Run the focused test**

Expected: PASS, with finite losses and the exact gradient boundary.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_4_pilot_executor.py src/joint_dispatch/formal_v4_4_training.py tests/test_joint_dispatch_formal_v4_4_pilot_executor.py
git commit -m "feat: orchestrate formal v4.4 pilot stages"
```

### Task 5: Add full-chronology 2019 rolling evaluation

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_rollout.py`
- Create: `tests/test_joint_dispatch_formal_v4_4_rollout.py`

**Interfaces:**
- Produce `rollout_v44_2019(*, model: nn.Module, materialized: MaterializedV44PilotData, indices: np.ndarray, normalization: NormalizationReceiptV42, parameters: Mapping[str, Any], artifact_root: Path, method_id: str) -> RolloutReceiptV44`.
- Return stored arrays for prediction, target, regime probabilities, prior probabilities, planned dispatch, settled dispatch, shortage, residuals, states, and timestamps.
- Generate the `transition_prior` diagnostic row by applying the frozen train-only transition probabilities to the causal last observed regime and combining them with the fixed P1 magnitude backbone; do not fit or calibrate it on 2019.

- [ ] **Step 1: Write failing first-step carry and chronology tests**

```python
def test_rollout_settles_first_step_and_carries_state(rollout_fixture):
    result = rollout_v44_2019(**rollout_fixture)
    assert np.all(np.diff(result.timestamps.astype("datetime64[ns]").astype("int64")) > 0)
    np.testing.assert_allclose(result.next_initial_soc[1], result.settled_dispatch[0, 0, 16])
    assert result.planned_dispatch.shape[-1] == 21

def test_rollout_rejects_nonchronological_origins(rollout_fixture):
    rollout_fixture["indices"] = rollout_fixture["indices"][::-1]
    with pytest.raises(ValueError, match="chronological"):
        rollout_v44_2019(**rollout_fixture)
```

- [ ] **Step 2: Run the focused test and verify it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_rollout.py -q
```

Expected: FAIL because the v4.4 rolling evaluator does not exist.

- [ ] **Step 3: Implement chronological settlement**

At each 2019 origin, use only the available history, evaluate the first
planned control interval against realized rigid demand and renewable output,
compute all residual families, update SOC/activity/CHP state, and carry that
state to the next origin. Never read 2020 arrays.

- [ ] **Step 4: Persist rollout arrays and run the focused test**

Expected: PASS with monotonically increasing timestamps and exact first-step
state carry.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_4_rollout.py tests/test_joint_dispatch_formal_v4_4_rollout.py
git commit -m "feat: add causal formal v4.4 rolling evaluator"
```

### Task 6: Connect the production executor and independent artifact audit

**Files:**
- Modify: `src/joint_dispatch/formal_v4_4_pilot.py`
- Modify: `scripts/run_rsc_pf_formal_v4_4_pilot.py`
- Modify: `src/joint_dispatch/formal_v4_4_artifacts.py`
- Modify: `scripts/audit_rsc_pf_formal_v4_4_pilot.py`
- Modify: `tests/test_joint_dispatch_formal_v4_4_pilot.py`
- Modify: `tests/test_joint_dispatch_formal_v4_4_adversarial.py`

**Interfaces:**
- `run_pilot_v44` passes a run-scoped `artifact_root` to the production executor.
- `execute_real_pilot_v44(...) -> Mapping[str, Any]` returns the exact fields required by `run_pilot_v44`: `prediction`, `target`, `probability`, `prior_probability`, `regimes`, `times`, `comparisons`, `joint`, `decoupled`, `physics`, and the five frozen rows.
- The CLI supplies `execute_real_pilot_v44` by default and has no code path that silently falls back to an injected fixture.

- [ ] **Step 1: Write failing CLI and adversarial tests**

```python
def test_cli_production_path_rejects_missing_executor(tmp_path, frozen_gate0):
    result = run_pilot_v44(**frozen_gate0, output_root=tmp_path, run_id="missing-executor")
    assert result.authorized_gate1 is False

def test_audit_rejects_hard_coded_success(valid_pilot_run):
    mutate_json(valid_pilot_run / "pilot" / "PILOT_RECEIPT.json", {"authorized_gate1": True, "comparisons": {}})
    assert audit_run(valid_pilot_run, CONTRACT).authorized_gate1 is False
```

- [ ] **Step 2: Run focused tests and verify the production integration fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_pilot.py tests\test_joint_dispatch_formal_v4_4_adversarial.py -q
```

Expected: the existing injected-fixture tests remain green, while the new
production-path test fails until the executor is connected.

- [ ] **Step 3: Connect the executor and write complete artifacts**

The executor must materialize data, train stages, build LP labels, roll all
2019 origins for every required method row, and return only measured arrays and
summaries. Persist one rollout NPZ per required row, not only the forecast
summary array. `run_pilot_v44` writes the canonical arrays and receipt; the
audit recomputes all authorization-critical metrics from disk without importing
the executor.

- [ ] **Step 4: Add resume and corruption checks**

Make a partial stage restart load only matching checkpoint and metadata hashes.
Make missing teacher rows, duplicate timestamps, 2020 timestamps, non-finite
arrays, and incomplete method rows fail closed.

- [ ] **Step 5: Run focused integration and adversarial tests**

Expected: PASS; forged success and corrupted artifacts must return
`authorized_gate1=false`.

- [ ] **Step 6: Make the independent audit recompute decision and gradient evidence**

The audit must load every row's stored planned/settled dispatch, shortage, and
physical-residual arrays to recompute penalized objective and shortage
comparisons. It must load the saved stage checkpoints and one immutable Pilot
batch to rerun the Joint/Fair-Decoupled gradient probe in a fresh process;
receipt-supplied gradient norms alone are not accepted. The audit must compare
its recomputed values with the receipt within the contract tolerance and fail
closed on disagreement.

- [ ] **Step 7: Commit**

```powershell
git add src/joint_dispatch/formal_v4_4_pilot.py scripts/run_rsc_pf_formal_v4_4_pilot.py src/joint_dispatch/formal_v4_4_artifacts.py scripts/audit_rsc_pf_formal_v4_4_pilot.py tests/test_joint_dispatch_formal_v4_4_pilot.py tests/test_joint_dispatch_formal_v4_4_adversarial.py
git commit -m "feat: connect auditable formal v4.4 pilot executor"
```

### Task 7: Run the complete regression suite and create a fresh Gate-0 run

**Files:**
- Modify only generated run artifacts under `reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_b/`.
- Create: `reports/joint_forecast_dispatch_formal_v4_4/source_manifests/formal_v4_4_20260905_b/SOURCE_MANIFEST.json`

- [ ] **Step 1: Run all v4.4, protected v4.2, decoder, and LP tests**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_contract.py tests\test_joint_dispatch_formal_v4_4_regime.py tests\test_joint_dispatch_formal_v4_4_pilot_data.py tests\test_joint_dispatch_formal_v4_4_pilot_materializer.py tests\test_joint_dispatch_formal_v4_4_teacher.py tests\test_joint_dispatch_formal_v4_4_model.py tests\test_joint_dispatch_formal_v4_4_loss.py tests\test_joint_dispatch_formal_v4_4_training.py tests\test_joint_dispatch_formal_v4_4_pilot_executor.py tests\test_joint_dispatch_formal_v4_4_rollout.py tests\test_joint_dispatch_formal_v4_4_metrics.py tests\test_joint_dispatch_formal_v4_4_pilot_gate.py tests\test_joint_dispatch_formal_v4_4_artifacts.py tests\test_joint_dispatch_formal_v4_4_gate0.py tests\test_joint_dispatch_formal_v4_4_pilot.py tests\test_joint_dispatch_formal_v4_4_adversarial.py tests\test_joint_dispatch_formal_v4_2_data.py tests\test_joint_dispatch_formal_v4_2_teacher.py tests\test_joint_dispatch_formal_v4_2_rollout.py tests\test_scheduling_proxy_decoder.py tests\test_dispatch_lp.py --basetemp D:\Paper\pytest_tmp_v44_executor -p no:cacheprovider -q
```

Expected: all tests pass; warnings are allowed only when they do not alter
authorization or metric computation.

- [ ] **Step 2: Generate a fresh v4.4 source manifest**

Use the current committed implementation and record the new run ID, Git commit,
contract hash, executor hashes, and exact source paths. Do not reuse the v4.2
manifest. The source manifest is staged outside the run directory with this
command:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\build_rsc_pf_formal_v4_4_source_manifest.py --contract configs\joint_forecast_dispatch_formal_v4_4.json --repo-root . --run-id formal_v4_4_20260905_b --output reports\joint_forecast_dispatch_formal_v4_4\source_manifests\formal_v4_4_20260905_b\SOURCE_MANIFEST.json
```

Expected: the manifest records the current commit, hashes the v4.4 source
allowlist, records the pre-existing iTransformer directory as allowed
untracked content, and sets `evaluation_year_accessed=false`.

- [ ] **Step 3: Run Gate 0 with a new run ID**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\run_rsc_pf_formal_v4_4_gate0.py --contract configs\joint_forecast_dispatch_formal_v4_4.json --source-manifest reports\joint_forecast_dispatch_formal_v4_4\source_manifests\formal_v4_4_20260905_b\SOURCE_MANIFEST.json --base-train-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_train.npz --base-selection-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_selection.npz --benchmark reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\benchmark\STANDARD_IES_BENCHMARK.yaml --capacity-receipt reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\CAPACITY_FREEZE.json --output-root reports\joint_forecast_dispatch_formal_v4_4 --run-id formal_v4_4_20260905_b
```

Expected: `authorized_pilot=true`, `evaluation_year_accessed=false`, all
implementation and physical checks measured, and a Gate-0 transition pointing
to Pilot.

- [ ] **Step 4: Verify the clean source state after Gate 0**

```powershell
git status --short
git log -1 --oneline
```

Generated reports remain run artifacts and are not mixed into source commits.

### Task 8: Run the real 4,096-window Pilot and independent audit

**Files:**
- Generate: `reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_b/pilot/`
- Generate: `reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_b/PILOT_TRANSITION.json`

- [ ] **Step 1: Launch the production Pilot**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\run_rsc_pf_formal_v4_4_pilot.py --contract configs\joint_forecast_dispatch_formal_v4_4.json --gate0-transition reports\joint_forecast_dispatch_formal_v4_4\formal_v4_4_20260905_b\GATE0_TRANSITION.json --source-manifest reports\joint_forecast_dispatch_formal_v4_4\source_manifests\formal_v4_4_20260905_b\SOURCE_MANIFEST.json --base-train-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_train.npz --base-selection-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_selection.npz --benchmark reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\benchmark\STANDARD_IES_BENCHMARK.yaml --capacity-receipt reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\CAPACITY_FREEZE.json --output-root reports\joint_forecast_dispatch_formal_v4_4 --run-id formal_v4_4_20260905_b
```

Expected: a non-placeholder `PILOT_RECEIPT.json`, stored 2019 arrays, stage
checkpoints, and either an authorization transition or a truthful failure.

- [ ] **Step 2: Run the independent audit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\audit_rsc_pf_formal_v4_4_pilot.py --run-dir reports\joint_forecast_dispatch_formal_v4_4\formal_v4_4_20260905_b --contract configs\joint_forecast_dispatch_formal_v4_4.json
```

Expected: the audit recomputes the same authorization boolean from persisted
artifacts. If it returns false, stop and report the measured failed criteria;
do not launch Gate 1.

- [ ] **Step 3: Preserve the outcome and stop**

Do not start Gate 1 automatically, even if the Pilot passes. Report runtime,
artifact paths, all failed or passed criteria, 2019-only access evidence, and
the exact next action for user review.

## Self-Review Checklist

- [x] The plan covers source lineage, data, teacher, training, rollout,
  artifacts, audit, regression, and real execution.
- [x] No step changes the frozen v4.4 model, data years, or thresholds.
- [x] The current v4.2 source-manifest gap is explicitly repaired before Pilot.
- [x] Every new interface is named with a concrete signature.
- [x] No placeholder tasks or unspecified “handle edge cases” instructions remain.
- [x] Every long-running command is bounded by a Gate transition and stops
  before Gate 1.
