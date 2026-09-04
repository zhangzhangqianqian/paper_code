# RSC-PF Formal-v4.1 Gate 0 Comprehensive Repair Implementation Plan

**Revision:** 1, incorporating the post-plan scientific and provenance audit.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair Tasks 1--12 as a new formal-v4.1 protocol, rebuild a causally and physically valid experiment foundation, and issue a new independently verifiable Gate 0 authorization without reading 2020 or starting neural-network training.

**Architecture:** Preserve the validated RSC-PF model and mathematical components, but rebuild the experiment provenance chain from a committed source closure through a 2015--2018-only benchmark, representative capacity audit, realized-settled device trajectory, capacity-bound data, train-only normalization and `C_ref`, executable method adapters, complete regression evidence, and a fail-closed Gate 0 orchestrator. Every generated artifact is immutable, stored below one new run root, and SHA-256-bound to its direct inputs.

**Tech Stack:** Python 3.9, PyTorch 2.8, NumPy, pandas, SciPy HiGHS, CVXPY/CVXPYlayers in `D:\Paper\envs\rsc_pf_diffopt_v4`, pytest, JSON/YAML/CSV/NPZ, Git, PowerShell.

## Global Constraints

- Canonical repository root for every command is `D:\Paper\github_work\paper-code`; Python source lives below `frame/`.
- The repair covers Tasks 1--12 and a new Gate 0 only. Do not start Gate 1, Gate 2, Gate 3, ablations, large neural training, or manuscript result updates.
- Training/normalization/benchmark derivation may read only 2015--2018; 2019 is selection-only; 2020 remains inaccessible until a future Gate 3 authorization.
- Preserve `formal_v4_20260903_retry3` and downstream trial artifacts unchanged; classify them as `invalid_pre_repair_trial` in a separate registry.
- Preserve `frame/configs/joint_forecast_dispatch_formal_v4.json` byte-for-byte. All repaired execution uses the new `frame/configs/joint_forecast_dispatch_formal_v4_1.json` with schema `joint-forecast-dispatch-formal-v4.1`.
- Never overwrite a generated artifact. Use a new run ID for each failed or successful Gate 0 attempt.
- Preserve unrelated user changes. Commit only files owned by the current task, and never reset or discard the dirty worktree.
- Forecast tasks remain electricity, cooling, heating, and station-side gas-consumption prior; only electricity/cooling/heating are rigid dispatch demands.
- Keep 24-hour history, four-hour horizon, 15 continuous scheduling controls, 21 decoded dispatch outputs, and six derived activity indicators.
- Do not introduce future binary commitment decisions or a price/carbon-response claim.
- A missing, unknown, stale, mismatched, skipped, or failed mandatory check must set `authorized_gate1=false` and return a nonzero process exit code.
- Every direct and transitive formal-v4.1 runtime dependency must be tracked, committed, clean, closure-listed, and hash-bound before authorization. Do not sweep unrelated dirty user files into the closure commit.
- Generated runtimes and timestamps never participate in scientific identity hashes.
- Before each commit, run `git diff --check` on the owned files.

---

### Task 1: Register Invalid Trials and Freeze the Formal-v4 Source Closure

**Files:**
- Create: `frame/configs/joint_dispatch_invalid_runs_v4.json`
- Create: `frame/configs/joint_forecast_dispatch_formal_v4_1.json`
- Create: `frame/configs/formal_v4_source_closure_v4_1.txt`
- Create: `frame/src/joint_dispatch/formal_v4_provenance.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_provenance.py`
- Modify: `frame/src/joint_dispatch/formal_protocol_v4.py`
- Test: `frame/tests/test_joint_dispatch_formal_protocol_v4.py`

**Interfaces:**
- Consumes: repository root, formal-v4 config, state/search configs, parameter ledger, third-party receipts, and a declared list of formal-v4 source files.
- Produces: `FormalV4SourceManifest`, `build_source_manifest(repo_root, paths, git_commit)`, `validate_source_manifest(...)`, and an immutable invalid-run registry.

- [ ] **Step 1: Write failing provenance tests**

Add tests proving that the source manifest rejects an untracked, missing, dirty, or hash-mismatched file in the formal-v4.1 source closure, rejects a missing benchmark rule file, rejects a transitive runtime dependency omitted from the closure, and ignores an unrelated dirty document outside the closure. Assert that loading the v4.1 protocol does not modify the original v4 configuration.

```python
def test_manifest_rejects_changed_formal_source(tmp_repo):
    manifest = build_source_manifest(tmp_repo, ["frame/src/joint_dispatch/formal_v4_data.py"], "abc")
    (tmp_repo / "frame/src/joint_dispatch/formal_v4_data.py").write_text("changed")
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_source_manifest(manifest, tmp_repo)
```

- [ ] **Step 2: Run the new tests and confirm failure**

Run:

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' -m pytest frame\tests\test_joint_dispatch_formal_v4_provenance.py -q --basetemp D:\Paper\pytest_tmp_v4_repair_t1 -p no:cacheprovider
```

Expected: FAIL because the provenance module and invalid-run registry do not exist.

- [ ] **Step 3: Add the invalid-run registry**

Record the pre-repair roots `formal_v4_20260903`, `formal_v4_20260903_retry1`, `formal_v4_20260903_retry2`, `formal_v4_20260903_retry3`, `formal_v4_20260903_retry3_gate1d`, `formal_v4_20260903_retry3_gate1e`, `formal_v4_20260903_retry3_gate1f`, `formal_v4_20260903_retry3_gate1_g`, `formal_v4_20260903_retry3_gate1_h`, `formal_v4_20260903_retry3_gate1_i`, `formal_v4_20260903_retry3_gate1_final`, `formal_v4_20260903_retry3_gate2`, `gate1_smoke`, and the sibling `joint_forecast_dispatch_formal_v4_retry_gate1` data root. Assign status `invalid_pre_repair_trial`, reasons, audit date, and `allowed_for_formal_results=false`. Do not edit the old run directories.

- [ ] **Step 4: Inventory the complete source closure and commit the reviewed baseline**

Create `formal_v4_source_closure_v4_1.txt` with repository-relative paths for every direct and transitive runtime dependency: source/config/tests, benchmark rules, parameter ledger, state ledger, builders, evaluator, Gate 0 and independent-auditor scripts, iTransformer receipt/imported sources, Differentiable-LP environment lock, and search budget. Planned files not yet created remain explicit required entries and prevent authorization until later tasks commit them. Review `git status --short` and `git diff -- <each existing closure path>` before staging. Commit only reviewed formal dependencies; preserve unrelated user edits. No generated receipt may claim a final source freeze while a closure path is missing, untracked, or dirty.

- [ ] **Step 5: Implement source-closure hashing and validation**

Hash raw file bytes, normalize paths relative to the Git root, record the Git commit, and reject path escape, duplicate paths, missing/untracked/dirty files, mismatched hashes, closure omissions, or invalid-run inputs.

- [ ] **Step 6: Create the v4.1 protocol schema**

Copy the scientific task/method identities into `joint_forecast_dispatch_formal_v4_1.json`, set schema `joint-forecast-dispatch-formal-v4.1`, and add exact fields for `invalid_run_registry`, `benchmark_rule_config`, `source_closure_file`, `source_manifest_required`, and the repaired Gate 0 schema. Treat the original v4 config as immutable legacy evidence.

- [ ] **Step 7: Run protocol and provenance tests**

Run the Task 1 tests plus `frame/tests/test_joint_dispatch_formal_protocol_v4.py`. Expected: PASS.

- [ ] **Step 8: Commit Task 1**

```powershell
git add -- frame/configs/joint_dispatch_invalid_runs_v4.json frame/configs/joint_forecast_dispatch_formal_v4_1.json frame/configs/formal_v4_source_closure_v4_1.txt frame/src/joint_dispatch/formal_v4_provenance.py frame/src/joint_dispatch/formal_protocol_v4.py frame/tests/test_joint_dispatch_formal_v4_provenance.py frame/tests/test_joint_dispatch_formal_protocol_v4.py
git commit -m "fix: freeze formal v4.1 source provenance"
```

---

### Task 2: Build a 2015--2018-only Formal-v4 Standard IES

**Files:**
- Create: `frame/configs/standard_ies_formal_v4_rules.yaml`
- Create: `frame/src/joint_dispatch/formal_v4_benchmark.py`
- Create: `frame/scripts/build_rsc_pf_formal_v4_benchmark.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_benchmark.py`
- Modify: `frame/configs/joint_forecast_dispatch_formal_v4_1.json`

**Interfaces:**
- Consumes: cleaned Kitakyushu 2015--2018 rows, frozen derivation rules, source hashes, and `scheduling_parameter_ledger_v2.csv`.
- Produces: `build_formal_v4_benchmark(frame, rules, ledger_hash) -> FormalV4Benchmark`, resolved YAML, and `FORMAL_V4_BENCHMARK_RECEIPT.json` below `run_root/gate0/benchmark/`.

- [ ] **Step 1: Write failing year-boundary and derivation tests**

Test exact rejection of any 2019/2020 row, correct P95-derived capacities, train-year counts, PV/WT rated-capacity derivation, source units, and parameter labels.

```python
def test_benchmark_rejects_selection_year(train_frame):
    bad = pd.concat([train_frame, selection_2019_row()])
    with pytest.raises(PermissionError, match="2015-2018"):
        build_formal_v4_benchmark(bad, RULES, LEDGER_HASH)
```

- [ ] **Step 2: Verify the tests fail**

Run the new test file. Expected: FAIL because the formal-v4 benchmark builder is absent.

- [ ] **Step 3: Freeze derivation rules without resolved values**

Copy the scientific rules from the legacy benchmark but not its 2015--2019 statistics or resolved capacities. Explicitly define every capacity ratio, efficiency, SOC/ramp convention, PV/WT energy-share rule, and source-label field.

- [ ] **Step 4: Implement the strict benchmark builder**

Require timestamps wholly within 2015--2018, calculate statistics only from those rows, resolve all device/PV/WT values, and include the raw dataset and ledger hashes. Refuse overwrite and write the resolved YAML and JSON receipt atomically.

- [ ] **Step 5: Add a legacy-isolation test**

Assert that `D:\Paper\standard_ies_benchmark_v1.yaml` is never modified and is not accepted as the repaired formal-v4 benchmark receipt.

- [ ] **Step 6: Run tests and a bounded builder smoke command**

Use a temporary run root and only 2015--2018. Expected: resolved benchmark with `source_years=[2015,2016,2017,2018]`; no 2019/2020 access.

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- frame/configs/standard_ies_formal_v4_rules.yaml frame/configs/joint_forecast_dispatch_formal_v4_1.json frame/src/joint_dispatch/formal_v4_benchmark.py frame/scripts/build_rsc_pf_formal_v4_benchmark.py frame/tests/test_joint_dispatch_formal_v4_benchmark.py
git commit -m "fix: rebuild formal v4.1 benchmark from train years"
```

---

### Task 3: Freeze Representative Capacity-audit Origins

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_capacity.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_capacity.py`
- Modify: `frame/configs/joint_forecast_dispatch_formal_v4_1.json`
- Modify: `frame/scripts/build_rsc_pf_formal_v4_data.py`

**Interfaces:**
- Consumes: valid 2015--2018 base series and the frozen capacity-stratification config.
- Produces: `select_capacity_origins(base, config) -> CapacityOriginManifest` containing exactly 500 deterministic origins and per-stratum coverage.

- [ ] **Step 1: Write failing representative-origin tests**

Test determinism, exactly 500 unique valid origins, coverage of every year/season, ordinary and high cooling, nonzero total cooling, high heating/electricity, and no timestamps outside 2015--2018.

- [ ] **Step 2: Verify the old first-500 behavior fails**

Add a regression fixture matching the January-zero-cooling pattern and assert that the old chronological-first selection is rejected.

- [ ] **Step 3: Implement deterministic stratified sampling**

Compute strata using training-only quantiles fixed before multiplier evaluation. Freeze these unique-origin quotas in config: 320 year-season-balanced origins (20 from each of 16 year×season cells, selected at evenly spaced timestamp ranks), 60 cooling top-decile origins, 40 heating top-decile origins, 40 electricity top-decile origins, and 40 uniformly timestamp-spaced origins from the remaining valid pool. Resolve ties by timestamp; when a ranked candidate is already selected, take the next eligible candidate in the same stratum. Fail if a quota cannot be filled or if the final manifest lacks weekday, weekend, and each six-hour time-of-day bin.

- [ ] **Step 4: Store the complete origin manifest**

Save actual indices, timestamps, stratum labels, demand summaries, source base hash and manifest SHA-256. Replace the current scalar `audit_origins=0` placeholder.

- [ ] **Step 5: Run Task 3 tests**

Expected: PASS and explicit proof that audited cooling demand is nonzero.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_capacity.py frame/tests/test_joint_dispatch_formal_v4_capacity.py frame/configs/joint_forecast_dispatch_formal_v4_1.json frame/scripts/build_rsc_pf_formal_v4_data.py
git commit -m "fix: stratify formal v4.1 capacity audit origins"
```

---

### Task 4: Rebuild the Capacity Audit with Correct Renewable Scaling

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_capacity.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_preflight.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_capacity.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_preflight.py`

**Interfaces:**
- Consumes: resolved formal-v4.1 benchmark, capacity-origin manifest, complete chronological 2015--2018 series, common PI-MPC solver, and multiplier list `[1.0,...,2.0]`.
- Produces: `run_capacity_audit(...) -> CapacityAuditReceipt` containing both the stratified diagnostic and complete chronological certificate, plus the selected `capacity_adequate_main` multiplier.

- [ ] **Step 1: Write failing audit tests**

Test identical origins for all multipliers, correctly rated PV/WT availability, fixed independent-window diagnostic states `SOC=0.5` and `previous_CHP_output=0`, nonzero-demand guard, per-stratum metrics, full chronological state carry, gap-only resets, smallest-passing-both selection, solver failure propagation, and mandatory failure when no multiplier passes both stages.

- [ ] **Step 2: Confirm current implementation fails the tests**

Expected failures include unit-rated PV/WT profiles, absent strata, and acceptance of a zero-cooling audit.

- [ ] **Step 3: Implement the common audit evaluator**

Use the resolved PV/WT rated capacities, frozen origin order, fixed diagnostic initial state, and the common LP. Record cooling shortage energy/rate overall and by stratum plus electric/heating feasibility diagnostics. Treat these 500 independent windows as a candidate filter only.

- [ ] **Step 4: Certify candidates on the full chronological training trajectory**

Evaluate candidates from smallest to largest over every valid 2015--2018 hour. Reset to `SOC=0.5` and `previous_CHP_output=0` at the series start and after a logged timestamp gap; otherwise carry both values between consecutive settled hours. Stop at the first candidate passing the frozen cooling shortage-energy and shortage-hour thresholds in both stages. If none passes, fail the audit and do not emit a selected multiplier.

- [ ] **Step 5: Build a deterministic capacity identity**

Hash the benchmark receipt, origin manifest, diagnostic initial states, full chronological timestamp range, state-reset log, solver identity, thresholds and both stages' scientific results. Exclude elapsed time, machine information and output paths from `capacity_scenario_hash`.

- [ ] **Step 6: Make capacity audit and protocol freeze mandatory blockers**

Add both names to the mandatory registry now, before the full preflight rewrite, and test `authorized_gate1=false` for either failure.

- [ ] **Step 7: Run capacity and preflight tests**

Expected: PASS; zero-cooling fixtures and failed audits must be denied.

- [ ] **Step 8: Commit Task 4**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_capacity.py frame/scripts/run_rsc_pf_formal_v4_preflight.py frame/tests/test_joint_dispatch_formal_v4_capacity.py frame/tests/test_joint_dispatch_formal_v4_preflight.py
git commit -m "fix: make capacity audit representative and mandatory"
```

---

### Task 5: Generate Realized-settled Causal Device Trajectories

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_history.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_history.py`
- Modify: `frame/scripts/build_rsc_pf_formal_v4_data.py`
- Modify: `frame/src/joint_dispatch/formal_v4_recourse.py`

**Interfaces:**
- Consumes: base train/selection series, capacity receipt, causal forecast rule, common LP, and `settle_first_step_v4`.
- Produces: `generate_settled_device_trajectory(...) -> SettledTrajectory` with raw 21-field hourly settlement, six activity fields, state transitions and audit metrics.

- [ ] **Step 1: Write failing causal-history tests**

Cover no future demand access, exactly one plan and one settlement per hour, realized-load balance, PV/WT availability, conversion identities, SOC recursion, CHP ramp carry, timestamp-gap reset, and status derivation from settled outputs.

```python
assert audit.max_balance_residual <= 1e-6
assert audit.future_label_reads == 0
assert np.array_equal(status, derive_device_status(settled_dispatch))
```

- [ ] **Step 2: Add a regression test for the current un-settled trajectory**

Construct different yesterday/current loads and prove that storing the LP plan directly violates realized balance.

- [ ] **Step 3: Implement the atomic transition**

Freeze the causal plan inputs before implementation: at decision hour `t`, use observed loads at `t-24:t-20` as the four-step seasonal-naive load plan and repeat PV/WT availability observed at `t-1` over the four planned hours. Current realized hour-`t` loads and renewables enter settlement only. Execute the first planned action, settle once, store `realized_dispatch`, and advance SOC/previous CHP only from the settled outcome.

- [ ] **Step 4: Implement the exact 48-hour causal bootstrap**

After a series start or timestamp gap, use hours 1--24 only to establish the seasonal-naive forecast; begin settled device transitions at hour 25; then require 24 settled device-history hours before emitting the first model sample at hour 49 (zero-based index 48). Repeat the burn-in after every gap. If December 2018 to January 2019 is continuous, carry SOC, previous CHP output, and the causal buffer across the boundary so 2019 selection may use past 2018 context, but never 2019 labels for fitting.

- [ ] **Step 5: Preserve raw units and exclusion rules**

Store all 21 settlement fields in the trajectory artifact; expose only the frozen first 17 plus six raw-derived statuses as model history. Keep slack/dump available for audit but excluded from model inputs.

- [ ] **Step 6: Add bounded real-data continuity tests**

Run a continuous July 2018 slice and a 2018--2019 boundary fixture. Assert all physical residuals, finite state advancement, first-origin index 48, gap reburn-in, allowed 2018 past context for early 2019 selection, and zero selection-label influence on fitted artifacts.

- [ ] **Step 7: Run history, recourse and state tests**

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_history.py frame/src/joint_dispatch/formal_v4_recourse.py frame/scripts/build_rsc_pf_formal_v4_data.py frame/tests/test_joint_dispatch_formal_v4_history.py
git commit -m "fix: settle causal device histories against realized loads"
```

---

### Task 6: Bind Data, Normalization, and C_ref into One Provenance Chain

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_artifacts.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_artifacts.py`
- Modify: `frame/src/joint_dispatch/formal_v4_data.py`
- Modify: `frame/src/joint_dispatch/formal_v4_objective.py`
- Modify: `frame/scripts/build_rsc_pf_formal_v4_data.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_data.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_objective.py`

**Interfaces:**
- Consumes: source manifest, benchmark/capacity receipts, settled trajectories and train/selection base series.
- Produces: immutable `train.npz`, `selection.npz`, `NORMALIZATION_RECEIPT.json`, `C_REF_RECEIPT.json`, and artifact-manifest hashes below one run root.

- [ ] **Step 1: Write failing hash-chain tests**

Tamper separately with benchmark, capacity receipt, trajectory, feature order, normalization, timestamps and `C_ref` source objectives. Each tamper must invalidate every direct descendant.

- [ ] **Step 2: Strengthen capacity-receipt validation**

Replace the boolean-only check with schema validation plus receipt SHA-256, selected multiplier, resolved-parameter hash and origin-manifest hash verification.

- [ ] **Step 3: Include provenance in every state hash**

Hash raw history arrays, origin timestamp, trajectory-rule version, trajectory hash and capacity receipt hash. Do not hash paths or normalized arrays.

- [ ] **Step 4: Materialize train and selection under the requested run root**

Remove fallback to `formal_v4_20260903`; require explicit `--run-root`. Reject sibling output roots, pre-existing files and any selection/evaluation split when its prerequisite receipt is absent.

- [ ] **Step 5: Fit and persist train-only normalization**

Record exact train timestamp range, feature order, source archive/capacity hashes and statistics. Assert that selection receives but never fits normalization.

- [ ] **Step 6: Calculate and persist training-only C_ref**

Evaluate all valid training origins, resumably and in chunks, with the common PI-LP and fixed step weights. Reject negative/nonfinite objectives and any 2019/2020 timestamp.

- [ ] **Step 7: Run data/objective/artifact tests**

Expected: PASS, including capacity-change invalidation and train-only timestamp checks.

- [ ] **Step 8: Commit Task 6**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_artifacts.py frame/src/joint_dispatch/formal_v4_data.py frame/src/joint_dispatch/formal_v4_objective.py frame/scripts/build_rsc_pf_formal_v4_data.py frame/tests/test_joint_dispatch_formal_v4_artifacts.py frame/tests/test_joint_dispatch_formal_v4_data.py frame/tests/test_joint_dispatch_formal_v4_objective.py
git commit -m "fix: bind formal v4 data normalization and objective scale"
```

---

### Task 7: Freeze Teacher-overlay and Stage-gradient Contracts

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_data.py`
- Modify: `frame/src/joint_dispatch/formal_v4_training.py`
- Modify: `frame/src/joint_dispatch/formal_v4_objective.py`
- Modify: `frame/configs/joint_forecast_dispatch_formal_v4_1.json`
- Test: `frame/tests/test_joint_dispatch_formal_v4_data.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_training.py`

**Interfaces:**
- Consumes: future Stage P checkpoint hash, capacity-bound archive/state hashes and frozen training curriculum.
- Produces: strict `SameInformationTeacherOverlay` validation and real-batch Stage P/S/J gradient receipts.

- [ ] **Step 1: Write failing stale-teacher tests**

Require checkpoint, train archive, capacity, normalization, timestamp, state and solver hashes. Change each one independently and assert rejection.

- [ ] **Step 2: Freeze one curriculum definition**

Move the curriculum into the v4.1 protocol config and remove conflicting code defaults. Freeze forecast weight at `1.0`; linearly reduce imitation from `1.0` at epoch 0 to `0.0` at epoch 18; linearly increase decision from `0.05` at epoch 0 to `1.0` at epoch 18; and keep forecast/imitation/decision at `1.0/0.0/1.0` after epoch 18. Specify the interpolation/boundary convention once and test epoch 0, an interior epoch, epoch 18, and later epochs exactly. In particular, reject the legacy `imitation_final=0.25` behavior.

- [ ] **Step 3: Preserve pre-Stage-P behavior**

Gate 0 must require `teacher_dispatch=None` and zero imitation when no Stage P checkpoint exists. It must reject any legacy teacher overlay rather than pretending it is current.

- [ ] **Step 4: Add real-materialized-batch gradient tests**

On a small train-only batch, prove Stage P updates only the forecaster, Stage S only the scheduler, Stage J joint has positive decision gradient to both, and Stage J decoupled has zero decision gradient to the forecaster.

- [ ] **Step 5: Run Task 7 tests**

Expected: PASS; the existing synthetic smoke tests remain as numerical checks but are no longer the only evidence.

- [ ] **Step 6: Commit Task 7**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_data.py frame/src/joint_dispatch/formal_v4_training.py frame/src/joint_dispatch/formal_v4_objective.py frame/configs/joint_forecast_dispatch_formal_v4_1.json frame/tests/test_joint_dispatch_formal_v4_data.py frame/tests/test_joint_dispatch_formal_v4_training.py
git commit -m "fix: freeze formal v4.1 teacher and gradient contracts"
```

---

### Task 8: Replace Metadata-only Baselines with Executable Adapters

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_baselines.py`
- Create: `frame/src/joint_dispatch/formal_v4_method_adapter.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_baseline.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_baselines.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_method_adapter.py`

**Interfaces:**
- Consumes: one formal-v4 window, rolling state, optional checkpoint and common decoder/LP.
- Produces: `FormalV4MethodAdapter.predict_and_dispatch(window, rolling_state) -> MethodStepResult` for every registered method.

- [ ] **Step 1: Write a shared adapter contract test**

For all nine method names, require forecast applicability, dispatch/demand shapes, finite next state, correct deployability/reference flags, optimizer calls and no future-label input except PI-MPC.

- [ ] **Step 2: Verify metadata-only baselines fail**

Expected: Scheme2R-PTO and State-Conditioned-PTO lack executable dispatch; the baseline runner only emits descriptors.

- [ ] **Step 3: Define `MethodStepResult` and the canonical adapter interface**

Use one result dataclass for forecast, dispatch, demand, target, next state, optimizer calls and component latency. Validate shapes and finite values centrally.

- [ ] **Step 4: Implement internal and reference adapters**

Wire RSC-PF, Decoupled-RSC-PF, Direct-Policy, Scheme2R-PTO, State-Conditioned-PTO, Seasonal-Naive-PTO and Perfect-Information-MPC to the common rolling state and settlement. PI-MPC alone receives realized future demand/renewables and remains nondeployable.

- [ ] **Step 5: Convert the baseline script from descriptor-only to bounded smoke execution**

Require a capacity-bound train archive and run a small train-only slice. Keep a separate `--describe` mode for metadata.

- [ ] **Step 6: Run adapter tests**

Expected: PASS and exact optimizer calls: zero for neural policies, one per window for PTO/PI methods.

- [ ] **Step 7: Commit Task 8**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_baselines.py frame/src/joint_dispatch/formal_v4_method_adapter.py frame/scripts/run_rsc_pf_formal_v4_baseline.py frame/tests/test_joint_dispatch_formal_v4_baselines.py frame/tests/test_joint_dispatch_formal_v4_method_adapter.py
git commit -m "fix: make formal v4 baselines executable"
```

---

### Task 9: Revalidate Official iTransformer and Differentiable-LP at Gate Time

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_itransformer.py`
- Modify: `frame/src/joint_dispatch/formal_v4_diffopt.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_diffopt_gate.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_itransformer.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_diffopt.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_method_adapter.py`

**Interfaces:**
- Consumes: current upstream source directory/receipt, isolated environment lock, train-only real batch and common adapter contract.
- Produces: current source/environment revalidation receipts and executable Official iTransformer-PTO/Differentiable-LP adapters.

- [ ] **Step 1: Write failing receipt-tamper tests**

Change one imported iTransformer source file, license hash, commit field, package version, environment executable, DPP status or gradient result and assert Gate 0 rejection.

- [ ] **Step 2: Make Gate-time iTransformer validation rehash source files**

Do not stop at `verified=true`. Instantiate the official backbone on a train-only batch and verify the `[B,4,4]` output.

- [ ] **Step 3: Re-run the Differentiable-LP numerical gate in the isolated environment**

Verify DPP, 100 finite solves, objective parity, residual at most `1e-6`, finite nonzero forecaster gradient, and current lock versions. Preserve SCS as the disclosed layer solver if ECOS remains unstable.

- [ ] **Step 4: Implement both external method adapters**

Official iTransformer-PTO performs one common LP call per window. Differentiable-LP performs one differentiable optimizer call per window and no neural scheduler call.

- [ ] **Step 5: Run Task 9 tests in both environments**

Expected: PASS in the PyTorch environment and in `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe` for DiffLP-specific tests.

- [ ] **Step 6: Commit Task 9**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_itransformer.py frame/src/joint_dispatch/formal_v4_diffopt.py frame/scripts/run_rsc_pf_formal_v4_diffopt_gate.py frame/tests/test_joint_dispatch_formal_v4_itransformer.py frame/tests/test_joint_dispatch_formal_v4_diffopt.py frame/tests/test_joint_dispatch_formal_v4_method_adapter.py
git commit -m "fix: revalidate formal v4 external adapters"
```

---

### Task 10: Repair Unified Closed-loop Evaluation and Optimizer Accounting

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_evaluation.py`
- Modify: `frame/src/joint_dispatch/formal_v4_method_adapter.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_evaluation.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_statistics.py`

**Interfaces:**
- Consumes: executable method adapter, chronological windows, rolling initial state and common realized settlement.
- Produces: forecast metrics, physical dispatch metrics, closed-loop trajectory totals, optimizer counts, component latency and paired statistical evidence.

- [ ] **Step 1: Write failing optimizer-count tests**

Prove that four PTO windows report four calls, four RSC-PF windows report zero, and a registered optimizer method cannot silently default to zero because of an attribute-name mismatch.

- [ ] **Step 2: Canonicalize the field name**

Use `online_optimizer_calls_per_window` throughout new code. Read `online_lp_calls_per_window` only through an explicit legacy adapter that raises if both fields disagree.

- [ ] **Step 3: Enforce chronological atomic rollout**

Require strictly increasing origins, no state reset, exactly one settlement per step, and returned next-state validation.

- [ ] **Step 4: Verify complete-trajectory PI comparisons**

Calculate differences only after aggregating matched closed-loop trajectories; prohibit per-window oracle-normalized regret.

- [ ] **Step 5: Add latency and method-role tests**

Validate forecast/scheduler/decoder/LP/recourse/end-to-end components and reference/deployable labels.

- [ ] **Step 6: Run evaluation and statistics tests**

Expected: PASS.

- [ ] **Step 7: Commit Task 10**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_evaluation.py frame/src/joint_dispatch/formal_v4_method_adapter.py frame/tests/test_joint_dispatch_formal_v4_evaluation.py frame/tests/test_joint_dispatch_formal_v4_statistics.py
git commit -m "fix: unify formal v4 closed loop accounting"
```

---

### Task 11: Add Access Logging and Resource Projection

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_access.py`
- Create: `frame/src/joint_dispatch/formal_v4_resources.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_access.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_resources.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_preflight.py`

**Interfaces:**
- Consumes: every formal-v4.1 input path/open event, archive-member event, 500 bounded real training operations, disk/memory information and method adapters.
- Produces: `DATA_ACCESS_RECEIPT.json`, `RESOURCE_PROJECTION.json`, and mandatory Gate 0 check results.

- [ ] **Step 1: Write failing access-control tests**

Attempt to open an evaluation/2020 artifact through every builder and adapter boundary and assert denial plus a logged blocked event. Attempt direct filesystem/NumPy/pandas/ZIP reads that bypass the canonical loader and require the static scan to fail. Confirm 2015--2018 access and read-only 2019 base preparation are classified correctly. For ZIP inputs, test separate hashes for the outer archive and exact opened member, and assert that Gate 0 never materializes an evaluation member.

- [ ] **Step 2: Implement centralized split-aware access**

Require builders, adapters, and Gate 0 components to obtain dataset paths through one canonical access controller with a runtime year-deny guard. Record split, years, purpose, path hash, caller and allow/deny decision. For archives, additionally record container hash, member path, member hash, and extraction/materialization status.

- [ ] **Step 3: Add an access-bypass static scan**

Scan the declared formal-v4.1 runtime closure for direct dataset-opening calls outside the canonical loader. Maintain a narrow reviewed allowlist for the loader itself. Any unapproved `open`, `Path.open`, pandas/NumPy loader, archive extraction, or equivalent data access blocks Gate 0. Require the evaluation archive/member to remain unmaterialized.

- [ ] **Step 4: Write failing resource-threshold tests**

Mock projected p95 above 24 hours and disk/memory margin below 20%; each case must block Gate 0.

- [ ] **Step 5: Implement bounded resource benchmarking**

Warm adapters, run fixed training-only slices, measure p50/p95 for required components, project one method/seed runtime, and report disk plus memory margins. Store machine-dependent results outside scientific hashes.

- [ ] **Step 6: Run Task 11 tests**

Expected: PASS.

- [ ] **Step 7: Commit Task 11**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_access.py frame/src/joint_dispatch/formal_v4_resources.py frame/scripts/run_rsc_pf_formal_v4_preflight.py frame/tests/test_joint_dispatch_formal_v4_access.py frame/tests/test_joint_dispatch_formal_v4_resources.py
git commit -m "fix: audit formal v4 access and resources"
```

---

### Task 12: Rewrite Gate 0 as a Transactional Fail-closed Orchestrator

**Files:**
- Create: `frame/src/joint_dispatch/formal_v4_gate0.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_preflight.py`
- Replace: `frame/tests/test_joint_dispatch_formal_v4_preflight.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_gate0_integration.py`

**Interfaces:**
- Consumes: committed source manifest and Tasks 2--11 builders/checks.
- Produces: one immutable `run_root/gate0/` tree, `GATE0_RECEIPT.json`, and `authorized_gate1` only after all mandatory checks pass.

- [ ] **Step 1: Define one exhaustive mandatory-check registry**

Include protocol/source freeze, tracked-and-clean runtime closure, invalid-run exclusion, benchmark boundary, stratified and full-chronology capacity certification, materialized data, trajectory physics/bootstrap, normalization, `C_ref`, teacher absence/alignment, exact curriculum, gradient boundaries, nine method adapters, iTransformer, DiffLP, data access/bypass scan, archive-member receipts, full tests and resources. Reject unknown or missing check IDs.

- [ ] **Step 2: Replace dummy-fixture tests with real fault injection**

Parameterize every registered check by injecting a failing component into the real orchestrator. Assert nonzero exit, diagnostic receipt and `authorized_gate1=false`.

```python
@pytest.mark.parametrize("failure", MANDATORY_CHECK_IDS)
def test_gate0_denies_every_mandatory_failure(gate0_fixture, failure):
    result = execute_gate0(gate0_fixture.with_failure(failure))
    assert not result.authorized_gate1
    assert result.checks[failure].passed is False
```

- [ ] **Step 3: Implement transactional ordering**

Create a new run directory, write an in-progress marker, execute source/benchmark/capacity/history/data/objective/adapters/tests/resources in dependency order, and write the final receipt last. On failure, retain diagnostics but never write an authorization token.

- [ ] **Step 4: Validate outputs rather than trusting booleans**

Each check reopens and hashes its artifacts, recalculates essential invariants, and records measured evidence. Remove all hard-coded `passed=true` branches.

- [ ] **Step 5: Prevent accidental reuse and overwrite**

Reject existing run IDs, invalid-run inputs, sibling data roots, legacy capacity receipts and stale smoke files. Ensure final authorization points only to artifacts within the same run root.

- [ ] **Step 6: Make the receipt satisfy the design contract**

Record all issue/check IDs, hashes, years, capacity result, state/normalization/`C_ref` identity, test counts, resource projection, access receipt and Git commit. `test_set_accessed=false` must be derived from the access log.

- [ ] **Step 7: Run Gate 0 unit and integration tests**

Expected: all pass; a clean bounded fixture authorizes and every injected failure denies.

- [ ] **Step 8: Commit Task 12**

```powershell
git add -- frame/src/joint_dispatch/formal_v4_gate0.py frame/scripts/run_rsc_pf_formal_v4_preflight.py frame/tests/test_joint_dispatch_formal_v4_preflight.py frame/tests/test_joint_dispatch_formal_v4_gate0_integration.py
git commit -m "fix: make formal v4 gate0 transactional and fail closed"
```

---

### Task 13: Repair the Complete Repository Regression Suite

**Files:**
- Modify: `frame/src/joint_dispatch/contract.py`
- Modify: `frame/scripts/run_joint_forecast_dispatch.py`
- Modify: `frame/tests/test_joint_dispatch_contract_v3.py`
- Modify: `frame/tests/test_joint_dispatch_runner.py`
- Modify: `frame/tests/test_scheme2r_proxy_carbon_tradeoff_runner.py`
- Create: `frame/scripts/run_repository_regression_v4.py`
- Create: `frame/tests/test_repository_regression_v4.py`

**Interfaces:**
- Consumes: canonical Git root and the complete `frame/tests` suite.
- Produces: one documented working-directory-independent regression command and immutable `REGRESSION_RECEIPT.json`.

- [ ] **Step 1: Reproduce and classify all current failures**

Record the current baseline: 835 passed, 14 failed, 5 skipped from the Git root; collection fails from `frame/`. Separate path-resolution failures from seal-fixture isolation failures.

- [ ] **Step 2: Write failing root-resolution tests**

Invoke contract loaders and CLI dry runs from both the Git root and `frame/`; require identical resolved config paths.

- [ ] **Step 3: Implement safe repository-relative path resolution**

Resolve known config defaults relative to `FRAME_ROOT`, not process CWD. Do not reinterpret arbitrary user paths or allow path escape.

- [ ] **Step 4: Isolate legacy seal fixtures**

When copying the live scientific contract into a fixture, explicitly clear every `SEAL_HASH_FIELDS` value before exercising first-seal behavior. Do not mutate the live contract.

- [ ] **Step 5: Prove legacy scientific behavior is unchanged**

Limit repairs to repository-relative path resolution, fixture isolation, or a demonstrated implementation regression. Before accepting each legacy-test change, compare the prior and repaired numerical outputs, expected metrics, and receipt/seal semantics on the same fixture. Do not alter frozen scientific calculations or expected results merely to obtain a passing test.

- [ ] **Step 6: Add the regression receipt runner**

Run the complete suite from `D:\Paper\github_work\paper-code`, capture command, environment, Git commit, pass/fail/skip counts and output hash. Any failure returns nonzero.

- [ ] **Step 7: Run the complete suite**

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' -m pytest frame\tests -q --basetemp D:\Paper\pytest_tmp_formal_v4_repair_full -p no:cacheprovider
```

Expected: zero failures and zero collection errors. Existing intentional skips must be enumerated and justified in the receipt.

- [ ] **Step 8: Commit Task 13**

```powershell
git add -- frame/src/joint_dispatch/contract.py frame/scripts/run_joint_forecast_dispatch.py frame/tests/test_joint_dispatch_contract_v3.py frame/tests/test_joint_dispatch_runner.py frame/tests/test_scheme2r_proxy_carbon_tradeoff_runner.py frame/scripts/run_repository_regression_v4.py frame/tests/test_repository_regression_v4.py
git commit -m "fix: restore complete repository regression suite"
```

---

### Task 14: Execute a Dry Gate 0, Freeze Repairs, and Run the New Gate 0

**Files:**
- Create: `frame/scripts/audit_rsc_pf_formal_v4_gate0.py`
- Create: `frame/tests/test_audit_rsc_pf_formal_v4_gate0.py`
- Modify only if a test exposes a defect: files owned by Tasks 1--13
- Output: `frame/reports/joint_forecast_dispatch_formal_v4_1/$RunId/gate0/`, where `$RunId` is generated once in Step 6.

**Interfaces:**
- Consumes: committed Tasks 1--13 implementation and a unique run ID supplied at execution time.
- Produces: failed dry-run evidence if defects remain, or a new successful Gate 0 receipt plus an independent audit receipt. It does not produce Gate 1 authorization outside the Gate 0 receipt and does not start training.

- [ ] **Step 1: Write the independent auditor tests**

The auditor must independently recompute hashes, reopen all artifacts, recalculate representative capacity coverage, sample physical-history residuals, verify test/resource/access receipts and deny any reference outside the run root.

- [ ] **Step 2: Implement the auditor without importing the Gate 0 decision function**

Shared low-level schemas/hash utilities are allowed; the auditor must not reuse the function that decided `authorized_gate1`.

- [ ] **Step 3: Test and commit the independent auditor before any final manifest**

Run the auditor unit tests and `git diff --check`, review the files, then commit the auditor. The later source-closure manifest must bind this commit and these exact auditor bytes; no auditor change is permitted between final manifest creation and independent verification.

```powershell
git add -- frame/scripts/audit_rsc_pf_formal_v4_gate0.py frame/tests/test_audit_rsc_pf_formal_v4_gate0.py
git commit -m "test: independently audit repaired formal v4.1 gate0"
```

- [ ] **Step 4: Run a deliberately nonauthorizing dry Gate 0**

Set `$DryRunId = "formal_v4_repair_dry_" + (Get-Date -Format "yyyyMMdd_HHmmss")` and run the bounded dry path with that ID. Confirm the orchestration path, artifact layout and failure receipts. Never reuse that ID.

- [ ] **Step 5: Review and commit dry-run fixes, then generate the final source manifest**

Run targeted tests, full regression, `git diff --check`, and commit only task-owned corrections. Recheck every path in `formal_v4_source_closure_v4_1.txt`: all must be tracked, committed, clean, and complete, including the auditor. Generate a fresh source manifest only after the final code commit. If any closure file changes afterward, regenerate the manifest and restart the final Gate 0 with a new run ID.

- [ ] **Step 6: Run the final Gate 0 with a fresh run ID**

From the Git root:

```powershell
$RunId = "formal_v4_1_repair_" + (Get-Date -Format "yyyyMMdd_HHmmss")
& 'D:\anaconda\envs\pytorch\python.exe' frame\scripts\run_rsc_pf_formal_v4_preflight.py --contract frame\configs\joint_forecast_dispatch_formal_v4_1.json --run-id $RunId
```

Expected: exit code 0 only after all mandatory artifacts/checks exist; no neural training process is launched.

- [ ] **Step 7: Run the independent audit**

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' frame\scripts\audit_rsc_pf_formal_v4_gate0.py --run-root (Join-Path 'frame\reports\joint_forecast_dispatch_formal_v4_1' $RunId)
```

Expected: `INDEPENDENT_GATE0_AUDIT.json` reports `verified=true`, reproduces every hash, and confirms no 2020 access.

- [ ] **Step 8: Perform the final acceptance review**

Confirm all of the following before reporting completion:

- old invalid runs are preserved and excluded;
- benchmark source years are exactly 2015--2018;
- capacity manifest has 500 representative origins and nonzero cooling coverage;
- the same selected capacity passes the 500-origin filter and complete chronological 2015--2018 certification;
- settled device histories have maximum physical residual at most `1e-6`;
- the 48-hour causal burn-in and 2018--2019 continuity rules are evidenced;
- train/selection, normalization and `C_ref` share one run root and complete hash chain;
- all nine adapters execute with correct optimizer counts;
- formal-v4 targeted and complete repository tests have zero failures;
- resource thresholds pass;
- 2020/evaluation access count is zero;
- no direct dataset-access bypass exists and no evaluation archive member was materialized;
- Gate 0 and independent audit both authorize/verify the same run.

---

## Execution Stop Point

Stop after Task 14 and report the repaired Gate 0 receipt, independent audit, selected capacity result, test counts, runtime projection and remaining limitations. Do not automatically enter Gate 1. A separate reviewed plan and explicit user instruction are required before any neural training begins.
