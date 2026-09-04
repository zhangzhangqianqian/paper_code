# RSC-PF Formal v4.1 Capacity Certification Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the formal v4.1 training-only cooling-capacity certification search so that the benchmark can select the smallest physically feasible multiplier without changing the pre-registered shortage thresholds or touching selection/evaluation data.

**Architecture:** Keep the existing uniform multiplier semantics for the electric and absorption chillers, but extend only the v4.1 candidate grid from 1.0--2.0 to 1.0--3.0 in increments of 0.1. Make the protocol validator schema-specific so legacy formal v4 remains byte-for-byte compatible. Re-run protocol tests, the complete repository regression, the training-only capacity audit, and a fresh fail-closed Gate 0; no training is allowed in this plan.

**Tech Stack:** Python 3.9, NumPy, pandas, SciPy HiGHS LP, PyYAML, pytest, JSON/NPZ evidence receipts, PowerShell.

## Global Constraints

- Training years are exactly 2015--2018; 2019 is selection-only and 2020 remains inaccessible before Gate 3.
- Cooling shortage thresholds remain `cooling_shortage_energy_ratio_max=0.005` and `cooling_shortage_hour_rate_max=0.01`.
- Formal v4 legacy configuration and receipts must not be modified.
- Capacity selection uses both the frozen 500-origin stratified audit and complete chronological 2015--2018 certification.
- Existing failed Gate 0 receipts remain immutable; every new run uses a fresh run ID.
- Do not start Gate 1, neural training, ablations, or manuscript edits.

---

### Task 1: Version the v4.1 Capacity Candidate Grid

**Files:**
- Modify: `frame/configs/joint_forecast_dispatch_formal_v4_1.json`
- Modify: `frame/src/joint_dispatch/formal_protocol_v4.py:362-365`
- Test: `frame/tests/test_joint_dispatch_formal_protocol_v4.py`

**Interfaces:**
- Consumes: the existing formal v4/v4.1 JSON contracts.
- Produces: schema-specific validation where v4 keeps `[1.0,...,2.0]` and v4.1 accepts `[1.0,1.1,...,3.0]`.

- [ ] **Step 1: Add a failing v4.1 contract test**

Add a test that loads `joint_forecast_dispatch_formal_v4_1.json`, asserts its candidate list equals `tuple(1.0 + 0.1 * i for i in range(21))`, and asserts the legacy v4 contract still equals `tuple(1.0 + 0.1 * i for i in range(11))`.

- [ ] **Step 2: Run the focused protocol test and observe the failure**

Run `D:\anaconda\envs\pytorch\python.exe -B -m pytest frame/tests/test_joint_dispatch_formal_protocol_v4.py -q -p no:cacheprovider`.

Expected: the new v4.1 assertion fails because the config has only eleven candidates.

- [ ] **Step 3: Update only the v4.1 JSON list**

Replace the v4.1 `capacity.candidate_multipliers` array with the 21 values from `1.0` through `3.0` at `0.1` increments. Leave both shortage thresholds and every other field unchanged.

- [ ] **Step 4: Make validator expectations schema-specific**

Use `range(21)` when `schema_version == SCHEMA_VERSION_V41`, and `range(11)` for legacy `SCHEMA_VERSION`. Preserve the existing threshold checks and error behavior.

- [ ] **Step 5: Run the focused protocol tests**

Run the command from Step 2.

Expected: all protocol tests pass, including explicit proof that legacy v4 still accepts only its original grid.

- [ ] **Step 6: Commit the contract repair**

```powershell
git add -- frame/configs/joint_forecast_dispatch_formal_v4_1.json frame/src/joint_dispatch/formal_protocol_v4.py frame/tests/test_joint_dispatch_formal_protocol_v4.py
git commit -m "fix: extend formal v4.1 capacity certification grid"
```

### Task 2: Re-run Regression and Static Closure Checks

**Files:**
- Use: `frame/scripts/run_repository_regression_v4.py`
- Use: `frame/configs/formal_v4_source_closure_v4_1.txt`
- Test: `frame/tests/test_repository_regression_v4.py`

**Interfaces:**
- Consumes: the committed Task 1 contract and validator.
- Produces: a current immutable regression receipt and a source manifest that binds the new commit.

- [ ] **Step 1: Run the complete repository regression**

Run `D:\anaconda\envs\pytorch\python.exe -B frame/scripts/run_repository_regression_v4.py --output frame/reports/joint_forecast_dispatch_formal_v4_1/capacity_grid_repair/audit/REGRESSION_RECEIPT.json` from `D:\Paper\github_work\paper-code`.

Expected: zero failures and zero collection errors; intentional skips remain enumerated.

- [ ] **Step 2: Verify the v4.1 source closure**

Run the focused preflight/closure tests and confirm the closure file still contains every required tracked path, including the changed config, validator, and protocol test.

Expected: closure status `pass`; no 2020/evaluation path is added.

- [ ] **Step 3: Commit only task-owned test/receipt adjustments if required**

If a regression exposes a path or receipt defect, repair only the affected task-owned file, rerun the focused test, and commit it before capacity execution. Do not edit scientific thresholds or legacy outputs.

### Task 3: Re-certify Training-only Cooling Capacity

**Files:**
- Use: `frame/src/joint_dispatch/formal_v4_capacity.py`
- Use: `frame/scripts/build_rsc_pf_formal_v4_data.py`
- Output: a fresh run-root `gate0/benchmark/FORMAL_V4_BENCHMARK_RECEIPT.json`, `gate0/CAPACITY_FREEZE.json`, and related receipts.

**Interfaces:**
- Consumes: 2015--2018 canonical data, the v4.1 benchmark rules, and the expanded candidate grid.
- Produces: an immutable capacity audit with the smallest candidate passing both stratified and chronological checks.

- [ ] **Step 1: Build the 2015--2018 benchmark base**

Run the existing base-builder in a new run root and verify that the benchmark receipt reports exactly source years 2015, 2016, 2017, and 2018, with no selection or evaluation year used for scaling.

- [ ] **Step 2: Run the expanded candidate audit**

Evaluate the 21 candidates `[1.0,...,3.0]` with unchanged thresholds, using the same 500 frozen origins and complete chronological stream. Do not write a capacity freeze receipt until the audit returns `status=pass` and a non-null selected candidate.

- [ ] **Step 3: Inspect the selected candidate**

Confirm that the first passing candidate is at least the observed 2.7 grid point, that its stratified and chronological rows both satisfy the thresholds, and that the receipt records solver identity, source hashes, timestamps, and capacity scenario hash.

- [ ] **Step 4: Preserve the old failed certificate**

Do not overwrite any prior `CAPACITY_FREEZE.json` or failed Gate 0 receipt. Store the new certificate only under the fresh run root.

### Task 4: Rebuild Evidence and Run Fresh Gate 0

**Files:**
- Use: `frame/scripts/run_rsc_pf_formal_v4_preflight.py`
- Use: `frame/scripts/audit_rsc_pf_formal_v4_gate0.py`
- Use: `frame/scripts/run_repository_regression_v4.py`
- Output: `frame/reports/joint_forecast_dispatch_formal_v4_1/<fresh_run_id>/`

**Interfaces:**
- Consumes: the newly certified capacity and all existing v4.1 evidence contracts.
- Produces: either a complete Gate 0 authorization plus independent verification, or a fail-closed diagnostic. It never starts training.

- [ ] **Step 1: Materialize settled histories and train-only normalization**

After the capacity receipt is authorized, run the existing materialization path for train and selection splits, preserving causal continuity and writing all receipts under the same fresh run root.

- [ ] **Step 2: Run targeted and complete regression again**

Require zero failures before Gate 0. Record the current Git commit and output hash in the run-root regression receipt.

- [ ] **Step 3: Run Gate 0 with a new run ID**

Run the fail-closed preflight from the Git root with the v4.1 contract. Confirm every mandatory check passes before any authorization marker is accepted.

- [ ] **Step 4: Run the independent Gate 0 auditor**

Run the independent auditor against the same run root. Require `verified=true`, matching receipt hash, no forbidden artifacts, and zero evaluation-data access.

- [ ] **Step 5: Acceptance review and stop**

Report the selected capacity, test counts, Gate 0 receipt, independent audit, and any remaining limitations. Stop before Gate 1 and before all neural training.
