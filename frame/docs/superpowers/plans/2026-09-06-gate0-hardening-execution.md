# RSC-PF Gate 0 Hardening Implementation Plan

> **For agentic workers:** Execute the tasks inline with a test after each task. Do not start Gate 1 or load formal 2019/2020 arrays.

**Goal:** Make Gate 0 fail closed, exercise the real baseline implementations on fixed synthetic inputs, and emit an auditable protocol/resource receipt before any formal training is authorized.

**Architecture:** Gate 0 remains a data-free readiness check. Its authorization is the conjunction of native smoke success, protocol consistency, duration authorization, and the configured disk safety margin. Baseline smokes call the actual model/adaptor functions with deterministic fixtures; formal arrays remain inaccessible. Receipts record source/config hashes and differentiable-LP solver-quality evidence.

**Tech Stack:** Python 3, PyTorch, NumPy, CVXPY/CVXPYLayers, pytest, JSON receipts.

## Global Constraints

- Do not read or write formal 2019/2020 evaluation arrays during Gate 0.
- Do not lower the 20 GiB post-checkpoint disk reserve or the 24-hour runtime limit; the fractional free-space value is diagnostic only.
- Do not start Gate 1 or delete user files.
- Keep RSC-PF architecture and the 15-dimensional latent / 21-dimensional dispatch contract unchanged.
- Keep gas as a station-side auxiliary prior and rigid demand as electricity/cooling/heating only.

### Task 1: Make Gate 0 authorization fail closed

**Files:**
- Modify: `scripts/run_rsc_pf_complete_formal_gate0.py`
- Test: `tests/test_rsc_pf_complete_formal_gate0.py`

- [x] Add a test proving the frozen 20 GiB absolute reserve controls authorization while the fractional free-space value remains diagnostic.
- [x] Change `audit_gate0_receipt` to require `resource_projection.authorized is True`, with authorization based on the frozen absolute reserve.
- [x] Include the resource-projection failure reason in the audit receipt.
- [x] Run the Gate 0 tests and verify the existing pass/fail fixtures remain meaningful.

### Task 2: Replace metadata-only baseline smokes

**Files:**
- Modify: `scripts/run_rsc_pf_complete_formal_gate0.py`
- Modify: `tests/test_rsc_pf_complete_formal_gate0.py`
- Modify: `tests/test_joint_dispatch_formal_v4_adapter_evidence.py`

- [x] Make `smoke_policy` instantiate the actual `DirectPolicyModel` with `JointForecastDispatchModel._test_parameters()` and fixed state tensors, then assert finite 15-control and 21-dispatch outputs plus a finite backward pass.
- [x] Make `smoke_naive` call `seasonal_naive_24h` with a deterministic 24-hour history and verify the four-step causal forecast shape and finite values.
- [x] Keep the exact LP smoke on a fixed Standard-IES-compatible fixture and include feasibility residual evidence in its result.
- [x] Preserve `data_arrays_loaded=false` and assert no formal-year paths are touched.
- [x] Run targeted smoke tests.

### Task 3: Record differentiable-LP solver quality

**Files:**
- Modify: `scripts/run_rsc_pf_complete_formal_gate0.py`
- Modify: `tests/test_rsc_pf_complete_formal_gate0.py`

- [x] Capture solver warning/status information during the CVXPYLayers probe without failing on a warning alone.
- [x] Record `solver_status_counts`, maximum primal residual, maximum dual/residual proxy, and warning count in the receipt.
- [x] Require finite gradients and residuals below the configured smoke tolerance; make non-finite or over-tolerance results fail Gate 0.
- [x] Add assertions for the new receipt fields.

### Task 4: Reconcile formal protocol identifiers and hashes

**Files:**
- Modify: `configs/joint_forecast_dispatch_formal_v4_6.json`
- Modify: `configs/rsc_pf_complete_formal_v1.json`
- Modify: `scripts/run_rsc_pf_complete_formal_gate0.py`
- Test: `tests/test_rsc_pf_complete_formal_contract.py`

- [x] Use `Decoupled-RSC-PF` consistently in the formal v4.6 source config.
- [x] Make the v4.6 source contract’s DiffLP inference-call declaration match the complete-formal contract (one LP call per origin).
- [x] Add hashes for the frozen benchmark rules and the relevant model/decoder source files to the Gate 0 receipt.
- [x] Add a consistency test that rejects mismatched method names or DiffLP call counts.

### Task 5: Freeze and audit Direct-Policy feasibility adaptation

**Files:**
- Modify: `src/joint_dispatch/formal_v4_models.py`
- Modify: `configs/rsc_pf_complete_formal_v1.json`
- Test: `tests/test_joint_dispatch_formal_v4_2_gate2_training.py`

- [x] Give the state-conditioned feasibility projection a named protocol identifier and expose it in the model evidence.
- [x] Ensure the Direct-Policy adapter reports the frozen physical-decoder/projection identifier; do not silently apply a special post-processing path.
- [x] Retain the adversarial carried-CHP ramp test and freeze the identifier in the complete contract.

### Task 6: Run the bounded verification only

**Files:**
- No formal experiment outputs are authorized.

- [x] Run targeted Gate 0, contract, model, adapter, and DiffLP tests.
- [x] Run `py_compile` and `git diff --check`.
- [x] Run Gate 0 with a fresh run id and confirm the receipt explicitly blocks Gate 1 for the current disk margin.
- [x] Report the exact status and do not launch Gate 1.
