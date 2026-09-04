# RSC-PF Formal v4.1 Gate 0 Evidence Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and independently validate the eleven immutable evidence files required by the twelve currently failing formal-v4.1 Gate 0 checks, then stop before Gate 1 or any large-scale training.

**Architecture:** Keep evidence production separate from authorization. Typed producers write receipts into one fresh staging run; a separate preflight run imports and validates them without generating scientific evidence. Every receipt is bound by hashes to the frozen benchmark, split-specific materialization, source commit, and protocol. Failures are fail-closed and previous roots are never reused.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy/HiGHS, CVXPY/CVXPYlayers in the isolated `D:\Paper\envs\rsc_pf_diffopt_v4` environment, pytest, JSON/NPZ artifacts, PowerShell orchestration, Git.

## Global Constraints

- Work from `D:\Paper\github_work\paper-code`.
- Use `D:\anaconda\envs\pytorch\python.exe` for the main project and the Python executable inside `D:\Paper\envs\rsc_pf_diffopt_v4` only for Differentiable-LP.
- Preserve the user's unrelated dirty worktree changes. Stage and commit only files listed in the active task.
- Never reuse or overwrite an earlier run root or receipt. Receipt writers must be atomic and fail if the target exists.
- Gate 0 may run deterministic data construction, LP normalization, 100-window mechanism probes, one-batch gradient probes, and resource timing. It may not run Stage P/S/J training, baseline training, ablations, or evaluation.
- Allow data years 2015–2018 for training and 2019 for selection. Deny 2020 at the access-controller boundary. Exclude 2021.
- Preserve the fixed 24-hour lookback, 4-hour horizon, four forecast channels, DS-TCN state encoder, 17 continuous controls, 6 activity indicators, and 21-dimensional physical dispatch output.
- Treat gas as a standardized station-side aggregate operating prior, not a rigid terminal gas-demand balance.
- Use first-step decision weights `[0.5, 1/6, 1/6, 1/6]` and forecast task weights `[1.0, 1.0, 1.0, 0.25]`.
- The clean capacity/data rerun must reproduce the frozen multiplier 2.7. Any other selected value is a stop condition requiring scientific review.
- A failed check must leave `authorized_gate1=false`. Do not continue to Gate 1 automatically as part of this plan.
- Design authority: `frame/docs/superpowers/specs/2026-09-04-rsc-pf-gate0-evidence-closure-design.md`.
- Starting regression reference: 945 passed, 5 skipped, 0 failed at commit `33c4260`.

---

### Task 1: Add Typed Gate 0 Receipt Contracts and Replace Schema-Only Validation

**Files:**

- Create: `frame/src/joint_dispatch/formal_v4_gate0_evidence.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_preflight.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_gate0_evidence.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_preflight.py`

- [ ] **Step 1: Write failing tests for strict receipt validation**

Add tests that start from the smallest valid payload for each receipt and then mutate one scientifically material field. At minimum, assert rejection of:

- a trajectory receipt with `future_label_reads=1` or any physical residual greater than `1e-6`;
- a teacher receipt that pretends to contain a Stage P checkpoint hash before Gate 1;
- a curriculum receipt with zero early decision weight;
- a gradient receipt with zero joint forecaster gradient or nonzero decoupled forecaster gradient;
- a method matrix that omits any of the nine frozen methods;
- a data/archive receipt containing 2020 or 2021;
- a resource receipt with projected runtime over 24 hours or less than 20% free memory/disk.

Use the public validator interface:

```python
validate_gate0_receipt(
    receipt_name: str,
    payload: Mapping[str, Any],
    *,
    run_root: Path,
) -> None
```

Run:

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest frame\tests\test_joint_dispatch_formal_v4_gate0_evidence.py -q
```

Expected: FAIL because the module and validators do not yet exist.

- [ ] **Step 2: Implement immutable receipt utilities and typed validators**

Implement:

```python
GATE0_RECEIPT_SCHEMAS: dict[str, str]
write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None
load_gate0_receipt(path: Path, receipt_name: str) -> dict[str, Any]
validate_gate0_receipt(
    receipt_name: str,
    payload: Mapping[str, Any],
    *,
    run_root: Path,
) -> None
```

Use exact schemas:

```text
formal-v4.1-trajectory-receipt-v1
formal-v4.1-teacher-alignment-v1
formal-v4.1-curriculum-receipt-v1
formal-v4.1-gradient-receipt-v1
formal-v4.1-method-adapter-receipt-v1
formal-v4.1-data-access-v1
formal-v4.1-archive-access-v1
formal-v4.1-resource-projection-v1
```

For every hash field, require a lowercase 64-character SHA-256 string and verify referenced files under `run_root`. Reject path traversal and references outside the run root. Reuse the existing C-ref, iTransformer, and Differentiable-LP validators rather than duplicating them.

- [ ] **Step 3: Make preflight a content-aware pure validator**

Replace `_file_check` schema-only acceptance for the eight new typed receipts with `validate_gate0_receipt`. Keep `--seed-root` import behavior, but ensure it copies only a declared allowlist and never invokes evidence builders.

Add a regression test that monkeypatches every evidence-builder entry point to raise if called and confirms preflight performs validation only.

- [ ] **Step 4: Run targeted tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_gate0_evidence.py `
  frame\tests\test_joint_dispatch_formal_v4_preflight.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 1 files**

```powershell
git add -- `
  frame/src/joint_dispatch/formal_v4_gate0_evidence.py `
  frame/scripts/run_rsc_pf_formal_v4_preflight.py `
  frame/tests/test_joint_dispatch_formal_v4_gate0_evidence.py `
  frame/tests/test_joint_dispatch_formal_v4_preflight.py
git commit -m "feat: validate formal v4.1 gate0 evidence"
```

---

### Task 2: Instrument Canonical Dataset Reads and Enforce Split-Specific Access

**Files:**

- Modify: `frame/src/kitakyushu_pipeline.py`
- Modify: `frame/src/joint_dispatch/formal_v4_access.py`
- Modify: `frame/scripts/build_rsc_pf_formal_v4_data.py`
- Modify: `frame/scripts/run_rsc_pf_formal_v4_capacity_audit.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_archive_access.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_access.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_data.py`

- [ ] **Step 1: Write failing tests for actual read-boundary logging**

Create miniature ZIP fixtures for electricity, gas, and weather. Assert that the canonical reader emits one event per actual member read with:

```python
{
    "source_kind": "electricity" | "gas" | "weather",
    "container_path": str,
    "container_sha256": str,
    "member_name": str,
    "member_sha256": str,
    "year": int,
}
```

Test that separate calls for 2015–2018 and 2019 remain separate in their receipts. Test that a 2020 request is denied before opening or hashing the 2020 member.

Run:

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_archive_access.py `
  frame\tests\test_joint_dispatch_formal_v4_access.py `
  -q
```

Expected: FAIL because the canonical loader has no audit sink.

- [ ] **Step 2: Add an optional audit sink without changing legacy callers**

Extend the canonical reader with a keyword-only optional callback:

```python
def read_kitakyushu_canonical(
    data_dir: Path,
    years: Sequence[int],
    *,
    audit_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> CanonicalDataset
```

Emit the event after reading the exact member bytes and before parsing them. Keep return values and behavior unchanged when `audit_sink is None`.

- [ ] **Step 3: Extend the access controller**

Add methods that bind events to a split and reject forbidden years before I/O:

```python
request_years(split: Literal["train", "selection"], years: Sequence[int]) -> None
record_archive_event(split: str, event: Mapping[str, Any]) -> None
build_data_access_receipt() -> dict[str, Any]
build_archive_access_receipt() -> dict[str, Any]
```

Make `build_data_access_receipt()` record an explicit denied evaluation request without reading evaluation contents. Both receipts must state `test_set_accessed=false`.

- [ ] **Step 4: Split the formal data builder into two canonical reads**

Modify the data/capacity path so training data are read with years `[2015, 2016, 2017, 2018]` and selection data are read separately with `[2019]`. Do not pass `[2015, 2016, 2017, 2018, 2019]` to one loader call.

The final builder must write, atomically:

```text
protocol/DATA_ACCESS_RECEIPT.json
protocol/ARCHIVE_ACCESS_RECEIPT.json
```

- [ ] **Step 5: Run targeted and legacy data tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_archive_access.py `
  frame\tests\test_joint_dispatch_formal_v4_access.py `
  frame\tests\test_joint_dispatch_formal_v4_data.py `
  frame\tests\test_joint_dispatch_data.py `
  -q
```

Expected: all tests pass, including legacy callers without an audit sink.

- [ ] **Step 6: Commit only Task 2 files**

```powershell
git add -- `
  frame/src/kitakyushu_pipeline.py `
  frame/src/joint_dispatch/formal_v4_access.py `
  frame/scripts/build_rsc_pf_formal_v4_data.py `
  frame/scripts/run_rsc_pf_formal_v4_capacity_audit.py `
  frame/tests/test_joint_dispatch_formal_v4_archive_access.py `
  frame/tests/test_joint_dispatch_formal_v4_access.py `
  frame/tests/test_joint_dispatch_formal_v4_data.py
git commit -m "feat: audit formal v4.1 archive access"
```

---

### Task 3: Bind the Physical Trajectory Receipt to Final Materialization

**Files:**

- Modify: `frame/src/joint_dispatch/formal_v4_data.py`
- Modify: `frame/scripts/build_rsc_pf_formal_v4_data.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_trajectory_receipt.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_data.py`

- [ ] **Step 1: Write a failing trajectory-receipt test**

Build a tiny settled trajectory and assert that the generated receipt binds:

- `CAPACITY_FREEZE.json` hash;
- generated benchmark hash;
- train and selection archive hashes;
- `trajectory_id`, `trajectory_sha256`, and exact rule version `formal-v4.1-causal-realized-settlement-v1`;
- all fields from `TrajectoryAudit`;
- `future_label_reads=0` and maximum violations at most `1e-6`.

Also assert that changing the NPZ after receipt creation invalidates the receipt.

- [ ] **Step 2: Implement trajectory receipt construction**

Add:

```python
build_trajectory_receipt(
    *,
    run_root: Path,
    settled_trajectory: SettledTrajectory,
    train_archive_path: Path,
    selection_archive_path: Path,
    capacity_receipt_path: Path,
    benchmark_path: Path,
) -> dict[str, Any]
```

Do not recalculate or suppress any physical-audit field. Write the receipt only after both archives have been closed and hashed.

- [ ] **Step 3: Make data construction emit the receipt**

Write `protocol/TRAJECTORY_RECEIPT.json` in the same fresh staging root as the split access receipts. A failed physical audit must abort before receipt creation.

- [ ] **Step 4: Run tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_trajectory_receipt.py `
  frame\tests\test_joint_dispatch_formal_v4_data.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 3 files**

```powershell
git add -- `
  frame/src/joint_dispatch/formal_v4_data.py `
  frame/scripts/build_rsc_pf_formal_v4_data.py `
  frame/tests/test_joint_dispatch_formal_v4_trajectory_receipt.py `
  frame/tests/test_joint_dispatch_formal_v4_data.py
git commit -m "feat: bind formal trajectory evidence"
```

---

### Task 4: Produce C-ref and Same-Information Teacher Mechanism Evidence

**Files:**

- Modify: `frame/src/joint_dispatch/formal_v4_artifacts.py`
- Modify: `frame/src/joint_dispatch/formal_v4_data.py`
- Create: `frame/scripts/build_rsc_pf_formal_v4_objective_evidence.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_objective_evidence.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_data.py`

- [ ] **Step 1: Write failing C-ref lineage tests**

Assert that C-ref uses all training windows and no selection/evaluation windows; records the objective code hash, benchmark hash, capacity hash, train archive hash, sample count, median-floor rule, and exact step weights; and fails if any bound file changes.

- [ ] **Step 2: Write failing teacher-probe tests**

Use 100 fixed train-only windows. Assert timestamp/state alignment, four-hour input/output shape, finite LP result, no realized future information beyond the declared teacher target, and these mandatory fields:

```json
{
  "probe_only": true,
  "production_overlay_deferred_until_stage_p": true,
  "stage_p_checkpoint_sha256": null,
  "window_count": 100,
  "test_set_accessed": false
}
```

Reject a non-null or invented Stage P checkpoint hash at Gate 0.

- [ ] **Step 3: Implement the objective-evidence builder**

Create a CLI that accepts only explicit run-root paths:

```text
--run-root
--benchmark-path
--capacity-receipt
--train-archive
--selection-archive
```

It must write:

```text
C_REF_RECEIPT.json
protocol/TEACHER_ALIGNMENT_RECEIPT.json
```

Reuse `fit_c_ref_from_train_split`, `build_c_ref_receipt`, and the existing same-information teacher implementation. Do not train or save a neural model.

- [ ] **Step 4: Run tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_objective_evidence.py `
  frame\tests\test_joint_dispatch_formal_v4_data.py `
  frame\tests\test_joint_dispatch_formal_v4_objective.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 4 files**

```powershell
git add -- `
  frame/src/joint_dispatch/formal_v4_artifacts.py `
  frame/src/joint_dispatch/formal_v4_data.py `
  frame/scripts/build_rsc_pf_formal_v4_objective_evidence.py `
  frame/tests/test_joint_dispatch_formal_v4_objective_evidence.py `
  frame/tests/test_joint_dispatch_formal_v4_data.py
git commit -m "feat: generate formal objective evidence"
```

---

### Task 5: Produce Curriculum and Gradient-Boundary Evidence

**Files:**

- Modify: `frame/src/joint_dispatch/formal_v4_training.py`
- Create: `frame/scripts/build_rsc_pf_formal_v4_training_evidence.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_training_evidence.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_training.py`

- [ ] **Step 1: Write failing curriculum tests**

Probe the first, middle, and final Stage J epochs without optimizer steps. Require finite nonnegative values, gas weight 0.25, rigid-task weights 1.0, exact horizon weights, and a positive decision component at the first joint epoch.

- [ ] **Step 2: Write failing gradient-boundary tests**

Create two identical disposable model copies and a deterministic train-only mini-batch. Backpropagate the decision component once and assert:

```text
initial model hashes are identical
RSC-PF forecaster gradient norm > 0
RSC-PF scheduler gradient norm > 0
Decoupled-RSC-PF forecaster gradient norm == 0
Decoupled-RSC-PF scheduler gradient norm > 0
all recorded norms are finite
```

Also assert that no checkpoint file exists afterward.

- [ ] **Step 3: Implement probe helpers and the CLI**

Expose deterministic functions:

```python
probe_curriculum(contract: FormalV4Contract) -> dict[str, Any]
probe_gradient_boundary(
    contract: FormalV4Contract,
    batch: Mapping[str, Tensor],
    *,
    seed: int,
) -> dict[str, Any]
```

The CLI writes:

```text
protocol/CURRICULUM_RECEIPT.json
protocol/GRADIENT_RECEIPT.json
```

Mark both as `probe_only=true`. Use one batch and no optimizer step.

- [ ] **Step 4: Run tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_training_evidence.py `
  frame\tests\test_joint_dispatch_formal_v4_training.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 5 files**

```powershell
git add -- `
  frame/src/joint_dispatch/formal_v4_training.py `
  frame/scripts/build_rsc_pf_formal_v4_training_evidence.py `
  frame/tests/test_joint_dispatch_formal_v4_training_evidence.py `
  frame/tests/test_joint_dispatch_formal_v4_training.py
git commit -m "feat: certify formal joint gradient boundary"
```

---

### Task 6: Freeze and Probe the Nine-Method Adapter Matrix

**Files:**

- Modify: `frame/src/joint_dispatch/formal_v4_method_adapter.py`
- Create: `frame/scripts/build_rsc_pf_formal_v4_adapter_evidence.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_adapter_evidence.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_method_adapter.py`

- [ ] **Step 1: Write failing registry-completeness tests**

Define the expected method IDs exactly once in the test and require a one-to-one registry match. For each method verify:

- deployable versus nondeployable role;
- whether forecast output exists;
- forecast shape `[4, 4]` when applicable;
- physical dispatch shape `[4, 21]`;
- online optimizer calls per rolling window;
- forecast-metric applicability.

Required call policy:

```text
RSC-PF, Decoupled-RSC-PF, Direct-Policy: 0
Scheme2R-PTO, State-Conditioned-PTO, Official iTransformer-PTO: 1
Differentiable-LP, Perfect-Information-MPC, Seasonal-Naive-PTO: 1
```

Mark Perfect-Information-MPC nondeployable and Direct-Policy forecast metrics not applicable.

- [ ] **Step 2: Add a declarative method contract**

Implement:

```python
@dataclass(frozen=True)
class FormalV4MethodContract:
    method_id: str
    role: str
    deployable: bool
    produces_forecast: bool
    forecast_metrics_applicable: bool
    online_optimizer_calls_per_window: int
    expected_forecast_shape: tuple[int, int] | None
    expected_dispatch_shape: tuple[int, int]

FORMAL_V4_METHODS: Sequence[FormalV4MethodContract]
```

Existing adapter implementations must reference this registry rather than maintaining separate names or roles.

- [ ] **Step 3: Implement disposable adapter probes**

The builder loads one fixed train-only window, uses disposable untrained neural modules where a trained checkpoint would otherwise be required, and verifies interface execution only. It must never report accuracy or save model state.

Write `protocol/METHOD_ADAPTER_RECEIPT.json` with all nine method rows, source hashes, shapes, call counts, and `probe_only=true`.

- [ ] **Step 4: Run tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_adapter_evidence.py `
  frame\tests\test_joint_dispatch_formal_v4_method_adapter.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 6 files**

```powershell
git add -- `
  frame/src/joint_dispatch/formal_v4_method_adapter.py `
  frame/scripts/build_rsc_pf_formal_v4_adapter_evidence.py `
  frame/tests/test_joint_dispatch_formal_v4_adapter_evidence.py `
  frame/tests/test_joint_dispatch_formal_v4_method_adapter.py
git commit -m "feat: freeze formal v4.1 method adapters"
```

---

### Task 7: Repair and Freeze the Official iTransformer Source Receipt

**Files:**

- Modify: `frame/scripts/prepare_rsc_pf_official_itransformer_v4.py`
- Modify: `frame/src/joint_dispatch/formal_v4_itransformer.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_itransformer.py`
- Modify: `frame/third_party/iTransformer/README_RSC_PF_ADAPTATION.md`

- [ ] **Step 1: Write failing receipt-field tests**

Require schema `formal-v4.1-itransformer-source-v1`, field `license_file` rather than `license_path`, repository URL `https://github.com/thuml/iTransformer`, recorded Git commit, backbone `model.iTransformer.Model`, source-root-relative imported files, and hashes for every imported file plus the license.

Test that changing any imported file, changing Git HEAD, or omitting the license invalidates the receipt.

- [ ] **Step 2: Normalize preparation and verification**

Make the preparation script accept:

```text
--source-root frame/third_party/iTransformer
--output-receipt $Gate0EvidenceRoot/protocol/ITRANSFORMER_SOURCE_RECEIPT.json
```

It must verify the existing local checkout and write once. It must not clone, fetch, or overwrite. If the local source does not match the recorded THUML commit `c2426e68ca13f74aaec08045c5c724d8ad328124`, stop and report the mismatch.

- [ ] **Step 3: Document adaptation boundaries**

Update the adaptation README to distinguish official backbone code from RSC-PF data-shape, training, and PTO wrappers. State that the benchmark is `Official iTransformer-PTO adaptation`, not a claim of exact reproduction of the source paper's entire experimental pipeline.

- [ ] **Step 4: Run tests and a local dry verification**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest frame\tests\test_joint_dispatch_formal_v4_itransformer.py -q
& D:\anaconda\envs\pytorch\python.exe frame\scripts\prepare_rsc_pf_official_itransformer_v4.py --help
```

Expected: tests pass and help exits 0 without network access.

- [ ] **Step 5: Commit only Task 7 files**

```powershell
git add -- `
  frame/scripts/prepare_rsc_pf_official_itransformer_v4.py `
  frame/src/joint_dispatch/formal_v4_itransformer.py `
  frame/tests/test_joint_dispatch_formal_v4_itransformer.py `
  frame/third_party/iTransformer/README_RSC_PF_ADAPTATION.md
git commit -m "fix: freeze official itransformer provenance"
```

---

### Task 8: Bind the Differentiable-LP Gate to the Current Run Root

**Files:**

- Modify: `frame/scripts/run_rsc_pf_formal_v4_diffopt_gate.py`
- Modify: `frame/src/joint_dispatch/formal_v4_diffopt.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_diffopt_runner.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_diffopt.py`

- [ ] **Step 1: Write failing run-root binding tests**

Assert that the gate refuses:

- the legacy `D:\Paper\standard_ies_benchmark_v1.yaml` when a run-root benchmark is required;
- data outside the supplied staging root;
- selection/evaluation windows;
- output overwrite;
- a main-environment invocation when the isolated environment is required.

- [ ] **Step 2: Repair the gate CLI**

Require explicit arguments:

```text
--run-root
--benchmark-path
--train-archive
--output-receipt
--window-count 100
```

Load the generated formal-v4.1 benchmark and 100 deterministic train windows from the materialized train archive. Do not reread raw source data.

- [ ] **Step 3: Enforce eligibility thresholds**

The typed receipt must require:

```text
DPP compliant
100/100 finite forward probes
objective parity relative gap <= 1e-4
maximum physical residual <= 1e-6
finite nonzero gradient
native CVXPYlayers probe passed
free disk and memory >= 20%
projected runtime <= 24 hours
test_set_accessed = false
```

Write an ineligible diagnostic receipt when possible, but never mark it eligible after a partial pass.

- [ ] **Step 4: Run main-environment unit tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_diffopt_runner.py `
  frame\tests\test_joint_dispatch_formal_v4_diffopt.py `
  -q
```

Expected: all tests pass without importing CVXPYlayers into the main environment.

- [ ] **Step 5: Run isolated-environment smoke verification**

Resolve the isolated interpreter without modifying either environment:

```powershell
$DiffPy = "D:\Paper\envs\rsc_pf_diffopt_v4\python.exe"
& $DiffPy -c "import cvxpy, cvxpylayers, diffcp, ecos, torch; print('diffopt-ok')"
```

Expected: `diffopt-ok`. If this fails, stop and report Gate 0 blocked; do not install into the main environment under this plan.

- [ ] **Step 6: Commit only Task 8 files**

```powershell
git add -- `
  frame/scripts/run_rsc_pf_formal_v4_diffopt_gate.py `
  frame/src/joint_dispatch/formal_v4_diffopt.py `
  frame/tests/test_joint_dispatch_formal_v4_diffopt_runner.py `
  frame/tests/test_joint_dispatch_formal_v4_diffopt.py
git commit -m "fix: bind diffopt gate to formal run root"
```

---

### Task 9: Produce a Method-Level Resource Projection

**Files:**

- Modify: `frame/src/joint_dispatch/formal_v4_resources.py`
- Create: `frame/scripts/run_rsc_pf_formal_v4_resource_gate.py`
- Create: `frame/tests/test_joint_dispatch_formal_v4_resource_gate.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_resources.py`

- [ ] **Step 1: Write failing multi-method projection tests**

Require one row for every method that can dominate runtime, including online LP, neural forward/decoder, Official iTransformer-PTO, Differentiable-LP, and Perfect-Information-MPC. Verify 500 representative train-only operations, p50/p95 timing, projected total duration, peak memory, free disk, and free-memory/free-disk fractions.

- [ ] **Step 2: Extend the resource schema**

Use:

```python
@dataclass(frozen=True)
class MethodResourceProjection:
    method_id: str
    sample_count: int
    p50_seconds: float
    p95_seconds: float
    projected_hours: float
    peak_memory_bytes: int

@dataclass(frozen=True)
class ResourceProjectionReceipt:
    rows: Sequence[MethodResourceProjection]
    worst_case_method_id: str
    worst_case_projected_hours: float
    free_memory_fraction: float
    free_disk_fraction: float
```

Eligibility requires all finite values, 500 samples per row, worst-case duration at most 24 hours, and both free fractions at least 0.20.

- [ ] **Step 3: Implement the resource gate**

Use only train materialization. Import the Differentiable-LP timing summary from its already-frozen receipt rather than importing CVXPYlayers into the main environment. Write `protocol/RESOURCE_PROJECTION.json` immutably.

- [ ] **Step 4: Run tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_resource_gate.py `
  frame\tests\test_joint_dispatch_formal_v4_resources.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 9 files**

```powershell
git add -- `
  frame/src/joint_dispatch/formal_v4_resources.py `
  frame/scripts/run_rsc_pf_formal_v4_resource_gate.py `
  frame/tests/test_joint_dispatch_formal_v4_resource_gate.py `
  frame/tests/test_joint_dispatch_formal_v4_resources.py
git commit -m "feat: add formal resource projection gate"
```

---

### Task 10: Add a Fail-Closed Evidence Orchestrator and Freeze Source Closure

**Files:**

- Create: `frame/scripts/run_rsc_pf_formal_v4_gate0_evidence.py`
- Modify: `frame/configs/formal_v4_source_closure_v4_1.txt`
- Create: `frame/tests/test_joint_dispatch_formal_v4_gate0_orchestrator.py`
- Modify: `frame/tests/test_joint_dispatch_formal_v4_preflight.py`

- [ ] **Step 1: Write failing orchestration tests**

Test the exact stage order:

```text
1. benchmark/capacity/data materialization
2. trajectory/access/archive receipts
3. C-ref and teacher-alignment receipts
4. curriculum and gradient receipts
5. method-adapter receipt
6. official iTransformer source receipt
7. isolated Differentiable-LP gate
8. resource projection
9. receipt inventory validation
```

Monkeypatch each stage and assert that a failure stops all later stages, no authorization marker is written, and the partial root remains for audit.

- [ ] **Step 2: Implement the no-training orchestrator**

The CLI creates its own timestamped fresh root below:

```text
frame/reports/joint_forecast_dispatch_formal_v4_1/
```

It records the root in its final JSON output and never accepts an existing directory. It launches the Differentiable-LP child process with the isolated interpreter. It must not import any Gate 1 runner.

After materialization, assert:

```text
selected_capacity_multiplier == 2.7
train_window_count == 34959
selection_window_count == 8733
test_set_accessed == false
```

If either window count changes because a previously documented count is corrected by the split repair, stop and require an explicit audit instead of rewriting expectations.

- [ ] **Step 3: Update the source-closure allowlist**

Add every new or modified runtime file from Tasks 1–10 to `formal_v4_source_closure_v4_1.txt`. Do not add tests, reports, generated receipts, or prior run roots to the runtime closure unless the existing closure convention explicitly includes them.

- [ ] **Step 4: Run targeted orchestration tests**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest `
  frame\tests\test_joint_dispatch_formal_v4_gate0_orchestrator.py `
  frame\tests\test_joint_dispatch_formal_v4_preflight.py `
  -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 10 files**

```powershell
git add -- `
  frame/scripts/run_rsc_pf_formal_v4_gate0_evidence.py `
  frame/configs/formal_v4_source_closure_v4_1.txt `
  frame/tests/test_joint_dispatch_formal_v4_gate0_orchestrator.py `
  frame/tests/test_joint_dispatch_formal_v4_preflight.py
git commit -m "feat: orchestrate formal v4.1 gate0 evidence"
```

---

### Task 11: Run Full Regression Before Creating Scientific Evidence

**Files:**

- No source changes expected.
- Generated temporary files must use a short explicit base path outside the repository.

- [ ] **Step 1: Run the complete test suite**

```powershell
& D:\anaconda\envs\pytorch\python.exe -m pytest frame\tests `
  --basetemp D:\Paper\pytest_tmp_gate0_closure `
  -p no:cacheprovider `
  -q
```

Expected: zero failures. The pass count may exceed the 945-test baseline because this plan adds tests; the 5 existing environment-dependent skips must be explained if their count changes.

- [ ] **Step 2: Confirm no hidden evaluation or training entry point ran**

```powershell
rg -n "2020|Gate 1|Stage P|Stage S|Stage J|train_stage" `
  frame\scripts\run_rsc_pf_formal_v4_gate0_evidence.py `
  frame\scripts\run_rsc_pf_formal_v4_preflight.py
```

Expected: only denial rules, labels, or explicit guards; no call that starts training or opens evaluation data.

- [ ] **Step 3: Stop on any failure**

Do not waive or xfail a newly failing test. Repair the responsible task, rerun its targeted tests, commit the repair, and restart the complete regression.

---

### Task 12: Generate All Evidence in One Fresh Staging Root

**Files:**

- Generated only: a new timestamped root below `frame/reports/joint_forecast_dispatch_formal_v4_1/`.

- [ ] **Step 1: Run the evidence orchestrator**

```powershell
$MainPy = "D:\anaconda\envs\pytorch\python.exe"
$DiffPy = "D:\Paper\envs\rsc_pf_diffopt_v4\python.exe"
& $MainPy frame\scripts\run_rsc_pf_formal_v4_gate0_evidence.py `
  --diffopt-python $DiffPy `
  --reports-root frame\reports\joint_forecast_dispatch_formal_v4_1
```

Expected: the command prints a new staging-root path, reports all producer stages complete, reproduces capacity multiplier 2.7, and does not start Gate 1.

- [ ] **Step 2: Verify the exact receipt inventory**

The staging root must contain exactly these Gate 0 evidence files in addition to its benchmark/data/source manifests:

```text
C_REF_RECEIPT.json
protocol/TRAJECTORY_RECEIPT.json
protocol/TEACHER_ALIGNMENT_RECEIPT.json
protocol/CURRICULUM_RECEIPT.json
protocol/GRADIENT_RECEIPT.json
protocol/METHOD_ADAPTER_RECEIPT.json
protocol/ITRANSFORMER_SOURCE_RECEIPT.json
protocol/DIFFERENTIABLE_LP_GATE.json
protocol/DATA_ACCESS_RECEIPT.json
protocol/ARCHIVE_ACCESS_RECEIPT.json
protocol/RESOURCE_PROJECTION.json
```

Expected: eleven unique files satisfy twelve preflight checks because the data-access receipt also proves no evaluation access.

- [ ] **Step 3: Inspect scientific invariants before authorization**

Check the orchestrator summary for:

```text
capacity multiplier = 2.7
train years = 2015–2018
selection year = 2019
evaluation access = false
future label reads = 0
maximum physical residuals <= 1e-6
C-ref fitted from training only
teacher probe windows = 100
joint forecaster decision gradient > 0
decoupled forecaster decision gradient = 0
Differentiable-LP eligible = true
resource worst-case projection <= 24 h
```

If any value differs, stop. Do not edit the receipt by hand.

---

### Task 13: Run a Fresh Pure Gate 0 Validation and Independent Audit

**Files:**

- Generated only: one fresh preflight root and its independent audit files.

- [ ] **Step 1: Resolve the staging root from the orchestrator output**

Use the exact absolute staging-root path printed by Task 12. Do not select an older root by modification time.

- [ ] **Step 2: Run preflight in seed-import validation mode**

```powershell
$MainPy = "D:\anaconda\envs\pytorch\python.exe"
& $MainPy frame\scripts\run_rsc_pf_formal_v4_preflight.py `
  --seed-root $Gate0EvidenceRoot `
  --reports-root frame\reports\joint_forecast_dispatch_formal_v4_1
```

Here `$Gate0EvidenceRoot` is assigned directly from the Task 12 JSON output in the same execution session; it must not be inferred from a wildcard.

Expected: twelve of twelve checks pass and the new preflight root contains `GATE0_AUTHORIZATION.json` with `authorized_gate1=true`.

- [ ] **Step 3: Run the independent auditor**

```powershell
& $MainPy frame\scripts\audit_rsc_pf_formal_v4_gate0.py `
  --run-root $Gate0PreflightRoot
```

Assign `$Gate0PreflightRoot` from the preflight command's JSON output. Expected:

```text
verified = true
authorized_gate1 = true
forbidden_artifacts = []
```

- [ ] **Step 4: Prove that execution stopped before Gate 1**

Confirm the preflight root contains no checkpoint, training history, evaluation prediction, baseline result, ablation result, or Gate 1 completion marker. The only Gate 1-related artifact allowed is the boolean authorization decision.

- [ ] **Step 5: Record the handoff summary**

Report:

- staging-root absolute path;
- preflight-root absolute path;
- source commit;
- full regression counts;
- capacity multiplier and split window counts;
- twelve check results;
- independent-audit result;
- explicit statement: `Gate 1 eligible but not started`.

Do not start Gate 1 until the user reviews this handoff and separately authorizes the next experimental stage.

---

## Final Self-Review Checklist

- [ ] Every missing Gate 0 check has a named producer, typed validator, targeted test, and final receipt path.
- [ ] The eleven physical receipts and twelve logical checks are distinguished correctly.
- [ ] Data access is split into train and selection reads, and 2020 is denied before I/O.
- [ ] Archive container and member hashes are captured at the real read boundary.
- [ ] The teacher receipt is a Gate 0 mechanism probe and does not invent a Stage P checkpoint.
- [ ] Gradient evidence proves both RSC-PF coupling and Decoupled-RSC-PF blocking.
- [ ] All nine method roles are frozen and unambiguous.
- [ ] Official iTransformer provenance and adaptation boundaries are explicit.
- [ ] Differentiable-LP uses the isolated environment and current run-root artifacts.
- [ ] The 2.7 capacity result must reproduce or execution stops.
- [ ] Full regression precedes evidence generation.
- [ ] The final authorization comes from a separate pure-validation root and independent auditor.
- [ ] No task starts Gate 1, large training, evaluation, baseline experiments, or ablations.
- [ ] No unresolved marker, placeholder, or hand-edited scientific receipt remains.
