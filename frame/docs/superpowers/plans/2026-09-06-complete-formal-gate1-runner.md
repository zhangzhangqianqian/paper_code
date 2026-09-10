# Complete Formal Gate 1 Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the real complete-v1 Gate 1 runner that trains 35 stochastic rows, evaluates all 37 rows across 8,709 chronological 2019 origins, and emits a fail-closed Gate 1 transition.

**Architecture:** Reuse the verified v4.2 training and chronological settlement kernels behind a narrow adapter. The adapter copies only train/selection windows, checks hashes and year boundaries, trains each frozen row with the complete-v1 seed roster, evaluates the resulting model or deterministic reference on 2019, and writes complete-v1 receipts. The existing generic audit independently recomputes row accounting before authorization.

**Tech Stack:** Python 3, NumPy, PyTorch, SciPy/HiGHS, CVXPYLayers, pytest, JSON/NPZ receipts.

## Global Constraints

- No 2020 or 2021 array may be loaded during Gate 1.
- Train years are exactly `(2015, 2016, 2017, 2018)`; selection year is exactly `2019`.
- The Gate 1 matrix is exactly 37 rows: 7 stochastic methods × 5 seeds plus two deterministic references.
- `RSC-PF` must retain a non-zero decision-to-forecaster gradient; `Decoupled-RSC-PF` must record the detached decision-gradient boundary.
- `Direct-Policy` has no forecast metric; its finite forecast transport array is only an interface placeholder.
- Synthetic/smoke rows never receive `paper_result=true` and never authorize Gate 2.
- Existing user changes and the legacy v4.2 runner remain untouched unless an adapter import is required.

---

### Task 1: Define the complete-v1 Gate 1 data and row interfaces

**Files:**
- Create: `src/joint_dispatch/complete_formal_gate1.py`
- Test: `tests/test_complete_formal_gate1.py`

**Interfaces:**
- `Gate1RunConfig(contract_path: Path, gate0_transition_path: Path, source_run_root: Path, output_root: Path, run_id: str, smoke: bool = False)`
- `Gate1DataBundle(train: FormalV4WindowSplit, selection: FormalV4WindowSplit, normalization: NormalizationReceiptV42, parameters: Mapping[str, Any], lineage: Mapping[str, str])`
- `expected_gate1_rows(contract: CompleteFormalContract) -> tuple[MethodSeedKey, ...]`
- `load_gate1_data(config: Gate1RunConfig, contract: CompleteFormalContract) -> Gate1DataBundle`

- [ ] **Step 1: Write failing tests for exact 37-row accounting and strict year checks.**

```python
def test_complete_gate1_matrix_is_37_rows():
    contract = CompleteFormalContract.from_path(CONTRACT)
    rows = expected_gate1_rows(contract)
    assert len(rows) == 37
    assert rows == contract.expected_rows("gate1")

def test_gate1_data_rejects_non_2019_selection(tmp_path):
    contract = CompleteFormalContract.from_path(CONTRACT_PATH)
    config = Gate1RunConfig(CONTRACT_PATH, GATE0_TRANSITION, SOURCE_RUN, tmp_path, "invalid", True)
    with pytest.raises(PermissionError, match="2019"):
        load_gate1_data(config, contract)
```

- [ ] **Step 2: Run the focused tests and verify the new interface fails.**

Run: `pytest tests/test_complete_formal_gate1.py -q`

Expected: FAIL because the Gate 1 adapter is not implemented.

- [ ] **Step 3: Implement data loading and source-run validation.**

`load_gate1_data` must read only `<source_run_root>/gate1/TRAIN_WINDOWS.npz`,
`SELECTION_WINDOWS.npz`, and `NORMALIZATION.json`; validate that train target
years are 2015--2018, selection target years are 2019, lengths are 34,959
and 8,709, timestamps are strictly hourly, and all arrays are finite. Read
capacity values from the frozen complete contract and copy the current source
run's base parameters only after exact capacity equality is confirmed. Record
SHA-256 hashes for both NPZ files and normalization in `lineage`.

- [ ] **Step 4: Run the focused tests and verify they pass.**

Run: `pytest tests/test_complete_formal_gate1.py -q`

Expected: PASS for row count, year boundary, length, and lineage tests.

### Task 2: Add the explicit training-matrix adapter

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1.py`
- Test: `tests/test_complete_formal_gate1.py`

**Interfaces:**
- `train_gate1_matrix(data: Gate1DataBundle, contract: CompleteFormalContract, output_dir: Path, *, smoke: bool = False) -> Mapping[MethodSeedKey, TrainedMethodArtifact | None]`

- [ ] **Step 1: Write tests for seed coverage, checkpoint lineage, and smoke non-authorization.**

```python
def test_training_matrix_writes_all_stochastic_rows(contract, gate1_data, tmp_path):
    artifacts = train_gate1_matrix(gate1_data, contract, tmp_path, smoke=True)
    assert len([key for key in artifacts if key.method_id in STOCHASTIC_METHOD_IDS]) == 35
    assert all(value is not None for value in artifacts.values())

def test_smoke_artifacts_are_not_paper_results(contract, gate1_config):
    result = run_complete_gate1(gate1_config, smoke=True)
    assert result["paper_result"] is False and result["authorized_gate2"] is False
```

- [ ] **Step 2: Run the focused tests and verify they fail.**

Run: `pytest tests/test_complete_formal_gate1.py::test_training_matrix_writes_all_stochastic_rows -q`

Expected: FAIL because no matrix adapter exists.

- [ ] **Step 3: Implement the matrix adapter.**

Construct a `Gate2DataBundle` view from the complete Gate1 bundle and pass an
explicit `StageBudgetV42` mapped from the frozen 30-epoch, batch-64 protocol.
For every seed call `train_rsc_family`, `train_direct_policy`,
`train_scheme2r_pto`, `train_official_itransformer_pto`, and
`train_differentiable_lp`; register all seven stochastic artifacts under the
complete row directories. The official iTransformer receipt and
Differentiable-LP environment receipt must be explicit inputs and their hashes
must be copied to each row lineage. In smoke mode use a tiny in-memory subset
and one epoch, but mark all receipts `synthetic=true`, `paper_result=false`.

- [ ] **Step 4: Run focused tests and verify all 35 stochastic keys are present.**

Run: `pytest tests/test_complete_formal_gate1.py -q`

Expected: PASS; no missing seeds and no checkpoint is silently fabricated.

### Task 2b: Materialize the predeclared 2019-only search budget

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1.py`
- Test: `tests/test_complete_formal_gate1_real.py`

**Interfaces:**
- `select_gate1_hyperparameters(data: Gate1DataBundle, contract: CompleteFormalContract, output_dir: Path, *, smoke: bool = False) -> Mapping[str, Any]`

- [ ] **Step 1: Write the search receipt test.**

```python
def test_smoke_search_is_explicitly_skipped(contract, gate1_data, tmp_path):
    receipt = select_gate1_hyperparameters(gate1_data, contract, tmp_path, smoke=True)
    assert receipt["status"] == "smoke-skipped"
    assert (tmp_path / "GATE1_SEARCH.json").is_file()
```

- [ ] **Step 2: Run the test and verify it fails before implementation.**

Run: `pytest tests/test_complete_formal_gate1_real.py::test_smoke_search_is_explicitly_skipped -q`

Expected: FAIL because the search interface is not present.

- [ ] **Step 3: Implement the frozen search.**

For a full run, evaluate every value in the complete-v1 RSC grid and
Differentiable-LP learning-rate grid using seed 2026 and 2019 selection only.
Write each trial below `gate1/search/`, select the lowest finite,
physically-feasible penalized objective, and persist `GATE1_SEARCH.json`.
Use the selected values for all five stochastic seeds. Smoke mode writes a
non-authorizing `smoke-skipped` receipt and does not claim a selected result.

- [ ] **Step 4: Run the focused test.**

Run: `pytest tests/test_complete_formal_gate1_real.py::test_smoke_search_is_explicitly_skipped -q`

Expected: PASS.

### Task 3: Add chronological 2019 evaluation and row receipts

**Files:**
- Modify: `src/joint_dispatch/complete_formal_gate1.py`
- Test: `tests/test_complete_formal_gate1_real.py`

**Interfaces:**
- `evaluate_gate1_row(key: MethodSeedKey, artifact: Any | None, data: Gate1DataBundle, output_dir: Path, *, smoke: bool = False) -> Mapping[str, Any]`
- `evaluate_gate1_matrix(data: Gate1DataBundle, contract: CompleteFormalContract, output_dir: Path, artifacts: Mapping[MethodSeedKey, TrainedMethodArtifact | None], *, smoke: bool = False) -> Mapping[str, Mapping[str, Any]]`

- [ ] **Step 1: Write failing tests for chronology, LP call counts, and 8,709-origin receipts.**

```python
def test_row_receipt_has_selection_origin_count_and_expected_lp_calls(selection_fixture):
    row = evaluate_gate1_row(selection_fixture.key, selection_fixture.artifact, selection_fixture.data, selection_fixture.output_dir, contract=selection_fixture.contract, smoke=True)
    assert row["origin_count"] == len(selection_fixture.data.selection)
    assert row["inference_lp_calls"] == row["origin_count"] * selection_fixture.expected_calls
    assert row["evaluation_year_accessed"] is False
```

- [ ] **Step 2: Run the focused test to confirm failure.**

Run: `pytest tests/test_complete_formal_gate1_real.py -q`

Expected: FAIL because the evaluator is not implemented.

- [ ] **Step 3: Implement evaluation using the existing causal rollout.**

Build the registered adapter from each trained artifact, call the existing
chronological settlement engine over the selection split, reopen the NPZ, and
recompute metrics from reopened arrays. For deterministic rows use the
seasonal-naive adapter and the dedicated realized-information MPC reference.
Write `ROLLOUT.npz`, `METRICS.json`, and `ROW_RECEIPT.json` with checkpoint,
rollout, metrics, source, normalization, and contract hashes. Set
`complete=true`, `synthetic=false`, `paper_result=true` only for a non-smoke
full run. Require exact origin count 8,709 and expected optimizer-call count.

- [ ] **Step 4: Run the focused evaluator tests.**

Run: `pytest tests/test_complete_formal_gate1_real.py -q`

Expected: PASS with a tiny chronology and exact call accounting.

### Task 4: Add fail-closed CLI and independent Gate1 audit

**Files:**
- Create: `scripts/run_rsc_pf_complete_formal_gate1_real.py`
- Modify: `src/joint_dispatch/complete_formal_gate1.py`
- Test: `tests/test_run_rsc_pf_complete_formal_gate1_real.py`

**Interfaces:**
- `run_complete_gate1(config: Gate1RunConfig, *, smoke: bool = False) -> Mapping[str, Any]`
- CLI options: `--contract`, `--gate0-transition`, `--source-run`, `--output-root`, `--run-id`, `--smoke`.

- [ ] **Step 1: Write failing tests for Gate0 authorization and synthetic refusal.**

```python
def test_cli_refuses_gate0_without_authorized_gate1(tmp_path):
    with pytest.raises(PermissionError, match="Gate 0"):
        run_complete_gate1(config_with_bad_gate0(tmp_path))

def test_smoke_never_writes_authorized_gate2(gate1_config):
    result = run_complete_gate1(gate1_config, smoke=True)
    assert result["authorized_gate2"] is False
```

- [ ] **Step 2: Run tests and verify failure.**

Run: `pytest tests/test_run_rsc_pf_complete_formal_gate1_real.py -q`

Expected: FAIL because the real CLI is not present.

- [ ] **Step 3: Implement the CLI and transition.**

Before data loading, call `CompleteFormalContract.authorize`-equivalent Gate0
validation requiring `authorized_gate1_training=true`, matching contract hash,
`status=pass`, and no evaluation access. After training/evaluation, call
`audit_complete_formal(contract, rows, origin_count=contract.selection_origin_count,
stage="gate1")`, write `ROWS.json`, `GATE1_EVIDENCE.json`,
`GATE1_AUDIT.json`, and `protocol/GATE1_TRANSITION.json`. The transition must
include `synthetic`, `paper_result`, `evaluation_year_accessed`,
`test_set_accessed`, audit hash, row count, and lineage hashes. Any failure
writes `GATE1_FAILURE.json` and returns a non-zero CLI code.

- [ ] **Step 4: Run all focused and existing formal tests.**

Run: `pytest tests/test_complete_formal_gate1_real.py tests/test_rsc_pf_complete_formal_gate0.py tests/test_rsc_pf_complete_formal_contract.py tests/test_rsc_pf_complete_formal_execution.py -q`

Expected: PASS, with smoke mode explicitly unauthorized.

### Task 5: Final static audit and operator command

**Files:**
- Modify: `scripts/run_rsc_pf_complete_formal_gate1.py` (documentation only, if needed)
- Create: `docs/superpowers/reports/2026-09-06-complete-formal-gate1-implementation.md`

- [ ] **Step 1: Run syntax, diff, and complete formal regression checks.**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m py_compile src/joint_dispatch/complete_formal_gate1.py scripts/run_rsc_pf_complete_formal_gate1_real.py
pytest tests/test_complete_formal_gate1_real.py tests/test_rsc_pf_complete_formal_gate0.py tests/test_rsc_pf_complete_formal_contract.py tests/test_rsc_pf_complete_formal_execution.py tests/test_rsc_pf_complete_formal_providers.py tests/test_rsc_pf_complete_formal_training.py -q
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [ ] **Step 2: Document the full-run command without starting the 18--23 hour experiment.**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts/run_rsc_pf_complete_formal_gate1_real.py `
  --contract configs/rsc_pf_complete_formal_v1.json `
  --gate0-transition reports/rsc_pf_complete_formal/complete_formal_gate0_20260906_i/gate0/GATE0_TRANSITION.json `
  --source-run reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_j `
  --output-root reports/rsc_pf_complete_formal `
  --run-id complete_formal_gate1_20260906_a
```

- [ ] **Step 3: Record that implementation is complete but the full run is operator-started.**

The code must not silently launch the long experiment during tests or module
import. The final report must state whether the full command was run and must
never call a smoke result a paper result.
