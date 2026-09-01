# RSC-PF Common Evaluation and Comparison Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate RSC-PF and the frozen external baselines on one common validation-window protocol and produce a directly comparable, provenance-checked result table.

**Architecture:** Keep the previously frozen external-baseline artifacts unchanged. Add a read-only evaluator for the RSC-PF checkpoints that uses the same 24-hour history, four-hour horizon, 21-variable dispatch representation, realized-demand shortage calculation, and metric definitions already used by the external evaluator. Aggregate the five RSC-PF seeds with the external receipts while preserving optimizer-role and unit metadata.

**Tech Stack:** Python 3.9, PyTorch, NumPy, CSV/JSON receipts, pytest.

## Global Constraints

- Use the train-fitted normalization only; never fit or read normalization from the sealed test split.
- Use validation data only; no path containing `test`, `2021`, or a sealed test artifact may be opened.
- Preserve the frozen external-baseline registry, checkpoints, source receipt, and validation freeze.
- Use the same 24-hour history, four-hour horizon, four-task order, 21 dispatch outputs, and 5 seeds `[2026, 2027, 2028, 2029, 2030]`.
- Report costs as per-window four-hour operating-cost sums and carbon as per-window physical-carbon sums; do not mix them with closed-loop cumulative totals.
- Record that RSC-PF and DigitalTwins-Policy use zero exact LP calls at inference, while iTransformer-PTO and DecisionFocused-Online use one exact LP per validation window.

---

### Task 1: Add common-evaluation contracts and checkpoint loading tests ✅

**Files:**
- Modify: `tests/test_rsc_pf_common_evaluation.py`
- Test: `tests/test_rsc_pf_common_evaluation.py`

**Interfaces:**
- Consumes: `external_validation_manifest.json`, `JointForecastDispatchModel`, train/validation split artifacts.
- Produces: failing tests for the common RSC-PF receipt schema, exact validation-window count, checkpoint provenance, and optimizer-role accounting.

- [x] **Step 1: Write the failing tests**

```python
def test_common_evaluation_rejects_test_paths(tmp_path):
    with pytest.raises(ValueError):
        evaluate_rsc_pf_validation(2026, config, data_root=tmp_path / "test")

def test_common_evaluation_receipt_has_frozen_contract():
    receipt = evaluate_rsc_pf_validation(2026, config)
    assert receipt["test_set_accessed"] is False
    assert receipt["metrics"]["windows"] == 8780
    assert receipt["exact_lp_calls"] == 0
    assert receipt["optimizer_role"] == "none at inference"
```

- [x] **Step 2: Run the focused test to verify it fails**

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' -m pytest tests/test_rsc_pf_common_evaluation.py -q
```

Expected: FAIL because the evaluator interface is not yet defined.

### Task 2: Implement read-only RSC-PF validation evaluation ✅

**Files:**
- Create: `scripts/evaluate_rsc_pf_common_validation.py`
- Modify: `tests/test_rsc_pf_common_evaluation.py`

**Interfaces:**
- Consumes: `reports/joint_forecast_dispatch_v1/data/{train,validation}.npz`, `reports/joint_forecast_dispatch_v2/validation/{From-Scratch-Joint,Warm-Start-Joint}/seed_*/best_checkpoint.pt`, `configs/joint_forecast_dispatch_contract_v2.json`.
- Produces: `evaluate_rsc_pf_validation(seed, config) -> dict[str, Any]` and one JSON/NPZ receipt per RSC-PF seed under `reports/rsc_pf_external_baselines_v1/implementation/validation/RSC_PF/seed_<seed>/evaluation/`.

- [x] **Step 1: Load train-fitted normalization and reject test-like paths**
- [x] **Step 2: Reconstruct `JointForecastDispatchModel` and load the selected RSC-PF checkpoint with provenance checks**
- [x] **Step 3: Run batched forward passes on all 8,780 validation windows and collect `[N,4,4]` forecasts and `[N,4,21]` direct dispatch**
- [x] **Step 4: Reuse the common forecast and dispatch metric definitions and write finite-output/test-isolation fields**
- [x] **Step 5: Run the focused tests and relevant external-regression tests**

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' -m pytest tests/test_rsc_pf_common_evaluation.py tests/test_rsc_pf_external_baseline_runner.py tests/test_joint_dispatch_model.py tests/test_joint_dispatch_data.py -q
```

### Task 3: Build the merged five-seed comparison table ✅

**Files:**
- Modify: `scripts/evaluate_rsc_pf_common_validation.py`
- Modify: `tests/test_rsc_pf_common_evaluation.py`

**Interfaces:**
- Consumes: 5 RSC-PF receipts and the frozen 15-entry external validation manifest.
- Produces: `common_validation_manifest.json`, `common_validation_summary.csv`, and method-level mean/SD table with optimizer roles, LP-call counts, and test-isolation status.

- [x] **Step 1: Test that all four methods cover five seeds and exactly 8,780 windows per seed**
- [x] **Step 2: Aggregate means and seed SDs without re-ranking or changing checkpoints**
- [x] **Step 3: Label the metric unit/protocol and preserve per-seed source receipt paths**
- [x] **Step 4: Add a command-line entry point that runs RSC-PF evaluation and table aggregation without reading test data**

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' scripts/evaluate_rsc_pf_common_validation.py --all-seeds --build-table
```

### Task 4: Self-review and hand off to figures/manuscript results ✅

**Files:**
- Create: `reports/rsc_pf_external_baselines_v1/implementation/task8_common_evaluation_self_review.json`
- Create: `reports/rsc_pf_external_baselines_v1/implementation/common_validation_handoff.md`

**Interfaces:**
- Consumes: all common-validation receipts, frozen external-baseline freeze receipt, and test results.
- Produces: a self-review receipt documenting tests, artifact hashes, metric units, optimizer roles, and residual risks.

- [x] **Step 1: Verify all receipts are finite and test-set-free**
- [x] **Step 2: Verify RSC-PF and external metrics are on the same open-loop window protocol**
- [x] **Step 3: Record that closed-loop cumulative v2 totals must not be mixed into this table**
- [x] **Step 4: Run `git diff --check` and commit only owned evaluator/tests/receipts**

### Task 5: Render the common comparison figure with publication QA ✅

**Files:**
- Create: `scripts/plot_rsc_pf_common_validation.py`
- Create: `tests/test_plot_rsc_pf_common_validation.py`
- Create: `reports/rsc_pf_external_baselines_v1/implementation/figures/rsc_pf_common_comparison_v1/figure_contract.json`

**Interfaces:**
- Consumes: `common_validation_summary.csv` and the common validation manifest.
- Produces: one 2×4 quantitative comparison figure with editable SVG/PDF, 600-dpi TIFF/PNG, source-data copy, and a machine-readable figure manifest.

- [x] **Step 1: Test source-table loading, method order, and five-seed error-bar fields**
- [x] **Step 2: Implement a Python/matplotlib quantitative-grid figure with forecast panels and decision panels**
- [x] **Step 3: Run the nature-figure source validator and render outputs**
- [x] **Step 4: Inspect the final-size PNG and record reviewer risks and statistics metadata**
- [x] **Step 5: Commit only the plotting source, tests, and plan changes**

```powershell
& 'D:\anaconda\envs\pytorch\python.exe' -m pytest tests/test_plot_rsc_pf_common_validation.py -q
& 'D:\anaconda\envs\pytorch\python.exe' C:\Users\张骞\.codex\skills\nature-figure\scripts\validate_figure.py scripts/plot_rsc_pf_common_validation.py --json
```

## Stop rules

Stop if an RSC-PF checkpoint cannot be loaded with an auditable state dict, the validation window count differs, metric units cannot be aligned, a test path is opened, or an optimizer-role/LP-call count is inconsistent. Do not expose or evaluate the sealed test split in this phase.
