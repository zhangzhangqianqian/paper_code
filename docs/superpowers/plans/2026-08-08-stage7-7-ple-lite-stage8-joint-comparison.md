# Stage 7.7 PLE-lite and Stage 8 Joint Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a lightweight two-level PLE baseline, restore Hard-Share and Dynamic Symmetric to the formal multi-task comparison, and refactor Stage 8 so its main table compares joint models while STL-H4 remains only the matched negative-transfer reference.

**Architecture:** PLE-lite reuses the existing `[loads, historical exogenous]` external-baseline interface and stacks two CGC layers with shared and task-specific experts. A new Stage 7.7 addendum contract runs three already-fixed joint baselines without modifying Stage 6.6 or existing Stage 7 results. Stage 8 loads accepted runs by explicit role, produces a joint-model leaderboard, and keeps Scheme2R-H4 versus STL-H4 transfer analysis separate.

**Tech Stack:** Python 3.9, PyTorch, NumPy, pandas, unittest, JSON/CSV manifests, PowerShell.

## Global Constraints

- Dataset: `kitakyushu_energy_station`.
- Task order: `electricity, cooling, heating, gas`.
- Input/output protocol: 24 historical hourly steps to 4 future hourly steps.
- Model output shape: `[batch, 4, 4]`.
- Historical exogenous variables are allowed; future exogenous variables are forbidden.
- Formal protocols: `full`, `small_sample`.
- Formal seeds: `2026, 2027, 2028, 2029, 2030`.
- Formal training budget follows the accepted Stage 7 protocol: full uses maximum 100 epochs, patience 12, batch size 256; small-sample uses maximum 200 epochs, patience 20, batch size 32. Both use CPU, AdamW, and SmoothL1Loss.
- Never overwrite Stage 6-R or Stage 7-R formal result directories.
- Never use test metrics to select a model or hyperparameter.
- Preserve unrelated dirty-worktree changes and stage only task-specific files.

## File Structure

- `frame/src/validation_selection.py`: validation ranking and WAPE tie grouping.
- `frame/src/external_models.py`: PLE expert, CGC layer, and `PLELiteBaseline` only.
- `frame/scripts/run_stage7_3.py`: reusable formal single-run function and PLE model construction.
- `frame/configs/stage7_7_joint_baselines_contract.json`: immutable Stage 7.7 addendum contract.
- `frame/scripts/run_stage7_7.py`: 30-run orchestration, resume, smoke, aggregation, manifest.
- `frame/src/stage8_analysis.py`: reusable metrics, paired comparison, bootstrap, FDR.
- `frame/scripts/run_stage8_analysis.py`: input validation and Stage 8 orchestration.
- `frame/src/stage8_figures.py`: figure data contracts and plotting helpers.
- `frame/scripts/render_stage8_figures.py`: Stage 8 figure CLI.
- `frame/tests/test_validation_selection.py`: ranking contract tests.
- `frame/tests/test_ple_lite.py`: isolated PLE structure and gradient tests.
- `frame/tests/test_stage7_7_plan.py`: addendum contract and run-matrix tests.
- `frame/tests/test_stage8_analysis.py`: role separation and output-contract tests.
- `frame/README.md`: commands and output roles.

---

### Task 1: Make Stage 6 WAPE tie tolerance executable

**Files:**
- Modify: `frame/src/validation_selection.py`
- Modify: `frame/tests/test_validation_selection.py`

**Interfaces:**
- Consumes: validation comparison rows containing `WAPE`, `validation_max_per_task_WAPE`, `validation_negative_transfer_rate`, `parameter_count`, `fit_seconds`, `model`, and `candidate_id`.
- Produces: `rank_validation_rows(rows, tie_tolerance_percentage_points=0.1) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing tie-group tests**

Add tests that prove rows within `0.1` WAPE percentage points are ordered by the first tie-breaker, while rows outside the tolerance remain ordered by WAPE:

```python
def test_rank_validation_rows_uses_max_task_wape_inside_tolerance(self):
    rows = [
        {"model": "a", "candidate_id": "H1", "WAPE": 10.00,
         "validation_max_per_task_WAPE": 20.0,
         "validation_negative_transfer_rate": 0.0,
         "parameter_count": 10, "fit_seconds": 2.0},
        {"model": "b", "candidate_id": "H1", "WAPE": 10.05,
         "validation_max_per_task_WAPE": 19.0,
         "validation_negative_transfer_rate": 0.0,
         "parameter_count": 10, "fit_seconds": 2.0},
    ]
    ranked = rank_validation_rows(rows, tie_tolerance_percentage_points=0.1)
    self.assertEqual([row["model"] for row in ranked], ["b", "a"])

def test_rank_validation_rows_keeps_wape_order_outside_tolerance(self):
    rows = make_rows(wapes=(10.00, 10.11), max_task_wapes=(20.0, 1.0))
    ranked = rank_validation_rows(rows, tie_tolerance_percentage_points=0.1)
    self.assertEqual([row["WAPE"] for row in ranked], [10.00, 10.11])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest frame.tests.test_validation_selection -v
```

Expected: FAIL because `rank_validation_rows` does not exist.

- [ ] **Step 3: Implement deterministic tolerance grouping**

Implement raw-WAPE sorting, then create groups anchored at the smallest WAPE in each group. Rows with `WAPE - anchor <= tolerance` share a group and are sorted by:

```python
(
    validation_max_per_task_WAPE,
    validation_negative_transfer_rate,
    parameter_count,
    fit_seconds,
    model,
    candidate_id,
)
```

Replace the inline `overall_rows.sort(...)` call with `rank_validation_rows`.

- [ ] **Step 4: Run tests and existing-result audit**

Run the unit test command from Step 2, then invoke the aggregation path against existing Stage 6 results in a temporary audit directory. Verify the first two rows remain `stl_matched/H3` and `scheme2r/H4`.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- frame/src/validation_selection.py frame/tests/test_validation_selection.py
git commit -m "fix Stage 6 WAPE tie tolerance"
```

---

### Task 2: Implement the two-level PLE-lite model

**Files:**
- Modify: `frame/src/external_models.py`
- Create: `frame/tests/test_ple_lite.py`

**Interfaces:**
- Produces: `PLECGCLayer.forward(task_inputs, shared_input) -> (task_outputs, shared_output, gates)`.
- Produces: `PLELiteBaseline.forward(loads, exog=None) -> Tensor[batch, horizon, task_count]`.
- Produces: `PLELiteBaseline.gate_weights(loads, exog) -> dict[str, Tensor]` with `layer1_task`, `layer1_shared`, and `layer2_task`.

- [ ] **Step 1: Write failing PLE shape and gate tests**

Create `frame/tests/test_ple_lite.py` with tests for:

```python
model = PLELiteBaseline(
    lookback=24, horizon=4, task_count=4, exog_dim=12,
    shared_expert_count=2, task_expert_count=1,
    expert_hidden_dim=32, representation_dim=32,
    head_hidden_dim=16, dropout=0.1,
)
loads = torch.randn(3, 24, 4)
exog = torch.randn(3, 24, 12)
self.assertEqual(tuple(model(loads, exog).shape), (3, 4, 4))
gates = model.gate_weights(loads, exog)
self.assertEqual(tuple(gates["layer1_task"].shape), (3, 4, 3))
self.assertEqual(tuple(gates["layer1_shared"].shape), (3, 6))
self.assertEqual(tuple(gates["layer2_task"].shape), (3, 4, 3))
torch.testing.assert_close(gates["layer1_task"].sum(-1), torch.ones(3, 4))
```

Add separate tests for missing exogenous input, task/exogenous dimension mismatch, finite gradients, and the absence of cross-task private experts in each task gate.

- [ ] **Step 2: Run tests and verify failure**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest frame.tests.test_ple_lite -v
```

Expected: FAIL because `PLELiteBaseline` is undefined.

- [ ] **Step 3: Implement `PLEExpert` and `PLECGCLayer`**

Use two-layer GELU MLP experts. For each task, concatenate only its own private expert outputs with the shared expert outputs before applying the task softmax gate. When `produce_shared=True`, the shared gate may combine all private and shared expert outputs. Validate stream counts and dimensions before computation.

- [ ] **Step 4: Implement `PLELiteBaseline`**

Flatten `[loads, exog]`, initialize four identical task streams plus one shared stream, apply a first CGC layer with a shared output, then a second CGC layer without a shared output. Map each final task representation through an independent `32 -> 16 -> 4` head and stack results on the final task axis.

- [ ] **Step 5: Run PLE and model regression tests**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest frame.tests.test_ple_lite frame.tests.test_models -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```powershell
git add -- frame/src/external_models.py frame/tests/test_ple_lite.py
git commit -m "add two-level PLE-lite baseline"
```

---

### Task 3: Reuse the formal training path for PLE-lite

**Files:**
- Modify: `frame/scripts/run_stage7_3.py`
- Modify: `frame/tests/test_ple_lite.py`

**Interfaces:**
- Extends: `_build_model(model_name, hyperparameters, exog_dim)` with public model name `ple-lite`.
- Extends: `_run_one(..., stage_label="7.3", stage_role=None)` without changing existing callers.
- Keeps the legacy Stage 5 external-baseline CLI unchanged; Stage 7.7 uses the strict formal `_run_one` path directly.

- [ ] **Step 1: Write failing builder and training-interface tests**

Assert `_build_model("ple-lite", PLE_FIXED_CONFIG, 12)` returns `PLELiteBaseline` and the formal input mode remains `loads_and_exog`.

- [ ] **Step 2: Run focused tests and verify failure**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest frame.tests.test_ple_lite -v
```

- [ ] **Step 3: Add PLE construction and manifest fields**

Use fixed fields:

```python
{
    "shared_expert_count": 2,
    "task_expert_count": 1,
    "expert_hidden_dim": 32,
    "representation_dim": 32,
    "head_hidden_dim": 16,
    "dropout": 0.1,
    "learning_rate": 0.001,
}
```

Persist `future_exogenous_used=False` and the Stage 7.7 evidence role in the formal run manifest. Do not extend the legacy Stage 5 fairness contract because PLE-lite is a preregistered Stage 7.7 addendum, not one of the historical three Stage 5 baselines.

- [ ] **Step 4: Parameterize `_run_one` stage metadata**

Add optional `stage_label` and `stage_role`; persist them in `run_manifest.json`. Defaults must reproduce existing Stage 7.3 manifests.

- [ ] **Step 5: Run focused tests**

Run the command from Step 2 and `frame.tests.test_stage7_3_plan`.

- [ ] **Step 6: Commit Task 3**

```powershell
git add -- frame/scripts/run_stage7_3.py frame/tests/test_ple_lite.py
git commit -m "integrate PLE-lite with formal training"
```

---

### Task 4: Add the Stage 7.7 contract and 30-run orchestrator

**Files:**
- Create: `frame/configs/stage7_7_joint_baselines_contract.json`
- Create: `frame/scripts/run_stage7_7.py`
- Create: `frame/tests/test_stage7_7_plan.py`
- Modify: `frame/README.md`

**Interfaces:**
- Produces: `build_stage7_7_run_plan(seeds=EXPECTED_SEEDS) -> tuple[dict[str, object], ...]`.
- Produces: `validate_stage7_7_contract(contract, freeze) -> None`.
- Produces: `stage7_7_manifest.json`, `metrics_by_run.csv`, `overall_comparison_mean_std.csv`.

- [ ] **Step 1: Write failing 30-run plan tests**

Test exact coverage:

```python
plan = build_stage7_7_run_plan()
self.assertEqual(len(plan), 30)
self.assertEqual({row["model"] for row in plan},
                 {"hard_share", "dynamic_symmetric", "ple-lite"})
self.assertEqual({row["protocol"] for row in plan}, {"full", "small_sample"})
self.assertEqual({row["seed"] for row in plan}, {2026, 2027, 2028, 2029, 2030})
self.assertEqual({(row["model"], row["candidate_id"]) for row in plan}, {
    ("hard_share", "H2"),
    ("dynamic_symmetric", "H1"),
    ("ple-lite", "ple_lite_fixed_v1"),
})
```

Add rejection tests for wrong seed, wrong candidate, future exogenous use, and duplicate run IDs.

- [ ] **Step 2: Run tests and verify failure**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest frame.tests.test_stage7_7_plan -v
```

- [ ] **Step 3: Create immutable JSON contract**

Record data/task protocol, exact model candidates, PLE fixed configuration, training budget, seeds, output shape, no-test-selection rule, expected run count 30, and expected files.

- [ ] **Step 4: Implement dry-run and contract validation**

`--dry-run` must read no data, create no checkpoints, and print the exact 30-run matrix.

- [ ] **Step 5: Implement formal/smoke execution and resume**

Reuse `_standardize_protocol`, `_run_one`, `_summary_row`, and `_aggregate_rows` from `run_stage7_3.py`. Add sample-limit and epoch overrides only under `--smoke`; reject them in formal mode. Under `--resume`, skip only runs whose manifest is `passed` and whose required files exist.

- [ ] **Step 6: Implement final manifest validation**

Require 30 passed formal runs, exact protocol/model/seed/candidate coverage, `[N,4,4]` predictions, `test_used_for_selection=false`, and no non-finite primary metrics.

- [ ] **Step 7: Run tests and dry-run**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest frame.tests.test_stage7_7_plan -v
& "D:\anaconda\envs\pytorch\python.exe" frame\scripts\run_stage7_7.py --dry-run
```

Expected: PASS and `run_count=30`.

- [ ] **Step 8: Commit Task 4**

```powershell
git add -- frame/configs/stage7_7_joint_baselines_contract.json frame/scripts/run_stage7_7.py frame/tests/test_stage7_7_plan.py frame/README.md
git commit -m "add Stage 7.7 joint baseline addendum"
```

---

### Task 5: Refactor Stage 8 input roles and joint-model statistics

**Files:**
- Modify: `frame/src/stage8_analysis.py`
- Modify: `frame/scripts/run_stage8_analysis.py`
- Modify: `frame/tests/test_stage8_analysis.py`

**Interfaces:**
- Produces: `load_joint_model_records(...) -> dict[str, dict[tuple[str, int], RunRecord]]`.
- Produces: `paired_model_comparison(target, reference_prediction, joint_prediction, ...) -> dict[str, object]`.
- Produces: `joint_model_comparison.csv` and `joint_model_significance.csv`.

- [ ] **Step 1: Write failing role-separation tests**

Create fixtures for five joint models and two STL roles. Assert:

```python
self.assertEqual(set(joint_records), {
    "hard_share", "dynamic_symmetric", "mmoe-lite", "ple-lite", "scheme2r"
})
self.assertEqual(audit_roles["stl_h3"], "validation_selection_audit_only")
self.assertEqual(audit_roles["stl_h4"], "matched_transfer_reference_only")
```

Assert no `best_stl_reference_comparison.csv` appears in the expected output set.

- [ ] **Step 2: Run tests and verify failure**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest frame.tests.test_stage8_analysis -v
```

- [ ] **Step 3: Add Stage 7.7 CLI input and explicit role loaders**

Add `--stage7-7-dir`. Load:

- Scheme2R-H4 from Stage 7.3;
- MMoE-lite from Stage 7.5;
- Hard-Share-H2, Dynamic-Symmetric-H1, and PLE-lite from Stage 7.7;
- STL-H3 as audit-only metadata;
- STL-H4 from Stage 7R.STL as the transfer reference.

Reject missing, duplicate, wrong-candidate, wrong-protocol, wrong-seed, wrong-task-order, or test-selected records.

- [ ] **Step 4: Implement joint leaderboard aggregation**

For every protocol/model, aggregate five-seed MAE, RMSE, WAPE and valid MAPE into mean/std fields. Join parameter count, fit time, and test evaluation time. Keep deterministic/general baselines in a separate group.

- [ ] **Step 5: Implement paired Scheme2R comparisons**

Pair by protocol and seed. Validate identical `target_times` and `target`. Calculate Scheme2R gain against each joint baseline and apply 24-hour block bootstrap plus family-scoped BH correction.

- [ ] **Step 6: Keep transfer analysis strictly H4-to-H4**

Preserve task, horizon, context, negative-transfer rate, and gate analyses, but require every transfer row to name `reference_candidate_id=H4` and `joint_candidate_id=H4`.

- [ ] **Step 7: Run focused Stage 8 tests**

Run the command from Step 2. Expected: PASS.

- [ ] **Step 8: Commit Task 5**

```powershell
git add -- frame/src/stage8_analysis.py frame/scripts/run_stage8_analysis.py frame/tests/test_stage8_analysis.py
git commit -m "refactor Stage 8 joint model analysis"
```

---

### Task 6: Update Stage 8 figures and final manifest

**Files:**
- Modify: `frame/src/stage8_figures.py`
- Modify: `frame/scripts/render_stage8_figures.py`
- Modify: `frame/tests/test_stage8_analysis.py`

**Interfaces:**
- Consumes: `joint_model_comparison.csv`, transfer, ablation, gate, and resource tables.
- Produces: figure files and `stage8_manifest.json` with a complete output inventory.

- [ ] **Step 1: Write failing manifest/figure contract tests**

Assert the manifest contains `joint_main_models`, `selection_audit_reference`, `matched_transfer_reference`, `training_started=False`, `test_used_for_selection=False`, bootstrap settings, input counts, and every expected output file.

- [ ] **Step 2: Run tests and verify failure**

Run `frame.tests.test_stage8_analysis`.

- [ ] **Step 3: Update figures**

Make the main performance figure read only the five joint models. Label SOFTS/DLinear and naive models as separate general baselines. Do not plot STL-H3 in the joint-model panel.

- [ ] **Step 4: Update manifest and table validation**

Require nonempty CSVs, unique key columns, finite MAE/RMSE/WAPE, valid MAPE count/fraction fields, and complete files before atomically replacing the output directory.

- [ ] **Step 5: Run tests**

Run `frame.tests.test_stage8_analysis` and any existing Stage 8 figure tests.

- [ ] **Step 6: Commit Task 6**

```powershell
git add -- frame/src/stage8_figures.py frame/scripts/render_stage8_figures.py frame/tests/test_stage8_analysis.py
git commit -m "update Stage 8 figures and manifest"
```

---

### Task 7: Regression, dry-run, CPU smoke, and documentation

**Files:**
- Modify: `frame/README.md`
- Modify: `plan/Methodology与算法框架-分阶段执行计划.md`

**Interfaces:**
- Produces: reproducible user commands and a final engineering acceptance report.

- [ ] **Step 1: Run all targeted tests**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest `
  frame.tests.test_validation_selection `
  frame.tests.test_ple_lite `
  frame.tests.test_training `
  frame.tests.test_stage7_7_plan `
  frame.tests.test_stage8_analysis -v
```

- [ ] **Step 2: Run full regression tests**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" -m unittest discover -s frame\tests -v
```

Expected: all tests pass; no existing Stage 6-R/7-R contract regressions.

- [ ] **Step 3: Run Stage 7.7 dry-run**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" frame\scripts\run_stage7_7.py --dry-run
```

Expected: 30 unique runs and no data read.

- [ ] **Step 4: Run bounded CPU smoke**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" frame\scripts\run_stage7_7.py `
  --smoke `
  --kitakyushu-data-dir "D:\Paper\Kitakyushu dataset" `
  --output-dir "frame\reports\stage7r_7_joint_baselines_smoke"
```

Expected: every model/protocol path creates a passed manifest, finite metrics, and `[N,4,4]` predictions without reading future exogenous variables.

- [ ] **Step 5: Run Stage 8 dry-run against the formal path contract**

```powershell
& "D:\anaconda\envs\pytorch\python.exe" frame\scripts\run_stage8_analysis.py `
  --stage7-7-dir "frame\reports\stage7r_7_joint_baselines_formal" `
  --dry-run
```

Expected: Stage 8 reports the five joint main models, STL-H3 audit-only, and STL-H4 transfer-only. Formal analysis remains blocked until Stage 7.7 formal runs exist.

- [ ] **Step 6: Update README and methodology plan**

Document Stage 7.7 status, PLE-lite scope, 30-run command, Stage 8 role separation, and the fact that existing formal result directories were not overwritten.

- [ ] **Step 7: Commit Task 7**

```powershell
git add -- frame/README.md "plan/Methodology与算法框架-分阶段执行计划.md"
git commit -m "document Stage 7.7 and revised Stage 8 workflow"
```

- [ ] **Step 8: Final review**

Run `git status --short`, inspect every task commit, verify no report/checkpoint/data files are staged, and provide the user with the formal Stage 7.7 command. Do not start the 30 formal runs without an explicit user command.
