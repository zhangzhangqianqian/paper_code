# Scheme2R-v1 Freeze and Manuscript Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the validated Scheme2R-H4 architecture as `Scheme2R-v1`, make its interfaces auditable, and hand the frozen specification to the final network figure and Methodology sections without retraining.

**Architecture:** Preserve four independent full-window DS-TCN task encoders, an exogenous-state encoder, forecast-step and task-role embeddings, two-level directed routing `G = rho * pi`, low-rank directed messages, stable residual fusion, and task-step forecast heads. The freeze artifact records exact dimensions, task order, split policy, and the selected H4 configuration; it does not alter model behavior or formal results.

**Tech Stack:** Python 3.9.25, PyTorch 2.8.0+cpu, JSON contracts, unittest, Markdown/PPT/Word manuscript assets.

## Global Constraints

- Dataset: Kitakyushu Energy Station; tasks remain `[electricity, cooling, heating, gas]`.
- Input/output: loads `[B,24,4]`, exogenous history `[B,24,12]`, prediction `[B,4,4]`.
- Formal H4 parameters are read from `frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json`.
- Test results are read-only evidence; no post-test architecture or hyperparameter selection is allowed.
- No Transformer, STAR, Frequency, STIM, or additional module is added to Scheme2R-v1.
- The negative-transfer claim is diagnostic and task-dependent; the paper must not claim universal negative-transfer elimination.

---

### Task 1: Create the Scheme2R-v1 freeze contract

**Files:**
- Create: `frame/configs/scheme2r_v1_freeze.json`
- Test: `frame/tests/test_scheme2r_freeze_contract.py`

**Interfaces:**
- Consumes: `stage6_selected_config.json` and the current `Scheme2RModel` constructor.
- Produces: a machine-readable frozen architecture contract with exact tensor shapes, dimensions, H4 settings, and claim boundaries.

- [ ] **Step 1: Record the frozen architecture and experiment boundary**

  The contract must state the task order, 24-to-4 protocol, exogenous feature count, DS-TCN kernel/dilations, state and embedding dimensions, `rho`/`pi`/`gates` shapes, low-rank rank, head dimensions, parameter count, H4 candidate, and prohibited future changes.

- [ ] **Step 2: Add contract tests**

  Tests must verify that the contract contains all four tasks in order, `lookback == 24`, `horizon == 4`, `exog_dim == 12`, output shape `[batch,4,4]`, gate shape `[batch,4,4,4]`, `scheme2r_rank == 8`, and `candidate_id == "H4"`.

- [ ] **Step 3: Run the focused tests**

  Run:

  ```powershell
  & D:\anaconda\envs\pytorch\python.exe -m unittest frame.tests.test_scheme2r_freeze_contract -v
  ```

  Expected: all contract tests pass without training.

### Task 2: Validate the code-to-contract interface

**Files:**
- Create: `frame/scripts/validate_scheme2r_freeze.py`
- Modify: `frame/tests/test_scheme2r_freeze_contract.py`

**Interfaces:**
- Consumes: `scheme2r_v1_freeze.json` and `Scheme2RModel.forward_with_details`.
- Produces: a JSON validation report showing finite output and exact representation, state, route, message, fusion, and prediction shapes.

- [ ] **Step 1: Validate a deterministic synthetic batch**

  Instantiate the model with `exog_dim=12`, `task_count=4`, `hidden_dim=32`, `lookback=24`, and `horizon=4`; run a batch of two samples under `torch.no_grad()` and check all required shapes and finite values.

- [ ] **Step 2: Check the self-route rule**

  Verify `gates[:, :, i, i] == 0` for all tasks and forecast steps.

- [ ] **Step 3: Run the complete model test suite**

  Run:

  ```powershell
  & D:\anaconda\envs\pytorch\python.exe -m unittest discover -s frame/tests -v
  ```

  Expected: focused freeze tests and all existing model/training tests pass.

### Task 3: Freeze manuscript-facing specifications

**Files:**
- Modify: `plan/Methodology与算法框架-分阶段执行计划.md`
- Modify: `frame/README.md`
- Create: `docs/scheme2r_v1_freeze_note.md`

**Interfaces:**
- Consumes: the validated freeze contract and formal Stage 8 result interpretation.
- Produces: synchronized wording for the network figure, Methodology, Results, and limitations.

- [ ] **Step 1: Add the freeze record and exact tensor convention**
- [ ] **Step 2: Replace stale “next experiment” status text with the completed Stage 7-R/8 status**
- [ ] **Step 3: State that Scheme2R is competitive among the tested joint models, not universally superior or universally free of negative transfer**

### Task 4: Final figure and paper handoff

**Files:**
- Modify: `D:/Paper/new_paper` manuscript assets only after the contract passes
- Preserve: existing formal result directories and all legacy results read-only

**Interfaces:**
- Consumes: `scheme2r_v1_freeze.json` and the validated output report.
- Produces: final editable Scheme2R structure figure, Methodology equations, and a Results paragraph whose claims match the evidence.

- [ ] **Step 1: Draw the editable architecture figure from the frozen tensor contract**
- [ ] **Step 2: Write the Problem Definition, Overview, Routing, Fusion, and Prediction Head subsections**
- [ ] **Step 3: Add the performance and negative-transfer limitations without changing the trained model**
- [ ] **Step 4: Commit the freeze contract and synchronized documentation as one versioned checkpoint**

## Self-review checklist

- [ ] Every tensor shape in the figure is present in the freeze contract.
- [ ] H4 is the only reported Scheme2R configuration in the final main tables.
- [ ] Stage 8 results are not regenerated or re-selected.
- [ ] No claim says Scheme2R universally beats STL or eliminates negative transfer.
- [ ] Legacy result directories remain excluded from final tables.
