# Formal-v4.2 Four-Hour Settlement Residual Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct four-hour post-recourse physical-residual accounting so settlement-only electric dump is not misclassified as a physical violation, then rebuild the formal Gate evidence chain from fresh checkpoints.

**Architecture:** Keep the canonical one-step recourse projection as the sole source of electricity, cooling, heating, and conversion residuals. The four-hour objective aggregates those residual tensors and independently verifies the cross-step BESS SOC recursion; it does not reconstruct balances from the 21-column dispatch, add a gas-demand balance, or relax the physical gate.

**Tech Stack:** Python 3.10, PyTorch, NumPy, pytest, PowerShell, immutable JSON/NPZ/PT Gate receipts.

## Global Constraints

- The forecast contract remains four tasks: electricity, cooling, heating, and station-side gas consumption.
- Physical settlement remains exactly three rigid terminal demands: electricity, cooling, and heating.
- Gas is an auxiliary forecast/scheduling prior; realized `g_chp + g_gb` contributes to operating cost and carbon but is not matched to a terminal gas-demand equality.
- Do not weaken the `1.0e-5` Gate 1 physical-residual threshold.
- Do not reuse or re-rank Stage J or Gate 2 checkpoints created before this correction.
- Do not add the intentional nested checkout `frame/third_party/iTransformer_source/` to the parent repository.
- Stop at the first Gate whose immutable receipt does not authorize the next Gate.
- Do not invoke Gate 3, ablations, manuscript generation, or read the 2020 evaluation split except through an authorized Gate 2 execution.

---

### Task 1: Reproduce the omitted-`p_dump` defect with a deterministic test

**Files:**
- Modify: `frame/tests/test_joint_dispatch_formal_v4_objective.py`
- Test: `frame/tests/test_joint_dispatch_formal_v4_objective.py`

**Interfaces:**
- Consumes: `settle_first_step_v4(planned_action, demand, renewable, parameters, *, initial_soc, previous_chp) -> FormalV4RealizedOutcome`.
- Consumes: `settle_formal_v4_four_hour(planned_dispatch, actual_demand, actual_renewables, initial_soc, previous_chp, parameters) -> FormalV4FourHourSettlement`.
- Produces: `test_four_hour_constraint_penalty_includes_settlement_only_dump_accounting()`.

- [ ] **Step 1: Extend the objective-test imports**

Add `settle_formal_v4_four_hour` to the import from `formal_v4_objective`, import `settle_first_step_v4`, and import `VARIABLES`:

```python
from src.joint_dispatch.formal_v4_objective import (
    STEP_WEIGHTS,
    FormalV4FourHourSettlement,
    clip_formal_v4_gradients,
    fit_training_objective_scale,
    formal_v4_curriculum_weights,
    formal_v4_joint_loss,
    settle_formal_v4_four_hour,
)
from src.joint_dispatch.formal_v4_recourse import settle_first_step_v4
from src.scheduling.dispatch_schema import VARIABLES
```

- [ ] **Step 2: Write the forced-surplus regression test**

Use the CHP ramp lower bound to create surplus that cannot be removed by reducing grid, PV, or WT:

```python
def test_four_hour_constraint_penalty_includes_settlement_only_dump_accounting():
    index = {name: position for position, name in enumerate(VARIABLES)}
    planned = torch.zeros(1, 4, len(VARIABLES))
    planned[..., index["p_chp"]] = PARAMETERS["chp_electric_capacity"]
    demand = torch.zeros(1, 4, 3)
    renewables = torch.zeros(1, 4, 2)
    initial_soc = torch.full((1, 1), 0.5)
    previous_chp = torch.full((1, 1), PARAMETERS["chp_electric_capacity"])

    first = settle_first_step_v4(
        planned[:, 0], demand[:, 0], renewables[:, 0], PARAMETERS,
        initial_soc=initial_soc, previous_chp=previous_chp,
    )
    settled = settle_formal_v4_four_hour(
        planned, demand, renewables, initial_soc, previous_chp, PARAMETERS,
    )

    assert first.p_dump.item() > 0.0
    assert first.balance_residuals.abs().max().item() <= 1.0e-7
    assert first.conversion_residuals.abs().max().item() <= 1.0e-7
    assert settled.constraint_penalty.item() <= 1.0e-7
```

- [ ] **Step 3: Run the new test and verify the existing implementation fails for the intended reason**

Run from `frame/`:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_objective.py::test_four_hour_constraint_penalty_includes_settlement_only_dump_accounting --basetemp D:\Paper\pytest_tmp_residual_red -p no:cacheprovider -q
```

Expected: FAIL because the canonical one-step residual is zero while the current four-hour constraint penalty is approximately `6.6667`.

---

### Task 2: Aggregate canonical residuals in the four-hour objective

**Files:**
- Modify: `frame/src/joint_dispatch/formal_v4_objective.py:244-281`
- Test: `frame/tests/test_joint_dispatch_formal_v4_objective.py`

**Interfaces:**
- Consumes: the four `FormalV4RealizedOutcome` objects already accumulated in local variable `outcomes`.
- Produces: unchanged public function signature `settle_formal_v4_four_hour(...) -> FormalV4FourHourSettlement` with corrected `constraint_penalty: Tensor[B]`.

- [ ] **Step 1: Replace the duplicated balance and conversion reconstruction**

Delete the `balance = torch.stack((...))` and `conversion = torch.stack((...))` formulas based on `settled_dispatch`. Replace them with the residuals from the canonical outcomes:

```python
    balance = torch.stack([item.balance_residuals for item in outcomes], dim=1)
    conversion = torch.stack([item.conversion_residuals for item in outcomes], dim=1)
```

Keep the existing explicit SOC recurrence and final aggregation:

```python
    capacity = _finite_scalar(parameters.get("bess_energy_capacity", 1.0), "bess_energy_capacity")
    eta = _finite_scalar(parameters.get("bess_roundtrip_efficiency", 1.0), "bess_roundtrip_efficiency") ** 0.5
    previous_energy = initial_soc[:, 0] * capacity
    soc_residual = []
    for step in range(4):
        soc = settled_dispatch[:, step, _I["soc"]]
        soc_residual.append(
            soc - previous_energy - eta * settled_dispatch[:, step, _I["p_charge"]]
            + settled_dispatch[:, step, _I["p_discharge"]] / max(eta, 1.0e-8)
        )
        previous_energy = soc
    soc_residual_tensor = torch.stack(soc_residual, dim=1)
    constraint_penalty = (
        balance.abs().mean(dim=(1, 2))
        + conversion.abs().mean(dim=(1, 2))
        + soc_residual_tensor.abs().mean(dim=1)
    )
```

- [ ] **Step 2: Run the forced-surplus test**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_objective.py::test_four_hour_constraint_penalty_includes_settlement_only_dump_accounting --basetemp D:\Paper\pytest_tmp_residual_green -p no:cacheprovider -q
```

Expected: `1 passed`; `p_dump` remains positive and the post-recourse penalty is within `1.0e-7`.

- [ ] **Step 3: Run all formal-v4 objective and Gate 1 tests**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_objective.py tests\test_joint_dispatch_formal_v4_2_gate1.py --basetemp D:\Paper\pytest_tmp_residual_focused -p no:cacheprovider -q
```

Expected: all tests pass, including the existing `[B,4,4] -> [B,4,3]` Gate 1 boundary test.

- [ ] **Step 4: Commit the implementation and regression test**

Run from the repository root:

```powershell
git add -- frame/src/joint_dispatch/formal_v4_objective.py frame/tests/test_joint_dispatch_formal_v4_objective.py
git commit -m "fix: aggregate canonical four-hour settlement residuals"
```

Expected: only those two files are committed; `frame/third_party/iTransformer_source/` remains untracked.

---

### Task 3: Audit interfaces and run the complete formal-v4.2 regression suite

**Files:**
- Verify: `frame/src/joint_dispatch/formal_v4_objective.py`
- Verify: `frame/src/joint_dispatch/formal_v4_2_gate2_training.py`
- Verify: `frame/scripts/run_rsc_pf_formal_v4_2_gate1.py`
- Verify: `frame/src/joint_dispatch/formal_v4_2_rollout.py`
- Verify: `frame/configs/formal_v4_source_closure_v4_2.txt`

**Interfaces:**
- Consumes: corrected `settle_formal_v4_four_hour` and the frozen formal-v4.2 source closure.
- Produces: a tested commit whose source manifest will invalidate all pre-fix Gate receipts.

- [ ] **Step 1: Audit every four-hour settlement caller**

Run:

```powershell
rg -n -C 2 "settle_formal_v4_four_hour\(" frame -g "*.py"
```

Expected:

- `formal_v4_joint_loss` passes `target_physical[..., :3]`;
- both Gate 2 training call sites pass `batch["target_physical"][..., :3]`;
- Gate 1 passes `_rigid_demand_target(four_task_target)`;
- no caller sends all four forecast tasks to rigid settlement.

- [ ] **Step 2: Verify the independent Gate 2 rollout retains explicit dump accounting**

Run:

```powershell
rg -n "derived_p_dump|p_dump|balance =" frame/src/joint_dispatch/formal_v4_2_rollout.py frame/src/joint_dispatch/formal_v4_recourse.py
```

Expected: the canonical one-step balance subtracts `p_dump`, and the independent rollout derives and subtracts `derived_p_dump` before reporting its balance residual.

- [ ] **Step 3: Verify source closure covers the changed production file**

Run:

```powershell
rg -n "frame/src/joint_dispatch/formal_v4_objective.py" frame/configs/formal_v4_source_closure_v4_2.txt
```

Expected: exactly one matching source-closure entry. No closure-file edit is required.

- [ ] **Step 4: Run the complete formal-v4.2 suite by explicit file list**

Run from `frame/`:

```powershell
$testFiles = @(Get-ChildItem -LiteralPath tests -Filter 'test_joint_dispatch_formal_v4_2_*.py' | ForEach-Object { $_.FullName })
$testFiles += (Resolve-Path 'tests\test_joint_dispatch_formal_v4_itransformer.py').Path
$testFiles += (Resolve-Path 'tests\test_joint_dispatch_formal_v4_objective.py').Path
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest @testFiles --basetemp D:\Paper\pytest_tmp_residual_formal_v42 -p no:cacheprovider -q
```

Expected: all selected tests pass. Do not replace this with collection of the entire legacy `tests/` directory, whose unrelated archived modules are incomplete in this worktree.

- [ ] **Step 5: Verify repository status before execution**

Run:

```powershell
git status --short
git log -3 --oneline
```

Expected: no tracked implementation changes remain. The only allowed untracked entry is `frame/third_party/iTransformer_source/`.

---

### Task 4: Rebuild Gate 0, Pilot, and Gate 1 from a fresh lineage

**Files:**
- Generate only: `frame/reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_j/`

**Interfaces:**
- Consumes: the committed corrected source manifest, fixed contract `frame/configs/joint_forecast_dispatch_formal_v4_2.json`, isolated Python environment, and local official iTransformer checkout.
- Produces: immutable Gate 0, Pilot, and Gate 1 receipts, plus `GATE1_TRANSITION.json` only if Gate 1 authorizes Gate 2.

- [ ] **Step 1: Confirm the new run ID is unused**

Run from the repository root:

```powershell
$RunId = 'formal_v4_2_20260905_j'
Test-Path "frame\reports\joint_forecast_dispatch_formal_v4_2\$RunId"
```

Expected: `False`. If it is `True`, increment the suffix before running any Gate.

- [ ] **Step 2: Run Gate 0**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_gate0.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
```

Expected: exit code `0` and JSON with `"authorized_pilot": true`. Otherwise stop and report the Gate 0 evidence or failure receipt.

- [ ] **Step 3: Run Pilot**

Run only after Gate 0 authorization:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_pilot.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
```

Expected: exit code `0` and JSON with `"authorized_gate1": true`. Treat its shortage rate as an engineering diagnostic, not a paper result. Otherwise stop.

- [ ] **Step 4: Run Gate 1 with newly trained checkpoints**

Run only after Pilot authorization:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_gate1.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
```

Expected: `"authorized_gate2": true` and a non-null selected candidate. Verify `evaluation_year_accessed=false`, all physical residuals are within the frozen threshold, and four-task forecast metrics remain present. Otherwise stop and report the candidate table and failed check.

---

### Task 5: Execute and independently audit Gate 2 only if authorized

**Files:**
- Generate only under: `frame/reports/joint_forecast_dispatch_formal_v4_2/<authorized-run-id>/gate2/`

**Interfaces:**
- Consumes: `GATE1_TRANSITION.json` with `authorized_gate2=true` and the selected Gate 1 training freeze.
- Produces: the frozen 23-row formal-v4.2 experiment, immutable per-row receipts, Gate 2 decision, and an independent audit result.

- [ ] **Step 1: Run Gate 2**

Run only when Gate 1 authorizes it:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/run_rsc_pf_formal_v4_2_gate2.py --contract frame/configs/joint_forecast_dispatch_formal_v4_2.json --output-root frame/reports/joint_forecast_dispatch_formal_v4_2 --run-id $RunId
```

Expected: all 23 frozen rows are trained/evaluated or Gate 2 emits an immutable failure receipt and stops.

- [ ] **Step 2: Run the independent Gate 2 audit**

Run only after Gate 2 completes:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' frame/scripts/audit_rsc_pf_formal_v4_2_gate2.py --run-root "frame/reports/joint_forecast_dispatch_formal_v4_2/$RunId"
```

Expected: the audit agrees with the Gate 2 decision, verifies 23-row completeness and lineage hashes, and reports no unauthorized test access.

- [ ] **Step 3: Stop and report the evidence**

Report:

- Gate 0, Pilot, Gate 1, Gate 2, and independent-audit decisions;
- the selected Gate 1 multiplier and residual/guardrail values;
- 23-row completeness and total runtime;
- primary RSC-PF versus Decoupled-RSC-PF decision metrics;
- any first failing receipt if execution stopped early.

Do not invoke Gate 3, ablations, or manuscript generation.

---

## Plan Self-Review Record

- **Spec coverage:** Tasks 1-2 correct and test canonical residual aggregation; Task 3 audits four-task/three-balance boundaries and source closure; Tasks 4-5 rebuild the immutable evidence chain without checkpoint reuse.
- **Placeholder scan:** No TBD, TODO, deferred implementation, or unspecified error-handling steps remain.
- **Type consistency:** `FormalV4RealizedOutcome.balance_residuals` is `[B,3]`, `conversion_residuals` is `[B,5]`, stacked four-hour tensors are `[B,4,3]` and `[B,4,5]`, and `constraint_penalty` remains `[B]`.
- **Scope:** The plan changes one production function and one test file, then stops after Gate 2 and its independent audit.
