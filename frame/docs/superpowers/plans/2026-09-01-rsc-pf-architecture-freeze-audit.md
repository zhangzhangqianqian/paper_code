# RSC-PF Architecture Freeze Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reproducible evidence that the implemented RSC-PF is one jointly trainable forecast–scheduling network with 24-hour device-state histories, dispatch-loss gradients reaching both trainable modules, a compact 15-control scheduling interface, and a 21-variable physically feasible output before the algorithm is declared frozen.

**Architecture:** Treat `configs/joint_forecast_dispatch_contract_v2.json` and the v2 formal pipeline as the experiment authority, while recording v1 as legacy compatibility code. Add one read-only audit entry point that combines source-contract checks, actual artifact inspection, forward-dependency tests, gradient tests, and provenance hashes; write its results into a new architecture-freeze directory without changing existing calibration, validation, test, or freeze receipts. A separate architecture-freeze manifest becomes `frozen` only when every mandatory gate passes.

**Tech Stack:** Python 3, PyTorch, NumPy, pytest, JSON, SHA-256, Windows PowerShell, existing RSC-PF code under `src/joint_dispatch` and `src/scheduling`.

## Global Constraints

- The formal model name is `RSC-PF: Jointly Multi-energy Forecasting and Scheduling based on a Rolling State-Conditioned and Physics-Feasible Network`.
- The audit must not retrain models, rerun LP data generation, access the network, or alter any completed experimental receipt.
- The v2 formal experiment contract is the candidate authority; v1 files remain readable only for provenance and compatibility.
- Historical inputs must include exactly 24 steps of four load channels, 12 exogenous channels, 21 continuous device-output channels, and six binary device-status channels.
- The prediction bottleneck must expose four future steps for four tasks: electricity, cooling, heating, and station-side gas prior.
- The scheduling proxy output is one compact tensor of shape `[B, 15]`, not `[B, 4, 15]`; its 15 values allocate decisions across the complete four-hour horizon.
- The physics decoder output must have shape `[B, 4, 21]` and must not produce future binary commitment decisions.
- Warm-start weights are initialization only. In `Warm-Start-Joint`, both forecaster and scheduler must remain trainable under one optimizer and one backward pass.
- A dispatch-derived loss with forecast-loss weight set to zero must still produce finite nonzero gradients in both the forecaster and scheduler.
- The decoder may be described as piecewise differentiable; the audit must not describe clamp, minimum, maximum, or conditional branches as globally smooth.
- Online inference must make zero exact LP calls. LP is permitted only for offline teacher labels, oracle evaluation, and audited data preparation.
- Existing completed experiment outputs under `reports/joint_forecast_dispatch_v2` are immutable inputs to this audit.
- No manuscript file is edited by this plan. Passing the audit only authorizes a subsequent synchronized Chinese–English method revision.

---

## File Structure

- Create `scripts/audit_rsc_pf_architecture_freeze.py`: read-only command-line audit orchestrator and JSON/Markdown report writer.
- Create `tests/test_rsc_pf_architecture_freeze.py`: focused tests for authority, data, shapes, dependencies, gradients, artifacts, and report status.
- Create `configs/rsc_pf_architecture_freeze_v1.json`: immutable declaration of the architecture gates and authoritative source paths; it does not replace or mutate the v2 experiment contract.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/architecture_inventory.json`: hashes and public interface inventory.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/data_gate.json`: actual data-artifact and causality evidence.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/shape_and_dependency_gate.json`: forward shapes and perturbation evidence.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/gradient_gate.json`: static and dynamic gradient-path evidence.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/training_gate.json`: optimizer, warm-start, loss, and no-online-LP evidence.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/artifact_gate.json`: checkpoint and completed-receipt compatibility evidence.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/freeze_receipt.json`: machine-readable aggregate verdict.
- Create `reports/joint_forecast_dispatch_v2/architecture_freeze/freeze_report.md`: human-readable evidence summary and authorized scientific claims.
- Read without modification: `configs/joint_forecast_dispatch_contract_v1.json`, `src/joint_dispatch/contract.py`, and all existing result artifacts.
- Modify only if a gate exposes a real implementation defect: `src/joint_dispatch/data.py`, `src/joint_dispatch/model.py`, `src/joint_dispatch/losses.py`, `src/joint_dispatch/training.py`, `src/joint_dispatch/formal_training.py`, or `src/scheduling/proxy_decoder.py`. Any such modification invalidates the first receipt and requires rerunning every audit and regression step.

---

### Task 1: Establish the v2 Authority and Freeze-Manifest Schema

**Files:**
- Create: `configs/rsc_pf_architecture_freeze_v1.json`
- Create: `scripts/audit_rsc_pf_architecture_freeze.py`
- Create: `tests/test_rsc_pf_architecture_freeze.py`
- Read: `configs/joint_forecast_dispatch_contract_v2.json`
- Read: `configs/joint_forecast_dispatch_contract_v1.json`
- Read: `src/joint_dispatch/formal_protocol.py:56-89,129-289`
- Read: `src/joint_dispatch/contract.py`

**Interfaces:**
- Consumes: `load_formal_experiment_spec(path: str | Path) -> FormalExperimentSpec` and the two existing JSON contracts.
- Produces: `load_freeze_spec(path: str | Path) -> Mapping[str, Any]`, `sha256_file(path: str | Path) -> str`, and `build_architecture_inventory(project_root: Path, freeze_spec: Mapping[str, Any]) -> dict[str, Any]`.
- Test helper: `load_v2_spec(frame_root: Path) -> FormalExperimentSpec` must call `load_formal_experiment_spec(frame_root / "configs/joint_forecast_dispatch_contract_v2.json")` without modifying the returned contract.
- Produces schema `rsc-pf-architecture-freeze-v1` with fields `status`, `authority_contract`, `legacy_contracts`, `source_files`, `shape_contract`, `required_gates`, and `prohibited_claims`.

- [ ] **Step 1: Write the failing authority and schema tests**

```python
def test_freeze_spec_designates_v2_as_authority(frame_root: Path) -> None:
    spec = load_freeze_spec(frame_root / "configs/rsc_pf_architecture_freeze_v1.json")
    assert spec["schema_version"] == "rsc-pf-architecture-freeze-v1"
    assert spec["status"] == "candidate"
    assert spec["authority_contract"] == "configs/joint_forecast_dispatch_contract_v2.json"
    assert "configs/joint_forecast_dispatch_contract_v1.json" in spec["legacy_contracts"]
    assert spec["shape_contract"]["control_logits"] == ["B", 15]
    assert spec["shape_contract"]["dispatch"] == ["B", 4, 21]


def test_inventory_hashes_authoritative_sources(frame_root: Path) -> None:
    spec = load_freeze_spec(frame_root / "configs/rsc_pf_architecture_freeze_v1.json")
    inventory = build_architecture_inventory(frame_root, spec)
    assert inventory["authority"]["schema_version"] == "joint-forecast-dispatch-v2"
    assert inventory["authority"]["contract_status"] in {"calibration_candidate", "frozen"}
    assert len(inventory["authority"]["sha256"]) == 64
    assert inventory["legacy"][0]["role"] == "legacy_compatibility_only"
```

- [ ] **Step 2: Run the new tests and verify the missing files fail**

Run:

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py::test_freeze_spec_designates_v2_as_authority tests\test_rsc_pf_architecture_freeze.py::test_inventory_hashes_authoritative_sources -q
```

Expected: FAIL because the freeze manifest and audit functions do not exist.

- [ ] **Step 3: Add the candidate manifest with exact shape and claim gates**

```json
{
  "schema_version": "rsc-pf-architecture-freeze-v1",
  "status": "candidate",
  "authority_contract": "configs/joint_forecast_dispatch_contract_v2.json",
  "legacy_contracts": ["configs/joint_forecast_dispatch_contract_v1.json"],
  "source_files": [
    "src/joint_dispatch/data.py",
    "src/joint_dispatch/model.py",
    "src/joint_dispatch/losses.py",
    "src/joint_dispatch/training.py",
    "src/joint_dispatch/formal_training.py",
    "src/scheduling/proxy_decoder.py"
  ],
  "shape_contract": {
    "load_history": ["B", 24, 4],
    "exog_history": ["B", 24, 12],
    "device_history": ["B", 24, 21],
    "device_status": ["B", 24, 6],
    "forecast": ["B", 4, 4],
    "scheduling_features": ["B", 4, 10],
    "control_logits": ["B", 15],
    "dispatch": ["B", 4, 21]
  },
  "required_gates": [
    "authority", "data", "dependency", "gradient", "training", "artifact", "regression"
  ],
  "prohibited_claims": [
    "globally_smooth_decoder",
    "future_binary_commitment_output",
    "online_lp_inference",
    "four_by_fifteen_control_logits"
  ]
}
```

- [ ] **Step 4: Implement strict loading and deterministic SHA-256 inventory generation**

The loader must reject unknown top-level keys, non-candidate initial status, missing files, an authority other than v2, or shape values different from the JSON above. The inventory must store relative paths, hashes, v2 schema/status, and the explicit legacy role; it must never overwrite either experiment contract.

- [ ] **Step 5: Run the authority tests and verify they pass**

Run the command from Step 2. Expected: 2 passed.

- [ ] **Step 6: Commit the authority boundary**

```powershell
git add configs/rsc_pf_architecture_freeze_v1.json scripts/audit_rsc_pf_architecture_freeze.py tests/test_rsc_pf_architecture_freeze.py
git commit -m "test: define RSC-PF architecture freeze authority"
```

---

### Task 2: Verify Full Historical Device Inputs and Causality

**Files:**
- Modify: `scripts/audit_rsc_pf_architecture_freeze.py`
- Modify: `tests/test_rsc_pf_architecture_freeze.py`
- Read: `src/joint_dispatch/data.py:57-68,132-224,227-382,531-612,614-682`
- Read: `configs/joint_forecast_dispatch_contract_v2.json`
- Inspect: v2 train, validation, and test `.npz` paths resolved through `FormalExperimentSpec.path(...)`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/data_gate.json`

**Interfaces:**
- Consumes: `load_joint_split(path) -> tuple[JointWindowSplit, JointNormalization | None, Mapping[str, object]]`.
- Produces: `audit_data_gate(spec: FormalExperimentSpec, project_root: Path) -> dict[str, Any]`.
- Test helper: `build_paired_causality_fixtures() -> tuple[dict[str, object], dict[str, object]]` must return two complete `build_joint_windows(...)` keyword dictionaries whose source rows are identical through the chosen origin and differ only in future target rows.
- Result keys: `passed`, `split_evidence`, `history_source`, `status_binary`, `status_consistency`, `causality_probe`, and `failures`.

- [ ] **Step 1: Write failing tests for exact shapes and binary-state derivation**

```python
@pytest.mark.parametrize("split_name", ["train", "validation", "test"])
def test_actual_v2_split_has_complete_24_hour_device_history(frame_root: Path, split_name: str) -> None:
    evidence = audit_data_gate(load_v2_spec(frame_root), frame_root)
    split = evidence["split_evidence"][split_name]
    assert split["load_history_tail"] == [24, 4]
    assert split["exog_history_tail"] == [24, 12]
    assert split["device_history_tail"] == [24, 21]
    assert split["device_status_tail"] == [24, 6]
    assert split["target_tail"] == [4, 4]
    assert split["context_tail"] == [4, 6]
    assert split["teacher_dispatch_tail"] == [4, 21]


def test_actual_status_is_binary_and_matches_canonical_dispatch(frame_root: Path) -> None:
    evidence = audit_data_gate(load_v2_spec(frame_root), frame_root)
    assert evidence["status_binary"] is True
    assert evidence["status_consistency"]["max_absolute_difference"] == 0.0
```

- [ ] **Step 2: Run the tests and verify the missing data audit fails**

Run:

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "complete_24_hour or status_is_binary" -q
```

Expected: FAIL because `audit_data_gate` has not been implemented.

- [ ] **Step 3: Implement artifact loading, shape checks, and finite-value checks**

For each split, load the artifact from the v2 contract, record its SHA-256, sample count, dtype, exact trailing dimensions, minimum and maximum status value, and counts of NaN/Inf. Compare `device_status` with `derive_device_status(device_history)` using the canonical 21-variable order. The audit must return `passed: false` with exact field names when any condition fails; it must not regenerate the split.

The v2 contract currently resolves `data_root` to `reports/joint_forecast_dispatch_v1/data`. Record this exact resolved path and its hashes as an intentionally reused immutable data artifact; do not treat that directory name as experiment-contract authority. Fail only if its stored schema, dimensions, provenance metadata, or hashes violate the v2 input contract.

- [ ] **Step 4: Add a causal-prefix probe**

Use a small synthetic time series with two copies that are identical through forecast origin `t` but sharply different after `t`. Call `build_joint_windows(...)` on both and assert that the selected sample's `load_history`, `exog_history`, `device_history`, and `device_status` are byte-identical while `target` differs. Record the tested origin and maximum history difference.

```python
def test_future_perturbation_cannot_change_model_history() -> None:
    left, right = build_paired_causality_fixtures()
    left_window = build_joint_windows(**left)
    right_window = build_joint_windows(**right)
    np.testing.assert_array_equal(left_window.device_history[0], right_window.device_history[0])
    np.testing.assert_array_equal(left_window.device_status[0], right_window.device_status[0])
    assert not np.array_equal(left_window.target[0], right_window.target[0])
```

- [ ] **Step 5: Enforce permitted history provenance**

Require training metadata `history_source` to be `causal_lp` or `joint_policy_rollin`. Require validation and test histories to be fixed, causal artifacts and reject metadata indicating future-truth construction. Store the literal metadata values in `data_gate.json`.

- [ ] **Step 6: Run the data tests and existing data regression tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "history or status or causal" tests\test_joint_dispatch_data.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the data gate**

```powershell
git add scripts/audit_rsc_pf_architecture_freeze.py tests/test_rsc_pf_architecture_freeze.py
git commit -m "test: audit RSC-PF historical device inputs"
```

---

### Task 3: Verify One Connected Forward Graph and the 15-to-21 Interface

**Files:**
- Modify: `scripts/audit_rsc_pf_architecture_freeze.py`
- Modify: `tests/test_rsc_pf_architecture_freeze.py`
- Read: `src/joint_dispatch/model.py:31-157,159-280`
- Read: `src/scheduling/proxy_decoder.py:20-29,293-327`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/shape_and_dependency_gate.json`

**Interfaces:**
- Consumes: `JointForecastDispatchModel.forward(...) -> JointForwardOutput`.
- Produces: `audit_forward_gate(model: JointForecastDispatchModel, batch: Mapping[str, Tensor]) -> dict[str, Any]`.
- Test fixture: `model_and_batch() -> tuple[JointForecastDispatchModel, dict[str, Tensor]]` must create a deterministic three-sample CPU batch with exact v2 dimensions and valid SOC/status values.
- Test helpers: `clone_batch(batch) -> dict[str, Tensor]` clones every tensor; `run_with_forecast_perturbation(model, batch, delta) -> JointForwardOutput` temporarily wraps `model.forecast_to_physical`, adds `delta` to its returned four-task forecast, runs the unchanged production `model.forward`, and restores the original method.
- Records shapes for `forecast_normalized`, `forecast_physical`, `physical_features`, `control_logits`, and `dispatch`, plus perturbation effects.

- [ ] **Step 1: Write the failing exact-interface test**

```python
def test_rsc_pf_public_forward_contract(model_and_batch) -> None:
    model, batch = model_and_batch
    output = model(**batch)
    assert output.forecast_physical.shape == (3, 4, 4)
    assert output.physical_features.shape == (3, 4, 10)
    assert output.control_logits.shape == (3, 15)
    assert output.dispatch.shape == (3, 4, 21)
    assert not hasattr(output, "future_commitment")
```

- [ ] **Step 2: Write dependency tests that cannot pass through disconnected modules**

```python
def test_forecast_bottleneck_changes_scheduler_and_dispatch(model_and_batch) -> None:
    model, batch = model_and_batch
    base = model(**batch)
    changed = run_with_forecast_perturbation(model, batch, delta=0.05)
    assert torch.max(torch.abs(changed.control_logits - base.control_logits)).item() > 0.0
    assert torch.max(torch.abs(changed.dispatch - base.dispatch)).item() > 0.0


def test_device_history_and_status_change_joint_state(model_and_batch) -> None:
    model, batch = model_and_batch
    base = model(**batch)
    perturbed = clone_batch(batch)
    perturbed["device_history"][:, -1, 0] += 0.1
    perturbed["device_status"][:, -1, 0] = 1.0 - perturbed["device_status"][:, -1, 0]
    changed = model(**perturbed)
    assert torch.max(torch.abs(changed.details["device_state"] - base.details["device_state"])).item() > 0.0
    assert torch.max(torch.abs(changed.control_logits - base.control_logits)).item() > 0.0
```

- [ ] **Step 3: Run the tests and verify helper functions are absent**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "public_forward or bottleneck_changes or history_and_status_change" -q
```

Expected: FAIL because the audit fixtures and controlled perturbation helper do not exist.

- [ ] **Step 4: Implement deterministic fixtures and forward audit**

Build a CPU float32 model from v2 contract dimensions, use an interior finite batch, fix the PyTorch seed, and execute in evaluation mode. The forecast perturbation helper must intervene at the explicit forecast bottleneck through a forward hook or a narrowly scoped test wrapper; it must not replace the scheduling proxy with a mock.

- [ ] **Step 5: Reject the incorrect `[B, 4, 15]` interpretation**

Add a manifest assertion that the 15 controls are partitioned according to `DECISION_GROUPS`: cooling `0:4`, CHP `4:8`, intermediate SOC `8:11`, and PV allocation `11:15`. Record that this is a horizon-level compact control vector decoded to four hourly rows.

- [ ] **Step 6: Run the new and existing model/decoder interface tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "forward or bottleneck or control" tests\test_joint_dispatch_model.py tests\test_scheduling_proxy_decoder.py -q
```

Expected: all selected tests pass and the report records `[B,15] -> [B,4,21]`.

- [ ] **Step 7: Commit the forward-contract gate**

```powershell
git add scripts/audit_rsc_pf_architecture_freeze.py tests/test_rsc_pf_architecture_freeze.py
git commit -m "test: freeze RSC-PF connected forward interface"
```

---

### Task 4: Prove Dispatch Gradients Reach Both Trainable Modules

**Files:**
- Modify: `scripts/audit_rsc_pf_architecture_freeze.py`
- Modify: `tests/test_rsc_pf_architecture_freeze.py`
- Read: `src/joint_dispatch/losses.py:19-160`
- Read: `src/joint_dispatch/training.py:24-171`
- Read: `src/scheduling/proxy_decoder.py:83-327`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/gradient_gate.json`

**Interfaces:**
- Consumes: `joint_forecast_dispatch_loss(...) -> JointLossBreakdown`, `build_joint_optimizer(...)`, and `joint_train_step(...) -> JointStepResult`.
- Produces: `audit_gradient_gate(model, batch, parameters, contract) -> dict[str, Any]`.
- Test fixture: `model_training_fixture() -> tuple[JointForecastDispatchModel, dict[str, Tensor], Mapping[str, Any], JointTrainingContract | None]` must use finite interior physical values and a teacher/oracle batch that yields a nonzero decision term.
- Test helpers: `_forward_inputs(batch) -> dict[str, Tensor]` returns only the six keyword inputs accepted by `JointForecastDispatchModel.forward`; `_parameter_grad_norm(parameters) -> float` computes the L2 norm of all present gradients; `_representative_dispatch_scalar(dispatch, group) -> Tensor` selects `q_ec`, `p_chp`, `soc`, or `pv_use` according to `group`.
- Result keys: `static_scan`, `decoder_group_gradients`, `dispatch_to_forecaster`, `dispatch_to_scheduler`, `optimizer_update`, `finite_difference`, `piecewise_differentiable`, and `failures`.

- [ ] **Step 1: Write a dispatch-only gradient test**

```python
def test_dispatch_only_loss_updates_forecaster_and_scheduler(model_training_fixture) -> None:
    model, batch, parameters, contract = model_training_fixture
    output = model(**_forward_inputs(batch))
    weights = CurriculumWeights(forecast=0.0, imitation=0.0, decision=1.0)
    loss = joint_forecast_dispatch_loss(
        output,
        batch["target_normalized"],
        batch["target_physical"],
        batch["teacher_dispatch"],
        batch["oracle_first_step_objective"],
        parameters,
        weights,
        normalize_decision=True,
    ).total
    loss.backward()
    assert _parameter_grad_norm(model.forecaster.parameters()) > 0.0
    assert _parameter_grad_norm(model.scheduler.parameters()) > 0.0
```

- [ ] **Step 2: Write one gradient test per compact decision group**

```python
@pytest.mark.parametrize("group", ["cooling", "chp", "soc", "renewable_pv"])
def test_each_control_group_has_finite_nonzero_decoder_gradient(decoder_fixture, group: str) -> None:
    logits, features, parameters = decoder_fixture
    logits = logits.clone().requires_grad_(True)
    dispatch = decode_feasible_dispatch(logits, features, parameters)
    _representative_dispatch_scalar(dispatch, group).backward()
    start, stop = DECISION_GROUPS[group]
    assert torch.isfinite(logits.grad[:, start:stop]).all()
    assert torch.count_nonzero(logits.grad[:, start:stop]).item() > 0
```

- [ ] **Step 3: Write a finite-difference comparison away from decoder kinks**

Use double precision, logits near zero, strictly interior feature values, centered difference `h=1e-5`, and require the representative autograd derivative and finite-difference derivative to satisfy `rtol=5e-3, atol=1e-6`. Test one coordinate from each decision group.

- [ ] **Step 4: Run the tests and verify the new audit helpers fail**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "dispatch_only or control_group or finite_difference" -q
```

Expected: FAIL because audit helpers and interior fixtures are absent.

- [ ] **Step 5: Add a static stop-gradient and external-solver scan**

Scan the forward-critical files for `.detach(`, `torch.no_grad`, `.numpy(`, `.item(`, `argmax`, `round`, SciPy optimizer imports, and LP-solver imports. Classify each occurrence by exact file and line. The normalization-only target scale detach in `losses.py` may pass only if the report states that it cannot disconnect `output.forecast_*`, `output.control_logits`, or `output.dispatch`. Any occurrence on those three paths fails the gate.

- [ ] **Step 6: Implement the dynamic gradient audit and one-step parameter update check**

Build one optimizer containing `model.forecaster.parameters()` and `model.scheduler.parameters()`. Snapshot both state dictionaries, execute one step with forecast weight zero and decision weight positive, then require at least one changed finite tensor in each submodule. Report gradient L2 norms and changed-parameter counts; never store parameter values in the report.

- [ ] **Step 7: Run gradient, loss, training, and decoder regression tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "gradient or dispatch_only or finite_difference" tests\test_joint_dispatch_losses.py tests\test_joint_dispatch_losses_v2.py tests\test_joint_dispatch_training.py tests\test_scheduling_proxy_physics.py -q
```

Expected: all selected tests pass with finite nonzero norms for both trainable modules.

- [ ] **Step 8: Commit the gradient gate**

```powershell
git add scripts/audit_rsc_pf_architecture_freeze.py tests/test_rsc_pf_architecture_freeze.py
git commit -m "test: prove dispatch gradients cross RSC-PF"
```

---

### Task 5: Verify Joint-Training Semantics and Inference Boundaries

**Files:**
- Modify: `scripts/audit_rsc_pf_architecture_freeze.py`
- Modify: `tests/test_rsc_pf_architecture_freeze.py`
- Read: `src/joint_dispatch/training.py:55-91,104-171,249-338`
- Read: `src/joint_dispatch/formal_training.py:123-251`
- Read: `scripts/run_joint_forecast_dispatch_v2.py:80-203,247-315,505-541`
- Read: `configs/joint_forecast_dispatch_contract_v2.json`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/training_gate.json`

**Interfaces:**
- Consumes: v2 method definitions for `From-Scratch-Joint`, `Warm-Start-Joint`, `Scheme2R-PTO`, and `Oracle-LP`.
- Produces: `audit_training_gate(spec, model_factory, batch, parameters) -> dict[str, Any]`.
- Records per-method `initialization`, `forecaster_trainable`, `scheduler_trainable`, `optimizer_parameter_ids`, `single_backward`, `online_exact_lp_calls`, and `primary_carbon_loss`.

- [ ] **Step 1: Write failing semantic tests for warm start and joint optimization**

```python
def test_warm_start_is_initialization_not_freezing(training_gate: Mapping[str, Any]) -> None:
    warm = training_gate["methods"]["Warm-Start-Joint"]
    assert warm["initialization"] == "pretrained_weights"
    assert warm["forecaster_trainable"] is True
    assert warm["scheduler_trainable"] is True
    assert warm["one_optimizer_contains_both"] is True
    assert warm["one_backward_updates_both"] is True


def test_joint_inference_has_no_exact_lp_calls(training_gate: Mapping[str, Any]) -> None:
    assert training_gate["online_exact_lp_calls"] == 0
    assert training_gate["future_binary_decisions"] is False
    assert training_gate["primary_carbon_loss"] is False
```

- [ ] **Step 2: Run the semantic tests and verify the training audit is absent**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "warm_start_is or inference_has" -q
```

Expected: FAIL because `audit_training_gate` has not been implemented.

- [ ] **Step 3: Implement method-by-method semantic inspection**

Instantiate each neural method using the runner's production factory. For both joint variants, require all forecaster and scheduler parameters to have `requires_grad=True`, require one optimizer to contain every unique trainable parameter exactly once, and execute one production `joint_train_step`. Record `Frozen-Forecaster` as an ablation and `Scheme2R-PTO` as a prediction-then-optimization baseline; neither may be used as evidence that the proposed model is jointly trained.

- [ ] **Step 4: Verify loss meaning from the formal contract**

Confirm that the production loss contains forecast supervision, teacher imitation when enabled, and decision terms under the configured curriculum. Confirm that the carbon term is reported/evaluated according to the contract and is not silently inserted as a primary training term while `primary_carbon_loss=false`.

- [ ] **Step 5: Verify inference imports and call graph**

Trace the formal-test neural path from checkpoint loading through `JointForecastDispatchModel.forward` and `decode_feasible_dispatch`. Fail if this path imports or calls `solve_dispatch_lp`, SciPy/HiGHS optimization, or another exact optimizer. Separately record offline LP usage as permitted for teacher, PTO, and oracle artifacts.

- [ ] **Step 6: Run formal-protocol, formal-training, runner, and evaluation tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "training or warm_start or inference" tests\test_joint_dispatch_formal_protocol.py tests\test_joint_dispatch_formal_training.py tests\test_joint_dispatch_runner_v2.py tests\test_joint_dispatch_evaluation.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the training-semantics gate**

```powershell
git add scripts/audit_rsc_pf_architecture_freeze.py tests/test_rsc_pf_architecture_freeze.py
git commit -m "test: audit RSC-PF joint training semantics"
```

---

### Task 6: Audit Existing v2 Checkpoints and Frozen Experimental Receipts

**Files:**
- Modify: `scripts/audit_rsc_pf_architecture_freeze.py`
- Modify: `tests/test_rsc_pf_architecture_freeze.py`
- Inspect without modification: `reports/joint_forecast_dispatch_v2/frozen/calibration_freeze_receipt.json`
- Inspect without modification: `reports/joint_forecast_dispatch_v2/frozen/validation_freeze_receipt.json`
- Inspect without modification: `reports/joint_forecast_dispatch_v2/validation/From-Scratch-Joint/seed_*/best_checkpoint.pt`
- Inspect without modification: `reports/joint_forecast_dispatch_v2/validation/Warm-Start-Joint/seed_*/best_checkpoint.pt`
- Inspect without modification: `reports/joint_forecast_dispatch_v2/test/test_manifest.json`
- Inspect without modification: `reports/joint_forecast_dispatch_v2/test/*/seed_*/test_receipt.json`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/artifact_gate.json`

**Interfaces:**
- Consumes: checkpoint payloads created by `save_joint_checkpoint(...)` and existing JSON receipts.
- Produces: `audit_artifact_gate(spec: FormalExperimentSpec, model_factory) -> dict[str, Any]`.
- Result keys: `passed`, `freeze_receipts`, `checkpoint_count`, `checkpoint_schema`, `state_coverage`, `one_batch_forward`, `source_hashes`, and `failures`.

- [ ] **Step 1: Write failing tests for checkpoint completeness and receipt immutability**

```python
def test_existing_joint_checkpoints_contain_both_trainable_modules(artifact_gate) -> None:
    assert artifact_gate["checkpoint_count"] >= 10
    assert artifact_gate["state_coverage"]["forecaster"] is True
    assert artifact_gate["state_coverage"]["scheduler"] is True
    assert artifact_gate["state_coverage"]["missing_keys"] == []
    assert artifact_gate["state_coverage"]["unexpected_keys"] == []


def test_existing_receipts_are_read_only_inputs(frame_root: Path) -> None:
    before = hash_existing_v2_receipts(frame_root)
    run_architecture_audit(frame_root, write_reports=False)
    after = hash_existing_v2_receipts(frame_root)
    assert after == before
```

- [ ] **Step 2: Run the tests and verify artifact inspection is absent**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "checkpoints_contain or receipts_are_read" -q
```

Expected: FAIL because `audit_artifact_gate` and receipt hashing are not implemented.

- [ ] **Step 3: Implement read-only checkpoint schema inspection**

Load every formal joint checkpoint on CPU, verify the stored variant and seed, and load its model state with `strict=True`. Require keys covering both `forecaster.` and `scheduler.` prefixes. Do not load optimizer state into an active optimizer. Record only metadata, counts, file hashes, and compatibility results.

- [ ] **Step 4: Run a one-batch forward pass from actual artifacts**

For one `From-Scratch-Joint` and one `Warm-Start-Joint` checkpoint, load the first two validation samples, run inference under `torch.no_grad()`, and require finite outputs with shapes `[2,4,4]`, `[2,15]`, and `[2,4,21]`. This is a compatibility check, not a performance comparison.

- [ ] **Step 5: Verify experimental-freeze chains**

Recompute the hashes named in `calibration_freeze_receipt.json` and `validation_freeze_receipt.json`. Confirm the sealed test manifest exists and that its recorded test access occurred after validation freezing. Any hash mismatch fails the artifact gate and must be reported as provenance failure rather than model failure.

- [ ] **Step 6: Define missing-artifact behavior**

If a required file named by the v2 contract is absent, emit `passed: false`, `failure_class: "missing_artifact"`, and its exact path. Never synthesize a passing replacement and never start a large experiment from this audit.

- [ ] **Step 7: Run artifact tests twice to verify idempotence**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "checkpoint or receipt or artifact" -q
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "checkpoint or receipt or artifact" -q
```

Expected: both runs pass and existing receipt hashes remain unchanged.

- [ ] **Step 8: Commit the artifact gate**

```powershell
git add scripts/audit_rsc_pf_architecture_freeze.py tests/test_rsc_pf_architecture_freeze.py
git commit -m "test: verify RSC-PF formal artifacts"
```

---

### Task 7: Generate the Architecture Freeze Receipt and Claim Boundary

**Files:**
- Modify: `scripts/audit_rsc_pf_architecture_freeze.py`
- Modify: `tests/test_rsc_pf_architecture_freeze.py`
- Read: `configs/rsc_pf_architecture_freeze_v1.json`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/architecture_inventory.json`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/freeze_receipt.json`
- Create during audit: `reports/joint_forecast_dispatch_v2/architecture_freeze/freeze_report.md`

**Interfaces:**
- Consumes: results from `build_architecture_inventory`, `audit_data_gate`, `audit_forward_gate`, `audit_gradient_gate`, `audit_training_gate`, and `audit_artifact_gate`.
- Produces: `run_architecture_audit(project_root: Path, freeze_spec_path: Path | None = None, output_dir: Path | None = None, *, write_reports: bool = True, regression_verified: bool = False) -> dict[str, Any]` and CLI exit code `0` only when all mandatory gates pass.
- Receipt status is exactly `frozen` or `failed`; the source manifest remains `candidate` so the receipt is independently reproducible and existing experiment-contract hashes remain valid.

- [ ] **Step 1: Write failing aggregate-verdict tests**

```python
def test_freeze_receipt_requires_every_gate(monkeypatch, frame_root: Path) -> None:
    monkeypatch.setattr(audit, "audit_gradient_gate", lambda *args, **kwargs: {"passed": False})
    receipt = run_architecture_audit(frame_root, write_reports=False)
    assert receipt["status"] == "failed"
    assert "gradient" in receipt["failed_gates"]


def test_passing_receipt_authorizes_only_verified_claims(frame_root: Path) -> None:
    receipt = run_architecture_audit(frame_root, write_reports=False, regression_verified=True)
    assert receipt["status"] == "frozen"
    assert receipt["authorized_claims"] == [
        "24-hour load, exogenous, device-output, and device-status histories are consumed",
        "forecast and scheduling are connected in one trainable forward graph",
        "dispatch-derived gradients update both forecaster and scheduler",
        "15 continuous controls decode to four-by-21 physically feasible dispatch outputs",
        "joint neural inference uses no online exact LP",
    ]
```

- [ ] **Step 2: Run the aggregate tests and verify report generation is absent**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -k "freeze_receipt or passing_receipt" -q
```

Expected: FAIL because aggregate receipt generation has not been implemented.

- [ ] **Step 3: Implement atomic JSON and Markdown report writing**

Write every report to a sibling `.tmp` file, flush and close it, then replace the destination. Sort JSON keys, reject NaN/Inf with `allow_nan=False`, and include `generated_at_utc`, Python/PyTorch versions, authority-contract hash, source hashes, gate verdicts, and the exact test command. The Markdown report must distinguish implementation facts, evidence paths, limitations, and claims not authorized by this audit.

- [ ] **Step 4: Preserve completed experiment provenance**

Do not change `configs/joint_forecast_dispatch_contract_v2.json` from `calibration_candidate` to `frozen`: that edit would change the contract SHA-256 already embedded in completed receipts. The independent architecture receipt carries the structure-freeze verdict while the existing calibration/validation freeze receipts continue to describe experimental selection.

- [ ] **Step 5: Add the explicit paper-writing gate**

The report must say that method text may use the five authorized claims only when `freeze_receipt.json` has `status: "frozen"`. It must also say that prediction performance, dispatch superiority, statistical significance, and baseline completeness are experiment questions and are not certified by the architecture audit.

- [ ] **Step 6: Run the audit in report-writing mode**

```powershell
& $Py scripts\audit_rsc_pf_architecture_freeze.py --project-root D:\Paper\github_work\paper-code\frame --freeze-spec configs\rsc_pf_architecture_freeze_v1.json --output-dir reports\joint_forecast_dispatch_v2\architecture_freeze
```

Expected before the full regression suite: exit code 2 with only the `regression` gate false. After Task 8 supplies `--regression-verified`, the command must exit 0, `freeze_receipt.json` must have `status: "frozen"`, and all seven mandatory gates must be true. If any non-regression gate fails, stop at the exact failed gate and do not describe the structure as frozen.

- [ ] **Step 7: Run the aggregate tests again**

```powershell
& $Py -m pytest tests\test_rsc_pf_architecture_freeze.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit the reproducible receipt generator and generated evidence**

```powershell
git add scripts/audit_rsc_pf_architecture_freeze.py tests/test_rsc_pf_architecture_freeze.py configs/rsc_pf_architecture_freeze_v1.json reports/joint_forecast_dispatch_v2/architecture_freeze
git commit -m "docs: freeze verified RSC-PF architecture"
```

---

### Task 8: Run Full Regression and Perform the Final Freeze Review

**Files:**
- Verify: `tests/test_rsc_pf_architecture_freeze.py`
- Verify: `tests/test_joint_dispatch_contract.py`
- Verify: `tests/test_joint_dispatch_formal_protocol.py`
- Verify: `tests/test_joint_dispatch_data.py`
- Verify: `tests/test_joint_dispatch_model.py`
- Verify: `tests/test_joint_dispatch_rollout.py`
- Verify: `tests/test_joint_dispatch_losses.py`
- Verify: `tests/test_joint_dispatch_losses_v2.py`
- Verify: `tests/test_joint_dispatch_training.py`
- Verify: `tests/test_joint_dispatch_formal_training.py`
- Verify: `tests/test_joint_dispatch_runner.py`
- Verify: `tests/test_joint_dispatch_runner_v2.py`
- Verify: `tests/test_joint_dispatch_evaluation.py`
- Verify: `tests/test_scheme2r.py`
- Verify: `tests/test_scheduling_proxy_decoder.py`
- Verify: `tests/test_scheduling_proxy_physics.py`
- Verify: `tests/test_dispatch_lp.py`
- Verify: `reports/joint_forecast_dispatch_v2/architecture_freeze/freeze_receipt.json`

**Interfaces:**
- Consumes: the complete audit and all existing joint forecast–dispatch regression suites.
- Produces: final evidence that the repository still passes its established tests and that the freeze receipt matches the current source hashes.

- [ ] **Step 1: Run the complete relevant regression suite**

```powershell
& $Py -m pytest `
  tests\test_rsc_pf_architecture_freeze.py `
  tests\test_joint_dispatch_contract.py `
  tests\test_joint_dispatch_formal_protocol.py `
  tests\test_joint_dispatch_data.py `
  tests\test_joint_dispatch_model.py `
  tests\test_joint_dispatch_rollout.py `
  tests\test_joint_dispatch_losses.py `
  tests\test_joint_dispatch_losses_v2.py `
  tests\test_joint_dispatch_training.py `
  tests\test_joint_dispatch_formal_training.py `
  tests\test_joint_dispatch_runner.py `
  tests\test_joint_dispatch_runner_v2.py `
  tests\test_joint_dispatch_evaluation.py `
  tests\test_scheme2r.py `
  tests\test_scheduling_proxy_decoder.py `
  tests\test_scheduling_proxy_physics.py `
  tests\test_dispatch_lp.py `
  --basetemp D:\Paper\pytest_tmp_rsc_pf_freeze `
  -p no:cacheprovider `
  -q
```

Expected: 100% pass. Record the exact test count and duration in `freeze_report.md`; do not preserve an old expected count because this plan adds tests.

- [ ] **Step 2: Rerun the architecture audit after the regression suite**

```powershell
& $Py scripts\audit_rsc_pf_architecture_freeze.py --project-root D:\Paper\github_work\paper-code\frame --freeze-spec configs\rsc_pf_architecture_freeze_v1.json --output-dir reports\joint_forecast_dispatch_v2\architecture_freeze --regression-verified
```

Expected: exit code 0 with `--regression-verified` and source hashes matching the current worktree.

- [ ] **Step 3: Verify no completed experiment artifact changed**

Compare the current hashes of all files named in the pre-existing calibration and validation freeze receipts with the hashes captured before Task 1. Expected: exact equality. Only files under `reports/joint_forecast_dispatch_v2/architecture_freeze` may be new or changed.

- [ ] **Step 4: Review the scientific claim boundary manually**

Confirm the report states all of the following and nothing stronger:

1. Complete 24-hour load/exogenous/device-output/device-status histories enter the trainable network.
2. A four-task, four-hour forecast bottleneck is explicit and measurable.
3. Future PV/WT availability, prices/carbon context, current SOC, and previous CHP state enter the scheduling side through the defined context interface.
4. The scheduling proxy outputs 15 continuous horizon-level controls.
5. The physics decoder outputs four hours by 21 continuous physical dispatch variables.
6. Dispatch loss can update both forecaster and scheduler under joint variants.
7. No future binary commitment output and no online exact LP are present.
8. Warm start does not by itself prove superiority; it is a permitted initialization for the same jointly fine-tuned graph.

- [ ] **Step 5: Review the Chinese–English terminology handoff**

Record the following fixed terminology in `freeze_report.md` for the subsequent manuscript task:

| English | Chinese |
|---|---|
| jointly trainable network | 联合可训练网络 |
| explicit forecast bottleneck | 显式预测瓶颈 |
| rolling state-conditioned encoding | 滚动状态条件编码 |
| scheduling proxy | 调度代理 |
| piecewise-differentiable physics-feasible decoder | 分段可微的物理可行解码器 |
| compact 15-dimensional control representation | 紧凑的15维控制表示 |
| 21-variable physical dispatch output | 21变量物理调度输出 |

- [ ] **Step 6: Commit any report-only corrections from the final review**

```powershell
git add reports/joint_forecast_dispatch_v2/architecture_freeze
git commit -m "docs: finalize RSC-PF architecture freeze evidence"
```

---

## Plan Self-Review Record

### Spec Coverage

- The 24-hour continuous device-output and binary-status requirement is covered by Task 2 against actual v2 data artifacts, not only synthetic tensors.
- The dispatch-to-forecast gradient requirement is covered by Task 4 with forecast loss disabled, per-control-group decoder gradients, finite differences, and one optimizer update.
- The interface requirement is covered by Task 3 and explicitly fixes the key dimensional fact: `[B,15]`, not `[B,4,15]`, is decoded to `[B,4,21]`.
- Joint-training semantics, warm-start interpretation, no-online-LP inference, and no future binary decision head are covered by Task 5.
- Existing trained checkpoints and sealed receipts are covered by Task 6, so source-level success cannot hide artifact incompatibility.
- v1/v2 ambiguity is resolved in Task 1 by naming v2 as authority and v1 as legacy compatibility code.
- Existing receipt hashes are protected by using an independent architecture-freeze manifest rather than editing the completed v2 contract.
- Synchronized manuscript revision is intentionally excluded; the plan produces an evidence-based terminology and claim handoff for that separate work.

### Reviewer Corrections Applied

1. Replaced every four-by-15 scheduler-output description with the implemented horizon-level 15-control representation.
2. Added actual checkpoint and receipt inspection rather than relying only on unit tests.
3. Added piecewise-differentiability wording because the decoder contains clamp/minimum/maximum branches.
4. Added a dispatch-only backward test so forecast-supervision gradients cannot masquerade as decision coupling.
5. Added explicit differentiation between warm-start initialization, joint fine-tuning, PTO baselines, and frozen-forecaster ablation.
6. Prevented contract-hash invalidation by keeping the existing v2 experiment contract immutable.
7. Separated architecture validity from empirical superiority; the freeze receipt cannot certify model performance or baseline completeness.
8. Recorded that the v2 pipeline intentionally reads immutable data from the `joint_forecast_dispatch_v1/data` directory, while v2 remains the experiment-contract authority.

### Type and Name Consistency

- `load_history`: `[B,24,4]`
- `exog_history`: `[B,24,12]`
- `device_history`: `[B,24,21]`
- `device_status`: `[B,24,6]`
- `forecast_physical`: `[B,4,4]`
- `physical_features`: `[B,4,10]`
- `control_logits`: `[B,15]`
- `dispatch`: `[B,4,21]`
- Aggregate function: `run_architecture_audit(project_root: Path, freeze_spec_path: Path | None = None, output_dir: Path | None = None, *, write_reports: bool = True, regression_verified: bool = False) -> dict[str, Any]`
- Final machine verdict: `reports/joint_forecast_dispatch_v2/architecture_freeze/freeze_receipt.json`

### Execution Stop Rule

If any mandatory gate fails, the receipt status must be `failed`, the current algorithm must remain unfrozen, and execution must stop with the exact failed assertion and evidence path. A source-code correction is permitted only after the failure is reproduced by a focused test; after correction, Tasks 2 through 8 must be rerun before any method claim is revised.
