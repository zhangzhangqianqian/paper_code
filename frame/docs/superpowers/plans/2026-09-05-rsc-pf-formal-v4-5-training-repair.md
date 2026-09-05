# RSC-PF Formal-v4.5 Training Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the formal RSC-PF training protocol so joint decision optimization cannot silently destroy four-task forecast semantics, then verify the repair with a bounded diagnostic before any large experiment.

**Architecture:** Keep the v4.4 residual-gated Scheme2R model, 15-dimensional continuous control, 21-dimensional physics decoder, and rolling evaluator unchanged. Add explicit train/early-stop loaders, stage-specific checkpoint selection, separate J optimizer groups, normalized curriculum losses, a differentiable P1 forecast anchor, and immutable validation/audit artifacts in a new formal-v4.5 namespace.

**Tech Stack:** Python 3, PyTorch, NumPy, PyYAML, pytest, existing formal-v4.4 artifact/provenance utilities.

## Global Constraints

- Do not modify or overwrite `reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_f`.
- Do not modify or stage `third_party/iTransformer_source/`.
- Preserve 2015--2018 training, 2019 Pilot selection, sealed 2020 evaluation, and excluded 2021.
- Preserve four forecast tasks, 15 continuous controls, 21 decoded dispatch values, no future binary decisions, and gas as a station-side auxiliary prior.
- Keep Gate-1 thresholds unchanged; a failed diagnostic or Pilot must fail closed.
- Do not start large baselines, ablations, multi-seed sweeps, or Gate 1 during this implementation.
- All new run/config/artifact identifiers must use `formal_v4_5`.

---

### Task 1: Add and validate the formal-v4.5 contract

**Files:**
- Create: `configs/joint_forecast_dispatch_formal_v4_5.json`
- Create: `src/joint_dispatch/formal_v4_5_contract.py`
- Test: `tests/test_joint_dispatch_formal_v4_5_contract.py`

**Interfaces:**
- `FormalV45Contract.from_path(path: str | Path) -> FormalV45Contract`
- `FormalV45Contract.validate() -> None`
- `FormalV45Contract.payload: Mapping[str, Any]`

- [ ] **Step 1: Write failing contract tests**

```python
def test_v45_contract_freezes_curriculum_and_year_boundary(tmp_path):
    contract = FormalV45Contract.from_path("configs/joint_forecast_dispatch_formal_v4_5.json")
    contract.validate()
    assert contract.payload["protocol_status"] == "frozen"
    assert contract.payload["train_years"] == [2015, 2016, 2017, 2018]
    assert contract.payload["selection_year"] == 2019
    assert contract.payload["evaluation_year"] == 2020
    assert contract.payload["joint_curriculum"] == {
        "forecast": 1.0, "anchor": 0.5, "decision_start": 0.05,
        "decision_final": 0.50, "imitation_start": 1.0,
        "imitation_final": 0.25, "ramp_epochs": 5,
    }

def test_v45_rejects_gate_threshold_changes():
    contract = FormalV45Contract.from_path("configs/joint_forecast_dispatch_formal_v4_5.json")
    contract.payload["pilot_thresholds"]["maximum_physical_residual"] = 1.0
    with pytest.raises(ValueError, match="threshold"):
        contract.validate()
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_contract.py -q`

Expected: FAIL because the v4.5 contract module and JSON do not exist.

- [ ] **Step 3: Implement the frozen contract**

Copy the v4.4 scientific fields, change schema/output namespace to v4.5, add:

```json
"joint_curriculum": {
  "forecast": 1.0,
  "anchor": 0.5,
  "decision_start": 0.05,
  "decision_final": 0.50,
  "imitation_start": 1.0,
  "imitation_final": 0.25,
  "ramp_epochs": 5
},
"forecast_guardrails": {
  "max_macro_f1_drop": 0.02,
  "max_inactive_leakage_relative_increase": 0.05
}
```

`FormalV45Contract.validate()` must reject changed Gate-1 thresholds, year
boundaries, task/control dimensions, or curriculum values. It must not import
the v4.4 Pilot receipt.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_contract.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add configs/joint_forecast_dispatch_formal_v4_5.json src/joint_dispatch/formal_v4_5_contract.py tests/test_joint_dispatch_formal_v4_5_contract.py
git commit -m "add formal v4.5 frozen training contract"
```

### Task 2: Implement validation-aware stage training and checkpoint restoration

**Files:**
- Create: `src/joint_dispatch/formal_v4_5_training.py`
- Test: `tests/test_joint_dispatch_formal_v4_5_training.py`

**Interfaces:**
- `StageValidationV45(metric: float, eligible: bool, details: Mapping[str, float])`
- `StageReceiptV45.best_epoch: int`
- `StageReceiptV45.validation_history: tuple[Mapping[str, float], ...]`
- `run_stage_with_validation_v45(model: nn.Module, train_batches: Sequence[Mapping[str, Any]], validation_batches: Sequence[Mapping[str, Any]], optimizer_factory: Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer], train_loss: Callable[[nn.Module, Mapping[str, Any], int], Tensor], validation_metric: Callable[[nn.Module, Mapping[str, Any]], float], eligibility: Callable[[Mapping[str, float]], bool], max_epochs: int, minimum_epochs: int, patience: int) -> StageReceiptV45`

- [ ] **Step 1: Write failing early-stop tests**

```python
def test_best_validation_state_is_restored():
    model = make_toy_stage_model(seed=7)
    receipt = run_stage_with_validation_v45(
        model=model,
        train_batches=make_toy_batches(metrics=[0.8, 0.6, 0.7]),
        validation_batches=make_toy_batches(metrics=[0.4, 0.2, 0.3]),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_loss=toy_train_loss,
        validation_metric=lambda current_model, batch: float(batch["metric"]),
        eligibility=lambda values: bool(np.isfinite(values["metric"])),
        max_epochs=4, minimum_epochs=1, patience=2,
    )
    assert receipt.best_epoch == 1
    assert receipt.stopping_reason == "early_stopped"
    assert receipt.final_sha256 == receipt.best_sha256

def test_validation_does_not_change_parameters_or_optimizer_steps():
    model = make_toy_stage_model(seed=8)
    train_batches = make_toy_batches(metrics=[0.8, 0.6])
    receipt = run_stage_with_validation_v45(
        model=model,
        train_batches=train_batches,
        validation_batches=make_toy_batches(metrics=[0.4, 0.3]),
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        train_loss=toy_train_loss,
        validation_metric=lambda current_model, batch: float(batch["metric"]),
        eligibility=lambda values: True,
        max_epochs=2, minimum_epochs=1, patience=2,
    )
    assert receipt.optimizer_steps == len(train_batches) * receipt.epochs
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_training.py -q`

Expected: FAIL because validation-aware training and receipts do not exist.

- [ ] **Step 3: Implement the validation loop**

Add a new v4.5 training module rather than changing v4.4 receipt schemas. The
loop must:

```python
for epoch in range(max_epochs):
    train_one_epoch(model, train_batches, optimizer, train_loss, epoch)
    validation = evaluate_without_grad(model, validation_batches, validation_metric)
    validation_history.append(validation)
    if eligible(validation) and validation.metric < best_metric:
        best_state = deepcopy(model.state_dict())
        best_epoch = epoch
        stale_epochs = 0
    else:
        stale_epochs += 1
    if epoch + 1 >= minimum_epochs and stale_epochs >= patience:
        break
model.load_state_dict(best_state)
```

Use `torch.no_grad()` for validation, save the selected and terminal hashes,
and fail closed if no eligible finite checkpoint exists. Provide stage adapters
for P0/P1/continuous/S so each stage uses its own validation loss.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_training.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_5_training.py tests/test_joint_dispatch_formal_v4_5_training.py
git commit -m "add validation-aware v4.5 stage selection"
```

### Task 3: Add normalized Joint curriculum, anchor, and optimizer-group audit

**Files:**
- Create: `src/joint_dispatch/formal_v4_5_loss.py`
- Modify: `src/joint_dispatch/formal_v4_5_training.py`
- Test: `tests/test_joint_dispatch_formal_v4_5_loss.py`

**Interfaces:**
- `curriculum_weights_v45(epoch: int, ramp_epochs: int) -> dict[str, float]`
- `JointNormalizationV45(forecast: float, imitation: float, decision: float)`
- `joint_loss_v45(output, parent_output, batch, normalization, weights, anchor_weight) -> JointLossV45`
- `build_j_optimizer_v45(model, contract) -> torch.optim.Optimizer`

- [ ] **Step 1: Write failing curriculum and gradient tests**

```python
def test_curriculum_is_monotone_and_positive():
    values = [curriculum_weights_v45(epoch, 5) for epoch in range(8)]
    assert values[0]["decision"] == pytest.approx(0.05)
    assert values[4]["decision"] == pytest.approx(0.50)
    assert values[0]["imitation"] == pytest.approx(1.0)
    assert values[4]["imitation"] == pytest.approx(0.25)
    assert all(row["forecast"] == 1.0 and row["anchor"] == 0.5 for row in values)

def test_joint_decision_gradient_reaches_forecast_and_decoupled_does_not():
    joint = run_one_j_batch(mode="joint")
    decoupled = run_one_j_batch(mode="decoupled")
    assert joint.gradient_norms["decision_to_base"] > 0
    assert joint.gradient_norms["decision_to_gate"] > 0
    assert joint.gradient_norms["decision_to_magnitude"] > 0
    assert decoupled.gradient_norms["decision_to_base"] <= 1e-12
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_loss.py -q`

Expected: FAIL because v4.5 curriculum, anchor, and optimizer audit do not exist.

- [ ] **Step 3: Implement normalized loss and optimizer groups**

Compute detached parent denominators on the training partition, lower-bounded
by `1e-8`. Implement:

```python
weights = curriculum_weights_v45(epoch, ramp_epochs)
total = (
    weights["forecast"] * forecast / normalization.forecast
    + weights["imitation"] * imitation / normalization.imitation
    + weights["decision"] * decision / normalization.decision
    + weights["anchor"] * smooth_l1(current_forecast, parent_forecast)
)
```

Build optimizer groups from `v44_parameter_groups()` with exactly
`j_forecaster_lr`, `j_head_lr`, and `j_scheduler_lr`. The Joint branch includes
base/gate/magnitude/scheduler; Fair Decoupled includes scheduler only and
detaches forecast-to-dispatch. Persist group names, counts, and effective rates.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_loss.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_5_loss.py src/joint_dispatch/formal_v4_5_training.py tests/test_joint_dispatch_formal_v4_5_loss.py
git commit -m "add v4.5 joint curriculum and forecast anchor"
```

### Task 4: Wire causal train/early-stop data and same-information teacher validation

**Files:**
- Create: `src/joint_dispatch/formal_v4_5_pilot_executor.py`
- Create: `src/joint_dispatch/formal_v4_5_pilot_data.py`
- Test: `tests/test_joint_dispatch_formal_v4_5_executor.py`

**Interfaces:**
- `build_v45_loaders(materialized, teacher, batch_size) -> dict[str, list[Mapping[str, Any]]]`
- `execute_training_stages_v45(*, materialized: MaterializedV44PilotData, contract: FormalV45Contract, artifact_root: str | Path, seed: int, parameters: Mapping[str, Any], prior: ThermalPriorReceiptV44 | None, teacher: TeacherReceiptV44 | None) -> TrainingBundleV45`

- [ ] **Step 1: Write failing wiring tests**

```python
def test_executor_exposes_disjoint_train_and_early_stop_loaders():
    loaders = build_v45_loaders(materialized_fixture(), teacher_fixture(), batch_size=4)
    assert set(loaders) == {"train", "early_stop"}
    train_origins = {int(batch["origin_index"]) for batch in loaders["train"]}
    validation_origins = {int(batch["origin_index"]) for batch in loaders["early_stop"]}
    assert train_origins.isdisjoint(validation_origins)

def test_validation_teacher_uses_p1_forecast_and_never_realized_load():
    bundle = execute_training_stages_v45(
        materialized=fixture(), contract=contract, artifact_root=tmp_path,
        seed=2026, parameters=parameters, prior=None, teacher=None,
    )
    assert bundle.teacher.train_forecast_hash != bundle.teacher.early_stop_forecast_hash
    assert bundle.teacher.input_years == [2015, 2016, 2017, 2018]
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_executor.py -q`

Expected: FAIL because the v4.5 executor and loaders do not exist.

- [ ] **Step 3: Implement the executor adapter**

Reuse v4.4 materialization and physical decoder adapters without copying their
receipts. Split `materialized.train` and `materialized.early_stop` into explicit
loaders, generate same-information teacher labels for both partitions using the
P1 forecast, run P0/P1/continuous/S with v4.5 validation, then run matched Joint
and Fair Decoupled branches. Do not load 2019 or 2020 during training.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_executor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_5_pilot_executor.py src/joint_dispatch/formal_v4_5_pilot_data.py tests/test_joint_dispatch_formal_v4_5_executor.py
git commit -m "wire v4.5 causal validation and executor"
```

### Task 5: Persist and independently audit v4.5 selection evidence

**Files:**
- Create: `src/joint_dispatch/formal_v4_5_artifacts.py`
- Test: `tests/test_joint_dispatch_formal_v4_5_audit.py`

**Interfaces:**
- `write_v45_stage_receipt(path, receipt, validation_history, optimizer_groups) -> None`
- `audit_v45_selection(run_root: str | Path, contract: FormalV45Contract) -> Mapping[str, Any]`

- [ ] **Step 1: Write failing audit tests**

```python
def test_audit_reconstructs_best_epoch_and_rejects_tampering(tmp_path):
    run_root = make_v45_artifacts(tmp_path)
    result = audit_v45_selection(run_root, contract)
    assert result["authorized_gate1"] is False
    tamper_validation_metric(run_root)
    with pytest.raises(ValueError, match="selection mismatch"):
        audit_v45_selection(run_root, contract)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_audit.py -q`

Expected: FAIL because v4.5 artifact and audit functions do not exist.

- [ ] **Step 3: Implement immutable receipts and audit**

Persist train/early-stop hashes, parent normalization constants, per-epoch
weights, anchor history, optimizer groups, guardrail eligibility, selected and
terminal checkpoint hashes. Recompute every selection-critical value from
persisted arrays; reject missing, non-finite, changed, or 2019/2020-derived data.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_audit.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_5_artifacts.py tests/test_joint_dispatch_formal_v4_5_audit.py
git commit -m "add auditable v4.5 selection artifacts"
```

### Task 6: Run regression tests and bounded diagnostic

**Files:**
- Create: `scripts/run_rsc_pf_formal_v4_5_diagnostic.py`
- Test: `tests/test_joint_dispatch_formal_v4_5_diagnostic.py`

**Interfaces:**
- `run_formal_v45_diagnostic(*, config: str | Path, output_root: str | Path, max_batches: int | None = None) -> Mapping[str, Any]`

- [ ] **Step 1: Add entrypoint and smoke test**

```python
def test_diagnostic_never_opens_selection_or_evaluation_year(tmp_path):
    receipt = run_formal_v45_diagnostic(
        config="configs/joint_forecast_dispatch_formal_v4_5.json",
        output_root=tmp_path, max_batches=2,
    )
    assert receipt["accessed_years"] == [2015, 2016, 2017, 2018]
    assert receipt["pilot_authorized"] is False
```

- [ ] **Step 2: Run the smoke test and verify it fails**

Run: `D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_diagnostic.py -q`

Expected: FAIL because the v4.5 entrypoint does not exist.

- [ ] **Step 3: Implement the bounded diagnostic runner**

Run deterministic fixture checks, then a small real-data train/early-stop
diagnostic only. Stop after gradient, loss, early-stop, guardrail, and artifact
checks. Do not evaluate 2019, do not access 2020, and do not launch baselines.

- [ ] **Step 4: Run all targeted tests and protected regression tests**

Run:

```powershell
D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_5_*.py -q
D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests/test_joint_dispatch_formal_v4_4_*.py -q
```

Expected: all targeted v4.5 tests pass and protected v4.4 tests remain green.
Unrelated legacy collection failures are reported separately and do not get
silently treated as v4.5 evidence.

- [ ] **Step 5: Run the bounded diagnostic and inspect the receipt**

Run:

```powershell
D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts/run_rsc_pf_formal_v4_5_diagnostic.py --config configs/joint_forecast_dispatch_formal_v4_5.json
```

Expected: a new `formal_v4_5` receipt with explicit early-stop selection,
optimizer-group, curriculum, anchor, gradient-boundary, and year-access
evidence. If any diagnostic criterion fails, stop and do not start a Pilot.

- [ ] **Step 6: Commit the verified implementation**

```powershell
git add configs/joint_forecast_dispatch_formal_v4_5.json src/joint_dispatch/formal_v4_5_*.py scripts/run_rsc_pf_formal_v4_5_diagnostic.py tests/test_joint_dispatch_formal_v4_5_*.py
git commit -m "implement and verify formal v4.5 training repair"
```

## Completion Handoff

After Task 6, report the diagnostic receipt path, test counts, accessed years,
whether all J optimizer groups were audited, whether forecast guardrails were
enforced, and whether the process stopped before any 2019 Pilot. A new Pilot is
a separate user-approved execution step after the diagnostic is reviewed.
