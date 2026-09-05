# RSC-PF Formal-v4.3 Regime-Aware Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the structurally unsuitable continuous cooling/heating heads with a differentiable three-state thermal-regime and conditional-magnitude forecast while preserving the four-task RSC-PF bottleneck, joint dispatch gradient, physical decoder, and sealed evaluation protocol.

**Architecture:** A new formal-v4.3 model reuses the state-conditioned Scheme2R encoder, 15-control scheduler, and 21-variable feasible decoder. A regime-aware head emits electricity, gas prior, three thermal-state logits, and conditional cooling/heating magnitudes; soft state probabilities form differentiable cooling/heating point forecasts. Formal-v4.3 adds regime-aware losses and full-chronology Gate 1 checks without changing gas into a rigid terminal demand.

**Tech Stack:** Python 3.10, PyTorch, NumPy, scikit-learn-compatible metric formulas implemented locally, pytest, PowerShell, immutable JSON/NPZ/PT receipts.

## Global Constraints

- Freeze formal-v4.2 source artifacts and receipts as diagnostic evidence; do not relabel any v4.2 result as v4.3.
- Training years are exactly 2015, 2016, 2017, and 2018; selection year is exactly 2019; evaluation year 2020 remains sealed until an immutable Gate 1 receipt authorizes Gate 2; 2021 is excluded.
- Preserve four forecast targets in order: electricity, cooling, heating, and station-side gas consumption prior.
- Preserve exactly three rigid terminal-demand balances: electricity, cooling, and heating. Gas is supervised and may condition scheduling, but it is not a terminal gas-demand equality.
- Preserve the scheduler interface `[B,4,10]`, 15 continuous controls, and 21 physical dispatch outputs.
- The thermal regime is learned from permitted inputs; do not introduce hard month masks, future realized weather, target leakage, or evaluation-year calibration.
- Soft regime probabilities remain on the training and scheduling path. Hard regime labels are diagnostics only.
- Do not round thermal predictions to the observed raw-data increments.
- RSC-PF and Decoupled-RSC-PF must use byte-identical Stage-P and Stage-S parents and differ only at the Stage-J dispatch-gradient boundary.
- The first run is a one-seed train/selection pilot. Do not run the full baseline matrix, Gate 2, Gate 3, ablations beyond the named continuous-head control, or manuscript generation until the pilot and Gate 1 authorize continuation.
- Stop at the first failed Gate and write a fail-closed receipt.

---

### Task 1: Freeze the formal-v4.3 contract and provenance boundary

**Files:**
- Create: `configs/joint_forecast_dispatch_formal_v4_3.json`
- Create: `src/joint_dispatch/formal_v4_3_contract.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_contract.py`

**Interfaces:**
- Consumes: `FormalV42Contract` parsing and canonical hashing conventions.
- Produces: `FormalV43Contract.load(path: str | Path) -> FormalV43Contract`.
- Produces: immutable contract sections `thermal_regime`, `forecast_loss`, `gate1_views`, and `candidate_grid`.

- [ ] **Step 1: Write contract tests that reject an unsafe protocol**

```python
def test_v43_contract_fixes_data_and_output_boundaries(tmp_path):
    contract = FormalV43Contract.load(CONTRACT)
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.excluded_years == (2021,)
    assert contract.forecast_tasks == ("electricity", "cooling", "heating", "gas")
    assert contract.rigid_demands == ("electricity", "cooling", "heating")
    assert contract.control_dim == 15
    assert contract.dispatch_dim == 21

def test_v43_contract_rejects_hard_calendar_mask(tmp_path):
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["thermal_regime"]["hard_calendar_mask"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hard calendar mask"):
        FormalV43Contract.load(path)
```

- [ ] **Step 2: Run the contract tests and verify they fail because v4.3 does not exist**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_contract.py --basetemp D:\Paper\pytest_tmp_v43_contract_red -p no:cacheprovider -q
```

Expected: FAIL on the missing `formal_v4_3_contract` module.

- [ ] **Step 3: Create the frozen JSON sections**

Start from the already audited v4.2 topology/data/source fields, change the protocol id to `formal-v4.3`, and add exactly:

```json
{
  "thermal_regime": {
    "classes": ["off", "cooling", "heating"],
    "active_epsilon": 1e-9,
    "probability_temperature": 1.0,
    "hard_calendar_mask": false,
    "hard_training_gate": false,
    "fail_on_simultaneous_thermal_targets": true,
    "class_weight_clip": [0.5, 2.0]
  },
  "forecast_loss": {
    "continuous_weight": 1.0,
    "regime_weight": 1.0,
    "active_magnitude_weight": 1.0,
    "point_weight": 0.5,
    "inactive_leakage_weight": 0.25,
    "gas_task_weight": 0.25,
    "transition_window_weight": 1.5
  },
  "gate1_views": {
    "full_chronology": {"enabled": true, "uniform_weight": true},
    "stress_sample": {"enabled": true, "secondary_only": true}
  },
  "candidate_grid": {
    "forecaster_lr_multiplier": [0.5, 1.0, 2.0],
    "decision_final": [0.25, 0.5, 1.0]
  }
}
```

- [ ] **Step 4: Implement `FormalV43Contract` validation and canonical hash**

Reject missing classes, reordered tasks, non-positive temperatures, hard masks/gates, invalid loss weights, any candidate not explicitly enumerated, and any data split that differs from the global constraints. Reuse `canonical_sha256` for `contract_sha256`.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_contract.py --basetemp D:\Paper\pytest_tmp_v43_contract -p no:cacheprovider -q
git add configs/joint_forecast_dispatch_formal_v4_3.json src/joint_dispatch/formal_v4_3_contract.py tests/test_joint_dispatch_formal_v4_3_contract.py
git commit -m "feat: freeze formal v4.3 regime-aware contract"
```

Expected: PASS; commit contains no experiment output.

---

### Task 2: Derive auditable thermal regimes and active-only statistics

**Files:**
- Create: `src/joint_dispatch/formal_v4_3_data.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_data.py`

**Interfaces:**
- Consumes: physical four-task targets with shape `[N,4,4]`.
- Produces: `derive_thermal_regimes(target, epsilon=1e-9) -> np.ndarray` with shape `[N,4]` and values `0,1,2`.
- Produces: `fit_thermal_magnitude_statistics(train_target, regimes) -> ThermalMagnitudeReceiptV43`.
- Produces: `thermal_transition_mask(regimes, last_observed_regime) -> np.ndarray` with shape `[N]`.

- [ ] **Step 1: Write failing label and leakage tests**

```python
def test_thermal_regimes_encode_off_cooling_and_heating():
    target = np.zeros((1, 4, 4), dtype=np.float32)
    target[0, 1, 1] = 111.425
    target[0, 2, 2] = 63.7934
    labels = derive_thermal_regimes(target)
    np.testing.assert_array_equal(labels, [[0, 1, 2, 0]])

def test_thermal_regimes_fail_closed_on_simultaneous_targets():
    target = np.zeros((1, 4, 4), dtype=np.float32)
    target[0, 0, 1:3] = 1.0
    with pytest.raises(ValueError, match="simultaneous cooling and heating"):
        derive_thermal_regimes(target)

def test_active_statistics_ignore_exact_zeros():
    target = np.zeros((2, 4, 4), dtype=np.float32)
    target[0, 0, 1] = 100.0
    target[1, 0, 1] = 300.0
    target[0, 1, 2] = 50.0
    target[1, 1, 2] = 150.0
    receipt = fit_thermal_magnitude_statistics(target, derive_thermal_regimes(target))
    np.testing.assert_allclose(receipt.mean, [200.0, 100.0])
```

- [ ] **Step 2: Run the tests and verify the module is missing**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_data.py --basetemp D:\Paper\pytest_tmp_v43_data_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement train-only labels, statistics, transition masks, and receipt hashing**

Use `cooling = target[..., 1] > epsilon` and `heating = target[..., 2] > epsilon`; raise before assigning labels when `(cooling & heating).any()`. Compute active means/scales separately and reject empty or zero-scale active sets. Store counts, fractions, means, scales, and the hash of the training target in the receipt.

- [ ] **Step 4: Add a source-integrity regression**

Load only the already materialized permitted training/selection windows and assert that the derived simultaneous count is zero and that statistics are fitted from the training receipt hash, never the selection array.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_data.py --basetemp D:\Paper\pytest_tmp_v43_data -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_3_data.py tests/test_joint_dispatch_formal_v4_3_data.py
git commit -m "feat: add thermal regime labels and active statistics"
```

---

### Task 3: Add the differentiable regime-aware RSC-PF forecast head

**Files:**
- Modify: `src/joint_dispatch/formal_v4_models.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_model.py`

**Interfaces:**
- Consumes: base Scheme2R forecast features `[B,4,4]` and DSTCN state `[B,32]`.
- Produces: `RegimeAwareForecastHead.forward(base, state) -> RegimeForecastV43`.
- Produces: `RegimeAwareRSCPFModel.forward(...) -> FormalV43ForwardOutput`.

- [ ] **Step 1: Write output-shape, non-negativity, and gradient tests**

```python
def test_regime_aware_model_preserves_four_task_dispatch_contract(batch, model):
    output = model(**batch)
    assert output.forecast_normalized.shape == (2, 4, 4)
    assert output.forecast_physical.shape == (2, 4, 4)
    assert output.regime_logits.shape == (2, 4, 3)
    assert output.regime_probabilities.shape == (2, 4, 3)
    assert output.thermal_magnitudes.shape == (2, 4, 2)
    assert output.controls.shape == (2, 4, 15)
    assert output.dispatch.shape == (2, 4, 21)
    assert torch.all(output.forecast_physical >= 0)
    torch.testing.assert_close(output.regime_probabilities.sum(-1), torch.ones(2, 4))

def test_dispatch_loss_reaches_regime_and_magnitude_parameters(batch, model):
    output = model(**batch)
    output.dispatch.square().mean().backward()
    assert model.regime_head.output.weight.grad is not None
    assert model.regime_head.output.weight.grad.norm().item() > 0.0
```

- [ ] **Step 2: Run the model tests and verify the classes are missing**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_model.py --basetemp D:\Paper\pytest_tmp_v43_model_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement the typed forecast head**

```python
@dataclass(frozen=True)
class RegimeForecastV43:
    electricity_normalized: Tensor
    gas_normalized: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    thermal_magnitude_normalized: Tensor
    thermal_magnitudes: Tensor

class RegimeAwareForecastHead(nn.Module):
    def __init__(self, state_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(4 + state_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.output = nn.Linear(hidden_dim, 7)  # e, gas, 3 regimes, cool magnitude, heat magnitude
```

Concatenate the four base horizon values with the expanded 32-dimensional state, split the seven outputs exactly as documented, and calculate probabilities with the fixed contract temperature.

- [ ] **Step 4: Implement physical reconstruction without changing scheduler inputs**

Use train-only task and active-magnitude buffers:

```python
electricity = F.softplus(task_mean[0] + task_scale[0] * e_norm)
gas = F.softplus(task_mean[3] + task_scale[3] * gas_norm)
cool_mag = F.softplus(active_mean[0] + active_scale[0] * cool_norm)
heat_mag = F.softplus(active_mean[1] + active_scale[1] * heat_norm)
cooling = probabilities[..., 1] * cool_mag
heating = probabilities[..., 2] * heat_mag
physical = torch.stack((electricity, cooling, heating, gas), dim=-1)
normalized = (physical - task_mean) / task_scale
```

Pass `physical` through the existing `_raw_physical_features`, scheduler, and physical decoder. Do not append probabilities to the ten scheduler features.

- [ ] **Step 5: Make `forecaster_parameters()` include the new head and verify detach semantics**

The joint branch must expose non-zero dispatch gradients on the new head; `detach_forecast_for_dispatch=True` must yield exactly zero dispatch gradient while supervised forecast gradients remain non-zero.

- [ ] **Step 6: Run model and existing formal-v4 tests, then commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_model.py tests\test_joint_dispatch_formal_v4_2_model.py --basetemp D:\Paper\pytest_tmp_v43_model -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_models.py tests/test_joint_dispatch_formal_v4_3_model.py
git commit -m "feat: add regime-aware RSC-PF forecast head"
```

---

### Task 4: Implement the regime-aware forecast objective and joint-loss injection

**Files:**
- Create: `src/joint_dispatch/formal_v4_3_training.py`
- Modify: `src/joint_dispatch/formal_v4_objective.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_training.py`
- Modify: `tests/test_joint_dispatch_formal_v4_objective.py`

**Interfaces:**
- Produces: `forecast_loss_v43(output, batch, receipt, weights) -> ForecastLossBreakdownV43`.
- Extends: `formal_v4_joint_loss(..., supervised_forecast_loss: Tensor | None = None)`; `None` preserves formal-v4.2 byte-level behavior.
- Produces: v4.3 `run_stage_p_v43`, `run_stage_s_v43`, and `run_stage_j_v43` receipts.

- [ ] **Step 1: Write exact component-loss tests**

```python
def test_v43_forecast_loss_is_zero_for_perfect_regime_and_magnitude(perfect_output, batch, receipt):
    loss = forecast_loss_v43(perfect_output, batch, receipt, WEIGHTS)
    assert loss.total.item() < 1e-6

def test_inactive_leakage_penalizes_false_thermal_output(output, off_batch, receipt):
    loss = forecast_loss_v43(output, off_batch, receipt, WEIGHTS)
    assert loss.inactive_leakage.item() > 0.0

def test_v42_joint_loss_default_is_unchanged(v42_case):
    before = formal_v4_joint_loss(**v42_case)
    after = formal_v4_joint_loss(**v42_case, supervised_forecast_loss=None)
    torch.testing.assert_close(before.total, after.total)
```

- [ ] **Step 2: Run the focused tests and verify failure**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_training.py tests\test_joint_dispatch_formal_v4_objective.py --basetemp D:\Paper\pytest_tmp_v43_loss_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement `ForecastLossBreakdownV43` and the five loss components**

Use class-weighted cross entropy on `[B,4,3]`; mask conditional magnitude losses by true class; compute the leakage term from predicted cooling on non-cooling targets and predicted heating on non-heating targets; multiply per-window components by `1.5` only when `thermal_transition_mask` is true. Return every component separately for audit.

- [ ] **Step 4: Add the optional forecast-loss injection to the physical objective**

```python
if supervised_forecast_loss is None:
    forecast_loss = existing_v42_forecast_loss
else:
    if supervised_forecast_loss.ndim != 0 or not torch.isfinite(supervised_forecast_loss):
        raise ValueError("supervised_forecast_loss must be a finite scalar")
    forecast_loss = supervised_forecast_loss
```

Do not alter imitation, settlement, shortage, cost, carbon, or physical-residual calculations.

- [ ] **Step 5: Implement v4.3 P/S/J wrappers with two optimizer parameter groups**

Stage P uses only `forecast_loss_v43.total`; Stage S reuses the existing teacher imitation; Stage J passes `forecast_loss_v43.total` into `formal_v4_joint_loss`. Joint mode sets all forecast and scheduler parameters trainable; decoupled mode detaches only `forecast_physical` before scheduler construction.

- [ ] **Step 6: Add a gradient-boundary receipt for regime parameters**

Record `decision_regime_gradient_norm`, `decision_magnitude_gradient_norm`, `decision_forecaster_gradient_norm`, and `scheduler_gradient_norm`. Joint values must be finite and positive; the two decision-to-forecast values must be exactly zero for the decoupled branch.

- [ ] **Step 7: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_training.py tests\test_joint_dispatch_formal_v4_objective.py tests\test_joint_dispatch_formal_v4_2_training.py --basetemp D:\Paper\pytest_tmp_v43_loss -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_3_training.py src/joint_dispatch/formal_v4_objective.py tests/test_joint_dispatch_formal_v4_3_training.py tests/test_joint_dispatch_formal_v4_objective.py
git commit -m "feat: train regime-aware forecasts with joint dispatch loss"
```

---

### Task 5: Add full-year, active, inactive, regime, and transition metrics

**Files:**
- Create: `src/joint_dispatch/formal_v4_3_metrics.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_metrics.py`

**Interfaces:**
- Produces: `compute_forecast_metrics_v43(prediction, target, regime_probability, target_regime, target_times) -> ForecastMetricsV43`.
- Produces: JSON-safe fields `all_hour`, `active_only`, `inactive_leakage`, `regime`, and `transition_by_horizon`.

- [ ] **Step 1: Write deterministic metric tests**

```python
def test_inactive_leakage_is_not_hidden_by_active_wape():
    target = np.array([[[0., 0., 0., 10.], [0., 100., 0., 10.]]])
    pred = np.array([[[0., 50., 0., 10.], [0., 100., 0., 10.]]])
    metrics = compute_forecast_metrics_v43(pred, target, PROB, REGIME, TIMES)
    assert metrics.active_only["cooling_wape"] == pytest.approx(0.0)
    assert metrics.inactive_leakage["cooling_mae"] == pytest.approx(50.0)

def test_regime_confusion_matrix_counts_all_three_classes():
    metrics = compute_forecast_metrics_v43(PRED, TARGET, PROB, REGIME, TIMES)
    assert np.asarray(metrics.regime["confusion_matrix"]).shape == (3, 3)
```

- [ ] **Step 2: Run tests and verify the module is missing**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_metrics.py --basetemp D:\Paper\pytest_tmp_v43_metrics_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement metrics without adding a new runtime dependency**

Calculate MAE, RMSE, WAPE, macro-F1, balanced accuracy, and the `3 x 3` confusion matrix with NumPy. Define zero-denominator WAPE as `NaN` and retain numerator/denominator fields so no undefined percentage is silently converted to zero.

- [ ] **Step 4: Add chronological and transition assertions**

Reject duplicate/out-of-order target times in the full-chronology view. Compute transition flags relative to the last observed thermal regime and report each of horizons `t+1` through `t+4` separately.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_metrics.py --basetemp D:\Paper\pytest_tmp_v43_metrics -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_3_metrics.py tests/test_joint_dispatch_formal_v4_3_metrics.py
git commit -m "feat: audit thermal regime and leakage metrics"
```

---

### Task 6: Build a dual-view Gate 1 evaluator and fail-closed eligibility rules

**Files:**
- Create: `scripts/run_rsc_pf_formal_v4_3_gate1.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_gate1.py`
- Modify: `src/joint_dispatch/formal_v4_3_data.py`

**Interfaces:**
- Consumes: full 2019 selection windows and the frozen stress-origin manifest.
- Produces: `_evaluate_full_chronology_v43(...)` and `_evaluate_stress_view_v43(...)`.
- Produces: `GATE1_CANDIDATES.json`, `GATE1_FREEZE.json`, and `GATE1_EVIDENCE.json` with `evaluation_year_accessed=false`.

- [ ] **Step 1: Write tests preventing the old sampling defect**

```python
def test_full_chronology_uses_every_selection_origin_once(selection):
    view = build_full_chronology_view(selection)
    np.testing.assert_array_equal(view.indices, np.arange(len(selection)))
    np.testing.assert_array_equal(view.weights, np.ones(len(selection)))

def test_stress_sample_cannot_authorize_candidate_by_itself(candidate):
    candidate["views"]["stress_sample"]["passed"] = True
    candidate["views"]["full_chronology"]["passed"] = False
    assert candidate_is_eligible_v43(candidate) is False
```

- [ ] **Step 2: Run the Gate 1 tests and verify failure**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_gate1.py --basetemp D:\Paper\pytest_tmp_v43_gate1_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement both selection views**

The full view uses `np.arange(len(selection))` and unit weights. The stress view reuses the frozen origin categories but stores its non-representative activity fractions and may never replace the full view.

- [ ] **Step 4: Freeze candidate eligibility**

Require all of the following booleans:

```python
eligible = all((
    views["full_chronology"]["forecast_guardrail_passed"],
    views["full_chronology"]["inactive_leakage_guardrail_passed"],
    views["full_chronology"]["regime_guardrail_passed"],
    views["stress_sample"]["forecast_guardrail_passed"],
    candidate["physical_feasibility_passed"],
    candidate["dispatch_improvement_passed"],
    candidate["gradient_boundary_passed"],
))
```

Numeric thresholds must come only from `joint_forecast_dispatch_formal_v4_3.json`; no threshold may be inferred from 2020 or modified after candidate execution.

- [ ] **Step 5: Enumerate the nine frozen Stage-J candidates**

Use the Cartesian product of forecaster learning-rate multipliers `[0.5,1.0,2.0]` and final decision weights `[0.25,0.5,1.0]`. Share one Stage-P and one Stage-S parent, use seed `2026`, record each candidate independently, and choose the eligible candidate with minimum full-chronology penalized objective. Do not rank an ineligible candidate.

- [ ] **Step 6: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_gate1.py --basetemp D:\Paper\pytest_tmp_v43_gate1 -p no:cacheprovider -q
git add scripts/run_rsc_pf_formal_v4_3_gate1.py src/joint_dispatch/formal_v4_3_data.py tests/test_joint_dispatch_formal_v4_3_gate1.py
git commit -m "feat: require chronological and stress evidence in gate one"
```

---

### Task 7: Add the one-seed mechanism pilot and continuous-head control

**Files:**
- Create: `scripts/run_rsc_pf_formal_v4_3_pilot.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_pilot.py`
- Modify: `src/joint_dispatch/formal_v4_models.py`

**Interfaces:**
- Produces four pilot rows: `stage_p_regime`, `rsc_pf_joint`, `decoupled_rsc_pf`, and `continuous_thermal_head`.
- Produces: `PILOT_EVIDENCE.json` and `PILOT_DECISION.json`.

- [ ] **Step 1: Write pilot-contract tests**

```python
def test_pilot_contains_exact_mechanism_rows(receipt):
    assert tuple(row["method_id"] for row in receipt["rows"]) == (
        "stage_p_regime", "rsc_pf_joint", "decoupled_rsc_pf", "continuous_thermal_head"
    )

def test_pilot_cannot_access_evaluation_year(receipt):
    assert receipt["evaluation_year_accessed"] is False
    assert receipt["authorized_gate1"] is receipt["checks"]["all_passed"]
```

- [ ] **Step 2: Run the tests and verify the pilot script is missing**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_pilot.py --basetemp D:\Paper\pytest_tmp_v43_pilot_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement the named continuous-head control**

Reuse `RSCPFModel` exactly as formal-v4.2 for the control. It must use the same train/selection origins, epoch budget, scheduler, teacher, and physical decoder as the regime-aware rows. Label it `continuous_thermal_head`; do not call it an external baseline.

- [ ] **Step 4: Implement pilot checks**

Authorize Gate 1 only if the regime-aware Stage P materially reduces both inactive cooling and inactive heating leakage versus the continuous head, retains acceptable active-only WAPE under the frozen contract, produces finite three-class metrics, preserves physical feasibility, and the joint row records positive dispatch-to-regime and dispatch-to-magnitude gradient norms.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_pilot.py tests\test_joint_dispatch_formal_v4_3_model.py tests\test_joint_dispatch_formal_v4_3_training.py --basetemp D:\Paper\pytest_tmp_v43_pilot -p no:cacheprovider -q
git add scripts/run_rsc_pf_formal_v4_3_pilot.py src/joint_dispatch/formal_v4_models.py tests/test_joint_dispatch_formal_v4_3_pilot.py
git commit -m "feat: add regime-aware mechanism pilot"
```

---

### Task 8: Preserve baseline fairness and artifact lineage

**Files:**
- Create: `src/joint_dispatch/formal_v4_3_artifacts.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_artifacts.py`
- Modify: `src/joint_dispatch/formal_v4_3_training.py`

**Interfaces:**
- Produces checkpoint lineage fields `contract_sha256`, `normalization_sha256`, `thermal_magnitude_receipt_sha256`, `parent_checkpoint_sha256`, `teacher_sha256`, `method_id`, and `seed`.
- Produces: `assert_shared_parent_v43(joint, decoupled) -> None`.

- [ ] **Step 1: Write lineage and fair-decoupling tests**

```python
def test_joint_and_decoupled_share_byte_identical_parents(joint, decoupled):
    assert joint["stage_p_parent_sha256"] == decoupled["stage_p_parent_sha256"]
    assert joint["stage_s_parent_sha256"] == decoupled["stage_s_parent_sha256"]
    assert joint["training_seed"] == decoupled["training_seed"]

def test_v42_checkpoint_cannot_be_claimed_by_v43(checkpoint):
    checkpoint["lineage"]["contract_version"] = "formal-v4.2"
    with pytest.raises(ValueError, match="formal-v4.3"):
        validate_v43_checkpoint(checkpoint)
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_artifacts.py --basetemp D:\Paper\pytest_tmp_v43_artifacts_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement immutable v4.3 receipt writers and parent checks**

Use write-once semantics; reject an existing output directory, missing hashes, v4.2 contract versions, unequal joint/decoupled parents, and any receipt claiming `paper_eligible=true` before Gate 2.

- [ ] **Step 4: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_artifacts.py --basetemp D:\Paper\pytest_tmp_v43_artifacts -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_3_artifacts.py src/joint_dispatch/formal_v4_3_training.py tests/test_joint_dispatch_formal_v4_3_artifacts.py
git commit -m "feat: seal formal v4.3 model lineage"
```

---

### Task 9: Build Gate 0 preflight and run the complete pre-experiment test suite

**Files:**
- Create: `scripts/run_rsc_pf_formal_v4_3_gate0.py`
- Create: `tests/test_joint_dispatch_formal_v4_3_gate0.py`

**Interfaces:**
- Produces: `GATE0_EVIDENCE.json` and `GATE0_TRANSITION.json`.
- Verifies contract, source hashes, split firewall, source thermal regimes, environment, model shapes, gradients, physical residuals, and pilot executability.

- [ ] **Step 1: Write fail-closed Gate 0 tests**

```python
@pytest.mark.parametrize("failed_check", [
    "contract", "split_firewall", "source_regime", "model_shape",
    "joint_gradient", "decoupled_gradient", "physical_residual", "pilot_entrypoint",
])
def test_gate0_denies_pilot_when_any_check_fails(failed_check, evidence):
    evidence["checks"][failed_check] = False
    decision = authorize_v43_pilot(evidence)
    assert decision.authorized_pilot is False
```

- [ ] **Step 2: Run the test and verify the Gate 0 module is missing**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_gate0.py --basetemp D:\Paper\pytest_tmp_v43_gate0_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement Gate 0 without training or evaluation access**

Read only the frozen contract, source manifests, 2015--2019 source/derived statistics, and synthetic unit batches. Gate 0 must not create learned checkpoints and must set `evaluation_year_accessed=false`.

- [ ] **Step 4: Run the complete v4.3 and compatibility suite**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_contract.py tests\test_joint_dispatch_formal_v4_3_data.py tests\test_joint_dispatch_formal_v4_3_model.py tests\test_joint_dispatch_formal_v4_3_training.py tests\test_joint_dispatch_formal_v4_3_metrics.py tests\test_joint_dispatch_formal_v4_3_gate1.py tests\test_joint_dispatch_formal_v4_3_pilot.py tests\test_joint_dispatch_formal_v4_3_artifacts.py tests\test_joint_dispatch_formal_v4_3_gate0.py tests\test_joint_dispatch_formal_v4_2_contract.py tests\test_joint_dispatch_formal_v4_2_training.py tests\test_joint_dispatch_formal_v4_objective.py --basetemp D:\Paper\pytest_tmp_v43_all -p no:cacheprovider -q
```

Expected: all selected tests PASS; no formal run directory is created.

- [ ] **Step 5: Commit**

```powershell
git add scripts/run_rsc_pf_formal_v4_3_gate0.py tests/test_joint_dispatch_formal_v4_3_gate0.py
git commit -m "feat: add formal v4.3 preflight gate"
```

---

### Task 10: Execute fresh Gate 0 and the one-seed pilot, then stop for evidence review

**Files:**
- Create at runtime: `reports/joint_forecast_dispatch_formal_v4_3/<new-run-id>/gate0/`
- Create at runtime only when authorized: `reports/joint_forecast_dispatch_formal_v4_3/<new-run-id>/pilot/`
- Review: `GATE0_EVIDENCE.json`, `PILOT_EVIDENCE.json`, and `PILOT_DECISION.json`

**Interfaces:**
- Consumes: committed v4.3 source and frozen contract.
- Produces: an explicit decision to proceed to Gate 1 or stop; it does not start Gate 1 automatically in this task.

- [ ] **Step 1: Choose a fresh immutable run id and run Gate 0**

```powershell
$Py = 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe'
$RunId = 'formal_v4_3_20260905_a'
& $Py scripts\run_rsc_pf_formal_v4_3_gate0.py --contract configs\joint_forecast_dispatch_formal_v4_3.json --output-root reports\joint_forecast_dispatch_formal_v4_3 --run-id $RunId
```

Expected: the transition receipt contains `authorized_pilot=true`, `evaluation_year_accessed=false`, and every named check is true. Otherwise stop.

- [ ] **Step 2: Run the one-seed pilot only after authorization**

```powershell
& $Py scripts\run_rsc_pf_formal_v4_3_pilot.py --contract configs\joint_forecast_dispatch_formal_v4_3.json --output-root reports\joint_forecast_dispatch_formal_v4_3 --run-id $RunId
```

Expected: a write-once pilot receipt with four exact mechanism rows and no access to 2020.

- [ ] **Step 3: Perform the evidence review before any Gate 1 run**

Verify:

```text
authorized_gate1 == true
evaluation_year_accessed == false
inactive cooling leakage improved vs continuous_thermal_head
inactive heating leakage improved vs continuous_thermal_head
active-only cooling and heating guardrails passed
regime metrics are finite for all three classes
joint dispatch-to-regime and dispatch-to-magnitude gradient norms > 0
decoupled dispatch-to-forecast gradient norms == 0
physical residual maximum <= contract threshold
```

- [ ] **Step 4: Stop and record the decision**

If any check fails, do not widen thresholds or access 2020. Diagnose from the pilot receipts and amend the design through a new reviewed plan. If every check passes, the next separately authorized action is a fresh formal-v4.3 Gate 1 calibration using the nine frozen candidates.

---

## Self-Review Record

- **Spec coverage:** The plan covers the four-task/gas boundary, learned three-state thermal head, soft differentiable reconstruction, active-only normalization, five-part forecast loss, Stage P/S/J behavior, gradient boundary, full-chronology and stress metrics, baseline fairness, lineage, Gate 0, and one-seed pilot. Gate 2 is intentionally excluded until Gate 1 authorization.
- **Placeholder scan:** No `TBD`, `TODO`, unspecified error handling, or deferred implementation steps remain. Optional hard gating and gradient surgery are explicitly outside the current implementation.
- **Type consistency:** Regime order is consistently `off=0`, `cooling=1`, `heating=2`; task order remains electricity/cooling/heating/gas; model shapes remain four horizons, four forecast tasks, 15 controls, and 21 dispatch variables.
