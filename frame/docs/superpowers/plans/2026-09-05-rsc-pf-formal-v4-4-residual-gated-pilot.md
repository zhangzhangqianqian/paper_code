# RSC-PF Formal-v4.4 Residual-Gated Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a version-isolated, auditable formal-v4.4 residual-gated RSC-PF implementation and execute only its truthful Gate 0 and bounded 2019 Pilot, stopping before the formal Gate 1 experiment.

**Architecture:** Preserve the committed Scheme2R four-task forecaster, state encoder, 15-control scheduler, and 21-output physical decoder. Add a zero-initialized three-state residual gate above a train-only causal transition prior and zero-initialized cooling/heating magnitude residuals, then compare matched joint and gradient-decoupled branches using measured full-chronology 2019 evidence.

**Tech Stack:** Python 3.10, PyTorch, NumPy, JSON/NPZ/PT immutable artifacts, pytest, PowerShell, existing formal-v4.2 data/teacher/decoder utilities with hash-verified lineage.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-09-05-rsc-pf-formal-v4-4-residual-gated-forecasting-design.md` at commit `bb876c4`.
- Formal-v4.3 source and failed receipts are diagnostic evidence; do not delete, overwrite, import, or relabel them as v4.4.
- Training years are exactly 2015--2018; selection is exactly 2019; evaluation year 2020 remains sealed; 2021 is excluded.
- Preserve forecast task order `[electricity, cooling, heating, gas_prior]`; gas is supervised auxiliary context, not a rigid terminal-demand balance.
- Preserve scheduler input shape `[B,4,10]`, 15 continuous controls, and 21 decoded dispatch outputs.
- Use soft three-state probabilities on the scheduling path. Hard regimes are diagnostics only. Do not add calendar masks, future labels, target-derived inputs, or thresholded forecasts.
- Fit normalization, active-magnitude statistics, class weights, and transition priors from the complete permitted 2015--2018 training split, never the Pilot subset or 2019.
- The Pilot uses one seed, 4,096 stratified training windows, purged internal validation blocks, and complete 2019 chronological evaluation.
- All transition decisions are fail closed and computed from stored arrays. Missing hashes, non-finite values, synthetic success flags, or pre-Gate-2 access to 2020 deny authorization.
- Stop immediately if Gate 0 or Pilot fails. Do not run formal Gate 1, Gate 2, the full baseline matrix, ablations beyond the continuous-head control, or manuscript generation under this plan.
- Keep all v4.4 source, scripts, tests, configuration, and reports in new versioned paths; avoid modifying the existing dirty v4.3 files.

---

### Task 0: Freeze the existing formal-v4.3 diagnostic implementation

**Files:**
- Stage without editing: `configs/joint_forecast_dispatch_formal_v4_3.json`
- Stage without editing: `docs/superpowers/specs/2026-09-05-rsc-pf-formal-v4-3-regime-aware-forecasting-design.md`
- Stage without editing: `docs/superpowers/plans/2026-09-05-rsc-pf-formal-v4-3-regime-aware-forecasting.md`
- Stage without editing: `scripts/run_rsc_pf_formal_v4_3_gate0.py`, `scripts/run_rsc_pf_formal_v4_3_pilot.py`
- Stage without editing: `src/joint_dispatch/formal_v4_3_artifacts.py`, `formal_v4_3_contract.py`, `formal_v4_3_data.py`, `formal_v4_3_gate1.py`, `formal_v4_3_metrics.py`, `formal_v4_3_pilot.py`, `formal_v4_3_training.py`
- Stage without editing: `src/joint_dispatch/formal_v4_models.py`, `src/joint_dispatch/formal_v4_objective.py`
- Stage without editing: the nine existing `tests/test_joint_dispatch_formal_v4_3_*.py` files enumerated in Step 3

**Interfaces:**
- Produces an immutable Git parent containing the exact failed v4.3 implementation used by run `formal_v4_3_20260905_f`.
- Does not stage `third_party/iTransformer_source/` or experiment reports.

- [ ] **Step 1: Inspect the exact v4.3-only diff**

```powershell
git diff -- src/joint_dispatch/formal_v4_models.py src/joint_dispatch/formal_v4_objective.py
git status --short
```

Confirm the two modified generic files contain only the previously tested v4.3 additions and the supervised-forecast-loss hook; do not edit them in this task.

- [ ] **Step 2: Re-run the frozen v4.3 regression set**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_3_contract.py tests\test_joint_dispatch_formal_v4_3_data.py tests\test_joint_dispatch_formal_v4_3_model.py tests\test_joint_dispatch_formal_v4_3_training.py tests\test_joint_dispatch_formal_v4_3_metrics.py tests\test_joint_dispatch_formal_v4_3_pilot.py tests\test_joint_dispatch_formal_v4_3_gate0.py tests\test_joint_dispatch_formal_v4_3_gate1.py tests\test_joint_dispatch_formal_v4_3_artifacts.py --basetemp D:\Paper\pytest_tmp_v43_freeze -p no:cacheprovider -q
```

Expected: all nine existing v4.3 test files pass. This validates code behavior only; it does not change the failed scientific Pilot decision.

- [ ] **Step 3: Stage only the v4.3 diagnostic source and tests**

```powershell
git add configs/joint_forecast_dispatch_formal_v4_3.json docs/superpowers/specs/2026-09-05-rsc-pf-formal-v4-3-regime-aware-forecasting-design.md docs/superpowers/plans/2026-09-05-rsc-pf-formal-v4-3-regime-aware-forecasting.md scripts/run_rsc_pf_formal_v4_3_gate0.py scripts/run_rsc_pf_formal_v4_3_pilot.py src/joint_dispatch/formal_v4_3_artifacts.py src/joint_dispatch/formal_v4_3_contract.py src/joint_dispatch/formal_v4_3_data.py src/joint_dispatch/formal_v4_3_gate1.py src/joint_dispatch/formal_v4_3_metrics.py src/joint_dispatch/formal_v4_3_pilot.py src/joint_dispatch/formal_v4_3_training.py src/joint_dispatch/formal_v4_models.py src/joint_dispatch/formal_v4_objective.py tests/test_joint_dispatch_formal_v4_3_artifacts.py tests/test_joint_dispatch_formal_v4_3_contract.py tests/test_joint_dispatch_formal_v4_3_data.py tests/test_joint_dispatch_formal_v4_3_gate0.py tests/test_joint_dispatch_formal_v4_3_gate1.py tests/test_joint_dispatch_formal_v4_3_metrics.py tests/test_joint_dispatch_formal_v4_3_model.py tests/test_joint_dispatch_formal_v4_3_pilot.py tests/test_joint_dispatch_formal_v4_3_training.py
git diff --cached --check
git commit -m "chore: freeze failed formal v4.3 pilot implementation"
```

---

### Task 1: Freeze the formal-v4.4 contract

**Files:**
- Create: `configs/joint_forecast_dispatch_formal_v4_4.json`
- Create: `src/joint_dispatch/formal_v4_4_contract.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_contract.py`

**Interfaces:**
- Consumes canonical JSON hashing from `src/joint_dispatch/formal_v4_2_contract.py`.
- Produces `FormalV44Contract.load(path: str | Path) -> FormalV44Contract`.
- Produces properties `train_years`, `selection_year`, `evaluation_year`, `forecast_tasks`, `rigid_demands`, `candidate_grid`, and `contract_sha256`.

- [ ] **Step 1: Write the failing contract tests**

```python
def test_v44_contract_freezes_scientific_boundaries():
    c = FormalV44Contract.load(CONTRACT)
    assert c.train_years == (2015, 2016, 2017, 2018)
    assert c.selection_year == 2019 and c.evaluation_year == 2020
    assert c.forecast_tasks == ("electricity", "cooling", "heating", "gas_prior")
    assert c.rigid_demands == ("electricity", "cooling", "heating")
    assert c.control_dim == 15 and c.dispatch_dim == 21
    assert c.pilot_train_windows == 4096
    assert len(c.candidate_grid) == 4

def test_v44_contract_rejects_calendar_mask(tmp_path):
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["thermal_regime"]["hard_calendar_mask"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="calendar"):
        FormalV44Contract.load(path)
```

- [ ] **Step 2: Run the contract test and confirm the missing-module failure**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_contract.py --basetemp D:\Paper\pytest_tmp_v44_contract_red -p no:cacheprovider -q
```

Expected: FAIL because `formal_v4_4_contract` is not defined.

- [ ] **Step 3: Create the exact v4.4 protocol additions**

Copy `configs/joint_forecast_dispatch_formal_v4_2.json` as the starting document. Preserve `lookback`, `horizon`, `latent_control_dim`, gas semantics, task/exogenous/dispatch/status orders, method registry, capacity policy, renewable forecast policy, and external-source paths. Change `schema_version` to `joint-forecast-dispatch-formal-v4.4`, change the report output root to `reports/joint_forecast_dispatch_formal_v4_4`, and replace the former Pilot/training selection fields with the following nested frozen sections. Add `rigid_demand_order: ["electricity", "cooling", "heating"]` explicitly.

```json
{
  "protocol": {"id": "formal-v4.4", "train_years": [2015, 2016, 2017, 2018], "selection_year": 2019, "evaluation_year": 2020, "excluded_years": [2021], "pilot_train_windows": 4096, "pilot_seed": 2026},
  "thermal_regime": {"classes": ["off", "cooling", "heating"], "epsilon": 1e-9, "laplace_alpha": 1.0, "hard_calendar_mask": false, "hard_scheduling_gate": false, "zero_initialize_residuals": true},
  "pilot_budget": {"batch_size": 32, "minimum_epochs": 5, "early_stopping_patience": 3, "p0_max_epochs": 20, "p1_max_epochs": 15, "s_max_epochs": 15, "j_max_epochs": 10, "p0_forecaster_lr": 0.001, "p1_base_lr": 0.0002, "p1_head_lr": 0.001, "s_scheduler_lr": 0.001, "j_forecaster_lr": 0.0002, "j_head_lr": 0.0005, "j_scheduler_lr": 0.0005, "weight_decay": 0.00001, "max_grad_norm": 1.0},
  "pilot_candidate": {"temperature": 1.00, "inactive_leakage_weight": 0.25},
  "candidate_grid": [{"temperature": 1.00, "inactive_leakage_weight": 0.25}, {"temperature": 0.75, "inactive_leakage_weight": 0.25}, {"temperature": 1.00, "inactive_leakage_weight": 0.50}, {"temperature": 0.75, "inactive_leakage_weight": 0.50}],
  "pilot_thresholds": {"minimum_leakage_reduction": 0.30, "maximum_active_wape_relative_degradation": 0.05, "maximum_electricity_gas_wape_relative_degradation": 0.02, "maximum_four_task_score_relative_degradation": 0.02, "minimum_transition_balanced_accuracy_gain": 0.01, "maximum_decoupled_decision_gradient": 1e-12, "maximum_physical_residual": 1e-6}
}
```

- [ ] **Step 4: Implement strict parsing and canonical hashing**

```python
@dataclass(frozen=True)
class FormalV44Contract:
    payload: Mapping[str, Any]
    contract_sha256: str

    @classmethod
    def load(cls, path: str | Path) -> "FormalV44Contract":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        validate_v44_payload(payload)
        return cls(payload, canonical_sha256(payload))
```

Reject reordered tasks, a gas rigid-demand equality, dimensions other than 15/21, any year mutation, unlisted candidates, non-positive temperatures, and enabled hard gates or masks.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_contract.py --basetemp D:\Paper\pytest_tmp_v44_contract -p no:cacheprovider -q
git add configs/joint_forecast_dispatch_formal_v4_4.json src/joint_dispatch/formal_v4_4_contract.py tests/test_joint_dispatch_formal_v4_4_contract.py
git commit -m "feat: freeze formal v4.4 contract"
```

---

### Task 2: Build train-only thermal labels and transition priors

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_regime.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_regime.py`

**Interfaces:**
- Consumes physical targets `[N,4,4]`, physical load histories `[N,24,4]`, and training timestamps.
- Produces `derive_thermal_regimes(target, epsilon) -> np.ndarray[N,4]`.
- Produces `derive_last_observed_regime(load_history, epsilon) -> np.ndarray[N]`.
- Produces `fit_thermal_prior(train_target, train_history, train_times, alpha=1.0) -> ThermalPriorReceiptV44`, whose transition array is `[4,3,3]`.

- [ ] **Step 1: Write failing prior and year-boundary tests**

```python
def test_transition_prior_is_normalized_and_horizon_conditioned():
    r = fit_thermal_prior(TARGET, HISTORY, TRAIN_TIMES, alpha=1.0)
    assert r.transition_probability.shape == (4, 3, 3)
    np.testing.assert_allclose(r.transition_probability.sum(-1), 1.0)
    assert r.years == (2015, 2016, 2017, 2018)

def test_prior_rejects_selection_year():
    times = TRAIN_TIMES.copy()
    times[0] = np.datetime64("2019-01-01T00")
    with pytest.raises(ValueError, match="training years"):
        fit_thermal_prior(TARGET, HISTORY, times)
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_regime.py --basetemp D:\Paper\pytest_tmp_v44_regime_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement fail-closed labels and Laplace-smoothed priors**

```python
counts = np.full((4, 3, 3), float(alpha), dtype=np.float64)
for horizon in range(4):
    np.add.at(counts[horizon], (last_regime, future_regime[:, horizon]), 1.0)
probability = counts / counts.sum(axis=-1, keepdims=True)
```

Raise on negative thermal targets, simultaneous cooling/heating, non-finite arrays, timestamps outside 2015--2018, or any class with no unsmoothed observation. Store counts, active-magnitude mean/scale, years, and input hashes.

- [ ] **Step 4: Add deterministic source-data regression and commit**

Load only the permitted base training artifact through the formal-v4.2 reader, calculate twice, and assert byte-identical canonical JSON and zero simultaneous cooling/heating observations.

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_regime.py --basetemp D:\Paper\pytest_tmp_v44_regime -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_regime.py tests/test_joint_dispatch_formal_v4_4_regime.py
git commit -m "feat: add v4.4 causal thermal priors"
```

---

### Task 3: Create the purged, stratified Pilot index contract

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_pilot_data.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_pilot_data.py`

**Interfaces:**
- Produces `build_pilot_indices(train_origins, train_last_regime, train_future_regime, selection_origins, selection_last_regime, selection_future_regime, n_train=4096, seed=2026) -> PilotSplitReceiptV44`.
- Produces index arrays `train`, `early_stop`, `selection_full`, and `selection_stress` plus hashes.
- Produces `build_v44_batches(materialized, normalization, indices, batch_size, teacher_dispatch=None) -> list[dict[str, Tensor]]` with causal `last_thermal_regime` and supervised `thermal_regime_target` fields.

- [ ] **Step 1: Write failing disjointness and determinism tests**

```python
def test_pilot_indices_are_deterministic_purged_and_year_safe():
    a = build_pilot_indices(TRAIN_TIMES, TRAIN_LAST, TRAIN_FUTURE, SELECTION_TIMES, SELECTION_LAST, SELECTION_FUTURE, n_train=4096, seed=2026)
    b = build_pilot_indices(TRAIN_TIMES, TRAIN_LAST, TRAIN_FUTURE, SELECTION_TIMES, SELECTION_LAST, SELECTION_FUTURE, n_train=4096, seed=2026)
    np.testing.assert_array_equal(a.train, b.train)
    assert len(a.train) == 4096
    assert set(a.train).isdisjoint(a.early_stop)
    assert set(years(SELECTION_TIMES[a.selection_full])) == {2019}
    assert min_time_distance(TRAIN_TIMES[a.train], TRAIN_TIMES[a.early_stop]) >= np.timedelta64(28, "h")

def test_last_regime_is_derived_before_normalization(materialized, normalization):
    batches = build_v44_batches(materialized, normalization, np.array([0]), 1)
    expected = derive_last_observed_regime(materialized.load_history[[0]])
    np.testing.assert_array_equal(batches[0]["last_thermal_regime"].numpy(), expected)
    assert batches[0]["thermal_regime_target"].shape == (1, 4)
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_pilot_data.py --basetemp D:\Paper\pytest_tmp_v44_split_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement validation blocks, purge, and deterministic strata**

Reserve 8--21 January, April, July, and October 2018 for early stopping and exclude training origins within 28 hours. Use all valid 2019 origins for `selection_full`.

```python
stratum = np.stack((season_id(train_origins), dominant_regime(train_future_regime), transition_flag(train_last_regime, train_future_regime)), axis=1)
train = deterministic_proportional_sample(candidate_indices, stratum, size=4096, seed=seed)
```

Build `selection_stress` from all 2019 transition windows plus an equal-sized seeded sample of active non-transition windows. Store exact indices and hashes.

Build batches from the unnormalized physical history first, then apply the existing train-only normalization to model inputs. Future regime labels and transition flags are loss/evaluation fields only; assert that neither is passed into the model's forecast or scheduler feature tensors.

- [ ] **Step 4: Add adversarial overlap/year tests, run, and commit**

```python
with pytest.raises(ValueError, match="purge"):
    validate_pilot_split(RECEIPT_WITH_OVERLAP, TRAIN_TIMES, SELECTION_TIMES)
with pytest.raises(ValueError, match="2020"):
    validate_pilot_split(RECEIPT_WITH_2020, TRAIN_TIMES, SELECTION_TIMES)
```

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_pilot_data.py --basetemp D:\Paper\pytest_tmp_v44_split -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_pilot_data.py tests/test_joint_dispatch_formal_v4_4_pilot_data.py
git commit -m "feat: add purged v4.4 pilot split"
```

---

### Task 4: Implement the residual-gated RSC-PF model

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_model.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_model.py`

**Interfaces:**
- Consumes committed `_FormalV4Base`, `_pad_device_history`, Scheme2R forecaster, and physical decoder from `src/joint_dispatch/formal_v4_models.py`.
- Produces `ResidualGatedRSCPFModel.forward(last_thermal_regime: Tensor[B], detach_forecast_for_dispatch: bool=False, **inputs) -> FormalV44ForwardOutput`.
- Produces parameter groups `base_forecaster_parameters()`, `gate_parameters()`, `magnitude_residual_parameters()`, and `scheduler_parameters()`.

- [ ] **Step 1: Write failing initialization, shape, and scheduler-input tests**

```python
def test_zero_residuals_preserve_scheme2r_magnitudes(model, batch):
    out = model(**batch)
    expected = model.core.forecast_to_physical(out.base_forecast_normalized)
    torch.testing.assert_close(out.thermal_magnitudes, expected[..., 1:3])

def test_v44_output_contract(model, batch):
    out = model(**batch)
    assert out.forecast_physical.shape == (2, 4, 4)
    assert out.regime_logits.shape == (2, 4, 3)
    assert out.thermal_residuals.shape == (2, 4, 2)
    assert out.controls.shape == (2, 4, 15)
    assert out.dispatch.shape == (2, 4, 21)
    torch.testing.assert_close(out.forecast_physical[..., [0, 3]], out.base_forecast_physical[..., [0, 3]])
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_model.py --basetemp D:\Paper\pytest_tmp_v44_model_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement the typed output and zero-initialized head**

```python
@dataclass(frozen=True)
class FormalV44ForwardOutput:
    base_forecast_normalized: Tensor
    base_forecast_physical: Tensor
    forecast_normalized: Tensor
    forecast_physical: Tensor
    regime_logits: Tensor
    regime_probabilities: Tensor
    thermal_magnitudes: Tensor
    thermal_residuals: Tensor
    controls: Tensor
    dispatch: Tensor

class ResidualThermalHead(nn.Module):
    def __init__(self, transition_probability: Tensor, temperature: float) -> None:
        super().__init__()
        self.register_buffer("transition_log_prior", transition_probability.clamp_min(1e-12).log())
        self.hidden = nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.LayerNorm(64))
        self.output = nn.Linear(64, 5)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
```

- [ ] **Step 4: Implement prior gathering and physical reconstruction**

```python
prior = torch.stack([self.transition_log_prior[h].index_select(0, last_thermal_regime) for h in range(4)], dim=1)
features = torch.cat((base_forecast, state[:, None].expand(-1, 4, -1)), dim=-1)
residual = self.output(self.hidden(features))
logits = prior + residual[..., :3]
probability = torch.softmax(logits / self.temperature, dim=-1)
thermal_norm = base_forecast[..., 1:3] + residual[..., 3:5]
thermal_mag = F.softplus(task_mean[1:3] + task_scale[1:3] * thermal_norm)
cool = probability[..., 1] * thermal_mag[..., 0]
heat = probability[..., 2] * thermal_mag[..., 1]
```

Use base physical electricity and gas unchanged. Reconstruct normalized outputs from the final physical tensor, detach only the final four-task forecast when requested, and pass exactly ten physical/context features to the existing scheduler.

- [ ] **Step 5: Add named gradient tests**

```python
def test_decision_gradient_reaches_all_joint_groups(model, batch):
    loss = model(**batch).dispatch.square().mean()
    loss.backward()
    for name, parameters in model.v44_parameter_groups().items():
        norm = sum(float(p.grad.norm()) for p in parameters if p.grad is not None)
        assert norm > 0.0, name
```

- [ ] **Step 6: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_model.py --basetemp D:\Paper\pytest_tmp_v44_model -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_model.py tests/test_joint_dispatch_formal_v4_4_model.py
git commit -m "feat: add residual-gated RSC-PF model"
```

---

### Task 5: Implement normalized curriculum forecast losses

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_loss.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_loss.py`

**Interfaces:**
- Produces `curriculum_weights(epoch: int, ramp_epochs: int, leakage_target: float) -> CurriculumWeightsV44`.
- Produces `forecast_loss_v44(output, batch, prior_receipt, weights) -> ForecastLossV44`.
- `ForecastLossV44` contains `total`, `electricity_gas`, `regime`, `active_magnitude`, `point`, and `inactive_leakage` tensors.

- [ ] **Step 1: Write failing masking and curriculum tests**

```python
def test_inactive_leakage_uses_final_soft_forecast_only():
    loss = forecast_loss_v44(OUTPUT, BATCH, PRIOR, WEIGHTS)
    expected = OUTPUT.forecast_physical[..., 1:3][INACTIVE_MASK].abs().mean()
    torch.testing.assert_close(loss.inactive_leakage_physical, expected)

def test_curriculum_is_fixed_and_monotone():
    early = curriculum_weights(0, ramp_epochs=5, leakage_target=0.25)
    late = curriculum_weights(5, ramp_epochs=5, leakage_target=0.25)
    assert early.point == 0.25 and late.point == 1.0
    assert early.inactive_leakage == 0.10 and late.inactive_leakage == 0.25
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_loss.py --basetemp D:\Paper\pytest_tmp_v44_loss_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement per-component normalized losses**

Use normalized Smooth-L1 for electricity, gas, point forecast, and active magnitudes; class-weighted cross entropy for regimes; and active-magnitude-scale-normalized physical leakage. Calculate components per sample before transition weights.

```python
progress = min(max(epoch / max(ramp_epochs, 1), 0.0), 1.0)
point_weight = 0.25 + 0.75 * progress
inactive_weight = 0.10 + (leakage_target - 0.10) * progress
total = eg + regime + active_magnitude + point_weight * point_loss + inactive_weight * leakage
```

Gas receives weight `0.25`. Clamp inverse-frequency class weights to `[0.5, 2.0]`. Raise on simultaneous thermal targets or missing physical/normalized labels.

- [ ] **Step 4: Add finite-gradient and scale-invariance tests**

Scale physical cooling/heating targets and active statistics by the same positive factor and assert unchanged normalized active/leakage losses. Backpropagate every component separately and assert finite gradients.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_loss.py --basetemp D:\Paper\pytest_tmp_v44_loss -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_loss.py tests/test_joint_dispatch_formal_v4_4_loss.py
git commit -m "feat: add v4.4 forecast curriculum"
```

---

### Task 6: Implement matched P0, P1, S, and J training

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_training.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_training.py`

**Interfaces:**
- Produces `run_stage_p0_v44`, `run_stage_p1_v44`, `run_continuous_control_v44`, `run_stage_s_v44`, and `run_stage_j_pair_v44`.
- Produces `StageReceiptV44` and `JointPairReceiptV44` containing parent hash, epochs, optimizer steps, loss history, stopping reason, and named gradient norms.
- Produces `named_autograd_norms(loss, groups, retain_graph=False) -> dict[str, float]` and `sha256_state_dict(model) -> str`.

- [ ] **Step 1: Write failing parent and gradient-boundary tests**

```python
def test_joint_pair_uses_identical_parents_and_steps(stage_s, batches):
    pair = run_stage_j_pair_v44(stage_s, batches, BUDGET)
    assert pair.joint.parent_sha256 == pair.decoupled.parent_sha256
    assert pair.joint.optimizer_steps == pair.decoupled.optimizer_steps
    assert pair.joint.gradient_norms["decision_to_gate"] > 0
    assert pair.joint.gradient_norms["decision_to_magnitude"] > 0
    assert pair.decoupled.gradient_norms["decision_to_base"] <= 1e-12
    assert pair.decoupled.gradient_norms["forecast_to_base"] > 0
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_training.py --basetemp D:\Paper\pytest_tmp_v44_training_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement P0 and the matched continuous control**

P0 trains the committed `RSCPFModel` forecast/state pathway. P1 loads P0 forecaster/state parameters into `ResidualGatedRSCPFModel`. The continuous control loads the byte-identical P0 parent and receives the same number of forecast parameter-group updates as P1.

```python
assert sha256_state_dict(p1_parent) == sha256_state_dict(control_parent)
assert p1_receipt.forecast_optimizer_steps == control_receipt.forecast_optimizer_steps
```

- [ ] **Step 4: Implement Stage S and a single matched J step per batch**

```python
joint_total = forecast_loss + imitation_loss + decision_loss + physical_penalty
decoupled_total = forecast_loss_dec + imitation_loss_dec + detached_decision_loss + physical_penalty_dec
```

Use one optimizer call per branch and batch. Joint decision gradients reach every forecast group; decoupled decision gradients do not. Learning rates, batch order, curriculum epoch, and seed are identical.

- [ ] **Step 5: Record group-specific gradients before updates**

```python
groups = {"base": model.base_forecaster_parameters(), "gate": model.gate_parameters(), "magnitude": model.magnitude_residual_parameters(), "scheduler": model.scheduler_parameters()}
decision_norms = named_autograd_norms(decision_loss, groups, retain_graph=True)
forecast_norms = named_autograd_norms(forecast_loss, groups, retain_graph=True)
```

Never assign one combined forecaster norm to several fields.

- [ ] **Step 6: Add deterministic resume and early-stopping tests**

Save model, optimizer, epoch, sampler state, and RNG state. Resume a two-epoch test after epoch one and assert equality with an uninterrupted two-epoch run. Early stopping may inspect only purged training-year validation indices.

- [ ] **Step 7: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_training.py --basetemp D:\Paper\pytest_tmp_v44_training -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_training.py tests/test_joint_dispatch_formal_v4_4_training.py
git commit -m "feat: add matched v4.4 training stages"
```

---

### Task 7: Implement full-chronology metrics and Pilot authorization

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_metrics.py`
- Create: `src/joint_dispatch/formal_v4_4_pilot_gate.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_metrics.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_pilot_gate.py`

**Interfaces:**
- Produces `compute_forecast_metrics_v44(prediction, target, probability, prior_probability, regimes, times) -> ForecastMetricsV44`.
- Produces `authorize_pilot_v44(receipt: Mapping[str, Any], contract: FormalV44Contract) -> PilotDecisionV44`.
- `PilotDecisionV44` contains `authorized_gate1`, every named criterion, and explicit failure reasons.

- [ ] **Step 1: Write failing hand-calculated metric tests**

```python
def test_inactive_leakage_and_active_wape_are_not_zero_dominated():
    metrics = compute_forecast_metrics_v44(PRED, TARGET, PROB, PRIOR_PROB, REGIME, TIMES)
    assert metrics.inactive_leakage["cooling_mae"] == pytest.approx(2.0)
    assert metrics.active_only["cooling_wape"] == pytest.approx(0.10)

def test_transition_accuracy_compares_with_prior():
    metrics = compute_forecast_metrics_v44(PRED, TARGET, PROB, PRIOR_PROB, REGIME, TIMES)
    assert metrics.transition["balanced_accuracy_gain"] == pytest.approx(0.02)
```

- [ ] **Step 2: Write failing corrupt-receipt tests**

```python
@pytest.mark.parametrize("mutation", [
    lambda r: r["joint"].update({"penalized_objective": float("inf")}),
    lambda r: r["lineage"].pop("source_manifest_sha256"),
    lambda r: r["accessed_years"].append(2020),
    lambda r: r["joint"]["gradient_norms"].update({"decision_to_gate": 0.0}),
])
def test_pilot_denies_corrupt_receipts(valid_receipt, mutation):
    mutation(valid_receipt)
    assert not authorize_pilot_v44(valid_receipt, CONTRACT).authorized_gate1
```

- [ ] **Step 3: Run both test files and confirm they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_metrics.py tests\test_joint_dispatch_formal_v4_4_pilot_gate.py --basetemp D:\Paper\pytest_tmp_v44_metrics_red -p no:cacheprovider -q
```

- [ ] **Step 4: Implement metrics and all nine criteria**

Calculate MAE, RMSE, WAPE, normalized four-task score, confusion matrix, macro-F1, balanced accuracy, per-horizon transition metrics, inactive leakage MAE/P95/total, and active-only metrics directly from arrays. A zero WAPE denominator is unavailable and denies authorization.

```python
criteria = {
    "leakage": cool_ratio <= 0.70 and heat_ratio <= 0.70,
    "active": cool_active_ratio <= 1.05 and heat_active_ratio <= 1.05,
    "unaffected": electricity_ratio <= 1.02 and gas_ratio <= 1.02,
    "overall": four_task_ratio <= 1.02,
    "regime": macro_f1 >= prior_macro_f1 and transition_gain >= 0.01,
    "decision": joint_objective <= decoupled_objective and not_worse_shortage,
    "gradient": joint_group_norms_positive and decoupled_decision_norms_below_tolerance,
    "physics": all_residuals_within_tolerance,
    "integrity": hashes_complete and values_finite and 2020 not in accessed_years,
}
authorized = all(criteria.values())
```

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_metrics.py tests\test_joint_dispatch_formal_v4_4_pilot_gate.py --basetemp D:\Paper\pytest_tmp_v44_metrics -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_metrics.py src/joint_dispatch/formal_v4_4_pilot_gate.py tests/test_joint_dispatch_formal_v4_4_metrics.py tests/test_joint_dispatch_formal_v4_4_pilot_gate.py
git commit -m "feat: add truthful v4.4 pilot criteria"
```

---

### Task 8: Add immutable artifacts and independent recomputation

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_artifacts.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_artifacts.py`

**Interfaces:**
- Produces `write_json_once(path, payload)`, `write_npz_once(path, arrays)`, `validate_lineage_v44(payload)`, and `recompute_pilot_receipt(report_dir, contract) -> dict`.

- [ ] **Step 1: Write failing immutability and audit tests**

```python
def test_json_writer_rejects_non_finite_and_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="finite"):
        write_json_once(path, {"x": float("inf")})
    write_json_once(path, {"x": 1.0})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"x": 2.0})

def test_recomputed_receipt_matches_saved_metrics(report_dir):
    recomputed = recompute_pilot_receipt(report_dir, CONTRACT)
    assert canonical_sha256(recomputed["metrics"]) == recomputed["metrics_sha256"]
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_artifacts.py --basetemp D:\Paper\pytest_tmp_v44_artifacts_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement strict write-once storage**

Use `json.dumps(payload, allow_nan=False, sort_keys=True)` and reject missing 64-character SHA-256 fields. Save full 2019 target, prediction, probability, dispatch, residual, sample-index, and timestamp arrays in compressed NPZ before creating the decision receipt.

- [ ] **Step 4: Implement independent recomputation**

`recompute_pilot_receipt` loads only stored arrays and the frozen contract, recalculates every metric and criterion, and compares hashes. It must not import the Pilot runner or reuse its in-memory metric objects.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_artifacts.py --basetemp D:\Paper\pytest_tmp_v44_artifacts -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_artifacts.py tests/test_joint_dispatch_formal_v4_4_artifacts.py
git commit -m "feat: add auditable v4.4 artifacts"
```

---

### Task 9: Build the measured Gate 0 preflight

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_gate0.py`
- Create: `scripts/run_rsc_pf_formal_v4_4_gate0.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_gate0.py`

**Interfaces:**
- Produces `run_gate0_v44(contract_path, source_manifest, base_train_data, base_selection_data, benchmark, capacity_receipt, output_root, run_id) -> Gate0ReceiptV44`.
- Produces `validate_gate0_receipt_v44(receipt_path, contract, expected_source_hashes) -> Gate0ReceiptV44`.
- Writes `gate0/GATE0_RECEIPT.json`, `gate0/THERMAL_PRIOR.json`, `gate0/PILOT_SPLIT.npz`, and root `GATE0_TRANSITION.json`.

- [ ] **Step 1: Write failing Gate 0 integration tests**

```python
def test_gate0_measures_every_check_and_authorizes_pilot(tmp_path, valid_inputs):
    receipt = run_gate0_v44(output_root=tmp_path, run_id="gate0_ok", **valid_inputs)
    assert receipt.authorized_pilot
    assert receipt.checks["physical_residual"].measured_max <= receipt.physical_tolerance
    assert receipt.source_manifest_sha256 != "0" * 64

def test_gate0_rejects_forbidden_year(tmp_path, valid_inputs):
    valid_inputs["base_selection_data"] = write_selection_fixture(tmp_path, year=2020)
    with pytest.raises(ValueError, match="2020"):
        run_gate0_v44(output_root=tmp_path, run_id="bad", **valid_inputs)
```

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_gate0.py --basetemp D:\Paper\pytest_tmp_v44_gate0_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement every measured Gate 0 check**

Verify contract/source/data hashes, permitted years, complete training statistics, transition priors, Pilot disjointness, zero-initialized residuals, shapes, scheduler feature count, joint/decoupled gradient boundaries, real balance/capacity/SOC/ramp residuals on a deterministic mini-batch, entry points, and output write permissions.

```python
authorized_pilot = all(check.measured and check.passed for check in checks.values())
```

The script requires every source path as an argument and contains no hard-coded old-run directory.

- [ ] **Step 4: Add source isolation checks**

Parse v4.4 imports and reject modules beginning `formal_v4_3`. Permit hash-verified generic v4.2 data, teacher, decoder, and settlement utilities; reject v4.2/v4.3 model checkpoints.

- [ ] **Step 5: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_gate0.py --basetemp D:\Paper\pytest_tmp_v44_gate0 -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_gate0.py scripts/run_rsc_pf_formal_v4_4_gate0.py tests/test_joint_dispatch_formal_v4_4_gate0.py
git commit -m "feat: add measured v4.4 gate0"
```

---

### Task 10: Build the bounded Pilot runner

**Files:**
- Create: `src/joint_dispatch/formal_v4_4_pilot.py`
- Create: `scripts/run_rsc_pf_formal_v4_4_pilot.py`
- Test: `tests/test_joint_dispatch_formal_v4_4_pilot.py`

**Interfaces:**
- Produces `run_pilot_v44(contract_path, gate0_transition, source_manifest, base_train_data, base_selection_data, benchmark, capacity_receipt, output_root, run_id) -> PilotDecisionV44`.
- Writes stage checkpoints, stored 2019 arrays, measured residual arrays, `pilot/PILOT_RECEIPT.json`, `pilot/PILOT_AUDIT.json`, and root `PILOT_TRANSITION.json`.

- [ ] **Step 1: Write a failing end-to-end synthetic Pilot test**

```python
def test_pilot_runs_all_stages_without_2020(tmp_path, synthetic_sources):
    gate0 = make_authorized_gate0(tmp_path, synthetic_sources)
    decision = run_pilot_v44(output_root=tmp_path, run_id="pilot", gate0_transition=gate0, **synthetic_sources)
    assert decision.accessed_years == (2015, 2016, 2017, 2018, 2019)
    assert set(decision.rows) == {"continuous_control", "residual_stage_p1", "rsc_pf_joint", "fair_decoupled", "transition_prior"}
    assert len(decision.audit_sha256) == 64
```

Use injected deterministic stage executors in this orchestration test so it verifies ordering, access, storage, and authorization without repeating the real 4,096-window training. Task 6 already tests the actual training functions.

- [ ] **Step 2: Run the test and confirm it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_pilot.py --basetemp D:\Paper\pytest_tmp_v44_pilot_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement explicit-source loading**

Require `--source-manifest`, `--base-train-data`, `--base-selection-data`, `--benchmark`, and `--capacity-receipt`. Verify their hashes against Gate 0 before reading arrays. Load only indices from `PILOT_SPLIT.npz`; never regenerate the split in the runner.

- [ ] **Step 4: Implement the exact stage sequence**

```text
P0 common Scheme2R -> clone P1 and continuous control
P1 residual-gate adaptation -> same-information LP teacher
S scheduler imitation -> clone byte-identical joint/decoupled parents
J matched joint and decoupled updates
full 2019 rolling evaluation -> stored arrays -> measured authorization -> independent audit
```

Generate teacher labels only for required training windows. Evaluation uses rolling state carry and existing measured settlement/residual utilities.

- [ ] **Step 5: Enforce stop semantics**

Exit `0` only when `authorized_gate1=true`, exit `3` for a measured scientific Pilot failure, and exit `2` for an integrity/runtime failure. Never invoke a Gate 1 script.

- [ ] **Step 6: Run tests and commit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_pilot.py --basetemp D:\Paper\pytest_tmp_v44_pilot -p no:cacheprovider -q
git add src/joint_dispatch/formal_v4_4_pilot.py scripts/run_rsc_pf_formal_v4_4_pilot.py tests/test_joint_dispatch_formal_v4_4_pilot.py
git commit -m "feat: add bounded v4.4 pilot runner"
```

---

### Task 11: Add adversarial audit and run protected regressions

**Files:**
- Create: `tests/test_joint_dispatch_formal_v4_4_adversarial.py`
- Create: `scripts/audit_rsc_pf_formal_v4_4_pilot.py`

**Interfaces:**
- The audit consumes a run directory and exits `0` only when independently recomputed evidence matches every primary receipt and authorization field.

- [ ] **Step 1: Write failing adversarial tests**

```python
@pytest.mark.parametrize("fault", [
    "infinite_objective", "missing_hash", "hardcoded_pass", "year_2020",
    "train_eval_overlap", "finite_only_physics", "duplicate_index",
    "joint_zero_gate_gradient", "decoupled_nonzero_decision_gradient",
])
def test_every_integrity_fault_denies_transition(valid_run, fault):
    corrupted = inject_fault(valid_run, fault)
    assert audit_run(corrupted).authorized_gate1 is False
```

- [ ] **Step 2: Run the test and confirm it fails before the audit exists**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_adversarial.py --basetemp D:\Paper\pytest_tmp_v44_adversarial_red -p no:cacheprovider -q
```

- [ ] **Step 3: Implement the independent command-line audit**

Read the contract, Gate 0 receipt, split indices, saved predictions/dispatch/residual arrays, stage receipts, and Pilot receipt. Recompute hashes, metrics, criteria, and the final boolean without importing `formal_v4_4_pilot.py`.

- [ ] **Step 4: Run all v4.4 and protected regression tests**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests\test_joint_dispatch_formal_v4_4_contract.py tests\test_joint_dispatch_formal_v4_4_regime.py tests\test_joint_dispatch_formal_v4_4_pilot_data.py tests\test_joint_dispatch_formal_v4_4_model.py tests\test_joint_dispatch_formal_v4_4_loss.py tests\test_joint_dispatch_formal_v4_4_training.py tests\test_joint_dispatch_formal_v4_4_metrics.py tests\test_joint_dispatch_formal_v4_4_pilot_gate.py tests\test_joint_dispatch_formal_v4_4_artifacts.py tests\test_joint_dispatch_formal_v4_4_gate0.py tests\test_joint_dispatch_formal_v4_4_pilot.py tests\test_joint_dispatch_formal_v4_4_adversarial.py tests\test_joint_dispatch_formal_v4_2_data.py tests\test_joint_dispatch_formal_v4_2_teacher.py tests\test_joint_dispatch_formal_v4_2_rollout.py tests\test_joint_dispatch_formal_v4_2_gate2_training.py tests\test_scheduling_proxy_decoder.py tests\test_dispatch_lp.py --basetemp D:\Paper\pytest_tmp_v44_full -p no:cacheprovider -q
```

Expected: all selected tests pass and no test accesses 2020.

- [ ] **Step 5: Commit the audit**

```powershell
git add tests/test_joint_dispatch_formal_v4_4_adversarial.py scripts/audit_rsc_pf_formal_v4_4_pilot.py
git commit -m "test: add adversarial v4.4 pilot audit"
```

---

### Task 12: Execute Gate 0 and the one-seed Pilot with stop gates

**Files:**
- Create on execution: `reports/joint_forecast_dispatch_formal_v4_4/<run-id>/...`
- Do not modify source files during this task.

**Interfaces:**
- Consumes explicit frozen source paths and the completed v4.4 implementation.
- Produces either a failed immutable receipt or an authorized Gate 1 transition; it never executes Gate 1.

- [ ] **Step 1: Confirm implementation state and record the code hash**

```powershell
git status --short
git rev-parse HEAD
```

Expected: only preserved v4.3 diagnostic files may remain uncommitted; every v4.4 implementation file is committed.

- [ ] **Step 2: Run formal-v4.4 Gate 0**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\run_rsc_pf_formal_v4_4_gate0.py --contract configs\joint_forecast_dispatch_formal_v4_4.json --source-manifest reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\protocol\SOURCE_MANIFEST.json --base-train-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_train.npz --base-selection-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_selection.npz --benchmark reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\benchmark\STANDARD_IES_BENCHMARK.yaml --capacity-receipt reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\CAPACITY_FREEZE.json --output-root reports\joint_forecast_dispatch_formal_v4_4 --run-id formal_v4_4_20260905_a
```

Expected: `authorized_pilot=true`. If false or non-zero, stop and report measured failures.

- [ ] **Step 3: Independently validate Gate 0 before training**

Load the written Gate 0 receipt with `validate_gate0_receipt_v44(...)` and confirm source, contract, prior, split, and implementation hashes match.

- [ ] **Step 4: Run the bounded Pilot only after Gate 0 passes**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\run_rsc_pf_formal_v4_4_pilot.py --contract configs\joint_forecast_dispatch_formal_v4_4.json --gate0-transition reports\joint_forecast_dispatch_formal_v4_4\formal_v4_4_20260905_a\GATE0_TRANSITION.json --source-manifest reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\protocol\SOURCE_MANIFEST.json --base-train-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_train.npz --base-selection-data reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\data\base_selection.npz --benchmark reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\benchmark\STANDARD_IES_BENCHMARK.yaml --capacity-receipt reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j\gate0\CAPACITY_FREEZE.json --output-root reports\joint_forecast_dispatch_formal_v4_4 --run-id formal_v4_4_20260905_a
```

Expected: exit `0` means Gate 1 is authorized; exit `3` means a scientific criterion failed; exit `2` means integrity/runtime failure. Both failure codes stop execution.

- [ ] **Step 5: Run the independent Pilot audit**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts\audit_rsc_pf_formal_v4_4_pilot.py --run-dir reports\joint_forecast_dispatch_formal_v4_4\formal_v4_4_20260905_a --contract configs\joint_forecast_dispatch_formal_v4_4.json
```

Expected: audit and primary decisions match on canonicalized criteria and hashes.

- [ ] **Step 6: Stop and report the Pilot result**

Report the leakage/active guardrails, regime comparison, joint-versus-decoupled objective, shortage, physical residuals, named gradient norms, and integrity result. Do not create or start a Gate 1 execution plan until the user reviews this evidence.
