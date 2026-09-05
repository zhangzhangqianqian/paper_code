# RSC-PF Matched Closed-Loop Evaluation Implementation Plan

**Goal:** Build and run a fail-closed 2019 chronological evaluation that compares the frozen single-seed RSC-PF pilot with five frozen seeds of each external baseline under one state-carry, settlement, metric, and optimizer-accounting protocol.

**Architecture:** Add a small metric module and a common chronological runner driven by method-specific action providers. Every provider consumes the current carried state, returns a four-hour forecast/plan, and delegates the executed first action to the existing canonical recourse and state-transition code. A separate CLI writes immutable per-run artifacts, while an independent auditor verifies provenance, metric definitions, method roles, the separate RSC-PF legacy replay, corrected first-origin equivalence, and zero 2020 access.

**Tech Stack:** Python 3.9, NumPy, PyTorch, SciPy/HiGHS LP, pytest, JSON/NPZ artifacts.

## Preflight Findings Incorporated into This Plan

1. The legacy RSC-PF rollout carried SOC into settlement, but rebuilt the next scheduler context from each materialized row's original SOC. The corrected matched protocol uses carried SOC consistently; therefore full legacy equality is a separate checkpoint-reconstruction test, not a requirement for corrected results.
2. The v4.6-b pilot receipt records a source-manifest SHA-256, but that exact manifest file is no longer present. This is disclosed as `receipt_hash_only`; checkpoint, contract, benchmark, capacity, and rollout artifacts remain independently hashable.
3. Existing iTransformer evaluation receipts incorrectly label the complete PTO pathway as having no inference optimizer even though one LP is solved per origin. The plan corrects future metadata without rewriting historical receipts or weights.
4. A per-method four-hour oracle first-step difference is not guaranteed to be non-negative regret. The plan instead runs one shared rolling Perfect-Information-MPC reference from the common initial state and reports signed cumulative objective gaps, final SOC, and terminal-stock-adjusted gaps.

## Global Constraints

- This is a 2019 selection-year protocol diagnostic, not a final paper comparison.
- Use all 8,709 consecutive origins from `selection_full.npz`; never open the 2020 sealed evaluation split or any 2021 artifact.
- Do not retrain or modify model weights, architectures, training losses, checkpoint selection, or manuscript text.
- RSC-PF uses one diagnostic seed (2026); each external method uses seeds 2026–2030. Do not run inferential statistics on this asymmetric diagnostic table.
- The gas target is a station-side auxiliary prior. It receives forecast metrics but is excluded from terminal balance, shortage, and no-shortage calculations.
- iTransformer-PTO and DecisionFocused-Online use one online LP per origin. RSC-PF and DigitalTwins-Policy use zero inference-time LP calls.
- One shared Perfect-Information-MPC trajectory is evaluation-only; its LP calls are counted separately from every method's inference LP calls.
- Planned physical feasibility is checked over all four planned hours, with a shadow SOC/CHP state advanced from one planned row to the next and without recourse. Settled feasibility is checked only for the executed first hour against realized demand/PV/WT.
- Use physical-residual tolerance `1e-6`, shortage tolerance `1e-8`, and RSC-PF replay tolerance `atol=1e-5`, `rtol=1e-6`.
- Existing independent-window receipts remain untouched and diagnostic-only.

---

### Task 1: Add unambiguous closed-loop metric primitives

**Files:**
- Create: `src/joint_dispatch/closed_loop_metrics.py`
- Test: `tests/test_rsc_pf_closed_loop_metrics.py`

**Interfaces:**
- Consumes: `PhysicalResidualsV42` from `src.joint_dispatch.formal_v4_2_rollout` and planned/settled dispatch arrays in canonical 21-column order.
- Produces:
  - `residual_vector(residuals: PhysicalResidualsV42) -> np.ndarray`
  - `physical_feasible(vector: np.ndarray, tolerance: float = 1e-6) -> bool`
  - `shortage_free(shortage: np.ndarray, tolerance: float = 1e-8) -> bool`
  - `recourse_distance(planned: np.ndarray, settled: np.ndarray, realized_demand: np.ndarray) -> tuple[float, float, np.ndarray]`
  - `terminal_stock_adjustment(initial_soc: float, final_soc: float, energy_capacity: float, roundtrip_efficiency: float, final_grid_price: float, final_carbon_price: float, grid_emission_factor: float) -> float`
  - `summarize_closed_loop(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests for the two feasibility concepts**

```python
def test_physical_feasibility_and_shortage_are_independent():
    assert physical_feasible(np.zeros(8)) is True
    assert shortage_free(np.array([0.0, 0.0, 2.0])) is False
    assert physical_feasible(np.array([0.0, 0.0, 2.0e-6])) is False
```

- [ ] **Step 2: Write failing tests for recourse normalization and summaries**

```python
def test_recourse_distance_uses_realized_three_carrier_demand():
    planned = np.zeros(21)
    settled = np.ones(21)
    absolute, normalized, by_channel = recourse_distance(planned, settled, np.array([10.0, 20.0, 30.0]))
    assert absolute == 9.0
    assert np.isclose(normalized, 9.0 / 60.0)
    assert by_channel.shape == (9,)

def test_summary_reports_planned_settled_and_shortage_separately():
    summary = summarize_closed_loop(make_metric_arrays())
    assert set(summary["rates"]) == {"planned_physical_feasibility", "settled_physical_feasibility", "no_shortage"}
    assert set(summary["recourse_adjustment"]) == {"absolute_mean", "absolute_median", "absolute_p95", "normalized_mean", "normalized_median", "normalized_p95"}
```

Add tests for both terminal-stock directions: ending below the common initial SOC adds replenishment cost using charge efficiency; ending above it receives only the dischargeable-energy credit using discharge efficiency. Invalid efficiency, negative capacity, or non-finite inputs fail closed.

- [ ] **Step 3: Run the focused tests and confirm the interfaces are absent**

Run:

```powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_closed_loop_metrics.py -q --basetemp D:\Paper\pytest_tmp_closed_loop_metrics -p no:cacheprovider
```

Expected: failures caused by missing `closed_loop_metrics` functions.

- [ ] **Step 4: Implement the metric primitives with fixed tolerances**

```python
PHYSICAL_TOLERANCE = 1.0e-6
SHORTAGE_TOLERANCE = 1.0e-8
RECOURSE_CHANNELS = (
    "grid", "pv_use", "wt_use", "p_chp", "q_gb",
    "q_ec", "q_ac", "p_charge", "p_discharge",
)

def residual_vector(residuals: PhysicalResidualsV42) -> np.ndarray:
    return np.asarray([
        np.max(np.abs(residuals.balance), initial=0.0),
        np.max(np.abs(residuals.capacity), initial=0.0),
        np.max(np.abs(residuals.conversion), initial=0.0),
        np.max(np.abs(residuals.soc), initial=0.0),
        np.max(np.abs(residuals.ramp), initial=0.0),
        np.max(np.abs(residuals.exclusivity), initial=0.0),
        np.max(np.abs(residuals.renewable_accounting), initial=0.0),
        np.max(np.abs(residuals.finite), initial=0.0),
    ], dtype=np.float64)

def physical_feasible(vector: np.ndarray, tolerance: float = PHYSICAL_TOLERANCE) -> bool:
    value = np.asarray(vector, dtype=np.float64)
    return bool(np.isfinite(value).all() and np.max(np.abs(value), initial=0.0) <= tolerance)

def shortage_free(shortage: np.ndarray, tolerance: float = SHORTAGE_TOLERANCE) -> bool:
    value = np.asarray(shortage, dtype=np.float64)
    return bool(np.isfinite(value).all() and np.max(value, initial=0.0) <= tolerance)

def recourse_distance(planned: np.ndarray, settled: np.ndarray, realized_demand: np.ndarray) -> tuple[float, float, np.ndarray]:
    indices = np.asarray([VARIABLES.index(name) for name in RECOURSE_CHANNELS])
    by_channel = np.abs(np.asarray(settled)[indices] - np.asarray(planned)[indices])
    absolute = float(by_channel.sum())
    denominator = max(float(np.asarray(realized_demand, dtype=np.float64).sum()), 1.0e-12)
    return absolute, absolute / denominator, by_channel
```

The nine channels are the non-derived operating decisions. Exclude fuel/conversion duplicates, curtailment complements, SOC, slack, and dump variables from the aggregate L1 distance; report shortage and dump separately. `summarize_closed_loop` must reject missing/non-finite arrays and compute means for forecast/cost/carbon/objective/shortage, cumulative raw objective, terminal SOC, terminal-stock adjustment, adjusted cumulative objective, three rates, mean/median/P95 aggregate recourse, per-channel recourse summaries, and median/P95 latency.

- [ ] **Step 5: Run the metric tests**

Expected: all tests in `test_rsc_pf_closed_loop_metrics.py` pass.

- [ ] **Step 6: Commit the metric boundary**

```powershell
git add src/joint_dispatch/closed_loop_metrics.py tests/test_rsc_pf_closed_loop_metrics.py
git commit -m "feat: separate closed-loop feasibility and adequacy metrics"
```

---

### Task 2: Implement the common chronological state-carry engine

**Files:**
- Create: `src/joint_dispatch/matched_closed_loop.py`
- Modify: `src/joint_dispatch/external_v46_data.py`
- Test: `tests/test_rsc_pf_matched_closed_loop.py`
- Test: `tests/test_rsc_pf_external_v46_data.py`

**Interfaces:**
- Consumes:
  - `ExternalV46Split`
  - `FormalV4ClosedLoopState`
  - `settle_first_step_v4`
  - `advance_formal_v4_state`
  - `calculate_all_residual_families`
  - Task 1 metric primitives.
- Produces:
  - `PlannedStep`
  - `ActionProvider` protocol
  - `MatchedClosedLoopResult`
  - `initial_state_from_selection(selection: ExternalV46Split) -> FormalV4ClosedLoopState`
  - `planned_horizon_residuals(plan: PlannedStep, state: FormalV4ClosedLoopState, parameters: Mapping[str, Any]) -> np.ndarray`
  - `run_matched_closed_loop(selection: ExternalV46Split, provider: ActionProvider, parameters: Mapping[str, Any], *, method_id: str, seed: int, warmup_origins: int = 100) -> MatchedClosedLoopResult`
  - `run_perfect_information_mpc_reference(selection: ExternalV46Split, parameters: Mapping[str, Any]) -> MatchedClosedLoopResult`

- [ ] **Step 1: Write a failing chronology test**

```python
def test_runner_rejects_non_hourly_selection(make_selection, provider, parameters):
    selection = make_selection(3)
    bad_times = selection.target_times.copy()
    bad_times[2] += np.timedelta64(2, "h")
    with pytest.raises(ValueError, match="consecutive"):
        run_matched_closed_loop(replace(selection, target_times=bad_times), provider, parameters, method_id="stub", seed=2026)
```

- [ ] **Step 2: Write a failing state-carry test**

```python
def test_next_origin_receives_previous_settled_soc_and_chp(make_selection, recording_provider, parameters):
    result = run_matched_closed_loop(make_selection(3), recording_provider, parameters, method_id="stub", seed=2026, warmup_origins=0)
    assert np.isclose(recording_provider.seen_soc[1], result.next_soc[0])
    assert np.isclose(recording_provider.seen_previous_chp[1], result.next_previous_chp[0])
    assert len(set(result.state_hashes.tolist())) == 3
```

Add a second test that loads the real `selection_full.npz`, requires one unique `trajectory_id == "capacity_bound_causal"`, and verifies that the newly constructed first state hash equals the first hash in the saved RSC-PF rollout. This prevents an apparently harmless new trajectory label from invalidating exact replay.

- [ ] **Step 3: Write a failing four-hour planned-feasibility test**

Construct a plan whose first row is feasible but whose third row violates the CHP ramp constraint. `planned_horizon_residuals` must return shape `[4,8]`, and the origin-level planned-feasibility flag must be false. The checker must inspect raw planned rows; it must not call recourse before computing planned residuals.

- [ ] **Step 4: Write failing tests for inference and reference-optimizer accounting**

```python
def test_pto_runner_counts_only_its_inference_lp_calls(make_selection, pto_provider, parameters):
    result = run_matched_closed_loop(make_selection(4), pto_provider, parameters, method_id="iTransformer-PTO", seed=2026, warmup_origins=0)
    assert result.inference_lp_calls == 4

def test_direct_provider_has_no_inference_lp_calls(make_selection, direct_provider, parameters):
    result = run_matched_closed_loop(make_selection(4), direct_provider, parameters, method_id="DigitalTwins-Policy", seed=2026, warmup_origins=0)
    assert result.inference_lp_calls == 0

def test_pi_mpc_is_one_separate_reference_trajectory(make_selection, parameters):
    result = run_perfect_information_mpc_reference(make_selection(4), parameters)
    assert result.reference_lp_calls == 4
    assert result.inference_lp_calls == 0
```

- [ ] **Step 5: Run the runner tests and confirm failure before implementation**

Run the new test file with the frozen Python environment. Expected: missing runner interfaces.

- [ ] **Step 6: Preserve trajectory metadata in the external split**

Extend `ExternalV46Split` with optional `trajectory_ids` and `source_state_hashes` arrays, carry them through `take`, and populate them from the NPZ loader when present. The matched runner requires both fields and fails closed if the selected chronology has zero or multiple trajectory IDs. Existing synthetic fixtures may omit them unless they call the matched runner.

- [ ] **Step 7: Implement the provider/result contracts**

```python
@dataclass(frozen=True)
class PlannedStep:
    forecast: np.ndarray             # [4,4]
    scheduler_demand: np.ndarray     # [4,4]
    renewable_forecast: np.ndarray   # [4,2]
    dispatch: np.ndarray             # [4,21]
    inference_lp_calls: int

class ActionProvider(Protocol):
    method_id: str
    optimizer_role: str
    def plan(self, step: ExternalV46Split, state: FormalV4ClosedLoopState) -> PlannedStep: ...

@dataclass(frozen=True)
class MatchedClosedLoopResult:
    method_id: str
    seed: int
    arrays: Mapping[str, np.ndarray]
    metrics: Mapping[str, Any]
    inference_lp_calls: int
    reference_lp_calls: int
```

- [ ] **Step 8: Implement chronological state construction and per-origin inputs**

```python
def initial_state_from_selection(selection):
    trajectory_ids = np.asarray(selection.trajectory_ids, dtype=str)
    unique = np.unique(trajectory_ids)
    if unique.tolist() != ["capacity_bound_causal"]:
        raise ValueError(f"unexpected selection trajectory IDs: {unique.tolist()}")
    return FormalV4ClosedLoopState(
        torch.as_tensor(selection.load_history[:1]),
        torch.as_tensor(selection.exog_history[:1]),
        torch.as_tensor(selection.device_history[:1]),
        torch.as_tensor(selection.device_status[:1]),
        torch.as_tensor(selection.scheduler_context[:1, 0, 5:6]),
        torch.as_tensor(selection.previous_chp[:1]),
        selection.target_times[0],
        str(unique[0]),
    )

def _step_view(selection, index, state):
    row = selection.take(np.asarray([index], dtype=np.int64))
    context = row.scheduler_context.copy()
    context[:, :, 5] = float(state.initial_soc[0, 0])
    return replace(
        row,
        load_history=state.load_history.detach().cpu().numpy(),
        exog_history=state.exog_history.detach().cpu().numpy(),
        device_history=state.device_history.detach().cpu().numpy(),
        device_status=state.activity_history.detach().cpu().numpy(),
        scheduler_context=context,
        previous_chp=state.previous_chp.detach().cpu().numpy(),
    )
```

- [ ] **Step 9: Implement four-hour planned checks and one canonical realized transition**

Inside `run_matched_closed_loop`, validate `np.diff(target_times) == np.timedelta64(1, "h")`, call `provider.plan`, and inspect all four raw planned rows against `scheduler_demand[:, :3]` and `renewable_forecast`. For the planned-only check, advance a shadow SOC/CHP state after every planned row with `advance_with_executed_first_hour`; never feed this shadow state into the real next origin. Then settle only `dispatch[0]` with `settle_first_step_v4`, compute settled residuals against realized first-hour demand/PV/WT, and advance the real state exactly once:

```python
outcome = settle_first_step_v4(
    torch.as_tensor(plan.dispatch[:1], dtype=torch.float64),
    torch.as_tensor(step.forecast_target[:, 0, :3], dtype=torch.float64),
    torch.as_tensor(step.renewable_realized[:, 0, :], dtype=torch.float64),
    parameters,
    initial_soc=state.initial_soc.to(torch.float64),
    previous_chp=state.previous_chp.to(torch.float64),
    grid_price=torch.as_tensor(step.scheduler_context[:, 0, 2], dtype=torch.float64),
    gas_price=torch.as_tensor(step.scheduler_context[:, 0, 3], dtype=torch.float64),
    carbon_price=torch.as_tensor(step.scheduler_context[:, 0, 4], dtype=torch.float64),
)
state = advance_formal_v4_state(
    state,
    outcome,
    realized_load=torch.as_tensor(step.forecast_target[:, 0, :], dtype=torch.float32),
    realized_exog=torch.as_tensor(selection.exog_history[min(index + 1, len(selection) - 1), -1:, :], dtype=torch.float32),
)
```

Time `provider.plan + settle_first_step_v4` with `time.perf_counter_ns`; if CUDA is used, synchronize immediately before and after the timed region. Execute the first 100 origins but omit their timing samples from median/P95 aggregation. Planned-residual calculation, metric aggregation, file I/O, and the separate reference run are outside this timed region.

`run_perfect_information_mpc_reference` starts from the identical first state, solves one four-hour LP per origin with realized future three-carrier demand and PV/WT, executes only its first action through the same settlement, and carries its own state. It writes no forecast metric and is explicitly labeled `deployable: false` and `reference_only: true`.

- [ ] **Step 10: Run the runner tests**

Expected: chronology, single transition, state carry, residual, recourse, LP-count, and gas-exclusion tests pass.

- [ ] **Step 11: Commit the common runner**

```powershell
git add src/joint_dispatch/matched_closed_loop.py src/joint_dispatch/external_v46_data.py tests/test_rsc_pf_matched_closed_loop.py tests/test_rsc_pf_external_v46_data.py
git commit -m "feat: add matched chronological closed-loop runner"
```

---

### Task 3: Add frozen action providers and RSC-PF replay verification

**Files:**
- Create: `src/joint_dispatch/matched_closed_loop_providers.py`
- Modify: `src/joint_dispatch/external_baselines.py`
- Modify: `src/joint_dispatch/external_baseline_training.py`
- Modify: `scripts/evaluate_rsc_pf_external_baselines.py`
- Modify: `scripts/run_rsc_pf_external_v46_pilot.py`
- Modify: `tests/test_rsc_pf_external_baselines.py`
- Test: `tests/test_rsc_pf_matched_closed_loop_providers.py`

**Interfaces:**
- Consumes: `PlannedStep`, `ActionProvider`, frozen v4.6 data normalization, external checkpoint loader, formal v4.6 checkpoint/risk-cap/contract artifacts.
- Produces:
  - `build_external_provider(method_id: str, seed: int, config: Mapping[str, Any]) -> ActionProvider`
  - `load_frozen_rsc_pf_provider(run_root: Path, contract_path: Path, benchmark_path: Path, capacity_receipt_path: Path) -> ActionProvider`
  - `solve_pto_from_forecast(method_id: str, forecast: np.ndarray, step: ExternalV46Split, state: FormalV4ClosedLoopState, parameters: Mapping[str, Any]) -> PlannedStep`
  - `run_and_verify_rsc_pf_legacy_replay(provider: ActionProvider, materialized_root: Path, legacy_rollout: Path, artifact_root: Path, *, atol: float = 1e-5, rtol: float = 1e-6) -> dict[str, Any]`
  - `verify_rsc_pf_first_origin(result: MatchedClosedLoopResult, legacy_rollout: Path, *, atol: float = 1e-5, rtol: float = 1e-6) -> dict[str, Any]`

- [ ] **Step 1: Write failing role tests for all providers**

```python
@pytest.mark.parametrize((method, role, calls), [
    ("iTransformer-PTO", "exact optimizer at inference", 1),
    ("DecisionFocused-Online", "exact optimizer at inference", 1),
    ("DigitalTwins-Policy", "none at inference", 0),
])
def test_external_provider_roles(method, role, calls, provider_factory, state, step):
    provider = provider_factory(method)
    planned = provider.plan(step, state)
    assert provider.optimizer_role == role
    assert planned.inference_lp_calls == calls
```

- [ ] **Step 2: Write a failing gas-semantics test**

```python
def test_predicted_gas_never_enters_pto_balance(step, state, parameters):
    forecast = step.forecast_target[0].copy()
    changed = forecast.copy()
    changed[:, 3] += 1.0e6
    first = solve_pto_from_forecast("iTransformer-PTO", forecast, step, state, parameters)
    second = solve_pto_from_forecast("iTransformer-PTO", changed, step, state, parameters)
    assert first.scheduler_demand.shape == (4, 4)
    assert np.array_equal(first.dispatch, second.dispatch)
```

This changes the predicted gas channel itself rather than the unused evaluation label. Also assert that gas forecast errors remain in the forecast table while gas never appears in terminal-balance, shortage, or no-shortage arrays.

- [ ] **Step 3: Add failing RSC-PF reconstruction and dual-path tests**

Load the real `J_joint.pt` checkpoint and require strict state-dict loading. First, run the unchanged legacy evaluator and require all saved legacy arrays to replay within `atol=1e-5`, `rtol=1e-6`. Second, run the common evaluator with carried SOC and require the first origin to reproduce nominal forecast, scheduler demand, controls, planned dispatch, settled dispatch, and state hash. Finally, assert that the second origin receives the first settled `next_soc`, not `selection.scheduler_context[1, 0, 5]`.

- [ ] **Step 4: Run provider tests and confirm they fail before implementation**

Expected: missing provider factory and RSC-PF loader.

- [ ] **Step 5: Correct stale optimizer-role metadata without changing frozen weights**

Set the complete deployed method role for `ExternalForecastPTO`/iTransformer-PTO to `"exact optimizer at inference"` in the model metadata and in both existing external evaluation receipt builders. Keep the already-produced receipts immutable; the new matched-run receipt records the corrected role and cites the frozen checkpoint hash. Add regression tests requiring both PTO methods to declare one inference LP and DigitalTwins/RSC-PF to declare zero.

- [ ] **Step 6: Implement external providers with corrected complete-method roles**

For iTransformer-PTO and DecisionFocused-Online, normalize the one-row carried-state view from training statistics, denormalize the forecast, and call the canonical LP once using the current SOC and CHP state. For DigitalTwins-Policy, use the raw causal view and the shared decoder without an LP.

```python
class PTOProvider:
    optimizer_role = "exact optimizer at inference"
    def plan(self, step, state):
        prediction = self.predict_physical(step)
        solved = solve_pto_windows(PTOForecasts(self.method_id, prediction[None], step.forecast_target), step, self.parameters)
        if solved.offline_exact_lp_calls != 1 or not bool(solved.success[0]):
            raise RuntimeError("PTO inference LP failed or call count is invalid")
        return PlannedStep(prediction, prediction, step.scheduler_context[0, :, :2], solved.dispatch[0], 1)
```

Before the LP call, rebuild the row's scheduler context from the carried `state.initial_soc` and `state.previous_chp`; never use the materialized row's stale SOC/CHP. The provider receipt must describe iTransformer as an official-source adaptation rather than an untouched reproduction.

- [ ] **Step 7: Implement strict RSC-PF reconstruction**

Load the contract from `configs/joint_forecast_dispatch_formal_v4_6.json`, the runtime parameters from the frozen benchmark plus capacity receipt, and the cap from `pilot/risk_caps/RISK_CAPS.npz`. Instantiate `RiskAdjustedRSCPFModelV46` using the architecture values read from the contract, uniform placeholder transition probabilities, the exact saved cap shape, and the frozen runtime decoder parameters. Then load the complete `state_dict` from `pilot/stages/J_joint.pt` with `strict=True`; the checkpoint overwrites all registered normalization, transition, and cap buffers. Verify the contract hash, checkpoint SHA-256, cap-array SHA-256, and available lineage hashes before exposing nominal forecast, risk-adjusted scheduler demand, controls, and dispatch through `PlannedStep` with zero inference LP calls. The matched provider always constructs its scheduler context with `state.initial_soc`; it must not reproduce the legacy stale-SOC behavior.

- [ ] **Step 8: Implement separate legacy replay and matched first-origin verification**

```python
def _compare_legacy_arrays(result, legacy_rollout, *, atol=1e-5, rtol=1e-6):
    with np.load(legacy_rollout, allow_pickle=False) as saved:
        checks = {
            "timestamps": np.array_equal(result.arrays["times"], saved["times"]),
            "forecast": np.allclose(result.arrays["forecast"], saved["forecast_nominal"], atol=atol, rtol=rtol),
            "scheduler_demand": np.allclose(result.arrays["scheduler_demand"], saved["scheduler_demand"], atol=atol, rtol=rtol),
            "planned_dispatch": np.allclose(result.arrays["planned_dispatch"], saved["planned_dispatch"], atol=atol, rtol=rtol),
            "settled_dispatch": np.allclose(result.arrays["settled_dispatch"], saved["settled_dispatch"], atol=atol, rtol=rtol),
            "state_hashes": np.array_equal(result.arrays["state_hashes"], saved["state_hashes"]),
        }
    if not all(checks.values()):
        raise ValueError(f"RSC-PF replay mismatch: {checks}")
    return checks
```

`run_and_verify_rsc_pf_legacy_replay` calls the unchanged `rollout_v46_2019` path and compares all 8,709 origins. It also compares `controls`, `initial_soc`, `previous_chp`, shortage, cost, carbon, objective, and realized inputs, and reports maximum absolute/relative differences for every floating array; timestamps and state hashes require exact equality. Its receipt records `scheduler_soc_source: materialized_origin` and is used only to prove checkpoint reconstruction.

`verify_rsc_pf_first_origin` compares index 0 only for the corrected common evaluator and records `scheduler_soc_source: carried_state`. It must explicitly reject any request to require full-array equality between the corrected matched run and the legacy rollout, because the legacy scheduler used materialized SOC at later origins.

- [ ] **Step 9: Run provider and replay tests**

Expected: all provider roles, one-step outputs, gas boundary, strict checkpoint load, and RSC-PF replay checks pass.

- [ ] **Step 10: Commit provider adapters**

```powershell
git add src/joint_dispatch/matched_closed_loop_providers.py src/joint_dispatch/external_baselines.py src/joint_dispatch/external_baseline_training.py scripts/evaluate_rsc_pf_external_baselines.py scripts/run_rsc_pf_external_v46_pilot.py tests/test_rsc_pf_external_baselines.py tests/test_rsc_pf_matched_closed_loop_providers.py
git commit -m "feat: adapt frozen methods to matched closed-loop evaluation"
```

---

### Task 4: Add fail-closed execution, artifact writing, and independent audit

**Files:**
- Create: `scripts/run_rsc_pf_matched_closed_loop_2019.py`
- Create: `scripts/audit_rsc_pf_matched_closed_loop_2019.py`
- Create: `configs/rsc_pf_matched_closed_loop_2019.json`
- Test: `tests/test_rsc_pf_matched_closed_loop_runner.py`

**Interfaces:**
- Consumes: Task 2 runner, Task 3 providers, the formal v4.6 paths, and existing frozen checkpoint hashes.
- Produces:
  - per-row `rollout.npz`
  - per-row `receipt.json`
  - `provenance/rsc_pf_legacy_replay.npz`
  - `provenance/RSC_PF_LEGACY_REPLAY.json`
  - `reference/perfect_information_mpc.npz`
  - `reference/PERFECT_INFORMATION_MPC.json`
  - `MATCHED_CLOSED_LOOP_MANIFEST.json`
  - `MATCHED_CLOSED_LOOP_AUDIT.json`

- [ ] **Step 1: Write a failing configuration-boundary test**

```python
def test_config_freezes_selection_and_refuses_evaluation_paths(load_config):
    config = load_config("configs/rsc_pf_matched_closed_loop_2019.json")
    assert config["selection_file"].endswith("selection_full.npz")
    assert config["selection_year"] == 2019
    assert config["evaluation_year"] == 2020
    assert config["allow_evaluation_access"] is False
```

- [ ] **Step 2: Write failing manifest audit tests**

```python
def test_manifest_requires_exact_row_matrix(valid_manifest):
    assert len(valid_manifest["rows"]) == 16  # one RSC-PF + 3 external methods x 5 seeds
    assert valid_manifest["reference"]["method_id"] == "Perfect-Information-MPC"

def test_audit_rejects_wrong_itransformer_optimizer_role(valid_manifest):
    broken = deepcopy(valid_manifest)
    row = next(value for value in broken["rows"] if value["method_id"] == "iTransformer-PTO")
    row["optimizer_role"] = "none at inference"
    with pytest.raises(ValueError, match="optimizer role"):
        audit_manifest(broken)
```

- [ ] **Step 3: Write failing access and artifact-integrity tests**

Test that any path containing the 2020 evaluation role, `sealed_test`, `test-set`, or excluded 2021 role is rejected before it is opened; mutate one NPZ byte and require the auditor to reject its hash. Add a provenance test requiring the receipt to distinguish `verified_file` from `receipt_hash_only`; the latter is the honest status for the v4.6-b source manifest, whose hash is recorded in `PILOT_RECEIPT.json` but whose manifest file is no longer present.

- [ ] **Step 4: Run runner/audit tests and confirm failure before implementation**

Expected: CLI/config/auditor interfaces are absent.

- [ ] **Step 5: Create the frozen 2019 configuration**

```json
{
  "schema_version": "rsc-pf-matched-closed-loop-2019-v1",
  "selection_year": 2019,
  "evaluation_year": 2020,
  "excluded_years": [2021],
  "allow_evaluation_access": false,
  "selection_file": "reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_f/pilot/data/selection_full.npz",
  "training_file": "reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_f/pilot/data/train.npz",
  "rsc_pf_contract": "configs/joint_forecast_dispatch_formal_v4_6.json",
  "rsc_pf_run_root": "reports/joint_forecast_dispatch_formal_v4_6/formal_v4_6_20260905_b/pilot",
  "rsc_pf_pilot_receipt": "reports/joint_forecast_dispatch_formal_v4_6/formal_v4_6_20260905_b/pilot/PILOT_RECEIPT.json",
  "rsc_pf_legacy_rollout": "reports/joint_forecast_dispatch_formal_v4_6/formal_v4_6_20260905_b/pilot/rollout/rsc_pf_joint.npz",
  "benchmark_path": "reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_j/gate0/benchmark/STANDARD_IES_BENCHMARK.yaml",
  "capacity_receipt_path": "reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_g/gate0/CAPACITY_FREEZE.json",
  "external_config": "configs/rsc_pf_external_baseline_implementation_v1.json",
  "external_source_receipt": "reports/rsc_pf_external_baselines_v46/implementation_final/sources/source_receipt.json",
  "external_output_root": "reports/rsc_pf_external_baselines_v46/implementation_final",
  "output_root": "reports/rsc_pf_matched_closed_loop_2019/v1",
  "external_seeds": [2026, 2027, 2028, 2029, 2030],
  "rsc_pf_seed": 2026,
  "warmup_origins": 100,
  "physical_tolerance": 1e-6,
  "shortage_tolerance": 1e-8,
  "replay_atol": 1e-5,
  "replay_rtol": 1e-6,
  "checkpoint_interval": 256
}
```

At config load, resolve every path relative to the repository root, verify it exists, record its SHA-256 before evaluation, and check the recorded v4.6 contract/capacity hashes against `PILOT_RECEIPT.json`. The absent v4.6-b source-manifest file is recorded as `receipt_hash_only`, never as independently verified; this limits the run to a protocol diagnostic and is repeated in the review.

- [ ] **Step 6: Implement resumable chunks and atomic finalization**

During each model row and the separate reference trajectory, save a `.partial` checkpoint every 256 origins containing the next index, complete carried-state tensors, trajectory ID/hash, accumulated arrays, timing samples, LP counters, and immutable input/checkpoint hashes. On `--resume`, verify every hash and the cursor/state hash before continuing; otherwise fail closed. Build the final NPZ and receipt inside a sibling `.writing` directory, close both, and atomically rename the directory into place. A valid completed row is skipped under `--resume`; an incomplete or hash-invalid completed row fails closed and is never overwritten automatically. Every receipt must contain both `test_set_accessed: false` and `evaluation_year_accessed: false`.

- [ ] **Step 7: Implement the CLI row matrix and smoke limit**

The CLI accepts `--config`, `--method`, `--seed`, `--all-methods`, `--all-seeds`, `--limit`, `--warmup-origins`, `--output-root`, `--resume`, and the mutually exclusive provenance mode `--verify-rsc-pf-legacy-only`. `--limit` and a warm-up override are allowed only when one normalized output-path segment is exactly `smoke`; the full run uses the frozen 100-origin warm-up. `--all-methods --all-seeds` expands to RSC-PF seed 2026 once plus the three external methods at seeds 2026–2030, exactly 16 model rows, and runs the Perfect-Information-MPC reference exactly once. Full execution requires exactly 8,709 origins. The loader additionally inspects `target_times` and refuses any year other than 2019, regardless of the file name.

- [ ] **Step 8: Implement independent audit recomputation**

The auditor reopens NPZ artifacts, recomputes forecast metrics through the existing canonical `evaluate_forecast` helper, recomputes all closed-loop summaries from raw arrays, verifies hashes and timestamp identity, enforces the 16-row method/seed matrix, checks inference LP counts (`8709` for each PTO row; `0` for RSC-PF/DigitalTwins), verifies one separate 8,709-call Perfect-Information-MPC reference, recomputes raw and terminal-stock-adjusted cumulative gaps, verifies gas exclusion and both access flags, and writes `MATCHED_CLOSED_LOOP_AUDIT.json` only if every numerical/protocol check passes. It additionally requires a full passing legacy-replay receipt and exact first-origin equality for the corrected RSC-PF row, while forbidding a false claim of full legacy equivalence for that corrected row. Provenance completeness is a separate field: the known `receipt_hash_only` v4.6 source-manifest gap must be disclosed but does not masquerade as a numerical audit failure.

- [ ] **Step 9: Run the runner/audit tests**

Expected: configuration, access, role, call-count, atomic-write, hash-tamper, resume, and audit-recomputation tests pass.

- [ ] **Step 10: Commit the execution boundary**

```powershell
git add scripts/run_rsc_pf_matched_closed_loop_2019.py scripts/audit_rsc_pf_matched_closed_loop_2019.py configs/rsc_pf_matched_closed_loop_2019.json tests/test_rsc_pf_matched_closed_loop_runner.py
git commit -m "feat: add fail-closed matched closed-loop evaluation CLI"
```

---

### Task 5: Verify, execute, audit, and review the 2019 diagnostic

**Files:**
- Create: `reports/rsc_pf_matched_closed_loop_2019/v1/MATCHED_CLOSED_LOOP_MANIFEST.json`
- Create: `reports/rsc_pf_matched_closed_loop_2019/v1/MATCHED_CLOSED_LOOP_AUDIT.json`
- Create: `reports/rsc_pf_matched_closed_loop_2019/v1/MATCHED_CLOSED_LOOP_REVIEW.md`
- Modify: `docs/superpowers/plans/2026-09-05-rsc-pf-matched-closed-loop-evaluation.md`

**Interfaces:**
- Consumes: Tasks 1–4 and all frozen 2019/checkpoint artifacts.
- Produces: the audited diagnostic comparison and the go/no-go recommendation for a separate five-seed/2020 plan.

- [ ] **Step 1: Run the complete relevant unit suite**

```powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest `
  tests\test_rsc_pf_closed_loop_metrics.py `
  tests\test_rsc_pf_matched_closed_loop.py `
  tests\test_rsc_pf_matched_closed_loop_providers.py `
  tests\test_rsc_pf_matched_closed_loop_runner.py `
  tests\test_rsc_pf_external_v46_data.py `
  tests\test_joint_dispatch_formal_v4_6_rollout.py `
  -q --basetemp D:\Paper\pytest_tmp_matched_closed_loop -p no:cacheprovider
```

Expected: all tests pass; only already-known PyTorch Transformer warnings are permitted.

- [ ] **Step 2: Reconstruct and verify the complete legacy RSC-PF rollout**

```powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\run_rsc_pf_matched_closed_loop_2019.py `
  --config configs\rsc_pf_matched_closed_loop_2019.json `
  --verify-rsc-pf-legacy-only
```

Expected: all 8,709 legacy origins replay within the frozen tolerance, timestamps/state hashes match exactly, and the receipt states `scheduler_soc_source: materialized_origin`. If this fails, stop before the matched smoke run because the checkpoint has not been reconstructed reliably.

- [ ] **Step 3: Run an eight-origin smoke matrix**

```powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\run_rsc_pf_matched_closed_loop_2019.py `
  --config configs\rsc_pf_matched_closed_loop_2019.json `
  --all-methods --all-seeds --limit 8 `
  --warmup-origins 2 `
  --output-root reports\rsc_pf_matched_closed_loop_2019\smoke
```

Expected: 16 complete model receipts plus one reference receipt; PTO rows each contain 8 inference LP calls, RSC-PF/DigitalTwins rows contain 0, the shared Perfect-Information-MPC receipt contains 8 reference LP calls, and neither access flag is true.

- [ ] **Step 4: Audit the smoke matrix and stop on any failure**

```powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\audit_rsc_pf_matched_closed_loop_2019.py `
  --config configs\rsc_pf_matched_closed_loop_2019.json `
  --output-root reports\rsc_pf_matched_closed_loop_2019\smoke `
  --expected-origins 8
```

Expected: `audit_passed: true`. The corrected RSC-PF row must equal the legacy rollout at origin 0 and must state `scheduler_soc_source: carried_state`. If false, do not run the full matrix.

- [ ] **Step 5: Record the measured resource projection**

Use smoke timings to record projected wall time and free disk space before the full run. The full matrix entails exactly 87,090 inference LP calls (two PTO methods × five seeds × 8,709 origins) and 8,709 reference-only Perfect-Information-MPC calls, for 95,799 LP solves total. Save the projection in the smoke audit so the eventual run duration is evidence-based rather than guessed.

- [ ] **Step 6: Run the full 2019 matrix**

```powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\run_rsc_pf_matched_closed_loop_2019.py `
  --config configs\rsc_pf_matched_closed_loop_2019.json `
  --all-methods --all-seeds
```

Expected: one RSC-PF row, fifteen external rows, and one separately labeled Perfect-Information-MPC reference, each with 8,709 aligned origins. This is sequential evaluation only; it does not retrain any model.

- [ ] **Step 7: Run the full independent audit**

```powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\audit_rsc_pf_matched_closed_loop_2019.py `
  --config configs\rsc_pf_matched_closed_loop_2019.json `
  --output-root reports\rsc_pf_matched_closed_loop_2019\v1 `
  --expected-origins 8709
```

Expected: `audit_passed: true`, 16 valid model rows plus one reference trajectory, exact timestamp alignment, correct LP roles/counts, successful legacy RSC-PF replay plus corrected first-origin equivalence, and zero evaluation-year access.

- [ ] **Step 8: Write the diagnostic review without inferential claims**

The review table separates forecast quality, four-hour planned feasibility, first-hour settled feasibility, recourse adjustment, no-shortage rate, shortage, raw and terminal-stock-adjusted cumulative objective, signed objective gap to the shared Perfect-Information-MPC reference, latency, and optimizer role. It states that RSC-PF has one seed, labels gas as an auxiliary prior, marks iTransformer/DecisionFocused/DigitalTwins as adaptations at their documented reproduction levels, discloses the missing v4.6-b source-manifest file, and calls the evidence “2019 protocol-validation diagnostic.” It must not call the signed gap guaranteed non-negative regret, and it must not compute p-values, confidence intervals, rank stability, or a final paper winner from the asymmetric seed matrix.

- [ ] **Step 9: Record the next-gate decision**

If every protocol check passes, recommend a new plan to complete RSC-PF seeds 2027–2030, restore the complete preregistered internal/external baseline roster, freeze five seeds for every stochastic method, and only then authorize one-time 2020 Gate 2 evaluation. If a protocol check fails or the legacy RSC-PF replay does not match, record the precise blocker and stop before any training or 2020 access.

- [ ] **Step 10: Mark completed plan items and commit code/document changes only**

Generated report artifacts remain under the report root according to the repository's ignore policy. Commit tracked source, tests, config, and the updated plan without staging the pre-existing untracked `third_party/iTransformer_source/` directory.

```powershell
git add docs/superpowers/plans/2026-09-05-rsc-pf-matched-closed-loop-evaluation.md
git commit -m "docs: record matched closed-loop diagnostic completion"
```
