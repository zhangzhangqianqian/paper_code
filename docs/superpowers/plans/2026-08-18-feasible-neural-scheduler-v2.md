# Feasible Neural Scheduler V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the infeasible independent 21-output scheduling proxy with a CPU-friendly supervised neural scheduler whose deterministic PyTorch decoder emits physically feasible four-hour, 21-variable dispatches without calling the exact LP at inference.

**Architecture:** Preserve the public `[B,4,10] -> [B,4,21]` scheduling interface and every v1 artifact. A residual MLP emits 15 control logits; a float64, horizon-reachable decoder constructs cooling allocation, a ramp-feasible CHP path, boiler output, a terminal-SOC-feasible battery path, renewable splits, grid import, and explicit capacity slacks. The HiGHS LP remains an offline teacher and benchmark only.

**Tech Stack:** Python 3.9, PyTorch, NumPy, SciPy/HiGHS for offline labels only, pytest, JSON/NPZ artifacts.

## Global Constraints

- Implement only the scheduling-proxy v2 subtask described in `docs/superpowers/specs/2026-08-18-physics-informed-neural-scheduling-design.md`.
- Do not modify Scheme2R, forecasting checkpoints, forecasting metrics, manuscript files, figures, formal scheduling results, `dispatch_lp.py`, or synthetic label generation.
- Preserve v1 contract, checkpoint loading, tests, diagnostic repairs, and artifacts unchanged.
- V2 inference must never call `solve_dispatch_lp`, an optimizer, a projection solver, or an exact fallback.
- External v2 input/output shapes are fixed at `[B,4,10]` and `[B,4,21]`; `LABEL_ORDER` is unchanged.
- Use float64 inside the physical decoder and return physical dispatch in float64.
- `gas_prior` is context only and never a balance constraint; mask it in the primary v2 smoke training path.
- Hyperparameter/checkpoint selection is validation-only; smoke work must not read or tune on the test split.
- Existing uncommitted user files `frame/tests/test_scheduling_proxy_repairs.py`, `frame/scripts/compare_scheduling_proxy_repairs.py`, and the older repair plan are out of scope and must not be reverted or reformatted.
- Use `apply_patch` for edits. Each task must end with its focused tests passing before proceeding.

---

## File map

- Create `frame/configs/scheduling_proxy_contract_v2.json`: immutable v2 control layout, decoder, loss, training, provenance, and no-fallback contract.
- Create `frame/src/scheduling/proxy_decoder.py`: input validation, 15-control grouping, feasible decoder, teacher-control recovery, invariant metadata.
- Modify `frame/src/scheduling/proxy_contract.py`: explicit v1/v2 parsing and cross-version rejection.
- Modify `frame/src/scheduling/proxy_model.py`: v2 model/config, 15-logit head, decoded prediction, versioned checkpoint loading while retaining v1 classes.
- Modify `frame/src/scheduling/proxy_physics.py`: decoded-dispatch v2 supervised/economic loss; keep v1 loss API.
- Modify `frame/src/scheduling/proxy_training.py`: version-dispatched train/load path and v2 provenance.
- Modify `frame/src/scheduling/proxy_adapter.py`: v2 decoded inference with no fallback; retain v1 behavior.
- Modify `frame/src/scheduling/proxy_evaluation.py`: v2 feasibility/economic/latency metrics.
- Modify `frame/scripts/run_scheduling_proxy_pipeline.py`: v2 smoke entry and artifacts.
- Modify `frame/scripts/audit_scheduling_proxy.py`: v2 no-LP and feasibility assertions.
- Create `frame/tests/test_scheduling_proxy_decoder.py` and extend focused proxy tests listed below.

---

### Task 1: Add an isolated v2 contract

**Files:**
- Create: `frame/configs/scheduling_proxy_contract_v2.json`
- Modify: `frame/src/scheduling/proxy_contract.py`
- Test: `frame/tests/test_scheduling_proxy_contract.py`

**Interfaces:**
- Consumes: existing `ProxyContract`, `FEATURE_ORDER`, `LABEL_ORDER`, and v1 JSON.
- Produces: `ProxyContract.schema_version`, `ProxyContract.is_v2`, validated `model.decision_groups`, and `safety.allow_exact_fallback == false` for v2.

- [ ] **Step 1: Write failing version-isolation tests**

```python
def test_v2_contract_freezes_feasible_decoder_and_no_fallback():
    c = load_proxy_contract("frame/configs/scheduling_proxy_contract_v2.json")
    assert c.schema_version == "scheduling-proxy-contract-v2"
    assert c.model["decision_dim"] == 15
    assert c.model["decision_groups"] == {
        "cooling": [0, 4], "chp": [4, 8],
        "soc": [8, 11], "renewable_pv": [11, 15],
    }
    assert c.safety["allow_exact_fallback"] is False
    assert c.safety["inference_exact_lp_calls"] == 0

def test_contract_rejects_wrong_v2_group_coverage(tmp_path):
    payload = json.loads(Path("frame/configs/scheduling_proxy_contract_v2.json").read_text(encoding="utf-8"))
    payload["model"]["decision_groups"]["renewable_pv"] = [11, 14]
    invalid = tmp_path / "invalid-v2.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decision"):
        load_proxy_contract(invalid)
```

- [ ] **Step 2: Run contract tests and confirm the new tests fail**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_contract.py -q`

Expected: failure because v2 contract/version checks do not exist.

- [ ] **Step 3: Add v2 JSON without altering v1 values**

Copy immutable feature/label/split hashes from v1. Set `schema_version`, 15-decision groups, `decoder_dtype=float64`, `control_temperature=0.25`, v2 loss weights, and no-fallback safety exactly as the design specification requires.

- [ ] **Step 4: Implement strict version validation**

Reject unknown versions, gaps/overlaps in decision ranges, any v2 decision dimension other than 15, any v2 horizon other than four, or any v2 fallback flag set true. Preserve the existing v1 parser behavior byte-for-behavior.

- [ ] **Step 5: Run focused tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_contract.py -q`

Expected: all tests pass.

### Task 2: Implement the feasible-by-construction decoder

**Files:**
- Create: `frame/src/scheduling/proxy_decoder.py`
- Create: `frame/tests/test_scheduling_proxy_decoder.py`

**Interfaces:**
- Produces: `decode_feasible_controls(controls, physical_features, parameters) -> Tensor[B,4,21]`.
- Produces: `decode_feasible_dispatch(logits, physical_features, parameters, temperature=0.25) -> Tensor[B,4,21]`.
- Produces: `recover_teacher_controls(dispatch, physical_features, parameters) -> Tensor[B,15]`.
- Produces: `DECISION_GROUPS` and `DECODER_SCHEMA_VERSION`.

- [ ] **Step 1: Write decoder shape, validation, and finite-output tests**

Test `[2,15]` controls and `[2,4,10]` features. Reject wrong shapes, NaN/Inf, negative demands/renewables, inconsistent horizon SOC, controls outside `[0,1]`, and nonpositive efficiencies/capacities.

- [ ] **Step 2: Write exact invariant tests before implementation**

For randomized normal, zero-renewable, capacity-edge, and overloaded inputs, decode then assert with existing helpers:

```python
assert dispatch.shape == (batch, 4, 21)
assert torch.isfinite(dispatch).all()
assert dispatch.min() >= -1e-10
assert balance_residuals(dispatch, features).abs().max() <= 1e-6
assert conversion_residuals(dispatch, params).abs().max() <= 1e-6
soc = soc_residuals(dispatch, features, params)
assert soc["state"].abs().max() <= 1e-6
assert soc["terminal"].abs().max() <= 1e-6
```

Also assert renewable use+curtailment equals availability, CHP ramps obey the zero prior and per-hour bound, capacities hold, and `min(p_charge,p_discharge) <= 1e-12`.

- [ ] **Step 3: Run decoder tests and confirm failure**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_decoder.py -q`

- [ ] **Step 4: Implement common checks and label assembly**

Use `LABEL_ORDER` indices rather than numeric literals. Cast controls/features to float64 without detaching gradients. Allocate the output with `features.new_zeros((B,4,21), dtype=torch.float64)`.

- [ ] **Step 5: Implement cooling and heat construction**

Implement the exact `Qac_safe`, `S_c`, `L_ec`, `U_ec`, conversion, boiler, heat slack, and heat dump equations from the design. Never hide overload by clipping demand.

- [ ] **Step 6: Implement backward-viable CHP construction**

Build `A_t` in reverse, then construct the ramp interval in forward order. Assert `V_chp >= L_chp - 1e-10`; do not silently swap invalid limits.

- [ ] **Step 7: Implement terminal-reachable SOC construction**

Construct the first three SOC states sequentially from local and remaining-horizon reachability bounds; fix the fourth state to initial energy. Derive mutually exclusive charge/discharge from signed SOC increments.

- [ ] **Step 8: Implement renewable/grid residual construction**

Allocate the maximum feasible total renewable energy, use the learned PV split only inside its exact feasible interval, and derive grid/slack from the remaining demand.

- [ ] **Step 9: Add gradient and no-LP dependency tests**

Use an interior float64 batch with `requires_grad=True`; backpropagate the sum of nonconstant dispatch entries and require finite gradients for all four decision groups. Inspect/import-test `proxy_decoder` with an LP function patched to raise and require decoding to succeed.

- [ ] **Step 10: Add teacher-control recovery tests**

Generate a small train/validation-only batch with the existing exact LP teacher, recover controls, decode them, and require maximum reconstruction error `<=1e-6`. Never load the test split.

- [ ] **Step 11: Run decoder tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_decoder.py frame/tests/test_dispatch_lp.py -q`

Expected: all tests pass.

### Task 3: Add a versioned v2 neural model and checkpoint path

**Files:**
- Modify: `frame/src/scheduling/proxy_model.py`
- Modify: `frame/tests/test_scheduling_proxy_model.py`

**Interfaces:**
- Produces: `FeasibleSchedulingProxyConfig` with `decision_dim=15`.
- Produces: `FeasibleSchedulingProxy.forward_logits(inputs_normalized) -> Tensor[B,15]`.
- Produces: `FeasibleSchedulingProxy.predict_dispatch(inputs_normalized, inputs_physical, parameters) -> Tensor[B,4,21]`.
- Keeps: `SchedulingProxy` and existing v1 checkpoint functions.

- [ ] **Step 1: Write failing v1/v2 coexistence tests**

Require v1 output `[B,4,21]`, v2 logits `[B,15]`, decoded output `[B,4,21]`, v1 checkpoint reload, v2 checkpoint reload, and cross-version load rejection.

- [ ] **Step 2: Run model tests and confirm failure**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_model.py -q`

- [ ] **Step 3: Implement the v2 head and decoded prediction**

Reuse the CPU-friendly flattened residual MLP trunk, but project to 15 unconstrained logits. `forward()` for v2 returns logits; only `predict_dispatch()` returns physical decisions through `proxy_decoder`.

- [ ] **Step 4: Version checkpoint metadata**

Persist model family, decoder schema, decision groups, contract SHA-256, benchmark SHA-256, and validation-only provenance. Reject missing or mismatched v2 metadata.

- [ ] **Step 5: Run focused model tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_model.py frame/tests/test_scheduling_proxy_contract.py -q`

### Task 4: Implement the v2 decoded-dispatch objective

**Files:**
- Modify: `frame/src/scheduling/proxy_physics.py`
- Modify: `frame/tests/test_scheduling_proxy_physics.py`

**Interfaces:**
- Produces: `feasible_proxy_loss(predicted_dispatch, teacher_dispatch, features, parameters, teacher_objective, teacher_carbon, weights) -> (total, components)`.
- Keeps: `physics_aware_loss` unchanged for v1.

- [ ] **Step 1: Write failing loss tests**

Require named scalar components `coordinate`, `renewable_split`, `objective`, `carbon`, `slack`, and `total`; exact teacher predictions must have near-zero loss; perturbed CHP/cooling/SOC must increase coordinate loss; nonzero slack must increase slack loss.

- [ ] **Step 2: Implement physical-scale Huber components**

Use capacity-normalized `q_ec`, `p_chp`, and `soc` coordinates; low-weight PV split; relative objective/carbon errors with denominator `max(abs(target),1)`; demand-normalized slack. Calculate every term from decoded physical dispatch.

- [ ] **Step 3: Keep physics identities as diagnostics, not trainable penalties**

Do not include balance, conversion, or SOC residual losses in v2 total. Expose their maximum absolute values for assertions/reporting.

- [ ] **Step 4: Run physics tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_physics.py frame/tests/test_scheduling_proxy_decoder.py -q`

### Task 5: Train and reload v2 without data leakage

**Files:**
- Modify: `frame/src/scheduling/proxy_training.py`
- Modify: `frame/tests/test_scheduling_proxy_training.py`

**Interfaces:**
- Extends: `train_proxy(train_split, validation_split, output_dir, contract, benchmark, stats=None, max_epochs=None, patience=None, batch_size=None, seed=None, learning_rate=None, weight_decay=None) -> TrainingResult` with explicit v2 dispatch while preserving v1.
- Extends: `load_trained_proxy(output_dir, contract=None, benchmark_path=None, expected_train_seed=None) -> tuple[SchedulingProxy | FeasibleSchedulingProxy, ProxyNormalizationStats, Mapping[str, Any]]` with contract/provenance validation.

- [ ] **Step 1: Write failing smoke-training and provenance tests**

Require that no test artifact is passed to the trainer; input normalization is fitted only on train; the primary path zeros/masks gas prior; best checkpoint is selected by validation loss; repeat runs with seed 2026 are deterministic.

- [ ] **Step 2: Add version-dispatched loaders and loss path**

For v2, call `forward_logits -> decode_feasible_dispatch -> feasible_proxy_loss`. Retain v1 execution when the v1 contract is explicitly used.

- [ ] **Step 3: Persist complete provenance**

Checkpoint/history must contain contract and benchmark hashes, split identity, `selection_split=validation`, `test_split_used_for_selection=false`, decoder schema, decision groups, and `inference_exact_lp_calls=0`.

- [ ] **Step 4: Run training tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_training.py -q`

### Task 6: Route v2 inference through the physical decoder only

**Files:**
- Modify: `frame/src/scheduling/proxy_adapter.py`
- Modify: `frame/tests/test_scheduling_proxy_adapter.py`

**Interfaces:**
- Keeps: `ProxyInferenceResult` and `[N,4,21]` arrays.
- V2 invariant: `raw_dispatch == safe_dispatch`, `fallback_mask == false`, `fallback_rate == 0`, `inference_exact_lp_calls == 0`.

- [ ] **Step 1: Write failing adapter tests with LP patched to raise**

Build a v2 adapter/model, patch `solve_dispatch_lp` to raise immediately, and require `infer()` to complete with feasible output. Require `allow_exact_fallback=True` to raise a clear configuration error for v2.

- [ ] **Step 2: Implement explicit v1/v2 branches**

The v1 branch keeps historical behavior. The v2 branch calls only `model.predict_dispatch`, evaluates feasibility, and fails closed if any decoded output is infeasible instead of repairing it.

- [ ] **Step 3: Run adapter tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_adapter.py frame/tests/test_scheduling_proxy_decoder.py -q`

### Task 7: Add v2 evaluation, pipeline, and audit artifacts

**Files:**
- Modify: `frame/src/scheduling/proxy_evaluation.py`
- Modify: `frame/scripts/run_scheduling_proxy_pipeline.py`
- Modify: `frame/scripts/audit_scheduling_proxy.py`
- Modify: `frame/tests/test_scheduling_proxy_pipeline.py`
- Modify: `frame/tests/test_scheduling_proxy_diagnostics.py`

**Interfaces:**
- Produces smoke artifacts under a caller-supplied new output directory only.
- Produces metrics: feasible rate, maximum balance/conversion/renewable/SOC/ramp/capacity residuals, charge-discharge overlap, objective gap, cost regret, carbon error, slack, fallback rate, exact-LP inference calls, and combined model+decoder CPU latency.

- [ ] **Step 1: Write failing pipeline artifact tests**

Require v2 manifest/checkpoint/history/validation metrics/predictions; require all version/hash/split fields; require no test result in smoke selection artifacts.

- [ ] **Step 2: Add combined model+decoder evaluation**

Warm up before timing, use the existing CPU thread policy, report median and p95, and time only neural forward plus decoder—not disk I/O or LP label loading.

- [ ] **Step 3: Add strict audit gates**

For smoke, require finite outputs, 100% feasibility, max physical residual `<=1e-6`, zero fallback, zero exact LP inference calls, and nonnegative slack. Economic quality is reported but is not a smoke pass/fail threshold.

- [ ] **Step 4: Run pipeline/audit tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_pipeline.py frame/tests/test_scheduling_proxy_diagnostics.py -q`

### Task 8: Run the complete focused and regression verification

**Files:**
- No production edits unless a test exposes an in-scope defect.

**Interfaces:**
- Produces: evidence that v2 works and v1/forecasting behavior remains intact.

- [ ] **Step 1: Run all scheduling-proxy tests**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_contract.py frame/tests/test_scheduling_proxy_dataset.py frame/tests/test_scheduling_proxy_model.py frame/tests/test_scheduling_proxy_decoder.py frame/tests/test_scheduling_proxy_physics.py frame/tests/test_scheduling_proxy_training.py frame/tests/test_scheduling_proxy_adapter.py frame/tests/test_scheduling_proxy_pipeline.py frame/tests/test_scheduling_proxy_diagnostics.py -q`

- [ ] **Step 2: Run the full repository test suite**

Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests -q`

- [ ] **Step 3: Run a v2 smoke pipeline in a new directory**

Run the existing pipeline entry with `--contract frame/configs/scheduling_proxy_contract_v2.json`, smoke mode, CPU, and a new directory such as `frame/reports/scheduling_proxy_v2/smoke`. Do not overwrite v1 reports.

- [ ] **Step 4: Audit smoke artifacts**

Require feasibility rate 1.0, zero fallback, zero inference LP calls, all invariant maxima `<=1e-6`, valid provenance, and no test-based selection.

- [ ] **Step 5: Inspect the final diff and preserve unrelated changes**

Run `git status --short` and `git diff --check`. Confirm no Scheme2R, manuscript, figure, LP teacher, formal-result, or existing repair file changed.

## Completion boundary

This plan ends after v2 code, tests, training entry, and a smoke validation bundle pass. It does **not** authorize formal full-data training, replacement of published/previous results, manuscript edits, figure changes, or deletion of the v1 scheduler.
