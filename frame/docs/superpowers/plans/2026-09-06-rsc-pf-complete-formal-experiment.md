# RSC-PF Complete Formal Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the completed 2019 matched closed-loop diagnostic into a complete, leakage-safe formal experiment that fairly compares RSC-PF with the frozen primary baseline roster, including a genuinely trained Differentiable-LP baseline, and opens the sealed 2020 evaluation split only after Gate 1 passes.

**Architecture:** Keep the RSC-PF v4.6 model architecture and the audited matched closed-loop settlement/metric engine unchanged. Add one canonical experiment contract, replace hard-coded four-method lists with a validated method registry, connect every primary method to the same training, checkpoint, chronological evaluation, and audit path, then execute Gate 0 (readiness), Gate 1 (2019 selection), and Gate 2 (one-time 2020 evaluation). Existing 2019 v2 outputs remain immutable diagnostic evidence and are never relabeled as final paper results.

**Tech Stack:** Python 3.9, PyTorch, NumPy, SciPy/HiGHS, CVXPY 1.3.2, CVXPYlayers 0.1.6, diffcp 1.0.23, official iTransformer source snapshot, pytest, JSON/NPZ/SHA-256 audit artifacts.

## Global Constraints

- Execute strictly one task at a time. After every task, run its focused tests, inspect the diff and artifacts, report the result, and stop until the user authorizes the next task.
- Tasks 1–6 prepare and test the execution chain; Task 7 is Gate 0; Task 8 performs training and 2019 Gate 1; Task 9 is the only task allowed to access 2020.
- Freeze the model architecture: 24-hour history, four forecast tasks, four-hour horizon, 15 continuous scheduling controls, 21 physical dispatch outputs, and no future binary commitment head.
- Gas remains a station-side auxiliary operating prior: predict and score it, but do not impose it as a rigid terminal gas-demand balance.
- Data boundary: train on 2015–2018; select/calibrate on the 8,709-origin 2019 chronology; evaluate once on the sealed 8,757-origin 2020 chronology only after an authorized Gate 1 receipt; never access 2021.
- Primary stochastic methods use seeds 2026–2030: RSC-PF, Decoupled-RSC-PF, Direct-Policy, Scheme2R-PTO, State-Conditioned-PTO, Official iTransformer-PTO, and Differentiable-LP.
- Deterministic rows are Seasonal-Naive-PTO and Perfect-Information-MPC. The latter is an evaluation-only ideal-information reference, not a deployable competitor.
- DecisionFocused-Online and DigitalTwins-Policy remain clearly labeled secondary adaptations. Preserve their diagnostic results, but do not substitute them for primary controls or call them exact official reproductions.
- Inference optimizer accounting is fixed: RSC-PF, Decoupled-RSC-PF, and Direct-Policy use zero exact LP calls; Scheme2R-PTO, State-Conditioned-PTO, Official iTransformer-PTO, Differentiable-LP, and Seasonal-Naive-PTO use one optimizer call per origin; Perfect-Information-MPC calls are reported separately.
- Differentiable-LP is a CVXPYlayers methodology adaptation. Its decision gradient must pass through the differentiable LP to its forecaster during training, and it still performs one optimization-layer solve per inference origin.
- All deployable methods receive identical causal inputs and carried SOC/previous-CHP state. Future realized targets and renewable realizations are inaccessible until settlement.
- Primary reporting uses raw cumulative realized objective, shortage, physical feasibility, forecast metrics, recourse distance, latency, and optimizer calls. Terminal-stock adjustment is sensitivity analysis only.
- Do not run ablations, generate manuscript figures, or edit either manuscript language version in this plan.
- Preserve the untracked third_party/iTransformer_source/ tree; do not delete, reset, or silently replace it.
- Existing artifacts under reports/rsc_pf_matched_closed_loop_2019/v1 and v2 are read-only.
- The common search budget is at most four validation-selected trials per trainable method, effective batch size 64, at most 30 epochs per trained stage, patience 5, and validation every epoch. DiffLP may micro-batch only when gradient accumulation preserves effective batch size 64. Any method-specific exception must already exist in a hashed frozen source contract and be copied into the canonical contract before Gate 0.
- The canonical configuration must embed, not infer at runtime, the frozen RSC-PF v4.6 training/loss values, the v4.2 DiffLP search budget, and the external iTransformer preprocessing/training receipt hashes. Gate 1 rejects any row whose realized settings differ from those frozen values.

---

### Task 1: Freeze a truthful pre-implementation inventory

**Files:**
- Create: scripts/audit_rsc_pf_complete_formal_inventory.py
- Create: tests/test_rsc_pf_complete_formal_inventory.py
- Generate: reports/rsc_pf_complete_formal/preimplementation/INVENTORY.json

**Interfaces:**
- Consumes existing contracts, DiffLP receipts, iTransformer sources, v2 audit, and checkpoint directories.
- Produces build_inventory(frame_root: Path) -> dict[str, Any].

- [ ] **Step 1: Write the failing inventory test**

~~~python
def test_inventory_does_not_confuse_contract_with_completed_result(frame_root):
    result = build_inventory(frame_root)
    diff = result["methods"]["Differentiable-LP"]
    assert diff["training_function_present"] is True
    assert diff["native_gradient_gate_present"] is True
    assert diff["formal_result_present"] is False
    assert diff["status"] == "missing_formal_result"
    assert result["matched_v2"]["formal_candidate"] is False
~~~

Add assertions that the official iTransformer snapshot exists, current RSC-PF has only seed 2026, v2 passed only as a protocol audit, and the inventory reads no 2020/2021 artifact.

- [ ] **Step 2: Run the test and verify it fails because the interface is absent**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_inventory.py -q --basetemp D:\Paper\pytest_tmp_complete_inventory -p no:cacheprovider
~~~

- [ ] **Step 3: Implement the read-only inventory**

~~~python
METHODS = (
    "RSC-PF", "Decoupled-RSC-PF", "Direct-Policy", "Scheme2R-PTO",
    "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP",
    "Seasonal-Naive-PTO", "Perfect-Information-MPC",
)

def classify(training_present, checkpoint_present, formal_result_present):
    if formal_result_present:
        return "formal_result"
    if checkpoint_present:
        return "checkpoint_available"
    if training_present:
        return "missing_formal_result"
    return "implementation_missing"
~~~

Hash every cited contract, checkpoint, source receipt, and audit. Record missing evidence as null and emit evaluation_year_accessed=false and excluded_year_accessed=false.

- [ ] **Step 4: Run the test and create the receipt**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_inventory.py -q --basetemp D:\Paper\pytest_tmp_complete_inventory -p no:cacheprovider
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\audit_rsc_pf_complete_formal_inventory.py --frame-root . --output reports\rsc_pf_complete_formal\preimplementation\INVENTORY.json
~~~

- [ ] **Step 5: Review and commit**

~~~powershell
git diff --check
git add scripts/audit_rsc_pf_complete_formal_inventory.py tests/test_rsc_pf_complete_formal_inventory.py reports/rsc_pf_complete_formal/preimplementation/INVENTORY.json
git commit -m "audit: freeze complete formal experiment inventory"
~~~

**Stop:** Report the inventory and stop. Do not modify models or experiment settings.

---

### Task 2: Create one canonical formal experiment contract

**Files:**
- Create: configs/rsc_pf_complete_formal_v1.json
- Create: src/joint_dispatch/complete_formal_contract.py
- Create: tests/test_rsc_pf_complete_formal_contract.py

**Interfaces:**
- Produces CompleteFormalContract.from_path(path), method(method_id), expected_rows(stage), MethodSpec, and MethodSeedKey.

- [ ] **Step 1: Write failing roster and accounting tests**

~~~python
def test_primary_matrix_is_frozen_and_complete(contract):
    assert contract.primary_method_ids == (
        "RSC-PF", "Decoupled-RSC-PF", "Direct-Policy", "Scheme2R-PTO",
        "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP",
        "Seasonal-Naive-PTO", "Perfect-Information-MPC",
    )
    assert len(contract.expected_rows("gate1")) == 37

def test_difflp_truthfully_declares_optimizer_at_inference(contract):
    spec = contract.method("Differentiable-LP")
    assert spec.reproduction_level == "cvxpylayers_methodology_adaptation"
    assert spec.inference_lp_calls_per_origin == 1

def test_selection_and_evaluation_origin_counts_are_not_conflated(contract):
    assert contract.selection_origin_count == 8709
    assert contract.evaluation_origin_count == 8757
~~~

The 37 rows are seven stochastic methods × five seeds plus two deterministic/reference rows. State-Conditioned-PTO is required to isolate the neural scheduling proxy from a state-conditioned forecast followed by the online optimizer. Add tests for years, gas semantics, architecture dimensions, exact seed order, and optimizer roles.

- [ ] **Step 2: Write a fail-closed sealed-year test**

~~~python
def test_evaluation_access_requires_authorized_gate1(contract, tmp_path):
    with pytest.raises(PermissionError, match="authorized Gate 1"):
        contract.authorize_evaluation(tmp_path / "missing_gate1_transition.json")
~~~

- [ ] **Step 3: Run tests and confirm failure**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_contract.py -q --basetemp D:\Paper\pytest_tmp_complete_contract -p no:cacheprovider
~~~

- [ ] **Step 4: Implement the parser and canonical JSON**

~~~python
@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    role: str
    stochastic: bool
    forecast_metrics_applicable: bool
    inference_lp_calls_per_origin: int
    reproduction_level: str

@dataclass(frozen=True)
class MethodSeedKey:
    method_id: str
    seed: int | None
~~~

Copy architecture/data semantics and the complete training/loss values from `configs/joint_forecast_dispatch_formal_v4_6.json`, the four-trial budgets from `configs/joint_dispatch_search_budget_v4.json`, the official iTransformer preprocessing/training hashes from its frozen receipts, and closed-loop tolerances from the audited 2019 diagnostic. Use effective batch size 64, at most 30 epochs per trained stage, patience 5, validation every epoch, exactly 8,709 consecutive 2019 origins, exactly 8,757 consecutive 2020 origins, and allow_evaluation_access_before_gate1=false. Use the existing code IDs `Decoupled-RSC-PF` and `Official iTransformer-PTO`; do not introduce spelling aliases. The display label for the former is “Decoupled RSC-PF”.

- [ ] **Step 5: Run tests and commit**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_contract.py -q --basetemp D:\Paper\pytest_tmp_complete_contract -p no:cacheprovider
git diff --check
git add configs/rsc_pf_complete_formal_v1.json src/joint_dispatch/complete_formal_contract.py tests/test_rsc_pf_complete_formal_contract.py
git commit -m "feat: freeze complete RSC-PF formal contract"
~~~

**Stop:** Report the 37-row matrix and naming/optimizer decisions; do not implement providers.

---

### Task 3: Replace hard-coded four-method lists with a provider registry

**Files:**
- Create: src/joint_dispatch/complete_formal_providers.py
- Test: tests/test_rsc_pf_complete_formal_providers.py

**Interfaces:**
- Produces CompletePlannedStep, CompleteActionProvider, CompleteProviderRegistry.register(), build(), and validate().

~~~python
@dataclass(frozen=True)
class CompletePlannedStep:
    forecast: np.ndarray | None
    scheduler_demand: np.ndarray
    renewable_forecast: np.ndarray
    dispatch: np.ndarray
    inference_lp_calls: int

class CompleteActionProvider(Protocol):
    method_id: str
    optimizer_role: str
    forecast_metrics_applicable: bool
    def plan(self, origin: CausalOriginInput) -> CompletePlannedStep: ...
~~~

- [ ] **Step 1: Write failing registry tests**

~~~python
def test_registry_requires_every_primary_method(contract, registry):
    registry._factories.pop("Differentiable-LP")
    with pytest.raises(ValueError, match="Differentiable-LP"):
        registry.validate(contract)

@pytest.mark.parametrize("method_id", DEPLOYABLE_METHOD_IDS)
def test_provider_receives_only_causal_origin(method_id, registry, causal_origin):
    provider = registry.build(MethodSeedKey(method_id, 2026), fixture_resources())
    step = provider.plan(causal_origin)
    assert step.dispatch.shape == (4, 21)
    if method_id == "Direct-Policy":
        assert step.forecast is None
    else:
        assert step.forecast.shape == (4, 4)
~~~

Use an access spy that raises if any provider reads realized labels or sealed years. The complete-formal wrapper defines `PlannedStep.forecast` as `np.ndarray | None`; the evaluator records forecast metrics as not applicable when it is `None`, never as zero.

- [ ] **Step 2: Run tests and confirm missing behavior**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_providers.py tests\test_rsc_pf_matched_closed_loop_runner.py -q --basetemp D:\Paper\pytest_tmp_complete_providers -p no:cacheprovider
~~~

- [ ] **Step 3: Register non-DiffLP providers**

~~~python
registry.register("RSC-PF", build_rsc_pf_provider)
registry.register("Decoupled-RSC-PF", build_decoupled_provider)
registry.register("Direct-Policy", build_direct_policy_provider)
registry.register("Scheme2R-PTO", build_scheme2r_pto_provider)
registry.register("State-Conditioned-PTO", build_state_conditioned_pto_provider)
registry.register("Official iTransformer-PTO", build_itransformer_pto_provider)
registry.register("Seasonal-Naive-PTO", build_seasonal_naive_pto_provider)
registry.register("Perfect-Information-MPC", build_pi_mpc_reference_provider)
~~~

RSC-PF and Decoupled-RSC-PF share architecture and inputs; their scientific difference is the training gradient boundary. Direct-Policy has no prediction output/forecast loss. Scheme2R-PTO uses the earlier forecaster, while State-Conditioned-PTO uses the new state-conditioned forecaster; both then call the canonical LP. Perfect-Information-MPC remains a separate reference provider. Do not alias secondary adaptations to primary methods.

- [ ] **Step 4: Prove the new registry leaves the legacy diagnostic untouched**

Do not edit `scripts/run_rsc_pf_matched_closed_loop_2019.py` or `src/joint_dispatch/matched_closed_loop_providers.py`. Run their existing tests unchanged and require the old v2 configuration to retain four methods and asymmetric seeds.

- [ ] **Step 5: Test and commit**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_providers.py tests\test_rsc_pf_matched_closed_loop_runner.py tests\test_rsc_pf_matched_closed_loop_providers.py -q --basetemp D:\Paper\pytest_tmp_complete_providers -p no:cacheprovider
git diff --check
git add src/joint_dispatch/complete_formal_providers.py tests/test_rsc_pf_complete_formal_providers.py
git commit -m "refactor: register complete formal method providers"
~~~

**Stop:** Demonstrate provider construction and causal isolation only; do not train.

---

### Task 4: Connect Differentiable-LP to the common pipeline

**Files:**
- Modify: src/joint_dispatch/complete_formal_providers.py
- Create: src/joint_dispatch/complete_formal_difflp.py
- Create: tests/test_rsc_pf_complete_formal_difflp.py

**Interfaces:**
- Produces build_difflp_provider() and train_complete_difflp().

- [ ] **Step 1: Write a native-gradient test**

~~~python
def test_difflp_decision_gradient_reaches_forecaster(native_layer, batch):
    model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4)
    artifact = one_difflp_training_step(model, native_layer, batch)
    assert artifact.lp_calls == 1
    assert artifact.forecaster_gradient_norm > 0.0
    assert np.isfinite(artifact.forecaster_gradient_norm)
~~~

- [ ] **Step 2: Write inference-accounting and gas-boundary tests**

~~~python
def test_difflp_provider_reports_one_solve_per_origin(provider, causal_origin):
    step = provider.plan(causal_origin)
    assert step.inference_lp_calls == 1

def test_difflp_rigid_balance_uses_only_three_carriers(problem_spec):
    assert problem_spec.rigid_demand_tasks == ("electricity", "cooling", "heating")
~~~

- [ ] **Step 3: Run tests and confirm common integration is absent**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_difflp.py -q --basetemp D:\Paper\pytest_tmp_complete_difflp -p no:cacheprovider
~~~

- [ ] **Step 4: Integrate through a new wrapper without changing legacy DiffLP code**

Import and wrap the existing four-hour, 21-output `DifferentiableIESLayer` and `train_differentiable_lp`; do not edit `formal_v4_diffopt.py` or `formal_v4_2_gate2_training.py`. Preserve effective batch size by gradient accumulation if micro-batching is needed.

~~~python
loss = forecast_weight * forecast_loss(prediction, target)
loss = loss + decision_weight * realized_dispatch_objective(lp_dispatch, realized_context)
loss.backward()
~~~

Persist package versions, Python path, DPP status, parity residual, gradient norm, optimizer calls, sample exposures, checkpoint hash, and reproduction level.

- [ ] **Step 5: Run native and regression tests**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_difflp.py tests\test_joint_dispatch_formal_v4_diffopt.py tests\test_joint_dispatch_formal_v4_diffopt_runner.py tests\test_joint_dispatch_formal_v4_2_gate2_training.py -q --basetemp D:\Paper\pytest_tmp_complete_difflp -p no:cacheprovider
~~~

- [ ] **Step 6: Commit**

~~~powershell
git diff --check
git add src/joint_dispatch/complete_formal_providers.py src/joint_dispatch/complete_formal_difflp.py tests/test_rsc_pf_complete_formal_difflp.py
git commit -m "feat: integrate differentiable LP formal baseline"
~~~

**Stop:** Run only one-batch/one-origin native evidence. Passing it is not a completed experiment.

---

### Task 5: Build resumable five-seed training and checkpoint lineage

**Files:**
- Create: src/joint_dispatch/complete_formal_training.py
- Create: scripts/train_rsc_pf_complete_formal.py
- Create: tests/test_rsc_pf_complete_formal_training.py

**Interfaces:**
- Produces train_row(contract, key, data, output_dir, resume=False), resume_row(), FrozenRowArtifact, and one TRAINING_RECEIPT.json per stochastic row.

~~~python
@dataclass(frozen=True)
class FrozenRowArtifact:
    key: MethodSeedKey
    checkpoint_path: Path
    checkpoint_sha256: str
    contract_sha256: str
    train_data_sha256: str
    source_sha256: str
    parent_checkpoint_sha256: str | None
    training_exposures: int
    decision_forecaster_gradient_norm: float | None
    complete: bool
~~~

- [ ] **Step 1: Write gradient-boundary tests**

~~~python
def test_joint_and_decoupled_have_different_forecaster_gradient_contract(tiny_data):
    joint = train_one_step("RSC-PF", tiny_data)
    decoupled = train_one_step("Decoupled-RSC-PF", tiny_data)
    assert joint.dispatch_to_forecaster_gradient_norm > 0.0
    assert decoupled.dispatch_to_forecaster_gradient_norm <= 1e-12
~~~

Also require Direct-Policy to have no forecast loss contract, PTO models to have no dispatch-to-forecast gradient, and every stochastic row to use its declared seed.

- [ ] **Step 2: Write resume/provenance tests**

~~~python
def test_resume_rejects_contract_or_data_hash_change(tmp_path, tiny_data, contract):
    train_row(contract, MethodSeedKey("RSC-PF", 2026), tiny_data, tmp_path)
    with pytest.raises(ValueError, match="hash"):
        resume_row(tmp_path, "wrong-contract-hash")
~~~

Require atomic checkpoints, epoch/optimizer/RNG restoration, exact exposure counts, and refusal to overwrite a completed row.

- [ ] **Step 3: Run tests and confirm missing orchestration**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_training.py -q --basetemp D:\Paper\pytest_tmp_complete_training -p no:cacheprovider
~~~

- [ ] **Step 4: Implement training through one row API**

RSC-PF uses warm-start initialization followed by fully unfrozen joint training. Decoupled-RSC-PF uses the same architecture, initialization, data, budget, and exact parent checkpoint hash, but detaches the forecast before scheduling. Scheme2R-PTO, State-Conditioned-PTO, and Official iTransformer-PTO train forecasting objectives only. Direct-Policy trains dispatch objectives only. DiffLP follows Task 4. Receipts must prove that the RSC-PF and Decoupled-RSC-PF parent checkpoint hashes are identical. State-Conditioned-PTO must use the same state-conditioned forecasting parent before its path switches to the canonical online LP.

The finite RSC-PF selection grid is [1.0, 1.5, 2.0, 3.0], matching the frozen maximum of four trials. Run it with seed 2026 and 2019 selection data only, then freeze the selected multiplier for all seeds. DiffLP uses forecaster learning rates [1e-5, 3e-5, 1e-4, 3e-4], also selected with seed 2026 on 2019. Other methods reuse their already frozen validation-selected configurations; a checkpoint is reusable only if its data, preprocessing, source, architecture, and training-config hashes match the new contract. Missing seeds are trained with that same frozen configuration, not retuned. No architecture, capacity, loss definition, search budget, or threshold changes are allowed after Task 2.

- [ ] **Step 5: Run tiny-data and regression tests**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_training.py tests\test_joint_dispatch_formal_v4_6_training.py tests\test_joint_dispatch_formal_v4_6_loss.py tests\test_scheme2r.py -q --basetemp D:\Paper\pytest_tmp_complete_training -p no:cacheprovider
~~~

- [ ] **Step 6: Commit**

~~~powershell
git diff --check
git add src/joint_dispatch/complete_formal_training.py scripts/train_rsc_pf_complete_formal.py tests/test_rsc_pf_complete_formal_training.py
git commit -m "feat: add resumable complete formal training"
~~~

**Stop:** Demonstrate tiny-data training and exact resume only; do not start full epochs.

---

### Task 6: Build the complete matrix runner and independent auditor

**Files:**
- Create: src/joint_dispatch/complete_formal_execution.py
- Create: scripts/run_rsc_pf_complete_formal_gate1.py
- Create: scripts/run_rsc_pf_complete_formal_gate2.py
- Create: scripts/audit_rsc_pf_complete_formal.py
- Create: tests/test_rsc_pf_complete_formal_execution.py
- Create: tests/test_rsc_pf_complete_formal_audit.py

**Interfaces:**
- Produces run_gate1(), run_gate2(), decide_gate1(), and audit_complete_formal().

~~~python
@dataclass(frozen=True)
class GateDecision:
    stage: str
    authorized_next_gate: bool
    expected_rows: tuple[MethodSeedKey, ...]
    missing_rows: tuple[str, ...]
    failures: tuple[str, ...]
    receipt_sha256: str

@dataclass(frozen=True)
class AuditReceipt:
    stage: str
    status: str
    execution_receipt_sha256: str
    recomputed_row_count: int
    failures: tuple[str, ...]
    evaluation_year_accessed: bool
    excluded_year_accessed: bool
~~~

- [ ] **Step 1: Write matrix-completeness tests**

~~~python
def test_gate1_rejects_one_missing_difflp_seed(contract, fake_rows):
    fake_rows.pop(("Differentiable-LP", 2030))
    decision = decide_gate1(contract, fake_rows)
    assert decision.authorized_gate2 is False
    assert "Differentiable-LP/2030" in decision.missing_rows
~~~

- [ ] **Step 2: Write access and accounting tests**

~~~python
def test_gate2_refuses_rejected_gate1(contract, tmp_path):
    with pytest.raises(PermissionError):
        run_gate2(contract, tmp_path, tmp_path / "GATE1_TRANSITION.json")

def test_audit_recomputes_lp_calls(contract, complete_rows):
    audit = audit_rows(contract, complete_rows, origin_count=8709)
    assert audit.rows["Differentiable-LP/2026"].inference_lp_calls == 8709
    assert audit.rows["RSC-PF/2026"].inference_lp_calls == 0
~~~

- [ ] **Step 3: Run tests and confirm failure**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_execution.py tests\test_rsc_pf_complete_formal_audit.py -q --basetemp D:\Paper\pytest_tmp_complete_execution -p no:cacheprovider
~~~

- [ ] **Step 4: Implement immutable, resumable row execution**

Each row writes TRAINING_RECEIPT.json, checkpoint.pt, rollout.npz, and EVALUATION_RECEIPT.json below reports/rsc_pf_complete_formal/<run-id>/<stage>/rows/<method>/<seed-or-deterministic>/. Reuse only when contract, data, source, checkpoint, and evaluator hashes match; otherwise fail instead of overwriting.

- [ ] **Step 5: Implement the independent auditor**

Recompute row completeness, years accessed, chronology, state-carry hashes, metrics, finite values, feasibility/shortage, optimizer calls, source/checkpoint hashes, and paired RSC-PF-versus-Decoupled results. Execution/audit agreement is required to create GATE1_TRANSITION.json or GATE2_AUDIT.json.

- [ ] **Step 6: Run regressions and commit**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_execution.py tests\test_rsc_pf_complete_formal_audit.py tests\test_rsc_pf_matched_closed_loop.py tests\test_rsc_pf_closed_loop_metrics.py -q --basetemp D:\Paper\pytest_tmp_complete_execution -p no:cacheprovider
git diff --check
git add src/joint_dispatch/complete_formal_execution.py scripts/run_rsc_pf_complete_formal_gate1.py scripts/run_rsc_pf_complete_formal_gate2.py scripts/audit_rsc_pf_complete_formal.py tests/test_rsc_pf_complete_formal_execution.py tests/test_rsc_pf_complete_formal_audit.py
git commit -m "feat: execute and audit complete formal matrix"
~~~

**Stop:** Complete synthetic/tiny-data matrices only; do not read real 2019 or 2020 arrays.

---

### Task 7: Run Gate 0 readiness and resource certification

**Files:**
- Create: scripts/run_rsc_pf_complete_formal_gate0.py
- Create: tests/test_rsc_pf_complete_formal_gate0.py
- Generate: reports/rsc_pf_complete_formal/<new-run-id>/gate0/

**Interfaces:**
- Produces GATE0_RECEIPT.json, GATE0_AUDIT.json, and GATE0_TRANSITION.json with authorized_gate1_training.

- [ ] **Step 1: Write tests requiring native family coverage**

~~~python
def test_gate0_requires_native_difflp_and_official_itransformer(receipt):
    assert receipt["difflp"]["native_gradient_nonzero"] is True
    assert receipt["itransformer"]["official_source_hash_verified"] is True
    assert receipt["all_method_families_smoked"] is True
~~~

- [ ] **Step 2: Implement a 16-window, two-epoch, one-seed smoke**

Measure memory and p50/p95 time for neural training, exact LP inference, and DiffLP forward/backward. Project the full 37-row training plus 2019 evaluation, subtracting only rows whose checkpoint reuse passes the hash audit. Require at least 20 GiB of disk space after the estimated checkpoint footprint; report the percentage as an informational diagnostic only. If estimated time exceeds 24 hours, report it and require explicit user authorization rather than falsifying a failure or silently shrinking the matrix.

- [ ] **Step 3: Run all affected tests**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_*.py tests\test_rsc_pf_matched_closed_loop*.py tests\test_joint_dispatch_formal_v4_diffopt*.py -q --basetemp D:\Paper\pytest_tmp_complete_gate0 -p no:cacheprovider
~~~

- [ ] **Step 4: Run Gate 0 with a fresh run ID**

~~~powershell
$RunId = 'complete_formal_20260906_a'
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\run_rsc_pf_complete_formal_gate0.py --contract configs\rsc_pf_complete_formal_v1.json --output-root reports\rsc_pf_complete_formal --run-id $RunId
~~~

- [ ] **Step 5: Inspect receipts and stop**

Confirm hashes, native dependencies, data boundaries, disk margin, duration estimate, and authorization. Do not start Task 8 automatically.

**Stop:** If Gate 0 fails, report the exact condition. If it passes, report runtime/storage and wait for authorization for full training.

---

### Task 8: Train the matrix and execute 2019 Gate 1

**Files:**
- Generate only: reports/rsc_pf_complete_formal/<run-id>/training/
- Generate only: reports/rsc_pf_complete_formal/<run-id>/gate1/

**Interfaces:**
- Consumes authorized Gate 0 and only 2015–2019 data.
- Produces five-seed checkpoints, 2019 rollouts, GATE1_DECISION.json, GATE1_AUDIT.json, and only on success GATE1_TRANSITION.json.

- [ ] **Step 1: Verify Gate 0 and sealed-year denial**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\audit_rsc_pf_complete_formal.py --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a --stage gate0
~~~

- [ ] **Step 2: Run seed-2026 RSC-PF training-grid selection using 2019 only**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\train_rsc_pf_complete_formal.py --contract configs\rsc_pf_complete_formal_v1.json --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a --method RSC-PF --seed 2026 --select-frozen-grid --resume
~~~

Freeze SELECTED_TRAINING_CONFIG.json before remaining seeds. The chosen candidate must satisfy unchanged forecast guardrails.

- [ ] **Step 3: Train every remaining stochastic row**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\train_rsc_pf_complete_formal.py --contract configs\rsc_pf_complete_formal_v1.json --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a --all-methods --all-seeds --resume
~~~

- [ ] **Step 4: Evaluate all 8,709 chronological 2019 origins**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\run_rsc_pf_complete_formal_gate1.py --contract configs\rsc_pf_complete_formal_v1.json --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a --resume
~~~

- [ ] **Step 5: Run the independent audit**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\audit_rsc_pf_complete_formal.py --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a --stage gate1
~~~

Gate 1 requires all 37 rows, five RSC-PF seeds, finite metrics, causal/state hashes, correct optimizer calls, zero 2020/2021 access, forecast guardrails, and at least three of five paired RSC-PF seed directions favorable to Decoupled-RSC-PF on the frozen primary rolling objective. Report effect sizes even if rejected.

All Gate-1 performance values are validation/model-selection evidence. They must be labeled as such and cannot appear as the final unbiased paper result.

**Stop:** Always stop after Gate 1. If rejected, do not open 2020 or change thresholds. If accepted, wait for explicit authorization for Gate 2.

---

### Task 9: Execute the one-time sealed 2020 Gate 2

**Files:**
- Generate only: reports/rsc_pf_complete_formal/<run-id>/gate2/

**Interfaces:**
- Consumes immutable checkpoints and an audited GATE1_TRANSITION.json with authorized_gate2=true.
- Produces final 2020 row artifacts, GATE2_DECISION.json, and GATE2_AUDIT.json.

- [ ] **Step 1: Verify authorization and freeze all input hashes**

Reject any checkpoint/config/source/evaluator hash mismatch. Do not construct a training optimizer. Create an append-only `SEALED_ACCESS_LEDGER.jsonl` before loading the split; it records run ID, process ID, contract hash, Gate-1 transition hash, start time, and row cursor.

Require exactly 8,757 consecutive 2020 origins before the first row is evaluated; any other count aborts Gate 2.

- [ ] **Step 2: Run sealed evaluation once**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\run_rsc_pf_complete_formal_gate2.py --contract configs\rsc_pf_complete_formal_v1.json --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a --gate1-transition reports\rsc_pf_complete_formal\complete_formal_20260906_a\gate1\GATE1_TRANSITION.json --resume
~~~

- [ ] **Step 3: Audit independently**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\audit_rsc_pf_complete_formal.py --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a --stage gate2
~~~

- [ ] **Step 4: Distinguish crash recovery from a second evaluation**

`--resume` may continue only from the last hash-verified row cursor and must append to the same sealed-access ledger. Starting a new run ID, changing any frozen input, or recomputing a completed row after the first sealed access fails closed.

- [ ] **Step 5: Stop without tuning**

Report completeness, feasibility, shortage, forecasting, objective, recourse, latency, optimizer calls, and paired five-seed effects. A disappointing sealed result remains the final result.

**Stop:** Do not start ablations or manuscript revision.

---

### Task 10: Produce the experiment-only statistical package

**Files:**
- Create: scripts/summarize_rsc_pf_complete_formal.py
- Create: tests/test_rsc_pf_complete_formal_summary.py
- Generate: reports/rsc_pf_complete_formal/<run-id>/summary/

**Interfaces:**
- Produces PRIMARY_RESULTS.csv, FORECAST_RESULTS.csv, PHYSICAL_RESULTS.csv, EFFICIENCY_RESULTS.csv, PAIRED_EFFECTS.csv, RESULTS_REVIEW.md, and SUMMARY_AUDIT.json.

- [ ] **Step 1: Write failing statistical-unit tests**

~~~python
def test_summary_uses_hierarchical_seed_then_week_resampling(summary):
    assert summary["independent_unit"] == "model_seed"
    assert summary["temporal_block"] == "paired_168h_within_seed"
    assert summary["bootstrap_order"] == ["seed", "paired_168h_block"]
    assert summary["bootstrap_replicates"] == 2000

def test_roles_are_not_overclaimed(summary):
    assert summary["roles"]["Perfect-Information-MPC"] == "ideal_information_reference"
    assert summary["roles"]["Differentiable-LP"] == "methodology_adaptation"
    assert summary["forecast_metrics"]["Direct-Policy"] == "not_applicable"
~~~

- [ ] **Step 2: Implement aggregation only**

Use five-seed mean ± SD and a hierarchical paired bootstrap with 2,000 replicates: resample the five paired model seeds first, then resample matched 168-hour time blocks within each selected seed. Never count weekly blocks as independent model replicates. Report 95% confidence intervals and multiplicity-aware secondary comparisons. Keep raw cumulative objective primary and terminal-stock adjustment as sensitivity. Separate forecast and dispatch outcomes.

- [ ] **Step 3: Test and generate**

~~~powershell
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe -m pytest tests\test_rsc_pf_complete_formal_summary.py -q --basetemp D:\Paper\pytest_tmp_complete_summary -p no:cacheprovider
& D:\Paper\envs\rsc_pf_diffopt_v4\python.exe scripts\summarize_rsc_pf_complete_formal.py --run-root reports\rsc_pf_complete_formal\complete_formal_20260906_a
~~~

- [ ] **Step 4: Audit counts/hashes and commit code only**

~~~powershell
git diff --check
git add scripts/summarize_rsc_pf_complete_formal.py tests/test_rsc_pf_complete_formal_summary.py
git commit -m "feat: summarize complete formal experiment"
~~~

**Stop:** Deliver experiment evidence only. Figures, ablations, and manuscript changes need separate reviewed plans.

---

## Mandatory Review Checkpoint After Every Task

Before Task N+1 begins, report exactly:

1. Files changed or artifacts generated.
2. Focused test command and exact pass/fail count.
3. Scientific contract checked, including data years and optimizer role.
4. Remaining limitation or blocker; write “none” only when evidence supports it.
5. Explicit statement: “Task N complete; Task N+1 not started.”

## Plan Self-Review Record

- **Spec coverage:** Addresses the diagnostic-only roster, missing DiffLP formal result, asymmetric RSC-PF seeds, the missing State-Conditioned-PTO isolation control, hard-coded method lists, incorrect optimizer metadata, missing full-matrix audit, rejected candidate, and sealed-test protection.
- **Scope control:** No ablation, figure, or manuscript work. Secondary adaptations are preserved but cannot displace primary controls.
- **Completeness scan:** No deferred or unspecified implementation step remains; tasks name files, interfaces, commands, assertions, and stop conditions.
- **Type consistency:** MethodSpec and MethodSeedKey are introduced in Task 2, FrozenRowArtifact in Task 5, and GateDecision/AuditReceipt in Task 6; downstream tasks consume those exact names and fields.
- **Risk controls:** 2020 is inaccessible before an audited Gate 1 authorization; 2021 is always excluded; existing diagnostic artifacts and iTransformer sources are preserved.
- **Execution mode:** Inline, one task at a time, with user review after every task.
