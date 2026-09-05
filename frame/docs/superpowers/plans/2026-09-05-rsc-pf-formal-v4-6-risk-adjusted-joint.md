# RSC-PF Formal-v4.6 Risk-Adjusted Joint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a statistically meaningful four-task nominal forecast while allowing one end-to-end RSC-PF network to learn a bounded state-conditioned scheduling risk adjustment that improves realized dispatch.

**Architecture:** Extend the residual-gated v4.5 model with a three-task risk head between the nominal forecast and scheduling proxy. The nominal electricity/cooling/heating/gas-prior tensor remains the reported forecast; the scheduler consumes nominal rigid demand plus a non-negative capped electricity/cooling/heating adjustment, while gas remains an unadjusted auxiliary prior. Joint training uses the full regime-aware forecast objective, decision/imitation terms, P1 anchoring, and explicit risk penalties; Fair Decoupled uses the identical interface but detaches every decision-to-forecast edge.

**Tech Stack:** Python 3, PyTorch, NumPy, PyYAML, pytest, existing formal-v4.4/v4.5 physics decoder, rollout, data, and immutable-artifact utilities.

## Global Constraints

- Do not modify or overwrite any `formal_v4_4` or `formal_v4_5` report, receipt, or config.
- Do not modify or stage `third_party/iTransformer_source/`.
- Use new `formal_v4_6` module, config, script, report, receipt, and schema names.
- Preserve 24-hour causal history, four-hour horizon, four nominal forecast tasks, 15 continuous controls, 21 decoded dispatch values, and no future binary decisions.
- Electricity, cooling, and heating are rigid terminal demands; gas remains `station_side_auxiliary_prior` and receives exactly zero risk adjustment.
- Use 2015--2018 for training, calibration, and early stopping; use 2019 only after configuration and source-manifest freeze; never open 2020 during this plan; exclude 2021.
- Fit risk caps only from frozen-P1 residuals on the purged early-stop partition at quantile `0.90`.
- Use risk-regularization multipliers exactly `[0.5, 1.0, 2.0]`; select among them only by eligible early-stop decision objective.
- Initialize the risk-head output bias to `-6.0`; use a 64-unit GELU/LayerNorm hidden layer and a sigmoid cap transform.
- Feed the risk head the same train-fitted normalized ten-feature scheduling context and normalized previous-CHP scalar used by the scheduler; never mix raw-capacity CHP units with normalized features.
- Nominal checkpoint guardrails are fixed at: four-task `1.02`, electricity `1.02`, gas `1.10`, active cooling/heating `1.05`, normalized inactive leakage `1.05`, macro-F1 drop `0.02`, and transition balanced-accuracy gain `0.01`.
- Fair Decoupled decision gradients into forecast base, regime gate, and magnitude head must be at most `1e-12`; maximum physical residual is `1e-6`.
- Do not start Gate 1 baselines, ablations, multi-seed experiments, or 2020 evaluation under this plan.
- If the v4.6 diagnostic or 2019 Pilot fails, persist the negative receipt and stop.

---

### Task 1: Freeze and validate the formal-v4.6 contract

**Files:**
- Create: `configs/joint_forecast_dispatch_formal_v4_6.json`
- Create: `src/joint_dispatch/formal_v4_6_contract.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_contract.py`

**Interfaces:**
- Consumes: `TASK_ORDER`, `EXOG_ORDER`, `STATUS_ORDER`, `DISPATCH_ORDER`, and the v4.5 method ordering.
- Produces: `FormalV46Contract`, `load_formal_v4_6_contract(path: str | Path) -> FormalV46Contract`, `FormalV46Contract.risk_adjustment`, and `FormalV46Contract.pilot_thresholds`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_v46_contract_freezes_risk_semantics_and_years():
    contract = load_formal_v4_6_contract(
        "configs/joint_forecast_dispatch_formal_v4_6.json"
    )
    assert contract.train_years == (2015, 2016, 2017, 2018)
    assert contract.selection_year == 2019
    assert contract.evaluation_year == 2020
    assert contract.risk_adjustment == {
        "tasks": ["electricity", "cooling", "heating"],
        "gas_adjustment": 0.0,
        "cap_quantile": 0.90,
        "hidden_width": 64,
        "initial_output_bias": -6.0,
        "regularization_multipliers": [0.5, 1.0, 2.0],
        "risk_size_base_weight": 0.10,
        "off_risk_base_weight": 0.50,
        "j_risk_lr": 0.0005,
    }

def test_v46_contract_rejects_gas_adjustment_and_evaluation_access(tmp_path):
    payload = json.loads(Path(CONFIG).read_text(encoding="utf-8"))
    payload["risk_adjustment"]["gas_adjustment"] = 0.1
    payload["allow_evaluation_access_before_gate2"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="gas|evaluation"):
        load_formal_v4_6_contract(path)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_contract.py -q
```

Expected: FAIL because the v4.6 contract and config do not exist.

- [ ] **Step 3: Implement the contract and config**

Create a strict v4.6 loader following `formal_v4_5_contract.py`. The JSON must
add the exact `risk_adjustment` object tested above and these thresholds:

```json
"pilot_thresholds": {
  "maximum_four_task_score_ratio": 1.02,
  "maximum_electricity_wape_ratio": 1.02,
  "maximum_gas_wape_ratio": 1.10,
  "maximum_active_thermal_wape_ratio": 1.05,
  "maximum_normalized_inactive_leakage_ratio": 1.05,
  "maximum_macro_f1_drop": 0.02,
  "minimum_transition_balanced_accuracy_gain": 0.01,
  "maximum_decoupled_decision_gradient": 1e-12,
  "maximum_physical_residual": 1e-6
}
```

The inherited `pilot_budget` retains all v4.5 values. The only new optimizer
rate is `risk_adjustment.j_risk_lr = 0.0005`; the contract validates it exactly.

`validate()` must compare exact frozen values, exact task/control orders, exact
years, and `protocol_status == "frozen"`.

- [ ] **Step 4: Run focused tests**

Run the Task 1 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add configs/joint_forecast_dispatch_formal_v4_6.json src/joint_dispatch/formal_v4_6_contract.py tests/test_joint_dispatch_formal_v4_6_contract.py
git commit -m "add formal v4.6 risk-adjustment contract"
```

### Task 2: Add a version-bound formal-v4.6 source manifest

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_provenance.py`
- Create: `scripts/build_rsc_pf_formal_v4_6_source_manifest.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_provenance.py`

**Interfaces:**
- Consumes: repository root, committed v4.6 contract, run ID, Git HEAD, the inherited v4.4/v4.5 runtime closure, and all new v4.6 production/config/script paths.
- Produces: `SOURCE_MANIFEST_SCHEMA_V46`, `V46_SOURCE_PATHS`, `build_source_manifest_v46(repo_root: str | Path, run_id: str, contract_sha256: str, source_paths: Sequence[str] = V46_SOURCE_PATHS) -> Mapping[str, Any]`, `_validate_manifest_entries_against_git_v46(payload: Mapping[str, Any], repo_root: Path) -> Mapping[str, Any]`, `validate_source_manifest_v46(payload: Mapping[str, Any], repo_root: str | Path, expected_run_id: str, expected_contract_sha256: str) -> Mapping[str, Any]`, and `write_source_manifest_v46(path: str | Path, payload: Mapping[str, Any]) -> str`.

- [ ] **Step 1: Write failing provenance tests**

```python
def test_v46_manifest_binds_contract_head_and_every_declared_file(tmp_path):
    repo = make_git_repo(tmp_path, files={"a.py": "x = 1\n", "b.json": "{}\n"})
    payload = build_source_manifest_v46(
        repo_root=repo, run_id="v46-test", contract_sha256="a" * 64,
        source_paths=("a.py", "b.json"),
    )
    checked = validate_source_manifest_v46(
        payload, repo, "v46-test", "a" * 64,
    )
    assert checked["entry_count"] == 2
    assert checked["git_commit"] == git_head(repo)

def test_v46_manifest_rejects_dirty_missing_or_post_hash_source(tmp_path):
    repo = make_git_repo(tmp_path, files={"a.py": "x = 1\n"})
    payload = build_source_manifest_v46(
        repo_root=repo, run_id="v46-test", contract_sha256="a" * 64,
        source_paths=("a.py",),
    )
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty|changed|hash"):
        validate_source_manifest_v46(payload, repo, "v46-test", "a" * 64)
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_provenance.py -q
```

Expected: FAIL because the v4.6 provenance module does not exist.

- [ ] **Step 3: Implement the manifest and builder**

Use schema `formal-v4.6-source-manifest-v1`. Start `V46_SOURCE_PATHS` with the
full `DEFAULT_SOURCE_PATHS` runtime closure from v4.4, add the v4.5 contract,
loss, training, Pilot-data, Pilot-executor, and artifact modules, then add the
v4.6 config and every production module/script created by this plan. Remove the
old v4.4 config and v4.4 gate/Pilot entry scripts from the resulting tuple so a
v4.6 manifest cannot be mistaken for a v4.4 run. Sort and deduplicate the final
tuple before hashing.

```python
SOURCE_MANIFEST_SCHEMA_V46 = "formal-v4.6-source-manifest-v1"

def validate_source_manifest_v46(payload, repo_root, expected_run_id,
                                 expected_contract_sha256):
    if payload.get("schema") != SOURCE_MANIFEST_SCHEMA_V46:
        raise ValueError("formal-v4.6 source-manifest schema mismatch")
    if payload.get("run_id") != expected_run_id:
        raise ValueError("formal-v4.6 source-manifest run-id mismatch")
    if payload.get("contract_sha256") != expected_contract_sha256:
        raise ValueError("formal-v4.6 source-manifest contract mismatch")
    return _validate_manifest_entries_against_git_v46(payload, Path(repo_root))
```

The validator must recompute the identity hash, every file hash and size,
require every declared path to be tracked and clean at the recorded HEAD, and
reject missing/extra entries. The writer uses `write_json_once`; the builder CLI
loads the v4.6 contract and prints the path and SHA-256 of the written manifest.

- [ ] **Step 4: Run focused provenance tests**

Run the Task 2 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_provenance.py scripts/build_rsc_pf_formal_v4_6_source_manifest.py tests/test_joint_dispatch_formal_v4_6_provenance.py
git commit -m "add formal v4.6 source manifest"
```

### Task 3: Fit immutable train-information-only risk caps

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_risk.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_risk.py`

**Interfaces:**
- Consumes: frozen-P1 physical predictions and targets shaped `[N,4,4]`, early-stop timestamps, contract hash, and parent model hash.
- Produces: `RiskCapReceiptV46`, `fit_risk_caps_v46(prediction: np.ndarray, target: np.ndarray, timestamps: np.ndarray, split_role: str, quantile: float, contract_sha256: str, parent_sha256: str) -> RiskCapReceiptV46`, `save_risk_caps_v46(root: str | Path, receipt: RiskCapReceiptV46) -> str`, and `load_risk_caps_v46(root: str | Path) -> RiskCapReceiptV46`.

- [ ] **Step 1: Write failing cap and lineage tests**

```python
def test_caps_are_early_stop_residual_quantiles_and_exclude_gas():
    prediction = torch.zeros(5, 4, 4)
    target = torch.arange(80, dtype=torch.float32).reshape(5, 4, 4)
    receipt = fit_risk_caps_v46(
        prediction=prediction,
        target=target,
        timestamps=np.array(["2018-01-01"] * 5, dtype="datetime64[ns]"),
        split_role="early_stop",
        quantile=0.90,
        contract_sha256="a" * 64,
        parent_sha256="b" * 64,
    )
    expected = torch.quantile(target[..., :3].abs(), 0.90, dim=0)
    assert np.allclose(receipt.cap, expected.numpy())
    assert receipt.cap.shape == (4, 3)
    assert receipt.residual.shape == (5, 4, 3)
    assert receipt.split_role == "early_stop"
    assert receipt.years == (2018,)

def test_caps_reject_2019_or_2020_and_tampered_lineage(tmp_path):
    with pytest.raises(ValueError, match="2015--2018"):
        make_cap_receipt(year=2019)
    receipt = make_cap_receipt(year=2018)
    save_risk_caps_v46(tmp_path, receipt)
    payload = json.loads((tmp_path / "RISK_CAPS.json").read_text())
    payload["cap"][0][0] += 1.0
    (tmp_path / "RISK_CAPS.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash|lineage"):
        load_risk_caps_v46(tmp_path)
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_risk.py -q
```

Expected: FAIL because the risk-cap module does not exist.

- [ ] **Step 3: Implement cap fitting and immutable receipt I/O**

```python
@dataclass(frozen=True)
class RiskCapReceiptV46:
    cap: np.ndarray                 # [4,3], electricity/cooling/heating
    residual: np.ndarray            # [N,4,3], persisted frozen-P1 errors
    timestamps: np.ndarray          # [N], early-stop origins
    split_role: str                 # exactly "early_stop"
    quantile: float                 # exactly 0.90
    years: tuple[int, ...]          # subset of 2015--2018 only
    contract_sha256: str
    parent_sha256: str
    arrays_sha256: str
    residual_sha256: str
    lineage_sha256: str

def fit_risk_caps_v46(prediction, target, timestamps, split_role, quantile,
                      contract_sha256, parent_sha256):
    if split_role != "early_stop":
        raise ValueError("risk caps require the purged early_stop split")
    residual = np.abs(np.asarray(target)[..., :3] -
                      np.asarray(prediction)[..., :3])
    cap = np.quantile(residual, 0.90, axis=0)
    cap = np.maximum(cap, 1e-6)
    arrays_sha256 = hashlib.sha256(
        np.asarray(cap, dtype="<f8").tobytes()
    ).hexdigest()
    residual_sha256 = hashlib.sha256(
        np.asarray(residual, dtype="<f8").tobytes()
    ).hexdigest()
    years = tuple(sorted(set(
        np.asarray(timestamps, dtype="datetime64[ns]")
        .astype("datetime64[Y]").astype(int) + 1970
    )))
    lineage = {
        "contract_sha256": contract_sha256,
        "parent_sha256": parent_sha256,
        "arrays_sha256": arrays_sha256,
        "residual_sha256": residual_sha256,
        "split_role": "early_stop",
        "quantile": 0.90,
        "years": list(years),
    }
    return RiskCapReceiptV46(
        cap=cap, residual=residual, timestamps=np.asarray(timestamps),
        split_role="early_stop", quantile=0.90, years=years,
        contract_sha256=contract_sha256, parent_sha256=parent_sha256,
        arrays_sha256=arrays_sha256, residual_sha256=residual_sha256,
        lineage_sha256=canonical_sha256(lineage),
    )
```

Use `write_json_once` and canonical SHA-256 helpers from v4.4. Persist the cap,
complete residual tensor, and timestamps in `RISK_CAPS.npz`; persist metadata
and both array hashes in `RISK_CAPS.json`. Refuse overwrite, non-finite data,
wrong shapes, a split role other than `early_stop`, non-0.90 quantile, and years
outside 2015--2018.

- [ ] **Step 4: Run focused tests**

Run the Task 3 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_risk.py tests/test_joint_dispatch_formal_v4_6_risk.py
git commit -m "add immutable v4.6 risk caps"
```

### Task 4: Add the bounded risk-adjusted RSC-PF model

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_model.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_model.py`

**Interfaces:**
- Consumes: v4.4 residual-gated forecast components, cap tensor `[4,3]`, existing scheduler context `[B,4,6]`, state `[B,32]`, and train-fitted previous-CHP mean/scale for a raw previous CHP tensor `[B,1]`.
- Produces: `RiskAdjustmentHeadV46`, `RiskAdjustedRSCPFModelV46`, `FormalV46ForwardOutput`, `FormalV46ForwardOutput.as_v44_forecast_output() -> FormalV44ForwardOutput`, and `v46_parameter_groups()` with `base`, `gate`, `magnitude`, `risk`, and `scheduler`.

- [ ] **Step 1: Write failing model-contract tests**

```python
def test_v46_exposes_distinct_nominal_and_scheduler_demand():
    model = RiskAdjustedRSCPFModelV46.for_test(risk_cap=torch.full((4, 3), 10.0))
    output = model(**make_v46_batch(batch_size=2))
    assert output.forecast_nominal_physical.shape == (2, 4, 4)
    assert output.risk_adjustment.shape == (2, 4, 3)
    assert output.risk_cap.shape == (2, 4, 3)
    assert output.scheduler_demand.shape == (2, 4, 4)
    assert output.controls.shape == (2, 4, 15)
    assert output.dispatch.shape == (2, 4, 21)
    assert torch.all(output.risk_adjustment >= 0)
    assert torch.all(output.risk_adjustment <= 10.0 + 1e-6)
    assert torch.equal(
        output.scheduler_demand[..., 3],
        output.forecast_nominal_physical[..., 3],
    )

def test_thermal_risk_is_probability_masked_and_decoupled_detaches_forecast():
    model = make_v46_model_with_known_probabilities()
    joint = model(**make_v46_batch(), detach_forecast_for_dispatch=False)
    decoupled = model(**make_v46_batch(), detach_forecast_for_dispatch=True)
    assert torch.all(joint.risk_adjustment[..., 1] <= joint.risk_cap[..., 1] * joint.regime_probabilities[..., 1] + 1e-6)
    assert torch.all(joint.risk_adjustment[..., 2] <= joint.risk_cap[..., 2] * joint.regime_probabilities[..., 2] + 1e-6)
    grad = torch.autograd.grad(decoupled.dispatch.sum(), model.base_forecaster_parameters(), allow_unused=True)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in grad)

@pytest.mark.parametrize("output_bias", [-40.0, 40.0])
def test_physics_remains_feasible_at_zero_and_maximum_risk(output_bias):
    model = RiskAdjustedRSCPFModelV46.for_test(
        risk_cap=torch.full((4, 3), 10.0)
    )
    torch.nn.init.zeros_(model.risk_head.output.weight)
    torch.nn.init.constant_(model.risk_head.output.bias, output_bias)
    batch = make_v46_batch(batch_size=2)
    output = model(**batch)
    features = model.core._raw_physical_features(
        output.scheduler_demand, batch["scheduler_context"]
    )
    soc = soc_residuals(output.dispatch, features, model.core.decoder_parameters)
    residuals = (
        balance_residuals(output.dispatch, features),
        conversion_residuals(output.dispatch, model.core.decoder_parameters),
        soc["state"], soc["terminal"],
    )
    assert max(float(value.abs().max()) for value in residuals) <= 1e-6
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_model.py -q
```

Expected: FAIL because the v4.6 model does not exist.

- [ ] **Step 3: Implement the risk head and forward graph**

```python
class RiskAdjustmentHeadV46(nn.Module):
    def __init__(self, risk_cap: Tensor, hidden_width: int = 64,
                 initial_output_bias: float = -6.0):
        super().__init__()
        self.register_buffer("risk_cap", torch.as_tensor(risk_cap).reshape(4, 3))
        self.hidden = nn.Sequential(
            nn.Linear(43, hidden_width), nn.GELU(), nn.LayerNorm(hidden_width)
        )
        self.output = nn.Linear(hidden_width, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, initial_output_bias)

    def forward(self, normalized_features, state, normalized_previous_chp,
                regime_probabilities):
        state_h = state[:, None, :].expand(-1, 4, -1)
        previous_h = normalized_previous_chp[:, None, :].expand(-1, 4, -1)
        raw = self.output(self.hidden(torch.cat(
            (normalized_features, state_h, previous_h), dim=-1
        )))
        mask = torch.stack((
            torch.ones_like(raw[..., 0]),
            regime_probabilities[..., 1],
            regime_probabilities[..., 2],
        ), dim=-1)
        cap = self.risk_cap.unsqueeze(0).expand(raw.shape[0], -1, -1)
        return torch.sigmoid(raw) * cap * mask, cap
```

Before calling the risk head, form the nominal ten-feature scheduling tensor
with `core._raw_physical_features`, normalize it with
`core.physical_feature_mean/physical_feature_scale`, bound raw previous CHP
exactly as `_FormalV4Base._schedule` does, and normalize it with
`previous_chp_mean/previous_chp_scale`. Construct `scheduler_demand` by adding
the three adjustments to the nominal rigid channels and copying nominal gas
unchanged; then rebuild the ten-feature tensor from `scheduler_demand` before
calling `_schedule`. `FormalV46ForwardOutput.risk_cap` stores the expanded
`[B,4,3]` cap returned above. When
`detach_forecast_for_dispatch=True`, detach nominal forecast, regime
probabilities, and state before both risk head and scheduler. Keep risk-head and
scheduler parameters trainable in the decoupled branch.
`as_v44_forecast_output()` must expose the nominal—not scheduler-adjusted—
`base_forecast_normalized`, `base_forecast_physical`, `forecast_normalized`,
`forecast_physical`, `regime_logits`,
`regime_probabilities`, `thermal_magnitudes`, and `thermal_residuals` while
reusing the current controls and dispatch only to satisfy the existing v4.4
dataclass contract.

- [ ] **Step 4: Run focused model tests**

Run the Task 4 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_model.py tests/test_joint_dispatch_formal_v4_6_model.py
git commit -m "add bounded v4.6 scheduling risk head"
```

### Task 5: Restore the full regime-aware forecast objective in J

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_loss.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_loss.py`

**Interfaces:**
- Consumes: `forecast_loss_v44`, `JointNormalizationV45`, v4.6 output, frozen P1 v4.4 output, teacher dispatch, and risk multiplier.
- Produces: `JointLossV46`, `joint_loss_v46(output: FormalV46ForwardOutput, parent_output: FormalV44ForwardOutput, batch: Mapping[str, Any], prior: ThermalPriorReceiptV44, normalization: JointNormalizationV45, curriculum: Mapping[str, float], risk_multiplier: float, decision_loss: Tensor) -> JointLossV46`, `risk_penalties_v46(output: FormalV46ForwardOutput, target_physical: Tensor) -> tuple[Tensor, Tensor]`, `_thermal_inactive_mask_v46(target_physical: Tensor) -> Tensor`, and `build_j_optimizer_v46(model: RiskAdjustedRSCPFModelV46, contract: FormalV46Contract, mode: str) -> torch.optim.Optimizer`.

- [ ] **Step 1: Write failing component and gradient tests**

```python
def test_joint_loss_contains_full_forecast_and_risk_terms():
    loss = joint_loss_v46(
        output=make_output(), parent_output=make_parent_output(),
        batch=make_labeled_batch(), prior=make_prior(),
        normalization=JointNormalizationV45(1.0, 1.0, 1.0),
        curriculum=curriculum_weights_v45(3, 5), risk_multiplier=1.0,
        decision_loss=torch.tensor(0.25, requires_grad=True),
    )
    assert loss.forecast.regime.item() > 0
    assert loss.forecast.active_magnitude.item() >= 0
    assert loss.forecast.inactive_leakage.item() >= 0
    assert loss.risk_size.item() >= 0
    assert loss.off_risk.item() >= 0
    assert torch.isfinite(loss.total)

def test_joint_decision_gradient_reaches_risk_and_forecast_groups():
    model = make_v46_model()
    terms = joint_loss_v46_from_model(model, detach=False)
    norms = named_autograd_norms(terms.decision, model.v46_parameter_groups())
    assert norms["base"] > 0
    assert norms["gate"] > 0
    assert norms["magnitude"] > 0
    assert norms["risk"] > 0
    assert norms["scheduler"] > 0
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_loss.py -q
```

Expected: FAIL because the v4.6 loss module does not exist.

- [ ] **Step 3: Implement the complete loss**

```python
def _thermal_inactive_mask_v46(target_physical: Tensor) -> Tensor:
    if target_physical.ndim != 3 or target_physical.shape[1:] != (4, 4):
        raise ValueError("target_physical must have shape [B,4,4]")
    return (target_physical[..., 1:3] <= 1e-9).to(target_physical.dtype)

forecast = forecast_loss_v44(
    output.as_v44_forecast_output(), batch, prior, forecast_weights
)
risk_fraction = output.risk_adjustment / output.risk_cap.clamp_min(1e-8)
risk_size = risk_fraction.mean()
off_mask = _thermal_inactive_mask_v46(batch["target_physical"])
off_risk = (risk_fraction[..., 1:3] * off_mask).sum() / off_mask.sum().clamp_min(1)
total = (
    curriculum["forecast"] * forecast.total / normalization.forecast
    + curriculum["imitation"] * imitation / normalization.imitation
    + curriculum["decision"] * decision / normalization.decision
    + curriculum["anchor"] * anchor
    + 0.10 * risk_multiplier * risk_size
    + 0.50 * risk_multiplier * off_risk
)
```

`build_j_optimizer_v46` must use `j_forecaster_lr`, `j_head_lr`,
`risk_adjustment.j_risk_lr`, and `j_scheduler_lr` for the separate base, head,
risk, and scheduler groups. Decoupled mode includes only risk and scheduler
groups.

- [ ] **Step 4: Run focused loss tests**

Run the Task 5 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_loss.py tests/test_joint_dispatch_formal_v4_6_loss.py
git commit -m "add full v4.6 joint forecast and risk loss"
```

### Task 6: Implement matched Joint and Fair Decoupled training

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_training.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_training.py`

**Interfaces:**
- Consumes: v4.5 P0/P1/S stage receipts, `RiskCapReceiptV46`, v4.6 model, v4.6 loss, explicit train/early-stop loaders, and one risk multiplier.
- Produces: `StageReceiptV46`, `JointPairReceiptV46`, `ZeroRiskReceiptV46`, `CandidateValidationV46`, `CalibratedCandidateV46`, `CalibrationReceiptV46`, `initialize_v46_parent(stage_s: StageReceiptV45, risk_cap: RiskCapReceiptV46, contract: FormalV46Contract) -> RiskAdjustedRSCPFModelV46`, `run_stage_j_pair_v46(stage_s: StageReceiptV45, loaders: Mapping[str, Sequence[Mapping[str, Any]]], budget: StageBudgetV45, prior: ThermalPriorReceiptV44, parameters: Mapping[str, Any], contract: FormalV46Contract, risk_cap: RiskCapReceiptV46, risk_multiplier: float, seed: int) -> JointPairReceiptV46`, `run_zero_risk_control_v46(stage_s: StageReceiptV45, loaders: Mapping[str, Sequence[Mapping[str, Any]]], budget: StageBudgetV45, prior: ThermalPriorReceiptV44, parameters: Mapping[str, Any], contract: FormalV46Contract, risk_cap: RiskCapReceiptV46, seed: int) -> ZeroRiskReceiptV46`, and `run_v46_calibration(stage_s: StageReceiptV45, loaders: Mapping[str, Sequence[Mapping[str, Any]]], budget: StageBudgetV45, prior: ThermalPriorReceiptV44, parameters: Mapping[str, Any], contract: FormalV46Contract, risk_cap: RiskCapReceiptV46, seed: int) -> CalibrationReceiptV46`.

- [ ] **Step 1: Write failing matched-training tests**

```python
def test_joint_and_decoupled_start_from_byte_identical_parent():
    pair = run_toy_v46_pair(risk_multiplier=1.0)
    assert pair.joint.parent_sha256 == pair.decoupled.parent_sha256
    assert pair.joint.optimizer_groups["risk"]["steps"] == pair.decoupled.optimizer_groups["risk"]["steps"]
    assert pair.joint.optimizer_groups["scheduler"]["steps"] == pair.decoupled.optimizer_groups["scheduler"]["steps"]

def test_decoupled_cuts_only_decision_edges_into_forecast():
    pair = run_toy_v46_pair(risk_multiplier=1.0)
    assert pair.joint.gradient_norms["decision_to_base"] > 0
    assert pair.joint.gradient_norms["decision_to_risk"] > 0
    assert pair.decoupled.gradient_norms["decision_to_base"] <= 1e-12
    assert pair.decoupled.gradient_norms["decision_to_gate"] <= 1e-12
    assert pair.decoupled.gradient_norms["decision_to_magnitude"] <= 1e-12
    assert pair.decoupled.gradient_norms["decision_to_risk"] > 0

def test_calibration_selects_only_eligible_early_stop_candidate():
    result = run_toy_calibration(multipliers=(0.5, 1.0, 2.0))
    assert result.selected_multiplier in (0.5, 1.0, 2.0)
    assert result.candidates[result.selected_multiplier].validation.eligible
    assert result.selection_role == "early_stop"
    assert result.selection_year_accessed is False

def test_zero_risk_control_keeps_adjustment_exactly_zero():
    result = run_toy_zero_risk_control()
    assert result.risk_trainable is False
    assert np.count_nonzero(result.risk_adjustment) == 0
    assert result.scheduler_demand_sha256 == result.forecast_nominal_sha256
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_training.py -q
```

Expected: FAIL because v4.6 matched training does not exist.

- [ ] **Step 3: Implement training and three-setting calibration**

`initialize_v46_parent` must construct the v4.6 class, load every compatible
P1+S base/gate/magnitude/scheduler tensor with `strict=False`, verify that the
only missing tensors belong to `risk_head`, initialize those tensors once with
the frozen zero-weight/`-6.0`-bias rule, and hash the resulting parent. Clone
that byte-identical parent into each candidate and branch. Seed before every
model construction. Run multipliers in the fixed order `(0.5, 1.0, 2.0)`.
Record train and early-stop histories, all component losses, best epoch,
selected and terminal hashes, optimizer groups, and gradient norms. Select:

```python
@dataclass(frozen=True)
class CandidateValidationV46:
    eligible: bool
    decision_objective: float
    nominal_metrics: Mapping[str, float]
    failure_names: tuple[str, ...]

@dataclass(frozen=True)
class CalibratedCandidateV46:
    risk_multiplier: float
    pair: JointPairReceiptV46
    validation: CandidateValidationV46

@dataclass(frozen=True)
class CalibrationReceiptV46:
    candidates: Mapping[float, CalibratedCandidateV46]
    selected_multiplier: float
    selected: CalibratedCandidateV46
    selection_role: str
    selection_year_accessed: bool

eligible = [candidate for candidate in candidates if candidate.validation.eligible]
if not eligible:
    raise ValueError("formal-v4.6 calibration has no eligible checkpoint")
selected = min(eligible, key=lambda item: item.validation.decision_objective)
```

No candidate may read `selection_full`, `selection_stress`, or any 2019/2020
file. `run_zero_risk_control_v46` uses the same parent, data, curriculum,
forecast/scheduler optimizer budgets, and J-stage loss, but fixes the risk
adjustment to an exact zero tensor and excludes risk-head parameters from the
optimizer. It is diagnostic-only evidence and must not be inserted into the
external baseline matrix by this plan.

- [ ] **Step 4: Run focused training tests**

Run the Task 6 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_training.py tests/test_joint_dispatch_formal_v4_6_training.py
git commit -m "add matched v4.6 joint calibration"
```

### Task 7: Define stable nominal-forecast metrics and Pilot authorization

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_metrics.py`
- Create: `src/joint_dispatch/formal_v4_6_pilot_gate.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_metrics.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_pilot_gate.py`

**Interfaces:**
- Consumes: nominal predictions/targets, regime probabilities/prior, transition labels, risk adjustments/caps, Joint/Fair Decoupled decision summaries, gradient receipts, and physical residuals.
- Produces: `ForecastMetricsV46`, `RiskMetricsV46`, `normalized_inactive_leakage_v46(prediction: np.ndarray, target: np.ndarray) -> Mapping[str, float]`, and `authorize_pilot_v46(receipt: Mapping[str, Any], contract: FormalV46Contract) -> PilotDecisionV46`.

- [ ] **Step 1: Write failing metric and gate tests**

```python
def test_normalized_inactive_leakage_uses_active_target_mass():
    prediction = np.array([[[0., 4., 0., 0.], [0., 0., 0., 0.]]])
    target = np.array([[[0., 0., 0., 0.], [0., 8., 0., 0.]]])
    result = normalized_inactive_leakage_v46(prediction, target)
    assert result["cooling"] == pytest.approx(4.0 / 8.0)

def test_gate_accepts_nominal_noninferiority_and_decision_improvement():
    receipt = make_passing_v46_receipt()
    decision = authorize_pilot_v46(receipt, load_contract())
    assert decision.authorized_gate1

@pytest.mark.parametrize("field", ["gas_adjustment", "cap_violation", "evaluation_year_accessed"])
def test_gate_fails_closed_on_semantic_or_lineage_violation(field):
    receipt = make_passing_v46_receipt()
    corrupt(receipt, field)
    assert not authorize_pilot_v46(receipt, load_contract()).authorized_gate1
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_metrics.py tests/test_joint_dispatch_formal_v4_6_pilot_gate.py -q
```

Expected: FAIL because v4.6 metrics and gate do not exist.

- [ ] **Step 3: Implement nominal metrics, risk audits, and fail-closed gate**

Compute normalized inactive leakage as:

```python
inactive_mass = np.abs(prediction[..., task][target[..., task] <= 1e-9]).sum()
active_mass = np.abs(target[..., task][target[..., task] > 1e-9]).sum()
value = inactive_mass / max(active_mass, 1e-12)
```

The gate must evaluate each frozen ratio separately, require Joint decision
objective and shortage not to exceed Fair Decoupled, validate all gradient
boundaries, enforce `risk_adjustment <= risk_cap * regime_mask + 1e-6`, enforce
zero gas adjustment, enforce physical residual, validate hashes, and reject any
2020 access. `RiskMetricsV46` must report adjustment mean and p95 for every
electricity/cooling/heating horizon cell, aggregate the same statistics by true
thermal regime, and report cap-utilization mean/p95/max plus a zero gas value.
It must return every measured value and every failure name.

- [ ] **Step 4: Run focused metric and gate tests**

Run the Task 7 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_metrics.py src/joint_dispatch/formal_v4_6_pilot_gate.py tests/test_joint_dispatch_formal_v4_6_metrics.py tests/test_joint_dispatch_formal_v4_6_pilot_gate.py
git commit -m "add stable v4.6 metrics and pilot gate"
```

### Task 8: Add risk-aware rolling settlement and Pilot executor

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_rollout.py`
- Create: `src/joint_dispatch/formal_v4_6_pilot_executor.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_rollout.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_executor.py`

**Interfaces:**
- Consumes: v4.5 cached data loaders, v4.6 cap/model/training modules, existing physical parameters, and full 2019 selection collection.
- Produces: `RolloutResultV46`, `FrozenCandidateV46`, `P1EarlyStopEvidenceV46`, `collect_p1_early_stop_evidence_v46(model: ResidualGatedRSCPFModel, early_stop_batches: Sequence[Mapping[str, Any]], early_stop_timestamps: np.ndarray) -> P1EarlyStopEvidenceV46`, `freeze_v46_candidate(candidate: CalibratedCandidateV46) -> FrozenCandidateV46`, `load_selection_view_v46(materialized_root: str | Path, train_data: str | Path, benchmark: str | Path, capacity_receipt: str | Path) -> MaterializedPilotV45`, `rollout_v46_2019(model: RiskAdjustedRSCPFModelV46, materialized: MaterializedPilotV45, indices: np.ndarray, normalization: NormalizationReceiptV42, parameters: Mapping[str, Any], artifact_root: str | Path, method_id: str) -> RolloutResultV46`, `run_matched_v46_rollouts(candidate: FrozenCandidateV46, materialized: MaterializedPilotV45, normalization: NormalizationReceiptV42, parameters: Mapping[str, Any], artifact_root: str | Path) -> Mapping[str, RolloutResultV46]`, `TrainingBundleV46`, and `execute_real_pilot_v46(materialized_training: MaterializedTrainingV45, selection_loader: Callable[[], MaterializedPilotV45], contract: FormalV46Contract, artifact_root: str | Path, benchmark: str | Path, capacity_receipt: str | Path, seed: int) -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing rollout and executor tests**

```python
def test_rollout_records_nominal_adjusted_and_first_step_state(tmp_path):
    result = rollout_v46_2019(
        model=make_v46_model(), materialized=make_tiny_materialized_2019(),
        indices=np.arange(8), normalization=make_normalization(),
        parameters=make_parameters(), artifact_root=tmp_path,
        method_id="rsc_pf_joint",
    )
    assert result.forecast_nominal.shape == (8, 4, 4)
    assert result.risk_adjustment.shape == (8, 4, 3)
    assert result.scheduler_demand.shape == (8, 4, 4)
    assert result.settled_dispatch.shape == (8, 21)
    assert np.max(result.physical_residual) <= 1e-6

def test_executor_calibrates_before_loading_2019(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(v46, "run_v46_calibration", lambda *a, **k: events.append("calibration") or make_calibration())
    monkeypatch.setattr(v46, "load_selection_view_v46", lambda *a, **k: events.append("selection") or make_selection())
    execute_real_pilot_v46(
        materialized_training=make_training_only(),
        selection_loader=lambda: v46.load_selection_view_v46(
            TRAIN_CACHE, TRAIN_DATA, BENCHMARK, CAPACITY
        ),
        contract=load_contract(), artifact_root=tmp_path,
        benchmark=write_benchmark(tmp_path),
        capacity_receipt=write_capacity_receipt(tmp_path), seed=2026,
    )
    assert events.index("calibration") < events.index("selection")
```

- [ ] **Step 2: Run the tests and verify they fail**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_rollout.py tests/test_joint_dispatch_formal_v4_6_executor.py -q
```

Expected: FAIL because the v4.6 rollout and executor do not exist.

- [ ] **Step 3: Implement rolling outputs and execution order**

Adapt the v4.4 rollout without changing it. At each origin, store the full
nominal forecast, risk adjustment, scheduler demand, controls, planned
dispatch, and only the settled first step. Carry SOC and previous CHP from the
settled dispatch. The executor order must be:

```python
normalization = fit_train_normalization(materialized_training.normalization_source.split)
prior = fit_thermal_prior(
    materialized_training.normalization_source.forecast_target,
    materialized_training.normalization_source.load_history,
    materialized_training.normalization_source.target_times,
)
base_model = build_v44_model(materialized_training, contract, prior, parameters)
loaders = build_v45_loaders(
    materialized_training, normalization,
    batch_size=contract.payload["pilot_budget"]["batch_size"],
)
p0 = run_stage_p0_v45(base_model, loaders, p0_budget, seed=2026)
p1 = run_stage_p1_v45(p0, loaders, p1_budget, prior=prior, seed=2026)
p1_early = collect_p1_early_stop_evidence_v46(
    p1.model, loaders["early_stop"], materialized_training.early_stop.timestamps
)
caps = fit_risk_caps_v46(
    p1_early.prediction, p1_early.target, p1_early.timestamps,
    "early_stop", 0.90,
    contract.contract_sha256, p1.final_sha256,
)
teachers = build_v45_teachers(
    model=p1.model, materialized=materialized_training,
    benchmark=benchmark, capacity_receipt=capacity_receipt,
    artifact_root=artifact_root / "teacher", contract=contract, seed=2026,
)
teacher_loaders = build_v45_loaders(
    materialized_training, normalization,
    batch_size=contract.payload["pilot_budget"]["batch_size"],
    teacher=teachers.dispatch,
)
stage_s = run_stage_s_v45(p1, teacher_loaders, s_budget, seed=2026)
v46_parent = initialize_v46_parent(stage_s, caps, contract)
calibration = run_v46_calibration(
    stage_s, teacher_loaders, j_budget, prior, parameters,
    contract, caps, seed=2026,
)
freeze = freeze_v46_candidate(calibration.selected)
selection = selection_loader()
rollouts = run_matched_v46_rollouts(
    freeze, selection, normalization, parameters, artifact_root,
)
```

The result mapping must expose Joint and Fair Decoupled forecast, risk,
decision, gradient, physics, and lineage evidence needed by Task 7.
`v46_parent` is used to verify that every calibrated Joint/Fair branch starts
from the same parent hash; it is not trained separately from the parent created
inside `run_v46_calibration`.

- [ ] **Step 4: Run focused rollout and executor tests**

Run the Task 8 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_rollout.py src/joint_dispatch/formal_v4_6_pilot_executor.py tests/test_joint_dispatch_formal_v4_6_rollout.py tests/test_joint_dispatch_formal_v4_6_executor.py
git commit -m "add v4.6 risk-aware rollout executor"
```

### Task 9: Persist calibration, checkpoint, and independent audit evidence

**Files:**
- Create: `src/joint_dispatch/formal_v4_6_artifacts.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_audit.py`

**Interfaces:**
- Consumes: contract, cap receipt, candidate histories, selected checkpoint, gradient evidence, rollouts, and Pilot receipt.
- Produces: `write_training_artifacts_v46(root: str | Path, bundle: TrainingBundleV46, contract: FormalV46Contract) -> Mapping[str, str]`, `audit_training_artifacts_v46(root: str | Path, contract: FormalV46Contract) -> Mapping[str, Any]`, and `audit_pilot_v46(root: str | Path, contract: FormalV46Contract) -> Mapping[str, Any]`.

- [ ] **Step 1: Write failing immutable-audit tests**

```python
def test_audit_reconstructs_cap_candidate_and_checkpoint_selection(tmp_path):
    artifacts = write_valid_v46_fixture(tmp_path)
    audit = audit_training_artifacts_v46(artifacts.root, load_contract())
    assert audit["cap_reconstructed"]
    assert audit["candidate_selection_reconstructed"]
    assert audit["checkpoint_selection_reconstructed"]
    assert audit["evaluation_year_accessed"] is False

def test_audit_rejects_adjusted_gas_and_changed_cap(tmp_path):
    artifacts = write_valid_v46_fixture(tmp_path)
    tamper_npz(artifacts.rollout, "scheduler_demand", channel=3, amount=1.0)
    with pytest.raises(ValueError, match="gas|cap|lineage"):
        audit_pilot_v46(artifacts.root, load_contract())
```

- [ ] **Step 2: Run the test and verify it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_audit.py -q
```

Expected: FAIL because the v4.6 artifact/audit module does not exist.

- [ ] **Step 3: Implement immutable writers and independent recomputation**

Write JSON/NPZ artifacts once only. Audit from persisted arrays rather than
trusting boolean fields in the receipt. Recompute the cap quantile, risk bounds,
normalized leakage, candidate eligibility, selected early-stop objective,
gradient boundary, physical residual, years, and all hashes. Reject partial or
mixed-version directories.

- [ ] **Step 4: Run focused audit tests**

Run the Task 9 command again.

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/joint_dispatch/formal_v4_6_artifacts.py tests/test_joint_dispatch_formal_v4_6_audit.py
git commit -m "add independent v4.6 risk audit"
```

### Task 10: Build and run the bounded train-only diagnostic

**Files:**
- Create: `scripts/run_rsc_pf_formal_v4_6_diagnostic.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_diagnostic.py`

**Interfaces:**
- Consumes: v4.6 config, cached 2015--2018 train/early-stop data, benchmark, capacity receipt, and deterministic one-batch/short-epoch limits.
- Produces: `run_formal_v46_diagnostic(config: str | Path, materialized_root: str | Path, train_data: str | Path, benchmark: str | Path, capacity_receipt: str | Path, output_root: str | Path, run_id: str, max_batches: int, epoch_cap: int) -> Mapping[str, Any]`, `DIAGNOSTIC_RECEIPT.json`, `DIAGNOSTIC_AUDIT.json`, and `diagnostic_authorized_pilot: bool`.

- [ ] **Step 1: Write failing firewall and gradient tests**

```python
def test_diagnostic_never_opens_2019_or_2020(tmp_path, monkeypatch):
    opened = []
    monkeypatch.setattr(Path, "open", record_opens(opened))
    receipt = run_formal_v46_diagnostic(
        config=CONFIG, materialized_root=TRAIN_CACHE,
        train_data=TRAIN_DATA, benchmark=BENCHMARK,
        capacity_receipt=CAPACITY, output_root=tmp_path,
        run_id="v46-test-diagnostic", max_batches=1, epoch_cap=2,
    )
    assert receipt["accessed_years"] == [2015, 2016, 2017, 2018]
    assert receipt["selection_year_accessed"] is False
    assert receipt["evaluation_year_accessed"] is False
    assert receipt["joint_gradient_norms"]["decision_to_risk"] > 0
    assert receipt["decoupled_gradient_norms"]["decision_to_base"] <= 1e-12
```

- [ ] **Step 2: Run the test and verify it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_diagnostic.py -q
```

Expected: FAIL because the diagnostic entry point does not exist.

- [ ] **Step 3: Implement the fail-closed diagnostic**

The diagnostic must use `load_v45_training_cache`, never
`load_v45_pilot_cache`. It must verify cap fitting, all output semantics, Joint
and Decoupled gradients, full J loss components, one eligible and one rejected
synthetic checkpoint, deterministic hashes, and physical feasibility. It must
also run `run_zero_risk_control_v46` under the same tiny budget and record its
objective, shortages, zero-adjustment proof, and parent hash as diagnostic-only
evidence. It sets
`diagnostic_authorized_pilot=true` only when every check passes.

- [ ] **Step 4: Run all v4.6 unit tests**

```powershell
$v46Tests = (Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_6_*.py').FullName
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest $v46Tests -q --tb=short
```

Expected: all v4.6 tests PASS.

- [ ] **Step 5: Run protected v4.5 regressions**

```powershell
$v45Tests = (Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_5_*.py').FullName
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest $v45Tests -q --tb=short
```

Expected: 13 tests PASS and no v4.5 artifact changes.

- [ ] **Step 6: Commit the diagnostic implementation before execution**

```powershell
git add scripts/run_rsc_pf_formal_v4_6_diagnostic.py tests/test_joint_dispatch_formal_v4_6_diagnostic.py
git commit -m "add bounded v4.6 train-only diagnostic"
```

- [ ] **Step 7: Run the bounded real-data diagnostic**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts/run_rsc_pf_formal_v4_6_diagnostic.py `
  --config configs/joint_forecast_dispatch_formal_v4_6.json `
  --materialized-root reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_f/pilot `
  --train-data reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_j/data/base_train.npz `
  --benchmark reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_j/gate0/benchmark/STANDARD_IES_BENCHMARK.yaml `
  --capacity-receipt reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_g/gate0/CAPACITY_FREEZE.json `
  --output-root reports/joint_forecast_dispatch_formal_v4_6 `
  --run-id formal_v4_6_diagnostic_20260905_a `
  --max-batches 1 `
  --epoch-cap 2
```

Expected: exit code 0 and a receipt with
`diagnostic_authorized_pilot=true`. If false, stop this plan.

### Task 11: Add the immutable Pilot runner and run one 2019 Pilot

**Files:**
- Create: `scripts/run_rsc_pf_formal_v4_6_pilot.py`
- Test: `tests/test_joint_dispatch_formal_v4_6_pilot.py`
- Output: `reports/joint_forecast_dispatch_formal_v4_6/<run-id>/pilot/`

**Interfaces:**
- Consumes: a successful diagnostic receipt, frozen config and source manifest, cached train/early-stop/2019 selection data, benchmark, and capacity receipt.
- Produces: `run_formal_v46_pilot(config: str | Path, diagnostic_receipt: str | Path, source_manifest: str | Path, materialized_root: str | Path, train_data: str | Path, benchmark: str | Path, capacity_receipt: str | Path, output_root: str | Path, run_id: str) -> Mapping[str, Any]`, `PILOT_RECEIPT.json`, `PILOT_AUDIT.json`, `PILOT_TRANSITION.json`, and `authorized_gate1: bool`.

- [ ] **Step 1: Write failing Pilot authorization and safe-restart tests**

```python
def test_pilot_requires_successful_diagnostic_and_never_reads_2020(tmp_path):
    with pytest.raises(ValueError, match="diagnostic"):
        run_formal_v46_pilot(
            config=CONFIG, diagnostic_receipt=make_failed_diagnostic(tmp_path),
            source_manifest=make_passed_source_manifest(tmp_path, "v46-reject"),
            materialized_root=TRAIN_CACHE, train_data=TRAIN_DATA,
            benchmark=BENCHMARK, capacity_receipt=CAPACITY,
            output_root=tmp_path, run_id="v46-reject",
        )
    receipt = run_formal_v46_pilot(
        config=CONFIG, diagnostic_receipt=make_passed_diagnostic(tmp_path),
        source_manifest=make_passed_source_manifest(tmp_path, "v46-accept"),
        materialized_root=TRAIN_CACHE, train_data=TRAIN_DATA,
        benchmark=BENCHMARK, capacity_receipt=CAPACITY,
        output_root=tmp_path, run_id="v46-accept",
    )
    assert receipt["accessed_years"] == [2015, 2016, 2017, 2018, 2019]
    assert receipt["evaluation_year_accessed"] is False

def test_pilot_restart_reuses_only_hash_matched_teacher_and_caps(tmp_path):
    root = write_teacher_and_cap_only_restart(tmp_path)
    receipt = run_formal_v46_pilot(
        config=CONFIG, diagnostic_receipt=make_passed_diagnostic(tmp_path),
        source_manifest=make_passed_source_manifest(tmp_path, "v46-restart"),
        materialized_root=TRAIN_CACHE, train_data=TRAIN_DATA,
        benchmark=BENCHMARK, capacity_receipt=CAPACITY,
        output_root=root, run_id="v46-restart",
    )
    assert receipt["teacher_reused"]
    assert receipt["risk_caps_reused"]
```

- [ ] **Step 2: Run the test and verify it fails**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_pilot.py -q
```

Expected: FAIL because the Pilot runner does not exist.

- [ ] **Step 3: Implement the Pilot runner**

Validate the formal-v4.6 source manifest against the current repository HEAD,
run ID, contract hash, and every declared file before opening 2019. Require the
manifest and diagnostic receipt hashes in the Pilot lineage. Write the selected
risk multiplier and cap hash before opening 2019. Permit safe restart only when
the directory contains complete, hash-matched teacher and cap caches and no
checkpoint, rollout, receipt, or transition artifact. Print one final JSON
summary and return exit code 0 only when `authorized_gate1=true`; use a distinct
non-zero code for a scientifically valid negative Pilot.

- [ ] **Step 4: Run focused Pilot tests and all v4.6 tests**

```powershell
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest tests/test_joint_dispatch_formal_v4_6_pilot.py -q
$v46Tests = (Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_6_*.py').FullName
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest $v46Tests -q --tb=short
```

Expected: all tests PASS.

- [ ] **Step 5: Commit the runner and tests before source freeze**

```powershell
git add scripts/run_rsc_pf_formal_v4_6_pilot.py tests/test_joint_dispatch_formal_v4_6_pilot.py
git commit -m "add formal v4.6 pilot runner"
```

- [ ] **Step 6: Freeze the v4.6 source manifest and run exactly one full Pilot**

Build the manifest only after every v4.6 runtime file is committed, then pass
that exact immutable file to the Pilot:

```powershell
$RunId = 'formal_v4_6_20260905_a'
$Manifest = "reports/joint_forecast_dispatch_formal_v4_6/$RunId/protocol/SOURCE_MANIFEST.json"
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts/build_rsc_pf_formal_v4_6_source_manifest.py `
  --contract configs/joint_forecast_dispatch_formal_v4_6.json `
  --repo-root . `
  --run-id $RunId `
  --output $Manifest
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts/run_rsc_pf_formal_v4_6_pilot.py `
  --config configs/joint_forecast_dispatch_formal_v4_6.json `
  --diagnostic-receipt reports/joint_forecast_dispatch_formal_v4_6/formal_v4_6_diagnostic_20260905_a/DIAGNOSTIC_RECEIPT.json `
  --source-manifest $Manifest `
  --materialized-root reports/joint_forecast_dispatch_formal_v4_4/formal_v4_4_20260905_f/pilot `
  --train-data reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_j/data/base_train.npz `
  --benchmark reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_j/gate0/benchmark/STANDARD_IES_BENCHMARK.yaml `
  --capacity-receipt reports/joint_forecast_dispatch_formal_v4_2/formal_v4_2_20260905_g/gate0/CAPACITY_FREEZE.json `
  --output-root reports/joint_forecast_dispatch_formal_v4_6 `
  --run-id $RunId
```

Expected: immutable Pilot and audit receipts. Inspect `authorized_gate1` rather
than assuming success. If false, record all failed criteria and stop; if true,
stop this plan after recording authorization and prepare a separate Gate 1 plan.

### Task 12: Final implementation and evidence review

**Files:**
- Review: all v4.6 files from Tasks 1--11
- Review: `reports/joint_forecast_dispatch_formal_v4_6/<run-id>/protocol/SOURCE_MANIFEST.json`
- Review: `reports/joint_forecast_dispatch_formal_v4_6/<run-id>/pilot/PILOT_RECEIPT.json`
- Review: `reports/joint_forecast_dispatch_formal_v4_6/<run-id>/pilot/PILOT_AUDIT.json`
- Review: `reports/joint_forecast_dispatch_formal_v4_6/<run-id>/PILOT_TRANSITION.json`

**Interfaces:**
- Consumes: completed implementation, test evidence, and immutable run artifacts.
- Produces: a factual completion report that states engineering status separately from scientific Pilot status.

- [ ] **Step 1: Run the complete protected test set**

```powershell
$protected = @(
  (Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_4_*.py').FullName
  (Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_5_*.py').FullName
  (Get-ChildItem tests -Filter 'test_joint_dispatch_formal_v4_6_*.py').FullName
)
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' -m pytest $protected -q --tb=short
```

Expected: all protected tests PASS.

- [ ] **Step 2: Verify repository scope**

```powershell
git status --short
git diff --check
git log --oneline -15
```

Expected: no uncommitted v4.6 changes; only the pre-existing untracked
`third_party/iTransformer_source/` may remain.

- [ ] **Step 3: Verify the independent evidence**

Confirm from persisted receipts:

```text
contract/config/source hashes match
source manifest matches the committed HEAD and complete v4.6 closure
risk caps were fitted only from allowed early-stop years
the selected multiplier came only from early-stop evidence
nominal and scheduler-demand tensors are distinct and gas is unchanged
Joint and Fair Decoupled gradient boundaries match the contract
physical residual is within 1e-6
2019 was accessed only after freeze
2020 was never accessed
authorized_gate1 equals the independent audit decision
```

- [ ] **Step 4: Record the outcome without expanding scope**

If authorized, state that formal-v4.6 may proceed to a separately reviewed
Gate 1 baseline plan. If unauthorized, state the exact measured failures and
stop. Do not launch baselines, ablations, or 2020 evaluation from this task.
