# Physics-Informed Neural Scheduling: Feasible-by-Construction Design

**Status:** proposed design; no production code or experiment artifact is changed by this document.

**Decision:** replace the v1 independent normalized-label head with a **horizon-reachable feasible dispatch decoder**.  The neural network emits 15 unconstrained control logits for a four-hour window; a small, deterministic PyTorch decoder maps them to the existing physical `[B, 4, 21]` dispatch contract.  The decoder never calls `solve_dispatch_lp` at inference.

## Scope and fixed boundaries

The research pipeline remains:

```text
Frozen Scheme2R four-step load forecast (+ PV/WT forecast, prices, initial SOC)
    -> supervised neural scheduler
    -> feasible [B,4,21] device dispatch -> cost and carbon accounting
```

- The exact HiGHS LP remains an **offline teacher** for synthetic standard-IES labels and an offline benchmark only.  It is not a runtime fallback, repair, projection, or inference dependency.
- Scheme2R, its forecasting loss, its checkpoints, and its historical forecasting metrics are unchanged.  Scheduler gradients must not flow into Scheme2R.
- The frozen v1 artifact remains reproducible.  This is a versioned v2 scheduler/contract, not a reinterpretation of its results.
- `proxy_repairs.physical_decoder` is not a candidate final scheduler.  It is retained only as historical diagnostic code.

## Evidence and root-cause diagnosis

The failure is structural, not a matter of increasing the residual-loss weight.

1. `proxy_model.SchedulingProxy` maps `[B,4,10]` to 84 independent sigmoid values (`[B,4,21]`).  Scaling and sigmoid bound many finite outputs, but cannot impose the three balance equalities, five conversion equalities, two renewable split equalities, the four-step SOC recursion/terminal equality, or CHP ramps across time.
2. The exact LP labels in `dispatch_lp.py` are highly coupled.  For example, `p_chp = eta_e*g_chp`, `q_chp = eta_h*g_chp`, `q_ec = COP_ec*p_ec`, `soc[t] = soc[t-1] + eta_b*p_charge[t] - p_discharge[t]/eta_b`, and PV/WT use plus curtailment must equal availability.  Treating each member as an unrelated regression target creates incompatible predictions even when each marginal error is small.
3. The completed raw-proxy diagnostic confirms the diagnosis: every 2,048 evaluated raw predictions violated balance, conversion, renewable-split, SOC-state, and terminal-SOC constraints; the reported maxima were 329.50, 106.17, 1,198.84, 146.27, and 135.36 respectively.  Bounds and CHP ramps happened to be within range, so simple clipping cannot fix the problem.
4. The validation repair comparison shows why the current deterministic decoder is not acceptable as the final model.  It reaches feasibility 1.0 without an LP call, but its relative objective gap is **47.180656x** and cost-regret relative is **47.901138x**.  Its construction sets CHP and BESS to zero and fills residuals with grid/slack, discarding the learned policy.  The closer `bound_and_split` candidate has a 3.50% objective gap but remains infeasible (feasible rate 0.0).
5. The exact validation labels are in the proposed decoder family.  A read-only validation-label audit found zero slack, zero absorption-chiller output, and zero heat dump in the current teacher labels.  More importantly, each label satisfied the control intervals below to numerical precision (largest interval/reconstruction discrepancy about `2.3e-13`).  The proposal therefore removes an artificial output-space obstacle rather than excluding the teacher policy.

The label set also has a scientific caveat: PV and WT have equal zero marginal use cost in the LP, so the individual renewable split can be a solver tie-break even when total renewable use and objective are identified.  It must not dominate model selection.

## Alternatives considered

| Design | Feasibility at inference | Main issue | Decision |
|---|---|---|---|
| Independent `[B,4,21]` head plus stronger physics/augmented-Lagrangian losses | No hard guarantee; equality residuals remain learned approximations | Repeats the observed 0% raw-feasibility failure | Reject |
| Differentiable LP/QP layer or projection at inference | Yes, subject to solver success | Is an online optimization fallback, adds solver/dependency/latency risk, and violates the no-LP-inference requirement | Reject |
| **Horizon-reachable feasible decoder with supervised controls** | **Yes for all LP equalities, bounds, renewable splits, SOC, and ramps; slack represents only unmet physical capacity** | Piecewise-smooth rather than globally smooth; needs a versioned contract | **Select** |

The selected approach is not claimed as a novel generic physics layer.  Its contribution is the task-specific integration of analytic IES feasibility, exact-LP supervision, and frozen Scheme2R forecasts in the fixed four-hour simulated dispatch track.

## Selected architecture and interfaces

### Inputs and network output

Keep the external feature order and physical input shape unchanged:

`X in R_+^[B,H,10]`, `H=4`, with

`(d_e, d_c, d_h, gas_prior, pv_available, wt_available, grid_price, gas_price, carbon_price, initial_soc)`.

`initial_soc` must be identical across the four horizon rows.  Negative forecast values remain clipped at the existing adapter boundary; non-finite inputs fail closed.  `gas_prior` stays an auxiliary feature only: it is never a balance equation or a decoder constraint.

The residual MLP keeps its CPU-friendly encoder/trunk but replaces the `84`-value label projection with 15 logits:

```text
z = f_theta(normalize_train_only(X))
z_cooling       : [B,4]
z_chp           : [B,4]
z_soc           : [B,3]
z_renewable_pv  : [B,4]
--------------------------------
z                : [B,15]
dispatch = decode(z, X, frozen_IES_parameters) : [B,4,21]
```

Use `u = sigmoid(z / 0.25)` for each control.  The low fixed temperature lets the model approach physically valid endpoint controls without a separate clipping repair.  The decoder should also expose `decode_controls(u, ...)` for exact unit tests with controls in `[0,1]`.

Internal API for v2:

```python
model.forward_logits(x_normalized: Tensor) -> Tensor  # [B,15]
decode_feasible_dispatch(logits: Tensor, x_physical: Tensor, parameters: Mapping[str, float]) -> Tensor  # [B,4,21]
model.predict_dispatch(x_normalized, x_physical, parameters) -> Tensor  # [B,4,21]
```

The public `Scheme2RProxyAdapter.infer(...)` input and `ProxyInferenceResult` dispatch shapes stay `[N,4,21]`.  For v2, `raw_dispatch == safe_dispatch`, `fallback_mask` is all false, and a request for `allow_exact_fallback=True` fails closed instead of silently invoking an LP.  V1 checkpoint loading remains available only through the explicit v1 path.

### Constants and notation

All constants below come from the frozen standard-IES parameter map:

- `Gbar`: grid-import capacity; `Pbar_chp`, `Qbar_chp`, `Qbar_gb`: CHP electric, CHP heat, and boiler heat capacities.
- `Qbar_ec`, `Qbar_ac`: electric and absorption chiller cooling capacities; `Pbar_b`, `Ebar_b`: BESS power and energy capacities.
- `eta_e`, `eta_h`, `eta_gb`, `COP_ec`, `COP_ac`, and `eta_b = sqrt(bess_roundtrip_efficiency)`.
- `rho = eta_h / eta_e`; CHP ramp `Rbar = chp_ramp_fraction * Pbar_chp`.
- `[a]_+ = max(a, 0)`.  Every operation is batched elementwise over `[B,H]` unless a time recurrence is shown.

Use float64 inside the decoder (logits may be cast from the float32 MLP) and return physical dispatch in float64.  This is small (`H=4`) and prevents cancellation at the `1e-3` feasibility tolerance; gradients through the cast remain defined.

## Feasible decoder equations

The order is intentional: each stage only needs earlier decisions, and every remaining LP variable is then deterministic.

### 1. Cooling allocation (`z_cooling`)

The absorption chiller can consume heat.  Bound it by boiler-safe heat so an arbitrary neural control cannot create artificial heat slack when the electric chiller can serve the load:

```text
Qac_safe = min(Qbar_ac, COP_ac * [Qbar_gb - d_h]_+)
S_c      = min(d_c, Qbar_ec + Qac_safe)
L_ec     = [S_c - Qac_safe]_+
U_ec     = min(S_c, Qbar_ec)
q_ec     = L_ec + u_cooling * (U_ec - L_ec)
q_ac     = S_c - q_ec
p_ec     = q_ec / COP_ec
q_ac_in  = q_ac / COP_ac
slack_c  = d_c - S_c
```

This gives `0 <= q_ec <= Qbar_ec`, `0 <= q_ac <= Qbar_ac`, cooling balance exactly, and no unnecessary cooling slack.  Under the present generator support, the electric chiller alone can meet the teacher cooling labels; the safe absorption bound does not exclude them.

### 2. Ramp- and demand-reachable CHP (`z_chp`)

Let `D0_t = d_e,t + p_ec,t`.  Enforce the local CHP cap before building a viable ramp envelope:

```text
U_chp,t = min(Pbar_chp, Qbar_chp / rho,
              (d_h,t + q_ac_in,t) / rho, D0_t)
F_t     = min(U_chp,t, (t + 1) * Rbar)       # start from p_chp,-1 = 0
A_H-1   = F_H-1
A_t     = min(F_t, A_t+1 + Rbar)             # t = H-2 ... 0

p_chp,-1 = 0
L_chp,t = max(0, p_chp,t-1 - Rbar)
V_chp,t = min(A_t, p_chp,t-1 + Rbar)
p_chp,t = L_chp,t + u_chp,t * (V_chp,t - L_chp,t)
g_chp,t = p_chp,t / eta_e
q_chp,t = rho * p_chp,t
```

`A_t` is a backward viability envelope.  It guarantees `V_chp,t >= L_chp,t`, capacity, zero-prior ramping, and the ability to respect every later time-varying cap.  The heat and electricity caps ensure CHP cannot create heat dump or electrical over-supply.  The model can still learn nonzero CHP where the teacher uses it; it is not hard-coded to zero.

### 3. Boiler, heat balance, and unavoidable heat slack

```text
R_h     = d_h + q_ac_in
q_gb    = min(Qbar_gb, [R_h - q_chp]_+)
g_gb    = q_gb / eta_gb
q_dump  = [q_chp + q_gb - R_h]_+
slack_h = [R_h - q_chp - q_gb]_+
```

Thus `q_chp + q_gb + slack_h - q_ac_in - q_dump = d_h` exactly.  The CHP cap above normally makes `q_dump=0`; it remains explicitly derived for numerical safety and out-of-support inputs.  Slack is nonnegative and represents only a real thermal-capacity deficit.

### 4. Terminal-SOC-reachable BESS (`z_soc`)

Set `s0 = initial_soc * Ebar_b` and `N_t = D0_t - p_chp,t`.  CHP construction gives `N_t >= 0`.  The available charge and discharge powers that do not create electrical over-supply are:

```text
cmax_t = min(Pbar_b, [Gbar + pv_available,t + wt_available,t - N_t]_+)
dmax_t = min(Pbar_b, N_t)
C_t    = eta_b * cmax_t             # largest positive SOC increment
D_t    = dmax_t / eta_b             # largest magnitude negative SOC increment
```

For `t=0,1,2`, define `s_-1=s0` and choose the post-step energy state through the interval that can still return to `s0`:

```text
L_soc,t = max(0, s_t-1 - D_t, s0 - sum_{j=t+1}^{H-1} C_j)
U_soc,t = min(Ebar_b, s_t-1 + C_t, s0 + sum_{j=t+1}^{H-1} D_j)
s_t     = L_soc,t + u_soc,t * (U_soc,t - L_soc,t)
s_H-1   = s0

Delta_t       = s_t - s_t-1
p_charge,t    = [Delta_t]_+ / eta_b
p_discharge,t = eta_b * [-Delta_t]_+
soc_t         = s_t
```

The final state is fixed to the initial state, not merely penalized.  The interval includes local SOC/power bounds and future reachability, so the last transition is feasible by construction.  Charge and discharge cannot be simultaneous because both come from the sign of one `Delta_t`.

### 5. Renewable split, grid, and electrical balance (`z_renewable_pv`)

```text
R_e    = N + p_charge - p_discharge
U_ren  = min(R_e, pv_available + wt_available)
L_pv   = [U_ren - wt_available]_+
U_pv   = min(pv_available, U_ren)
pv_use = L_pv + u_renewable_pv * (U_pv - L_pv)
wt_use = U_ren - pv_use
pv_curt = pv_available - pv_use
wt_curt = wt_available - wt_use
grid    = min(Gbar, R_e - U_ren)
slack_e = [R_e - U_ren - Gbar]_+
```

This gives exact renewable availability splits and

```text
grid + pv_use + wt_use + p_chp + p_discharge + slack_e
    - p_ec - p_charge = d_e.
```

When available grid plus renewable supply can meet the net electric demand, `cmax` ensures a charging decision cannot manufacture slack; `slack_e` is then exactly zero.  If an out-of-support forecast exceeds all real supply, the decoder remains feasible by reporting the LP's explicit unserved-energy slack rather than producing an invalid dispatch.

### Existing 21-label output order

| LP label | Derived by |
|---|---|
| `grid` | electric residual after renewable use, capped at `Gbar` |
| `pv_use`, `pv_curt`, `wt_use`, `wt_curt` | renewable total and feasible PV allocation interval |
| `g_chp`, `p_chp`, `q_chp` | reachable CHP trajectory and exact CHP efficiencies |
| `g_gb`, `q_gb` | heat residual and boiler efficiency |
| `p_ec`, `q_ec`, `q_ac_in`, `q_ac` | bounded cooling allocation and exact COPs |
| `p_charge`, `p_discharge`, `soc` | reachable SOC path and signed state increments |
| `slack_e`, `slack_c`, `slack_h`, `q_dump` | nonnegative balance residuals only |

No member of `LABEL_ORDER` is independently regressed.  All 21 remain in the same order, so downstream device-dispatch, cost, carbon, metrics, and artifact readers continue to receive the established physical interface.

## Why feasibility is guaranteed

For finite, nonnegative input features with `initial_soc in [0,1]` and valid positive frozen parameters:

1. Cooling, CHP, boiler, and BESS formulas enforce their nonnegativity and finite capacity bounds directly.
2. CHP is inside a start-reachable/backward-viable ramp interval, so all absolute ramp inequalities hold.
3. The SOC interval makes each state reachable from the previous one and from which `s0` is reachable at the horizon end.  Its signed increment formulas satisfy the exact LP recursion and terminal equality.
4. Renewable use and curtailment are complementary partitions of availability.
5. The three balance equations are identities after the residual variables are derived.

The only non-smooth operations are `min`, `max`, and ReLU at physical active-set boundaries.  PyTorch supplies valid subgradients away from the measure-zero ties; the decoder is differentiable enough for supervised optimization, while feasibility does not depend on optimization convergence.

## Supervision, losses, and data hygiene

### Labels

Keep the current exact LP labels: `[N,4,21]`, `teacher_objective`, `teacher_cost`, and `teacher_carbon` from the pure-simulation train/validation/test splits.  No label generation change is required for v2.  The decoder's direct-control test utility should recover normalized controls from a feasible teacher dispatch and reconstruct it before training; this verifies that the chosen family contains the teacher policy.

The output-label normalization currently used to make 21 sigmoid targets is no longer an output activation.  Keep train-only input normalization.  Retain physical capacity scales from `ProxyNormalizationStats` only as fixed, train-derived loss scales and for artifact compatibility.

### Fixed v2 training loss

Compute every term from the **decoded physical dispatch**, not raw logits:

```text
L_coord = Huber((q_ec_hat - q_ec*) / Qbar_ec)
        + Huber((p_chp_hat - p_chp*) / Pbar_chp)
        + Huber((soc_hat - soc*) / Ebar_b)

L_ren_split = Huber((pv_use_hat - pv_use*) / max(pv_available, 1))

J_hat = operating_cost(dispatch_hat, X)
      + sum_t carbon_price_t * carbon_emissions_per_step(dispatch_hat)
L_obj    = Huber((J_hat - J*) / max(abs(J*), 1))
L_carbon = Huber((carbon_hat - carbon*) / max(abs(carbon*), 1))
L_slack  = mean_b(sum_{t,k in {e,c,h}} slack_k[b,t]
                  / max(sum_{t,k in {e,c,h}} demand_k[b,t], 1))

L_total = 1.00*L_coord + 0.10*L_ren_split
        + 0.20*L_obj + 0.05*L_carbon + 1.00*L_slack
```

Use mean-reduced Smooth L1/Huber terms.  `L_ren_split` is deliberately low weight because PV/WT individual allocation can be a zero-cost LP tie-break.  `L_obj`, carbon, and slack keep the learned feasible policy economically aligned rather than rewarding solver-specific variable identities alone.

Remove the v1 balance/conversion/SOC residual penalties and the gas-prior imitation loss from the v2 optimization objective: the former are exact decoder invariants, and the latter is not a physical law.  Retain them as assertions/metrics.  `physics_terms` continues to provide cost and carbon calculations.

### Forecast loss is separate and unchanged

Scheme2R keeps its existing multi-task forecast objective and frozen historical results.  Train the scheduler after the forecast model is frozen; never back-propagate `L_total` through Scheme2R.  The scheduler consumes its four-step electricity/cooling/heating forecasts (and its current auxiliary gas forecast only if an explicit ablation enables it), then produces dispatch.

For the primary v2 result, mask the current synthetic label-derived gas-prior channel during scheduler training/validation (`gas_prior=0`, mask=0), because it is generated after the LP teacher solution and is not a physical input.  A separate, locked ablation may use the frozen Scheme2R gas forecast at both training and inference, but it must never be selected using test data and must remain outside all balance equations.

## Contract, provenance, and inference behavior

Create `scheduling_proxy_contract_v2.json`; do not mutate the v1 contract used by existing artifacts.  The v2 contract must declare:

```json
{
  "schema_version": "scheduling-proxy-contract-v2",
  "model": {
    "output_parameterization": "horizon_reachable_feasible_v2",
    "decision_order": ["cooling", "chp", "soc_0", "soc_1", "soc_2", "renewable_pv"],
    "decision_dim": 15,
    "decoder_dtype": "float64",
    "control_temperature": 0.25,
    "device": "cpu"
  },
  "safety": {
    "feasibility_tolerance": 0.001,
    "allow_exact_fallback": false,
    "inference_exact_lp_calls": 0
  }
}
```

The full grouped layout—not only the shorthand `decision_order` above—must be stored in checkpoint metadata: `cooling=[0:4]`, `chp=[4:8]`, `soc=[8:11]`, and `renewable_pv=[11:15]`.  Include decoder schema/version, contract SHA-256, benchmark SHA-256, train/validation split identities, `selection_split="validation"`, and `test_split_used_for_selection=false` in checkpoints, history, metrics, and prediction artifacts.

At v2 inference the adapter must not import or call `solve_dispatch_lp`.  Its `fallback_rate` must always be exactly zero.  Offline LP labels may still be read to score a completed test experiment; that is evaluation, not inference.

## Exact files affected

| File | Required v2 responsibility |
|---|---|
| `frame/configs/scheduling_proxy_contract_v2.json` | New immutable v2 model/decoder/loss/safety contract; v1 stays untouched. |
| `frame/src/scheduling/proxy_contract.py` | Parse/validate v1 and v2 explicitly; validate 15-control layout and no-fallback safety flag. |
| `frame/src/scheduling/proxy_model.py` | Replace v2 output projection with 15-logit head; add `forward_logits` and `predict_dispatch`; keep explicit v1 loading path. |
| `frame/src/scheduling/proxy_decoder.py` (new) | Pure PyTorch float64 feasible decoder, control recovery utility, input/parameter checks, and no LP imports. |
| `frame/src/scheduling/proxy_physics.py` | Reuse cost/carbon helpers; add v2 decoded-dispatch loss while retaining v1 loss for historical artifacts. |
| `frame/src/scheduling/proxy_training.py` | Train logits -> decoder -> v2 loss; select only validation checkpoint; persist decoder metadata. |
| `frame/src/scheduling/proxy_adapter.py` | Call v2 `predict_dispatch`; preserve output shapes; prohibit exact fallback for v2. |
| `frame/src/scheduling/proxy_evaluation.py` | Measure combined model+decoder latency; report invariant residuals, objective/cost/carbon, slack, and zero fallback. |
| `frame/scripts/run_scheduling_proxy_pipeline.py` | Produce v2 artifacts and record decoder/no-LP inference provenance. |
| `frame/scripts/audit_scheduling_proxy.py` | Assert v2 feasible output, decoder metadata, all-false fallback mask, and zero inference LP calls. |
| `frame/src/scheduling/proxy_repairs.py` and `frame/scripts/compare_scheduling_proxy_repairs.py` | Leave unchanged as v1 historical comparison only; do not route v2 inference through either. |
| `frame/src/scheduling/dispatch_lp.py`, `synthetic_scenarios.py`, Scheme2R forecasting modules | No implementation change.  LP is retained only for offline teacher/evaluation labels. |

Tests to add/update: `test_scheduling_proxy_decoder.py` (new), plus `test_scheduling_proxy_model.py`, `test_scheduling_proxy_physics.py`, `test_scheduling_proxy_adapter.py`, `test_scheduling_proxy_training.py`, `test_scheduling_proxy_contract.py`, `test_scheduling_proxy_pipeline.py`, and `test_scheduling_proxy_diagnostics.py`.

## Staged implementation and validation gates

### Gate 0 — frozen baseline and contract isolation

- Preserve v1 source behavior and load its existing checkpoint successfully.
- Add v2 beside it; reject a v1 checkpoint under a v2 contract and vice versa.
- Verify no Scheme2R source file, checkpoint, forecast loss, or historical forecast metric is modified.

### Gate 1 — decoder unit proof (no data split needed)

- For randomized valid inputs, capacity-edge inputs, zero renewable inputs, and intentionally overloaded inputs, decode `[B,15]` controls and run `evaluate_raw_feasibility`.
- Require finite output, nonnegative variables, all renewable/conversion/balance/SOC/ramp residuals `<= 1e-6` in float64, and bounds `<= 1e-9` above tolerance.
- Verify the terminal SOC equals the input energy state, no charge/discharge overlap exceeds `1e-12`, and overload creates explicit nonnegative slack rather than an infeasible vector.
- Autograd-check an interior double-precision batch; test gradients are finite for all four control groups away from active-set kinks.
- Mock any LP symbol in v2 adapter/decoder to raise; `predict` and `infer` must still complete without accessing it.

### Gate 2 — teacher-family coverage (train and validation only)

- Recover controls from each existing exact teacher label, invoke `decode_controls`, and require max dispatch reconstruction error `<= 1e-6` on train and validation artifacts.
- Require no recovered control outside `[0,1]` beyond `1e-8`; report any active endpoint separately.
- This is a structural compatibility audit, not hyperparameter selection.  Do not read the test split during it.

### Gate 3 — smoke training

- Run the existing small synthetic smoke pipeline using the v2 contract on CPU.
- Require deterministic repeatability for the same seed, a valid checkpoint/provenance bundle, `fallback_rate=0`, and 100% raw feasibility.
- Add the new decoder version, decision dimension, and `inference_exact_lp_calls=0` to smoke artifacts.

### Gate 4 — formal train/validation selection

Lock architecture, loss weights, seeds, and thresholds before this run.  Choose epoch/checkpoint and any allowed gas-prior ablation only on validation.

- Hard feasibility: validation feasible rate `=1.0`, maximum invariant residual `<=1e-6`, fallback rate `=0`, and no exact LP inference calls.
- Standard-generator service: validation mean and maximum slack must be `<=1e-6`; current exact labels have zero slack and the proposed family can represent them.
- Economic gate: validation mean relative absolute objective gap `<=0.05` and relative cost regret `<=0.05`.  This is stricter than rejecting the 47.18x physical-decoder failure while allowing a realistic imitation target.
- Carbon gate: validation relative carbon error `<=0.10`.
- CPU gate: report combined model+decoder median/p95 latency on the same CPU/batch protocol as the prior comparison; median must be no slower than 10 ms per batch of 2,048 scenarios on that reference machine.

### Gate 5 — one locked test evaluation

After Gate 4 is locked, evaluate the untouched test split once.  Report the same metrics and all feasibility invariants, but do not revise architecture, weights, thresholds, or selection from that result.  Store raw/safe arrays separately for compatibility even though they are equal in v2.

## Research limitations and claims boundary

- The labels come from a continuous, deterministic, fixed-parameter LP on synthetic standard IES scenarios.  They do not establish performance on real operational dispatch, integer commitment, startup/shutdown costs, export, uncertainty, network flow limits, maintenance, or time-varying tariff optimization.
- Standard-IES capacities are calibrated from Kitakyushu **training-period aggregates**, but the synthetic scenarios are not real Kitakyushu operating windows.  A low LP-imitation gap is not evidence of real-site generalization.
- The teacher's scalar per-scenario prices and its possible renewable tie-breaks limit economic and variable-level claims.  Report objective/cost/carbon and slack alongside dispatch RMSE; do not present PV-versus-WT split error alone as policy quality.
- Feasibility is relative to the frozen continuous LP model.  Out-of-distribution forecasts remain algebraically feasible because of explicit slack; their economics and service quality require separate robustness studies.
- The scheduler is a task-specific supervised integration, not a claim that a generic physics-informed layer is novel.

## Residual risks and mitigations

| Risk | Mitigation |
|---|---|
| Sigmoid controls approach but do not equal endpoint labels | Use `decode_controls` for exact family tests, temperature `0.25`, Huber/objective losses, and report endpoint mass.  Feasibility is unaffected. |
| Solver-degenerate PV/WT labels create noisy split targets | Keep renewable-split loss low weight and privilege total cost/objective/feasibility metrics. |
| The synthetic gas prior is derived after labels | Mask it for the primary result; use only a clearly labeled frozen-Scheme2R gas ablation. |
| Piecewise decoder has active-set kinks | Use float64 decoder arithmetic, interior grad tests, and standard PyTorch subgradients; no implicit differentiation through an LP is required. |
| Forecasts lie outside synthetic capacity support | Keep explicit nonnegative slacks and report them rather than silently repairing with an LP. |

This design is intentionally minimal: it changes the scheduler parameterization and its training/inference bridge, while preserving the existing four-step forecasting stage, the LP teacher, the 21-variable downstream dispatch interface, and the pure-simulation provenance protocol.
