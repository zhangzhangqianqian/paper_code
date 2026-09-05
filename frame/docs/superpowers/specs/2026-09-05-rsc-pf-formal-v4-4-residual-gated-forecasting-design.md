# RSC-PF Formal-v4.4 Residual-Gated Forecasting Design

## 1. Decision and Version Boundary

Formal-v4.3 is frozen as a failed development branch. Its failed Pilot receipt remains immutable diagnostic evidence and is not overwritten or presented as a scientific result. Formal-v4.4 is a new protocol, model contract, artifact namespace, and run chain.

The v4.3 Pilot does not establish that thermal-regime modeling is invalid. It used a newly initialized seven-output head that relearned electricity, gas, cooling magnitude, heating magnitude, and regime probabilities; it fitted thermal statistics on the 128-window Pilot subset; it trained for three epochs; and it evaluated the same training windows. Several Pilot receipt fields were constants rather than measured results. Formal-v4.4 corrects both the model and the evaluation protocol.

The selected architecture is a residual-gated retrofit of Scheme2R. It preserves the established four-task forecasting pathway and adds only the thermal-state structure required by the data.

## 2. Scientific Scope

RSC-PF remains one jointly trained forecasting-and-scheduling network with an explicit four-task forecast bottleneck. It predicts electricity, cooling, heating, and station-side gas-consumption prior; passes the resulting four-hour forecasts and exogenous scheduling context to the neural scheduling proxy; and uses the differentiable physical decoder to produce 21-dimensional feasible device schedules.

Gas remains a supervised fourth prediction task and an auxiliary operating prior. It is not reinterpreted as a rigid terminal gas-demand balance. Standard-IES simulation supplies device trajectories, SOC states, and offline LP scheduling labels; it does not replace the observed multi-energy load targets.

The redesign changes only the representation and training of cooling/heating occurrence and magnitude. The scheduler topology, 15-dimensional continuous control representation, 21-dimensional decoded schedule, four-hour horizon, rolling state carry, and gas boundary remain unchanged.

## 3. Data Boundary

- Training and development statistics: 2015--2018 only.
- Model selection and Pilot evaluation: 2019 only.
- Sealed final evaluation: 2020 only, inaccessible until an authorized Gate 2 transition exists.

All task normalization statistics, cooling/heating active-magnitude statistics, regime class weights, and thermal transition priors are calculated from the full permitted 2015--2018 training split. Pilot subsampling must not change these statistics.

The three thermal labels are derived from the supervised cooling and heating targets:

```text
off     := cooling <= 1e-9 and heating <= 1e-9
cooling := cooling >  1e-9 and heating <= 1e-9
heating := cooling <= 1e-9 and heating >  1e-9
```

Simultaneously positive cooling and heating observations fail closed under the three-state contract. The last observed regime is derived only from the available load history and is therefore causal at inference time.

## 4. Model Architecture

### 4.1 Preserved Scheme2R pathway

The existing state-conditioned Scheme2R forecaster continues to produce normalized four-task base forecasts

```text
base_forecast: [B, 4, 4] = [electricity, cooling, heating, gas_prior].
```

Electricity and gas use their existing Scheme2R base outputs. Formal-v4.4 does not place a randomly initialized replacement head above them.

### 4.2 Causal three-state gate

For every horizon, a small gate predicts `off`, `cooling`, and `heating`. Its logits combine a train-only transition prior and a learned state-conditioned residual:

```text
z_tau = log P_train(r_(t+tau) | r_t) + delta_z_tau
p_tau = softmax(z_tau / temperature).
```

`r_t` is the last observed thermal regime. `P_train` is estimated only from 2015--2018. The residual gate consumes the shared Scheme2R representation and causal historical state. Its output layer is zero-initialized, so the first forward pass reproduces the empirical transition prior instead of an arbitrary one-third probability for every state.

Month masks, evaluation-year calibration, target-regime inputs, straight-through estimators, and hard scheduling gates are prohibited. Hard `argmax` regimes are diagnostics only.

### 4.3 Residual thermal magnitudes

The original normalized Scheme2R cooling/heating forecasts remain the magnitude backbone. Two zero-initialized residuals refine them using the original all-hour task normalization parameters:

```text
m_c = softplus(mu_task,c + sigma_task,c * (base_c + delta_c))
m_h = softplus(mu_task,h + sigma_task,h * (base_h + delta_h)).
```

The final differentiable physical forecasts are

```text
cooling_hat = p_cooling * m_c
heating_hat = p_heating * m_h.
```

At initialization, `delta_c = delta_h = 0`; therefore the magnitude computation is exactly equal to the existing Scheme2R physical conversion. Active-only cooling/heating statistics are used to normalize the magnitude loss, not to convert model outputs. Soft expected forecasts are both reported and supplied to the scheduler. No thresholded forecast is substituted during evaluation.

### 4.4 Required outputs

The model reports:

```text
base_forecast_normalized: [B, 4, 4]
forecast_normalized:      [B, 4, 4]
forecast_physical:        [B, 4, 4]
regime_logits:            [B, 4, 3]
regime_probabilities:     [B, 4, 3]
thermal_magnitudes:       [B, 4, 2]
thermal_residuals:        [B, 4, 2]
controls:                 [B, 4, 15]
dispatch:                 [B, 4, 21]
```

The scheduler continues to receive the four physical forecasts plus the previously frozen renewable, price, carbon, and state context. Regime probabilities are not an extra scheduler input.

## 5. Training Design

### 5.1 Stage P0: common Scheme2R warm start

Within the new v4.4 run, train the preserved four-task Scheme2R forecaster with the ordinary normalized forecast objective. The resulting P0 checkpoint is the common parent of the residual-gated branch and the continuous-head mechanism control. No v4.2 or v4.3 model checkpoint is admitted as v4.4 evidence.

### 5.2 Stage P1: residual-gate adaptation

Attach the zero-initialized gate and magnitude residual heads to P0. Optimize an auditable forecast loss:

```text
L_forecast =
    lambda_eg * L_electricity_and_gas
  + lambda_regime * L_three_state_cross_entropy
  + lambda_mag * L_active_only_magnitude
  + lambda_point * L_four_task_point_forecast
  + lambda_zero * L_inactive_thermal_leakage.
```

Class weights are inverse-frequency weights calculated from 2015--2018 and clipped to the frozen contract range. Active-magnitude loss is evaluated only where the corresponding thermal service is active. Inactive leakage is calculated on the final soft expected cooling/heating forecasts.

Training uses a fixed curriculum: first establish the gate and active magnitudes while retaining the P0 forecast anchor, then increase the point-forecast and inactive-leakage weights. The curriculum, temperature, weights, and learning rates are contract fields and cannot be changed after 2019 selection.

### 5.3 Stage S: scheduler imitation

Freeze the forecast pathway and train the neural scheduling proxy against the same-information offline LP teacher. The differentiable physical decoder remains in the forward path, and imitation is applied to normalized 21-dimensional decoded schedules.

### 5.4 Stage J: matched joint and decoupled branches

Clone the same P1 and Stage S checkpoints into two branches.

For RSC-PF, a single joint loss updates the forecaster, gate, residual magnitude pathway, and scheduler:

```text
L_joint = lambda_f * L_forecast
        + lambda_i * L_imitation
        + lambda_d * L_realized_decision
        + lambda_phy * L_physical_penalty.
```

For Fair Decoupled, use the same batches, initialization, epochs, and per-parameter-group optimizer-step counts. On each batch, form one matched total from the supervised forecast loss and the scheduler loss evaluated with a detached forecast bottleneck, then execute one optimizer step. Decision loss therefore has zero gradient into Scheme2R, the regime gate, and the magnitude residuals, while forecast-only gradients remain non-zero. This isolates the scientific question of whether scheduling feedback improves the forecast representation and realized decisions without granting either branch more updates.

## 6. Pilot Design

The v4.4 Pilot is a one-seed, train/selection-only decision about whether a full Gate 1 experiment is justified. It is not paper evidence.

### 6.1 Pilot construction

- Fit all statistics and priors on the full 2015--2018 training split.
- Select 4,096 fixed training windows stratified by season, thermal regime, and transition/non-transition status.
- Reserve time-disjoint, purged training-year blocks for early stopping. A purge covers at least the 24-hour history and four-hour forecast horizon.
- Evaluate the final Pilot models on the complete 2019 chronology with uniform weights.
- Generate a secondary fixed transition/active stress view, but never substitute it for full chronology.
- Train the continuous-head mechanism control from the same P0 parent and with a matched number of forecast optimizer steps.

### 6.2 Pilot authorization criteria

Every condition below is required:

1. **Inactive leakage:** residual-gated cooling and heating inactive-period MAE are each at least 30% lower than the matched continuous-head control on full 2019 chronology.
2. **Active accuracy:** active-only cooling and heating WAPE each degrade by no more than 5% relative to the continuous control.
3. **Unaffected tasks:** electricity and gas WAPE each degrade by no more than 2% relative to the continuous control.
4. **Overall forecast guardrail:** the normalized four-task forecast score degrades by no more than 2%.
5. **Regime utility:** full-chronology three-class macro-F1 is not below the train-only transition-prior baseline; transition-window balanced accuracy exceeds that baseline by at least one percentage point.
6. **Decision behavior:** the matched RSC-PF penalized rolling objective is no worse than Fair Decoupled on the full 2019 chronology, and neither shortage nor any physical violation is increased.
7. **Gradient boundary:** RSC-PF decision loss produces finite positive gradients separately in the Scheme2R base, regime gate, thermal residual heads, and scheduler. Fair Decoupled has decision-gradient norm at most `1e-12` in all forecast components while retaining a positive supervised-forecast gradient.
8. **Physical feasibility:** measured maximum normalized electricity, cooling, heating, capacity, SOC, and CHP-ramp residuals satisfy the existing formal tolerance. Finiteness alone is not a pass.
9. **Integrity:** all objectives and metrics are finite, all required hashes are present, no receipt field is a placeholder or hard-coded success flag, and the 2020 split is not accessed.

Failure of any required criterion produces `authorized_gate1=false`. A failed Pilot may lead to one documented redesign or bounded hyperparameter correction, but it must never automatically launch Gate 1.

## 7. Gate Chain

### Gate 0: engineering and protocol preflight

Gate 0 validates the v4.4 contract, source manifest, data-year allowlist, train-only statistics, transition matrix, checkpoint lineage, model shapes, zero-initialization invariants, independent gradient paths, entry points, and actual physical residual calculations on a deterministic mini-batch. Gate 0 does not train or select a scientific model.

Gate 0 passes only when every check is measured and an immutable transition receipt authorizes the Pilot.

### Pilot: bounded architecture decision

The Pilot executes the procedure in Section 6. It authorizes Gate 1 only when every Pilot criterion passes.

### Gate 1: formal training and 2019 selection

Gate 1 runs the complete retained model matrix on 2015--2018 training data and selects configurations only on 2019. It uses the frozen formal seed set. Candidate variation is limited to the four predeclared `(gate_temperature, inactive_leakage_weight)` combinations `(1.00, 0.25)`, `(0.75, 0.25)`, `(1.00, 0.50)`, and `(0.75, 0.50)` after component-wise loss normalization. Architecture, data access, method identities, and evaluation metrics are immutable.

Gate 1 requires multi-seed matched comparisons. RSC-PF must have a lower mean primary penalized rolling objective than Fair Decoupled, and the upper endpoint of the predeclared paired 95% bootstrap confidence interval for `RSC-PF minus Fair Decoupled` must not exceed zero. Forecast and physical guardrails must also pass. Comparisons across the four candidates use the frozen multiple-comparison correction. Only an immutable `authorized_gate2=true` receipt opens 2020.

### Gate 2: sealed final evaluation

Gate 2 loads the selected frozen checkpoints and configurations exactly once on 2020. No hyperparameter, model, threshold, normalization statistic, baseline, or narrative claim may be changed in response to 2020 results. Gate 2 produces paper-eligible artifacts only if lineage, feasibility, and completeness checks pass.

## 8. Baseline and Ablation Boundary

The redesign does not silently modify PTO or external baselines. The retained formal comparison remains:

- RSC-PF residual-gated joint model;
- Fair Decoupled residual-gated model;
- Direct-Policy;
- Scheme2R-PTO;
- official iTransformer adaptation with PTO;
- Differentiable-LP baseline;
- Seasonal-Naive-PTO;
- Perfect-Information-MPC as a non-deployable information upper reference.

The continuous thermal head is a named mechanism ablation. A transition-prior-only model is a diagnostic baseline for the learned gate, not a main scheduling competitor. No additional model family is introduced during v4.4 implementation.

## 9. Evidence and Artifact Contract

Every stage writes immutable, run-scoped artifacts containing:

- contract, source-manifest, data, normalization, transition-prior, implementation, and parent-checkpoint SHA-256 hashes;
- accessed years and explicit confirmation that 2020 remains sealed before Gate 2;
- sample IDs, timestamps, split role, purge boundaries, and sampling weights;
- per-epoch losses, optimizer steps, learning rates, stopping decision, and random seed;
- forecast, regime, leakage, dispatch, shortage, and feasibility metrics computed from stored predictions;
- separate gradient norms for Scheme2R, gate, magnitude residuals, scheduler, and decoupled boundaries;
- wall-clock runtime and termination state.

Receipt generation is fail closed. Missing, non-finite, duplicated, constant-placeholder, or lineage-inconsistent fields prohibit transition authorization. Prior receipts are never overwritten; reruns require a new run ID.

## 10. Testing and Review Requirements

Unit tests cover label derivation, transition-prior estimation, zero-initialized residual equivalence, output shapes, causal data boundaries, loss masks, and receipt validation. Gradient tests demonstrate the exact joint/decoupled boundary separately for the base forecaster, gate, and magnitude residuals. Integration tests run P0, P1, S, and J on deterministic small data and recompute every receipt metric from saved arrays.

Adversarial tests must fail authorization when a success field is hard-coded, an objective is infinite, a hash is missing, a 2020 timestamp appears before Gate 2, train and validation windows overlap after purging, the physical residual is only checked for finiteness, or the same samples are used for training and Pilot evaluation.

Before execution, the implementation is reviewed against this design and the formal-v4.2 frozen protocol. Existing v4.3 files may be read for diagnosis but cannot be imported as the v4.4 scientific implementation except for generic, version-independent utilities whose hashes and behavior are explicitly audited.

## 11. Success Definition

Formal-v4.4 is ready for an implementation plan when this document is approved. Implementation completion means Gate 0 and the bounded Pilot can produce truthful, independently recomputable receipts; it does not mean the scientific model has passed. Gate 1 begins only after the measured Pilot authorizes it, and Gate 2 begins only after the measured Gate 1 authorizes it.
