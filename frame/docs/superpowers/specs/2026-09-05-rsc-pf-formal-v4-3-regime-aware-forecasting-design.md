# RSC-PF Formal-v4.3 Regime-Aware Forecasting Design

## Context

Formal-v4.2 treats electricity, cooling, heating, and station-side gas consumption as four continuous forecast targets. Its final physical conversion applies `softplus` to every task. That is incompatible with the observed thermal-load process: in the untouched source Excel, cooling and heating contain long exact-zero intervals and positive observations occur on discrete-looking levels. Across the 2015--2018 training split and 2019 selection split, cooling and heating are never simultaneously positive.

The defect is consequential rather than cosmetic. Full chronological 2019 diagnostics show substantial inactive-period thermal leakage, and joint Stage-J candidates can reduce the dispatch objective by degrading cooling forecasts. The existing Gate 1 stress sample does not expose the full chronological error because its selected origins over-represent active thermal periods.

Formal-v4.2 therefore remains frozen as diagnostic evidence. Formal-v4.3 changes the forecast representation and Gate 1 calibration before any access to the sealed 2020 evaluation split.

## Selected Approach

Use one learned three-class thermal-service regime per forecast horizon:

1. `off`;
2. `cooling`;
3. `heating`.

The shared state-conditioned Scheme2R representation produces:

- continuous electricity output;
- continuous station-side gas-prior output;
- three thermal-regime logits;
- a non-negative conditional cooling magnitude;
- a non-negative conditional heating magnitude.

With regime probabilities `p = softmax(logits / temperature)`, the differentiable point forecasts are

```text
cooling_hat = p_cooling * cooling_magnitude
heating_hat = p_heating * heating_magnitude
```

Training and scheduling use these soft expected values. Hard `argmax` states are diagnostics only and never sit on the dispatch-gradient path. This preserves exact end-to-end differentiation without a straight-through estimator. Temperature is fixed by the v4.3 contract and may be selected only on 2019.

Electricity and gas keep independent continuous heads. Gas remains a fourth supervised prediction target and auxiliary scheduling prior; it is not a rigid terminal-demand balance.

## Rejected Alternatives

1. **Keep four ordinary continuous heads.** Rejected because `softplus` structurally leaks positive cooling and heating into inactive periods and has already failed full-year diagnostics.
2. **Discard inactive seasons or train separate seasonal models.** Rejected as the main method because RSC-PF is a single full-year rolling model and must handle transition periods.
3. **Hard-code month or temperature masks.** Rejected because the result would be site-specific and would confound learned forecasting performance with manually supplied operating rules.
4. **Use two unrelated binary thermal gates.** Rejected because the development data support a mutually exclusive three-state process; the categorical head expresses that structure directly with fewer inconsistent states.
5. **Round magnitudes to observed increments.** Rejected because the raw quantization is verified, but its physical origin is not sufficiently documented to claim that increments equal device counts.
6. **Use a hard or straight-through gate in the main model.** Deferred because it introduces biased gradient estimation. It may be reconsidered only if the soft-gate pilot leaves material inactive-period leakage.

## Model Contract

For each horizon `tau`, `RegimeAwareRSCPFModel` reports:

```text
forecast_normalized: [B, 4, 4]
forecast_physical:   [B, 4, 4]
regime_logits:       [B, 4, 3]
regime_probabilities:[B, 4, 3]
thermal_magnitudes:  [B, 4, 2]
controls:            [B, 4, 15]
dispatch:            [B, 4, 21]
```

The four-task order remains `[electricity, cooling, heating, gas_prior]`. The scheduler receives the same ten physical/context features as formal-v4.2. Regime probabilities are not added to the scheduler input in v4.3, so comparisons do not gain an extra information channel.

## Labels and Normalization

Thermal regime labels are derived solely from target values:

```text
off     := cooling <= 1e-9 and heating <= 1e-9
cooling := cooling >  1e-9 and heating <= 1e-9
heating := cooling <= 1e-9 and heating >  1e-9
```

A simultaneous positive cooling/heating target is fail-closed during v4.3 development because no such observation exists in the permitted 2015--2019 data and the three-state contract cannot represent it.

Electricity and gas use train-only mean and scale. Cooling and heating conditional magnitudes use train-only statistics calculated from their respective active observations. Selection and evaluation values never influence normalization.

## Forecast Objective

The supervised forecast loss contains four auditable components:

```text
L_continuous = Huber(electricity) + 0.25 * Huber(gas_prior)
L_regime     = class-weighted cross entropy(off/cooling/heating)
L_magnitude  = active-only Huber(cooling magnitude) + active-only Huber(heating magnitude)
L_point      = all-hour Huber([electricity, cooling, heating, gas_prior])
L_zero       = mean predicted cooling/heating magnitude on truly inactive targets
```

`L_forecast` is their contract-weighted sum. Class weights are computed from the 2015--2018 training split only, normalized to mean one, and clipped to a conservative range fixed in the contract. Target windows that contain an actual regime transition receive a fixed, modest training weight; the transition label is never an input feature.

Stage P trains the entire forecast/state pathway with `L_forecast`. Stage S trains the scheduler by teacher imitation with the forecast path frozen. Stage J starts from those checkpoints, unfreezes both pathways in the joint branch, and minimizes the existing dispatch objective plus `L_forecast` and the scheduled imitation term. The decoupled branch uses the identical model and initialization but detaches the forecast at the dispatch bottleneck.

The forecaster and scheduler use separate non-zero learning rates during joint training. A lower forecaster learning rate and the explicit regime/zero anchors are the primary safeguards against destructive dispatch gradients. Gradient-surgery methods are not added unless a v4.3 gradient-conflict audit demonstrates that these safeguards are insufficient.

## Evaluation and Gate 1

Gate 1 contains two immutable 2019 views:

1. **Full chronology:** every valid rolling origin, in time order, with uniform weight.
2. **Stress sample:** the existing fixed seasonal/active sample, used only as a secondary robustness view.

Both views report:

- four-task MAE, RMSE, and WAPE;
- active-only cooling and heating WAPE;
- inactive cooling and heating leakage MAE and total leaked energy;
- three-class macro-F1, balanced accuracy, and confusion matrix;
- transition-window metrics by horizon;
- dispatch objective, shortage, and physical residuals.

A Stage-J candidate is eligible only when it passes the frozen full-chronology forecast guardrails, the stress-view guardrails, the physical-feasibility gate, and the dispatch-improvement rule. Gate 1 candidate selection may tune only the explicitly enumerated v4.3 candidate fields. The 2020 evaluation split remains inaccessible until the resulting immutable receipt contains `authorized_gate2=true`.

## Baseline Fairness

- `RSC-PF` and `Decoupled-RSC-PF` use the same regime-aware forecaster, scheduler, warm start, data, and budget. Their only intended difference is whether dispatch loss updates forecast parameters during Stage J.
- `Continuous-Thermal-Head` retains the former four-continuous-head representation as the named mechanism ablation.
- PTO baselines keep their declared forecasting implementations and feed their forecasts to the same online optimizer. They are not silently upgraded to the proposed regime-aware head.
- Direct-Policy, Perfect-Information-MPC, Seasonal-Naive-PTO, official iTransformer adaptation, and Differentiable-LP keep their previously frozen roles; no 2020 result is used to revise them.

## Execution Boundary

The first execution after implementation is a one-seed v4.3 pilot on train/selection only. It compares Stage P, regime-aware joint Stage J, regime-aware decoupled Stage J, and the continuous-head ablation. It does not run the complete baseline matrix.

Only if the pilot demonstrates materially reduced inactive leakage, acceptable active-period accuracy, correct regime behavior, non-zero dispatch gradients into the forecaster, and exact physical feasibility may a fresh v4.3 Gate 0/Pilot/Gate 1 chain be executed. Gate 2 and the sealed 2020 split remain blocked until Gate 1 explicitly authorizes them.

## Success Criteria

- Four supervised forecast tasks remain present, including the independent gas-prior head.
- Thermal state and magnitude outputs are differentiable through the scheduler and physical decoder.
- The main model never uses month-based hard masking or evaluation-year calibration.
- Full-year inactive leakage is explicitly measured and materially lower than the continuous-head ablation.
- Active-period performance is not hidden by all-year zero dominance.
- RSC-PF and Decoupled-RSC-PF differ only at the dispatch-gradient boundary.
- No formal-v4.2 checkpoint is reused as v4.3 evidence.
- The sealed 2020 split remains untouched until an authorized Gate 2 run.
