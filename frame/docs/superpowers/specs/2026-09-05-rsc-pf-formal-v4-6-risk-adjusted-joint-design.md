# RSC-PF Formal-v4.6 Nominal-Forecast and Risk-Adjusted Joint Design

## 1. Purpose and Evidence

Formal-v4.5 is retained as an immutable negative Pilot result. On 8,709 rolling
2019 windows, RSC-PF Joint reduced the mean penalized dispatch objective from
6,395.25 for Fair Decoupled to 1,880.16 and reduced mean shortage from 55.94 to
10.21, while increasing ordinary operating cost from 800.76 to 859.17. The
physics residual remained below `4.6e-13`, and audited decision gradients reached
the forecast base, thermal regime gate, magnitude head, and scheduler.

The same run failed the leakage, active-cooling, and regime gates. Joint training
raised inactive cooling output from 6.55 for the frozen P1 parent to 23.42,
raised active-cooling WAPE by 16%, and produced regime macro-F1 of 0.842 versus
0.852 for the causal transition prior. The increased cooling output was
concentrated in the cooling season and accompanied an 82% reduction in
shortage. This is evidence of an objective conflict rather than a broken
decoder: one tensor was being asked to represent both a statistically calibrated
point forecast and the conservative demand level preferred by an asymmetric
shortage penalty.

Formal-v4.6 separates these two meanings inside one differentiable network. It
preserves an auditable nominal four-task forecast and introduces a bounded,
state-conditioned risk adjustment used only by the scheduling path.

## 2. Considered Approaches

Three approaches were considered:

1. **Nominal forecast plus bounded risk adjustment (selected).** Preserve the
   supervised forecast as the reported prediction, then derive a transparent
   scheduling demand through a small risk-adjustment head. This resolves the
   statistical-versus-operational meaning conflict while retaining end-to-end
   decision gradients.
2. **Loss reweighting only (retained as a diagnostic comparator).** Restore all
   regime-aware forecast losses in J and increase leakage/anchor weights without
   changing the model. This is minimally invasive, but it forces one forecast
   tensor to remain both a point estimate and a decision-optimal conservative
   estimate, so it may erase the dispatch gain rather than resolve the conflict.
3. **Relax the failed Pilot thresholds (rejected).** The v4.5 thresholds remain
   part of the historical receipt. Any v4.6 metric changes must correct the
   semantic comparison, be frozen before the next Pilot, and receive a new
   contract hash; no threshold is altered merely to relabel v4.5 as successful.

## 3. Frozen Scientific Boundary

Formal-v4.6 preserves:

- one end-to-end RSC-PF neural network and one differentiable forward graph;
- 24 hours of causal load, exogenous, device-output, and activity history;
- a four-hour forecasting and scheduling horizon;
- nominal predictions for electricity, cooling, heating, and station-side gas
  prior;
- electricity, cooling, and heating as the only rigid terminal demands;
- gas as a reported auxiliary station-side prior, never a terminal gas-demand
  balance;
- the 15-dimensional continuous control representation, 21-dimensional decoded
  dispatch, continuous device operation, and no future binary commitment output;
- the physics-feasible decoder and rolling first-step settlement;
- 2015--2018 for training and early stopping, 2019 as the explicitly observed
  development/Pilot-selection year, 2020 as the sealed final evaluation year,
  and exclusion of 2021;
- Fair Decoupled as the matched gradient-boundary comparator.

Formal-v4.5 artifacts are never overwritten. Formal-v4.6 uses new config,
manifest, run, receipt, and artifact namespaces.

## 4. Model Semantics and Data Flow

For each four-hour horizon, the existing state-conditioned Scheme2R path emits
the nominal forecast

\[
\widehat{\mathbf L}^{nom}=
[\widehat e,\widehat c,\widehat h,\widehat g^{prior}].
\]

The nominal tensor is the only forecast used for MAE, RMSE, WAPE, regime,
active-magnitude, and inactive-leakage reporting. It retains the same physical
units and target semantics as P1.

A new risk head consumes the detached-or-attached nominal rigid forecast,
encoded device state, thermal-regime probabilities, renewable forecast, prices,
carbon weight, SOC, and previous CHP output. It emits three non-negative raw
risk scores. These are transformed into bounded adjustments:

\[
\Delta\mathbf L^{risk}_{h,k}
=\sigma(z_{h,k})B_{h,k}M_{h,k},\qquad
k\in\{e,c,h\}.
\]

`B` is a fixed train-information-only cap. For each task and horizon it equals
the 90th percentile of the absolute frozen-P1 residual on the purged early-stop
partition. This is computed once after P1, serialized, and never fitted from
2019 or 2020. `M` equals one for electricity, the predicted cooling-regime
probability for cooling, and the predicted heating-regime probability for
heating. The risk head has no gas output.

The scheduler receives

\[
\widetilde{\mathbf L}^{sched}
=
[\widehat e+\Delta e,\widehat c+\Delta c,
 \widehat h+\Delta h,\widehat g^{prior}],
\]

together with the existing scheduler context and state. The model output
explicitly exposes `forecast_nominal`, `risk_adjustment`,
`scheduler_demand`, `controls`, and `dispatch`. No adjusted scheduling demand is
reported as a forecast.

## 5. End-to-End and Comparator Boundaries

RSC-PF Joint performs one forward pass and one backward pass. Decision loss
flows through the scheduler, risk head, nominal forecast, residual thermal head,
and shared encoder. The receipt must show finite positive decision-gradient
norms for the Scheme2R base, regime gate, magnitude head, risk head, and
scheduler.

Fair Decoupled uses the byte-identical P1+S parent and the same risk-head and
scheduler architecture. Its nominal forecast and shared state are detached at
the scheduling interface. The risk head and scheduler may learn operational
adjustments, but decision gradients into the Scheme2R base, gate, and magnitude
head must be at most `1e-12`. This isolates the value of the end-to-end gradient
edge without giving RSC-PF a larger scheduling interface.

The loss-reweighting-only comparator uses the same J training protocol with the
risk adjustment fixed to zero. It is diagnostic evidence about the need for the
new interface and is not added to the full external-baseline matrix unless the
Pilot succeeds.

## 6. Joint Objective

The J-stage forecast term is the full state-aware P1 objective, not generic
smooth-L1 alone. It contains electricity/gas error, regime cross-entropy,
active thermal magnitude error, four-task point error, and inactive thermal
leakage. The complete objective is

\[
\begin{aligned}
\mathcal L_J={}&
\lambda_f\widetilde{\mathcal L}_{forecast}^{full}
+\lambda_i\widetilde{\mathcal L}_{imitation}
+\lambda_d\widetilde{\mathcal L}_{decision}\\
&+\lambda_a\mathcal L_{anchor}
+\lambda_b\mathcal L_{risk-size}
+\lambda_o\mathcal L_{off-risk}.
\end{aligned}
\]

The first three terms use detached train-partition normalization constants.
`L_anchor` compares the nominal forecast with the frozen P1 nominal forecast.
`L_risk-size` penalizes adjustment divided by its cap. `L_off-risk` penalizes
cooling and heating adjustment mass assigned where the supervised target is
inactive during training; it is never evaluated using future labels at
inference.

The existing v4.5 decision/imitation curriculum remains the default calibration
centre. Formal-v4.6 permits exactly three predeclared early-stop calibration
settings: risk regularization multipliers `0.5`, `1.0`, and `2.0`, with all
other loss weights fixed. The setting with the lowest early-stop decision
objective among forecast-eligible checkpoints is selected. No 2019 metric may
select among these settings.

## 7. Checkpoint Eligibility

Every J epoch is evaluated on the purged 2015--2018 early-stop split. A
checkpoint is eligible only when the nominal forecast satisfies all of the
following relative to its frozen P1 parent:

- four-task score no worse than `1.02` times the parent;
- electricity WAPE no worse than `1.02` times the parent;
- gas-prior WAPE no worse than `1.10` times the parent;
- active cooling and heating WAPE each no worse than `1.05` times the parent;
- normalized inactive leakage for cooling and heating each no worse than
  `1.05` times the parent;
- regime macro-F1 no more than `0.02` below the transition prior;
- transition-window balanced accuracy at least `0.01` above the transition
  prior.

Normalized inactive leakage is defined per task as total absolute nominal
prediction during truly inactive target positions divided by total true load
during active positions. Absolute inactive mean and p95 remain mandatory
descriptive metrics. This avoids unstable ratios whose denominator is an
already tiny absolute leakage value while preserving an interpretable audit of
false thermal output.

Among eligible checkpoints, select the lowest normalized early-stop decision
objective. If no checkpoint is eligible, the run fails closed; it is never
silently replaced by P1.

## 8. Formal-v4.6 Pilot Gate

The 2019 Pilot is run once only after implementation, diagnostic tests,
train/early-stop calibration, contract hashing, and source-manifest freezing.
The Pilot authorizes Gate 1 only when:

1. nominal forecast guardrails from Section 7 hold on 2019;
2. RSC-PF Joint penalized objective and total shortage do not exceed Fair
   Decoupled;
3. Joint decision gradients reach all intended forecast and scheduling groups,
   while Fair Decoupled decision gradients do not reach forecast groups;
4. maximum physical residual is at most `1e-6`;
5. risk adjustments are finite, non-negative, within serialized caps, absent
   from gas, and reported by task, horizon, regime, mean, and p95;
6. source, split, teacher, cap, checkpoint, and rollout lineage are complete;
7. no 2020 artifact is opened.

The old requirement that Joint must reduce each raw inactive-leakage mean by
30% relative to P1 is not reused. That comparison confounded the architectural
P1 effect with the J-stage gradient effect and became unstable when P1 leakage
was already small. The new nominal-forecast preservation rule and normalized
leakage definition are frozen prospectively in the v4.6 contract. The v4.5
failure remains unchanged in its original receipt.

## 9. Artifacts and Fail-Closed Audit

Formal-v4.6 persists:

- P1 residual distributions and the resulting per-task/per-horizon cap matrix;
- hashes for training, early-stop, benchmark, capacity, teacher, contract, and
  source manifest;
- every calibration setting and its complete validation history;
- nominal forecasts, bounded adjustments, scheduler demands, controls,
  dispatch, carried states, shortages, costs, carbon, and physical residuals;
- per-epoch optimizer groups, learning rates, curriculum weights, normalized
  loss terms, eligibility values, and selected checkpoint hash;
- Joint and Fair Decoupled gradient-boundary receipts;
- one immutable Pilot receipt, independent audit, and transition receipt.

Missing caps, cap violations, adjusted gas, non-finite values, selection from
2019, evaluation-year access, comparator asymmetry, or inconsistent hashes fail
closed.

## 10. Testing and Execution Order

Implementation tests must cover:

- output shapes and semantics for nominal forecast, risk adjustment, scheduler
  demand, controls, and dispatch;
- non-negative bounded risk adjustments and exactly zero gas adjustment;
- thermal probability masking of cooling/heating adjustments;
- full regime-aware forecast loss inside J;
- positive Joint decision gradients through all intended groups;
- zero Fair Decoupled decision gradients into forecast groups;
- train-information-only cap fitting and immutable cap receipts;
- normalized leakage calculation and checkpoint eligibility;
- deterministic safe restart, artifact integrity, and 2020 firewall;
- physical feasibility under both zero and maximum allowed adjustment.

Execution order is: deterministic unit tests, protected v4.5 regressions,
train-only gradient diagnostic, three-setting early-stop calibration, contract
and manifest freeze, then one full 2019 Pilot. Gate 1 baselines and ablations do
not start unless the Pilot receipt authorizes them. The 2020 archive remains
sealed until all later gates explicitly authorize final evaluation.

## 11. Completion Criteria

The formal-v4.6 repair is engineering-complete only when all targeted and
protected tests pass, the independent audit reconstructs cap fitting,
checkpoint eligibility, gradient boundaries, and Pilot authorization, and all
artifacts are immutable and reproducible from their lineage.

Scientific success is not assumed by implementation completion. If the new
2019 Pilot fails any frozen criterion, the process stops before Gate 1 and the
negative result is retained.
