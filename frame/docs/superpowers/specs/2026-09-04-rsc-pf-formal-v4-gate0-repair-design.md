# RSC-PF Formal-v4.1 Gate 0 Comprehensive Repair Design

**Date:** 2026-09-04  
**Status:** Approved design, review revision 1
**Scope:** Repair Tasks 1--12 and issue a scientifically valid Gate 0 authorization. Gate 1--3 training and manuscript-result updates are out of scope.

## 1. Objective

Repair the formal-v4 experiment foundation as a new formal-v4.1 protocol so that every artifact entering Gate 1 is causally generated, bound to the frozen 2015--2018 training protocol, physically interpretable, reproducible from immutable inputs, and verified by a genuinely fail-closed Gate 0. The original formal-v4 contract and its receipts remain immutable historical evidence; repaired runs must use a new `joint-forecast-dispatch-formal-v4.1` schema and configuration.

The repair preserves validated model components. It does not redesign the RSC-PF research idea, change the 24-to-4 forecasting task, introduce binary future commitment decisions, or access the locked 2020 evaluation split.

## 2. Disposition of Existing Work

The existing `formal_v4_20260903_retry3` Gate 0 receipt and all downstream Gate 1/2 artifacts are retained for audit but classified as `invalid_pre_repair_trial`. They must never be overwritten, deleted, merged into formal result tables, or used to initialize a repaired formal run unless an artifact is independently revalidated and copied through a new hash-checked receipt.

The following components remain conceptually valid and should be repaired in place rather than rewritten:

- RSC-PF joint forecaster--scheduler architecture;
- causal DS-TCN state encoder;
- 15-dimensional continuous scheduling representation and 21-dimensional decoded dispatch;
- Stage P, Stage S, and Stage J gradient-boundary definitions;
- fixed training-only `C_ref` objective scaling rule;
- differentiable physical recourse and constraint accounting;
- official THUML iTransformer backbone adaptation;
- isolated CVXPYlayers Differentiable-LP environment and numerical gate;
- the 2015--2018/2019/2020 train/selection/evaluation split.

## 3. Frozen Scientific Boundary

The repaired protocol uses:

- training and normalization: 2015--2018 only;
- model and hyperparameter selection: 2019 only;
- final evaluation: 2020, inaccessible until Gate 2 authorizes Gate 3;
- lookback: 24 hourly observations;
- forecast and scheduling horizon: 4 hours;
- forecast tasks: electricity, cooling, heating, and station-side gas-consumption prior;
- rigid dispatch demands: electricity, cooling, and heating only;
- PV/WT forecast: one-hour last-value persistence repeated across the four-hour horizon;
- no future binary commitment decisions;
- no price/carbon-response claim in the core experiment.

The station-side gas series remains a supervised auxiliary forecasting target and contextual prior. It is not interpreted as a rigid terminal gas demand and does not enter a fabricated gas-balance equation.

## 4. Repair Architecture

The repair is organized around one immutable provenance chain:

```text
Git source commit + frozen protocol + raw source hashes
                         |
                         v
2015--2018 benchmark parameter receipt
                         |
                         v
representative training-only capacity audit receipt
                         |
                         v
causally planned and realized-settled device trajectory
                         |
                         v
train/selection windows + train-only normalization + C_ref
                         |
                         v
same-information teacher overlays and method adapters
                         |
                         v
complete tests + resource projection + Gate 0 receipt
```

Every downstream receipt stores the SHA-256 of every direct upstream artifact. Changing a source commit, protocol file, benchmark value, capacity multiplier, raw source, trajectory rule, normalization, or objective-scale rule invalidates all descendants.

## 5. Protocol and Repository Freeze

Before rebuilding data, all formal-v4.1 code, configuration, tests, third-party source receipts, and repair documentation must be committed. The repair creates `joint_forecast_dispatch_formal_v4_1.json` and must not rewrite `joint_forecast_dispatch_formal_v4.json`. Gate 0 records the Git commit and refuses an untracked, missing, hash-mismatched, or dirty file in the formal-v4.1 runtime closure. Unrelated user changes may remain dirty only if the Gate 0 manifest proves that none of their paths are consumed by formal-v4.1.

The runtime closure is enumerated explicitly in `formal_v4_source_closure_v4_1.txt`, reviewed path by path, and committed before any repaired artifact is certified. It includes every direct or transitive Python module, configuration, test contract, benchmark rule, parameter ledger, state-feature ledger, third-party source receipt, imported upstream source file, environment lock, search budget, builder, evaluator, and Gate 0/auditor script used by the formal run. Existing unrelated user edits are preserved and are never swept into this commit without review.

The protocol freeze must be parsed and validated, not merely checked for existence. Its stored hashes must equal the current files. Dataset access receipts for container sources, including ZIP archives, bind both the container hash and the exact member path/hash used; hashing only the outer archive is insufficient.

## 6. Formal-v4.1 Standard IES Rebuild

A formal-v4-specific benchmark is rebuilt from 2015--2018 only. The existing 2015--2019 benchmark remains untouched as a legacy artifact.

The new receipt records:

- raw input hashes and exact training timestamps;
- derivation rule for every capacity;
- resolved device capacities and efficiencies;
- PV/WT profile rules and resolved rated capacities;
- source units and real/simulated/derived classification;
- parameter-ledger hash;
- confirmation that 2019 and 2020 were not read.

The benchmark generator must reject a timestamp outside 2015--2018 and refuse to overwrite an existing receipt.

## 7. Representative Capacity Audit

Capacity certification is a two-stage training-only procedure. The first stage no longer takes the first 500 chronological windows. A deterministic, precomputed origin manifest selects exactly 500 valid origins from 2015--2018 with coverage of:

- all four seasons and all four training years;
- weekday/weekend and major hour-of-day strata;
- ordinary, upper-decile, and extreme cooling-demand regimes;
- ordinary and high heating/electricity regimes;
- valid continuous 24-hour history plus four-hour future windows.

At least one registered stratum must contain nonzero cooling demand, and the audit fails if total audited cooling demand is zero or if high-cooling strata are absent. The same frozen origins are used for every multiplier from 1.0 to 2.0.

Each stage-one candidate is evaluated using the resolved formal-v4.1 benchmark, correctly scaled PV/WT availability, the common PI-MPC solver, and fixed independent-window diagnostic states `SOC=0.5` and `previous_CHP_output=0`. This stage is a stratified filter, not final capacity evidence.

The second stage tests candidates from smallest to largest on the complete chronological 2015--2018 trajectory. At the series start and after each documented timestamp gap, reset to `SOC=0.5` and `previous_CHP_output=0`; otherwise both values carry from one settled hour to the next. A multiplier is eligible only if it satisfies the cooling shortage-energy ratio of at most 0.5% and shortage-hour rate of at most 1.0% in both stages. The first multiplier passing both becomes `capacity_adequate_main`; multiplier 1.0 remains `benchmark_capacity_stress`. If no candidate passes both audits, Gate 0 fails rather than selecting the best observed candidate.

The capacity receipt contains the origin-manifest hash, resolved parameter hash, diagnostic initial states, per-stratum results, full-trajectory results, candidate-level pass/fail evidence, selected multiplier, solver version, state-reset log, and deterministic scenario identity. Runtime measurements are stored separately and are excluded from scientific identity hashes.

## 8. Causal Realized-Settled Device History

For every chronological hour, the historical simulator performs exactly one atomic transition:

1. construct a forecast from information available at or before the decision time;
2. solve the common four-hour planning problem;
3. execute only the first planned action;
4. settle that action once against the realized electricity, cooling, heating, PV and WT observations for the current hour;
5. store the realized 17 continuous device outputs;
6. derive the six activity indicators from those raw realized outputs using the frozen `1e-6` threshold;
7. advance SOC and previous CHP output from the settled result;
8. reset state only across a documented timestamp gap.

Forecast construction and history bootstrap are frozen as follows:

- at decision hour `t`, the four-step load plan uses the seasonal-naive values observed at `t-24:t-20`; it is available only after 24 continuous prior hours;
- the PV/WT plan repeats the availability observed at `t-1` across the four planned hours;
- realized demand and renewable availability at hour `t` are used only to settle the first action and never enter the planning input at `t`;
- after the beginning of a series or any timestamp gap, hours 1--24 are load-forecast warm-up and settled device transitions begin at hour 25;
- a model sample additionally requires 24 settled device-history hours, so the first eligible model origin is hour 49, equivalently zero-based index 48;
- the same 48-hour burn-in is repeated after every gap.

The stored history must satisfy electricity, cooling and heating balance residuals at most `1e-6`, conversion identities, device capacities, SOC recursion and bounds, CHP ramp limits, renewable availability identities, and the charge/discharge exclusivity convention. Slack and dump remain explicit settlement outputs but are excluded from the 17 historical device input fields according to the frozen ledger.

Selection-year states are generated only after the training trajectory and normalization are frozen. If December 2018 and January 2019 are timestamp-continuous, SOC, previous CHP output, and the causal history buffer carry across that boundary; the first 2019 sample may therefore use 2018 observations as past context. No 2019 label or future observation may affect training normalization, benchmark construction, capacity selection, or `C_ref`. A genuine gap triggers the same 48-hour burn-in rule.

## 9. Materialized Data, Normalization, and Objective Scale

The repaired run stores its artifacts under one run root. It must not write capacity-bound files to a sibling or legacy directory.

The train and selection archives include:

- 24-hour four-task load history;
- 24-hour exogenous history;
- 24-hour PV/WT availability history;
- 24-hour 17-field settled device history;
- 24-hour six-field activity history;
- four-hour forecast targets;
- three rigid future dispatch demands;
- PV/WT forecasts and realized availability;
- scheduler context, initial SOC and previous CHP output;
- timestamps, trajectory IDs and state hashes.

Normalization is fitted once from the capacity-bound 2015--2018 train archive. Its receipt stores training timestamp bounds, source archive hash, capacity receipt hash, feature order, mean/scale arrays and a normalization hash. Selection uses the frozen training normalization without refitting.

`C_ref` is fitted from finite, nonnegative, training-only PI-LP objectives using `max(median(weighted_objective), 1.0)` and the frozen step weights `[0.5, 1/6, 1/6, 1/6]`. Its receipt includes the train archive hash, capacity hash, raw mean, median, IQR, sample count and objective implementation hash.

State hashes include the capacity receipt hash, trajectory-rule version, origin timestamp and raw input arrays. A receipt path or boolean authorization alone is insufficient.

## 10. Teacher and Training-Stage Contracts

Stage P remains prediction pretraining, Stage S remains scheduler imitation with the forecaster frozen, and Stage J remains the formal joint/decoupled comparison. Stage J uses one frozen curriculum: forecast weight is `1.0` throughout; imitation weight decreases linearly from `1.0` at epoch 0 to `0.0` at epoch 18; decision weight increases linearly from `0.05` at epoch 0 to `1.0` at epoch 18; after epoch 18 the weights are forecast `1.0`, imitation `0.0`, and decision `1.0`. Epoch interpolation and boundary convention are defined once in the protocol and asserted exactly by tests. A code default such as an imitation final weight of `0.25` is incompatible and must not override the protocol.

Teacher overlays are regenerated from the same capacity-bound state and the exact Stage P checkpoint. Each overlay binds:

- Stage P checkpoint SHA-256;
- train archive and state hashes;
- capacity and normalization hashes;
- target timestamps;
- teacher implementation and solver hashes.

Stage J joint mode must show nonzero decision-loss gradient to both forecaster and scheduler. Decoupled mode must show zero decision-loss gradient to the forecaster and nonzero gradient to the scheduler. Smoke evidence uses real materialized batches in addition to synthetic numerical tests.

## 11. Baseline and Evaluation Repair

Every registered method must provide one executable adapter interface:

```text
predict_and_dispatch(window, rolling_state) -> {
    forecast, dispatch, demand, target, next_state, optimizer_calls, latency
}
```

Metadata-only descriptors do not qualify as implemented baselines. Scheme2R-PTO, State-Conditioned-PTO, Official iTransformer-PTO, Differentiable-LP, Perfect-Information-MPC and Seasonal-Naive-PTO must use the common LP and common rolling settlement where applicable. RSC-PF, Decoupled-RSC-PF and Direct-Policy must report zero online exact-LP calls; every PTO, Differentiable-LP and PI-MPC window must report exactly one.

One canonical field, `online_optimizer_calls_per_window`, is used everywhere. Compatibility aliases may be read only at legacy boundaries and must never silently default a registered optimizer-based method to zero.

The unified evaluator verifies chronological state carry, forecast shapes, physical metrics, optimizer calls, latency decomposition and complete-trajectory comparison with PI-MPC. Gate 0 exercises each adapter on a small real training-only slice; it does not produce paper results.

## 12. Genuine Fail-Closed Gate 0

Gate 0 executes checks rather than trusting prewritten booleans. Authorization is granted only when every mandatory item passes:

- protocol and source manifest match the frozen Git commit, and every runtime dependency is tracked, committed, clean, and listed in the source closure;
- formal-v4 benchmark is derived only from 2015--2018;
- both the 500-origin stratified audit and complete chronological capacity certification have representative nonzero cooling coverage and the same valid selection;
- train/selection state archives exist under the same run root;
- trajectory, normalization and `C_ref` hashes form a complete chain;
- all physical-history invariants pass at `1e-6` tolerance;
- teacher alignment and Stage P/S/J gradient boundaries pass;
- all nine registered method identities and executable adapters pass;
- official iTransformer source and current imported-file hashes pass;
- Differentiable-LP eligibility and current environment lock pass;
- all dataset opens pass through the canonical access controller, a static bypass scan finds no direct formal-v4.1 dataset access, and no 2020/evaluation artifact or archive member was opened or materialized;
- formal-v4 and complete legacy regression suites have zero failures;
- projected p95 time for any method/seed is at most 24 hours;
- available disk and memory margins are each at least 20%.

`capacity_audit`, `protocol_freeze`, data materialization, normalization, `C_ref`, full regression tests and resource checks are mandatory blockers. Unknown or missing checks fail authorization. Gate 0 exits nonzero on failure and still writes an immutable diagnostic receipt with `authorized_gate1=false`.

The successful receipt lists every planned issue identifier, measured evidence, artifact paths and hashes, test counts, resource projections, timestamps and the explicit statement `test_set_accessed=false` supported by an access log. For an archive source, the log identifies and hashes the opened member as well as the container. The evaluation archive remains unmaterialized through Gate 0.

## 13. Test Strategy

Tests are layered as follows:

1. unit tests for hashes, split rejection, capacity strata, settlement, physical identities, normalization and optimizer-call accounting;
2. integration tests that deliberately tamper with each upstream artifact and confirm descendant rejection;
3. real-data tests over bounded 2015--2018 slices, including summer cooling periods;
4. adapter contract tests for all nine methods;
5. Gate 0 fault-injection tests covering every mandatory failure;
6. the complete repository regression suite from one documented working directory;
7. a static access-bypass scan and archive-member receipt tests;
8. an independent Gate 0 auditor that is committed and hash-bound before the final source manifest is generated.

Tests must prove that a failed capacity audit, missing protocol receipt, stale teacher, changed benchmark, mismatched normalization, dirty formal source, hidden 2020 path, incorrect optimizer count, failed legacy test or insufficient resource margin each produces `authorized_gate1=false`.

Repairs to legacy tests are limited to path resolution, fixture isolation, or demonstrated regressions. They must not change frozen scientific calculations, expected metrics, or prior receipt semantics merely to make the suite pass; numerical and receipt-behavior comparisons are required before accepting such a test repair.

## 14. Completion Criteria

The repair is complete only when:

- the old runs are visibly marked invalid but preserved;
- all formal-v4.1 runtime dependencies are tracked, committed, clean, closure-listed, and hash-recorded while the original formal-v4 contract remains unchanged;
- a new 2015--2018 benchmark receipt exists;
- the representative 500-origin audit and the complete chronological 2015--2018 audit both produce interpretable nonzero-demand results and certify the same selected multiplier;
- settled state histories pass physical audits;
- train/selection archives, normalization and `C_ref` are stored under one new run root;
- every baseline executes through the common interface with correct optimizer accounting;
- formal-v4 targeted tests and the full repository suite have zero failures;
- a fresh Gate 0 run with a new run ID produces `authorized_gate1=true`;
- the previously committed independent post-run auditor reproduces every hash and check without reading or materializing 2020.

Only after these criteria are met may a separate Gate 1 execution plan be activated.
