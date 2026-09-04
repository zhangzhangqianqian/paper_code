# RSC-PF Formal-v4.2 Repair and Rerun Design

## 1. Purpose

Formal-v4.1 Gate 0 established useful data, capacity, provenance, and dependency evidence, but its Gate 1 and Gate 2 execution paths did not implement the frozen scientific protocol. The existing Gate 2 result is therefore retained only as an invalid diagnostic run. It must not authorize 2020 evaluation or support manuscript claims.

Formal-v4.2 will repair the complete executable path before rerunning experiments. The primary objective is to determine whether RSC-PF improves closed-loop rolling dispatch relative to its strictly matched decoupled comparator while retaining acceptable four-task forecasting performance and physical reliability.

No 2020 evaluation data may be read until the repaired Gate 2 authorizes Gate 3. The year 2021 remains outside this protocol.

## 2. Version and Artifact Boundary

- Preserve all formal-v4.1 artifacts byte-for-byte.
- Register `formal_v4_1_gate2_20260904_104000` and its corrected receipt as `invalid_diagnostic`; neither is a paper result.
- Create a new formal-v4.2 contract, source closure, run root, and immutable receipts.
- Reuse source data and the capacity-rule design only through newly hash-bound v4.2 artifacts. Do not write new caches into a prior Gate 0 root.
- A change to any protocol, data, benchmark, normalization, source, checkpoint, or method configuration requires a new unique run identifier and invalidates downstream authorization from the older run.

## 3. Frozen Scientific Question

The primary comparison is:

> Does allowing dispatch loss to update the state-conditioned Scheme2R forecasting pathway improve realized rolling dispatch relative to an otherwise identical branch whose forecaster remains frozen after Stage S?

The two branches must have identical architecture, data, Stage P checkpoint, Stage S checkpoint, scheduler initialization, random seed, physical decoder, and training budget. Their only intended difference is the Stage J forecast-gradient boundary.

Secondary comparisons cover a decision-only policy, forecast-then-optimize methods, an official iTransformer forecasting adaptation, a differentiable-LP adaptation, a deterministic seasonal baseline, and a non-deployable perfect-information reference.

## 4. Data and Information Boundary

- Training: 2015--2018 only.
- Model/configuration/checkpoint selection: 2019 only.
- Locked out-of-time evaluation: 2020 only after Gate 2 authorization.
- Excluded: 2021.
- History length: 24 hours; forecast and planning horizon: 4 hours.
- Forecast targets: electricity, cooling, heating, and station-side aggregate gas-use prior.
- Electricity, cooling, and heating are rigid physical demands. Gas remains a standardized auxiliary operating prior and never becomes a terminal gas-balance demand.
- PV/WT planning inputs use the frozen last-value persistence forecast. Realized future PV/WT are used only for supervised labels, realized settlement, and the Perfect-Information-MPC reference.
- All normalization statistics are fitted on 2015--2018 and reused unchanged for 2019 and 2020.
- Cooling and heating are zero-inflated. Primary metrics remain frozen, with active-hour and seasonal diagnostics added without replacing the primary endpoint.

## 5. Training Architecture

### 5.1 Persistent optimization state

Each training stage owns a persistent optimizer for the duration of that stage. Optimizer state must not be recreated per mini-batch. Checkpoints include model state, optimizer state, epoch, early-stopping state, source hashes, data hashes, and RNG state.

### 5.2 Sequential stages

For each seed:

1. Stage P fully trains the state-conditioned four-task forecaster.
2. A seed-specific same-information teacher overlay is generated from the frozen Stage P checkpoint, its predicted rigid demands, frozen PV/WT persistence forecasts, prices, and causal carried state.
3. Stage S freezes the forecast pathway and trains the neural scheduling pathway against the hash-bound teacher overlay and physical objective.
4. The completed Stage S checkpoint is copied byte-for-byte into RSC-PF and Decoupled-RSC-PF branches.
5. Stage J trains both branches under the same curriculum and budget. RSC-PF permits dispatch gradients into the forecaster; Decoupled-RSC-PF blocks those gradients while continuing to update the scheduler.

Stage P, Stage S, and Stage J are not interleaved. Early stopping and checkpoint selection use 2019 only and follow a frozen scoring rule. The decision curriculum must run long enough to reach its registered terminal weight unless early stopping is triggered by the registered rule after the required minimum epoch.

### 5.3 Scheduled roll-in

After the registered start fraction, training refreshes the device-state history using the model's own first-step realized actions. Refreshed histories are detached before the next forward pass; no cross-hour BPTT is claimed. When a state-specific same-information teacher is not recomputed, imitation loss for that refreshed sample is zero.

### 5.4 Direct-Policy and external baselines

- Direct-Policy receives the same causal histories and scheduler context but has no explicit forecast output or forecast loss. It has its own decision/imitation training routine.
- Scheme2R-PTO trains the original Scheme2R forecaster on the v4.2 split, then invokes one exact online LP per rolling window.
- State-Conditioned-PTO loads the seed-matched Stage P checkpoint byte-for-byte and invokes the same online LP. It is the strict same-information PTO anchor.
- Official iTransformer-PTO uses the verified upstream THUML backbone with a disclosed four-task adaptation, train-only normalization, and one exact online LP call per window.
- Differentiable-LP uses the verified CVXPYlayers environment and a trained forecasting pathway. Its optimizer-at-inference requirement is reported explicitly.
- Seasonal-Naive-PTO is deterministic and runs once.
- Perfect-Information-MPC uses realized future demand and renewable availability, is non-deployable, and runs once as a reference rather than a competitive model.

## 6. Closed-Loop Evaluation

All methods share one chronological evaluator. At each origin it:

1. constructs the causal 24-hour history from the current rolling state;
2. produces a four-hour forecast and/or four-hour dispatch plan;
3. executes only the first planned hour;
4. settles that action against realized demand and realized PV/WT with the canonical one-pass physical recourse;
5. records cost, carbon, shortage, curtailment, physical residuals, latency, and optimizer calls;
6. advances SOC, previous CHP output, device-output history, activity history, load history, and exogenous history by exactly one hour.

Adapters must use the supplied rolling state rather than stale state stored in a source window. The next state comes from the executed first action, never from the fourth planned action.

The evaluator computes, rather than defaults, balance, capacity, conversion, SOC, ramp, charge/discharge exclusivity, renewable accounting, and finite-value checks. Slack is reported as shortage and is not mislabeled as an equality residual. `p_dump` remains a realized-settlement diagnostic outside the learned 21-column output contract.

Forecast metrics use actual future targets. Dispatch metrics use realized first-step outcomes. Overlapping four-hour plans are not counted as four independent executed trajectories.

## 7. Gate Design

### Gate 0: repaired evidence and real resource probe

Gate 0 rebuilds source manifests and immutable train/selection artifacts, confirms the capacity scenario without 2019/2020 leakage, validates official-source and differentiable-LP dependencies, and benchmarks real forward, backward, LP, and differentiable-LP operations. Synthetic array operations cannot authorize the resource gate.

### Engineering pilot

Before Gate 1, run a one-seed, bounded, train-only, explicitly non-paper pilot. It must demonstrate decreasing Stage P/S/J losses, persistent optimizer steps, correct Stage S cloning, nonzero RSC-PF decision gradients, zero Decoupled forecaster decision gradients, first-step state carry, finite outputs, and non-catastrophic shortage. Pilot results never select a final model or enter paper tables.

### Gate 1: representative calibration

Gate 1 uses a pre-registered 2019 origin manifest covering cooling-active, heating-active, shoulder-season, and remaining chronological periods. It records exact indices, sampling weights, task activity fractions, timestamps, and hashes. A block with an almost-zero cooling or heating denominator cannot authorize Gate 2.

### Gate 2: complete 2019 selection matrix

Gate 2 runs all registered stochastic methods for seeds 2026, 2027, and 2028, plus deterministic references once. It saves seed-specific checkpoints and chronological first-step arrays. Authorization requires:

- complete method/seed coverage;
- unchanged Gate 1 configuration and data hashes;
- all registered forecast guardrails;
- finite physical metrics and registered tolerances;
- RSC-PF mean penalized objective below Decoupled-RSC-PF with at least two of three seed directions favorable;
- RSC-PF shortage energy no greater than 1.05 times the true seed-matched State-Conditioned-PTO value;
- no 2020 or 2021 access;
- complete checkpoint, runtime, optimizer-call, and failure receipts.

Training epochs, patience, candidate values, and validation frequency are read only from the frozen contract and Gate 1 receipt. Gate 2 exposes no command-line option that can change them after calibration.

If any condition fails, Gate 2 records a failure and stops. Thresholds, epochs, and candidate sets cannot be changed in place after viewing the result.

### Gate 3: locked 2020 evaluation

Only a valid Gate 2 receipt can create the one-time 2020 authorization envelope. Gate 3 evaluates the frozen five-seed models and deterministic references without training, retuning, or checkpoint selection on 2020. It stops after main evaluation; ablations, manuscript edits, and 2021 access remain separate work.

## 8. Metrics and Statistical Reporting

Forecasting reports task-by-horizon MAE, RMSE, and WAPE; rigid-task macro WAPE; gas WAPE; active-hour diagnostics; and seasonal diagnostics. Raw MAE/RMSE are never averaged across incompatible source units.

Dispatch reports realized operating cost, physical carbon, penalized objective, carrier-specific shortage energy/rate, curtailment, `p_dump`, `q_dump`, all physical residual families, runtime, and online optimizer calls.

The primary contrast is seed-matched RSC-PF minus Decoupled-RSC-PF. Gate 3 uncertainty uses paired seeds and contiguous 168-hour blocks with 2,000 bootstrap replicates. Five seeds are the independent model replicates; rolling windows are not treated as independent models. Deterministic references receive time-block uncertainty only.

## 9. Provenance, Resume, and Failure Handling

- One v4.2 run root contains Gate 0 through Gate 3 lineage.
- Generated teachers, checkpoints, metrics, and caches live under their producing gate, never in an earlier authorized root.
- Every cache validates split, timestamp, state, capacity, benchmark, normalization, model/checkpoint, implementation, and source hashes before reuse.
- Completed artifacts are immutable. Resume continues only incomplete work and verifies existing hashes before proceeding.
- Each stochastic row records the actual seed; no hard-coded seed labels are accepted.
- Every failed method/seed writes a failure receipt. Missing rows block authorization.
- Hardware, Python, package versions, CPU thread settings, determinism settings, runtime, and memory are recorded.

## 10. Test Strategy

The repaired suite must add integration tests for:

- persistent optimizer state across batches;
- sequential Stage P/S/J execution and identical Stage S branch cloning;
- exact joint versus decoupled gradient boundaries;
- same-information teacher inputs and cache hash rejection;
- adapter normalization and checkpoint loading;
- rolling-state consumption and first-step state advancement;
- realized closed-loop metrics and non-default physical violations;
- full Gate 1 and Gate 2 method/seed manifests;
- correct State-Conditioned-PTO identity;
- fail-closed Gate 2 authorization;
- real resource probes rather than synthetic no-op timing;
- immutable artifact roots and no evaluation access before Gate 3.

Existing unit tests remain useful but cannot authorize an experimental gate without these end-to-end tests.

## 11. Success Criteria

The repair is complete only when:

1. the old v4.1 run is preserved and registered as invalid diagnostic evidence;
2. all v4.2 focused and repository regression tests pass;
3. a real resource probe provides a credible runtime envelope;
4. the one-seed engineering pilot satisfies its numerical and gradient checks;
5. Gate 1 freezes a representative calibration configuration;
6. Gate 2 completes the full registered matrix and independently verifies its authorization decision;
7. no 2020 data are accessed unless Gate 2 authorizes Gate 3;
8. no result is labeled a paper result until the corresponding gate and independent audit pass.
