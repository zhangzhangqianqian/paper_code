# RSC-PF Formal-v4.2 Gate 2 Execution Closure Design

## 1. Purpose

Formal-v4.2 has valid engineering evidence for Gate 0 and the Pilot, but its current Gate 2 entry point only validates a pre-existing result payload. It does not yet train, evaluate, persist, and audit the complete 23-row comparison matrix. This change closes that execution gap before the first complete formal comparison is started.

The scientific protocol remains formal-v4.2. The model architecture, information boundary, data years, physical decoder, rolling settlement, method identities, primary comparison, and Gate 3 evaluation boundary do not change. This work completes the executable implementation and freezes previously missing resource budgets.

## 2. Scope and Stop Boundary

The implementation covers:

1. the final Gate 2 training and evaluation budget;
2. executable training paths for every stochastic method;
3. deterministic execution paths for the two references;
4. checkpoint, rollout, metric, provenance, runtime, and failure artifacts;
5. an independent Gate 2 audit;
6. a new immutable formal-v4.2 run from Gate 0 through Gate 2.

The execution stops immediately when Gate 0, Pilot, Gate 1, Gate 2, or the independent Gate 2 audit fails. It also stops after a successful Gate 2 so that the results can be reviewed before Gate 3. Gate 3, 2020 access, ablations, manuscript edits, and figure generation are outside this change.

## 3. Frozen Data and Information Boundary

- Stage P, Stage S, Stage J, Direct-Policy, Scheme2R, iTransformer, and Differentiable-LP training use 2015--2018 only.
- Normalization is fitted on 2015--2018 only.
- Gate 1 calibration and Gate 2 rolling selection evaluation use 2019 only. Gate 1 uses a fixed representative calibration manifest; Gate 2 evaluates the complete eligible 2019 chronology after checkpoint selection.
- No process may read, scan, materialize, normalize, train on, tune on, or evaluate against 2020 before a successful Gate 2 transition.
- The year 2021 remains excluded.
- Causal input history is 24 hours and prediction/planning horizon is 4 hours.
- Electricity, cooling, and heating are rigid demands. Gas is the standardized station-side aggregate gas-use prior and never becomes a terminal gas-balance demand.
- PV and WT planning inputs use the frozen persistence forecast. Realized future PV and WT are used only for supervised labels, realized settlement, and Perfect-Information-MPC.

## 4. Frozen Gate 2 Budget

The final contract records the following values and exposes no command-line override for them:

- formal training windows: every eligible causal window from the complete 2015--2018 chronology, recorded in one shared immutable manifest;
- Gate 1 calibration origins: 1,000 pre-registered representative origins from 2019;
- Gate 2 selection evaluation: every eligible chronological rolling origin from 2019, using the checkpoint selected by the frozen Gate 1 rule;
- batch size: 64;
- Stage P maximum epochs: 30;
- Stage S maximum epochs: 30;
- Stage J maximum epochs: 30;
- minimum Stage J epochs: 18;
- early-stopping patience: 5;
- validation interval: every epoch;
- Gate 2 stochastic seeds: 2026, 2027, and 2028;
- deterministic references: one execution each;
- Gate 1 candidate parameter: Stage J forecaster learning-rate multiplier;
- Gate 1 candidate values: 1.0, 1.25, 1.5, 2.0, 2.5, and 3.0.

All trainable comparison methods consume the same frozen training-window identities and calibration-origin identities. All methods are evaluated on the same complete eligible 2019 chronology. Method-specific optimization mechanics may differ, but they may not receive extra data, evaluation-year access, additional seed searches, or an unregistered checkpoint-selection opportunity.

Differentiable-LP receives the same complete training manifest, number of sample exposures, maximum epoch count, validation schedule, early-stopping rule, and seed set. It may split a batch of 64 into deterministic micro-batches only when required by solver memory; gradient accumulation must preserve the effective batch of 64 and the receipt must record the micro-batch size. Its receipt additionally records differentiable training solves, inference solves, failed solves, gradient finiteness, peak memory, runtime, and the exact CVXPYlayers environment identity.

## 5. Method Matrix and Training Semantics

Gate 2 produces exactly 23 rows.

### 5.1 Stochastic methods: three seeds each

1. **RSC-PF** executes Stage P, same-information teacher generation, Stage S, and joint Stage J. The realized dispatch loss updates both forecaster and scheduler during Stage J.
2. **Decoupled-RSC-PF** starts from the byte-identical Stage S checkpoint used by RSC-PF. During Stage J the forecaster is frozen and the scheduler remains trainable. This gradient boundary is the only intended difference from RSC-PF.
3. **Direct-Policy** maps the same causal histories and exogenous scheduling context directly to controls. It has no forecast head and no forecast loss. It uses a dedicated decision/imitation training loop rather than the RSC-PF Stage P routine.
4. **Scheme2R-PTO** trains the registered Scheme2R forecaster on the frozen training windows and invokes the canonical exact online LP once per rolling origin.
5. **State-Conditioned-PTO** loads the seed-matched Stage P forecasting checkpoint and invokes the same canonical exact online LP once per rolling origin.
6. **Official iTransformer-PTO** uses the frozen THUML iTransformer backbone commit recorded by Gate 0 through the disclosed four-task adaptation, trains on the same data boundary, and invokes the same canonical exact online LP once per rolling origin. It is labeled an adaptation, not an exact reproduction of the source paper's full experimental pipeline.
7. **Differentiable-LP** trains its forecasting pathway through the registered CVXPYlayers optimization layer and records both differentiable training solves and its optimizer-at-inference requirement. Its reported online-optimizer count is the actual number of optimization-layer solves used to produce a rolling plan.

### 5.2 Deterministic references: one execution each

1. **Seasonal-Naive-PTO** uses the registered deterministic seasonal forecast and the canonical exact online LP.
2. **Perfect-Information-MPC** uses realized future demand and renewable availability. It is explicitly non-deployable and is reported only as a reference.

No method row may be produced from an untrained randomly initialized neural model. Every stochastic row must reference a seed-specific checkpoint whose lineage resolves to the frozen contract, data, normalization, source, parent checkpoint, and training receipt.

## 6. Execution Architecture

The Gate 2 command becomes an orchestrator rather than a JSON-only validator. It performs the following sequence:

1. validate the Gate 1 transition, contract, source manifest, data, normalization, capacity, benchmark, and checkpoint hashes;
2. materialize or verify the frozen train and selection window manifests;
3. execute shared seed-specific Stage P and Stage S work where the protocol requires byte-identical ancestry;
4. execute each stochastic method and seed in an isolated method directory;
5. execute both deterministic references once;
6. save checkpoints and chronological first-step rollout arrays before computing summaries;
7. compute all forecast, dispatch, physical, latency, and optimizer-call metrics from the saved arrays;
8. write one immutable row receipt per method and seed;
9. assemble `gate2_rows.json` only from validated row receipts;
10. run the authorization decision;
11. run a separate independent audit that recomputes matrix completeness and authorization-critical comparisons from the row receipts and rollout arrays.

The orchestrator supports resume only at method/seed boundaries. A completed row is reused only when every registered lineage hash matches. A partial or mismatched row is never silently reused.

## 7. Artifact Layout

Each Gate 2 run stores artifacts under its own immutable run root:

```text
gate2/
  contract/
  manifests/
  shared/<seed>/
  methods/<method>/<seed-or-deterministic>/
    TRAINING_RECEIPT.json
    CHECKPOINT.pt
    ROLLOUT.npz
    METRICS.json
    ROW_RECEIPT.json
    FAILURE.json
  gate2_rows.json
  GATE2_DECISION.json
  GATE2_AUDIT.json
```

`FAILURE.json` is written only for a failed row. A failure receipt, missing checkpoint, missing rollout, non-finite result, hash mismatch, or incomplete matrix blocks authorization.

## 8. Authorization Rules

Gate 2 authorizes Gate 3 only when all of the following are true:

- all 23 registered rows are present and independently validated;
- every stochastic row has a trained seed-specific checkpoint and non-empty training receipt;
- RSC-PF and Decoupled-RSC-PF share the required byte-identical Stage S ancestor for each seed;
- RSC-PF has a nonzero decision-to-forecaster gradient and Decoupled-RSC-PF has a zero decision-to-forecaster gradient;
- all forecast guardrails pass for methods with an explicit forecast output; Direct-Policy and Perfect-Information-MPC are marked not applicable rather than assigned fabricated forecast scores;
- all physical metrics are finite and within the frozen tolerances;
- optimizer-call identities match the method definitions;
- RSC-PF mean rolling penalized objective is lower than Decoupled-RSC-PF and at least two of three seed directions are favorable;
- RSC-PF shortage energy is no greater than 1.05 times the seed-matched State-Conditioned-PTO shortage energy;
- neither 2020 nor 2021 was accessed;
- the independent audit agrees with the primary decision.

Thresholds and budgets are not changed after seeing Gate 2 results. A scientific failure produces a failure decision and stops execution.

## 9. Error Handling and Resource Safety

- Each method/seed executes transactionally in a new directory.
- Exceptions produce a typed failure receipt containing the method, seed, stage, exception class, message, source hash, contract hash, data hash, and elapsed time.
- Out-of-memory, non-finite gradients, infeasible differentiable solves, missing official-source code, or incorrect checkpoint ancestry are hard failures.
- The parent orchestrator stops on the first hard failure and never fabricates the remaining rows.
- CPU thread settings, package versions, device, peak memory, and wall-clock duration are recorded.
- Existing formal-v4.2 run directories are never overwritten.
- Gate 0 rejects the final run when the projected complete Gate 2 wall-clock time exceeds the frozen 24-hour resource envelope or available disk margin. The training data or budget is not reduced after this result; such a failure stops the run for a new, explicitly reviewed protocol decision.

## 10. Verification Strategy

Before the final run, focused tests must prove:

- the contract rejects a missing or overridden Gate 2 budget;
- the Gate 2 CLI exposes no budget-changing arguments;
- every stochastic adapter rejects missing, random, or mismatched checkpoints;
- Direct-Policy uses its dedicated training objective;
- Official iTransformer-PTO executes the frozen upstream backbone adapter;
- Differentiable-LP produces finite nonzero gradients through CVXPYlayers;
- all trainable methods consume the same complete 2015--2018 train manifest and the same Gate 1 calibration manifest, and all methods use the same complete eligible 2019 evaluation chronology;
- the common evaluator advances only the realized first step;
- row receipts are computed from persisted rollouts rather than caller-supplied summaries;
- resume accepts exact completed rows and rejects mismatched rows;
- one missing or failed row blocks authorization;
- any 2020 access before Gate 3 fails closed;
- the independent audit recomputes, rather than trusts, critical fields.

A bounded non-paper smoke run then executes one small train-only method path for every method family. The smoke configuration is compiled into tests and cannot be selected by the formal Gate 2 command.

## 11. Final Execution Sequence

After code, tests, dependencies, and the contract are frozen, a new formal-v4.2 run identifier is created. The sequence is:

1. Gate 0;
2. Engineering Pilot;
3. Gate 1;
4. Gate 2 full execution;
5. independent Gate 2 audit;
6. stop and report the result.

No later stage starts after a failed stage. A successful Gate 2 does not automatically start Gate 3.

## 12. Completion Criteria

This change is complete only when:

1. the frozen contract contains every Gate 2 and Differentiable-LP budget;
2. focused and regression tests pass in the isolated formal-v4.2 environment;
3. the bounded multi-family smoke test passes;
4. the final run produces a continuous Gate 0-to-Gate 2 lineage;
5. Gate 2 produces real training artifacts and exactly 23 validated rows, or stops with an immutable failure receipt;
6. the independent audit agrees with the Gate 2 decision;
7. no 2020 data were accessed;
8. execution stops after Gate 2 for result review.
