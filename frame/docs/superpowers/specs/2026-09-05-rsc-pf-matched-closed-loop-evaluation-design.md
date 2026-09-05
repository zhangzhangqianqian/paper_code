# RSC-PF Matched Closed-Loop Evaluation Design

## Goal

Produce a scientifically comparable 2019 closed-loop evaluation for the frozen RSC-PF model and the three frozen external baselines without retraining any model or accessing the sealed test split.

The evaluation must correct two remaining ambiguities:

1. all methods must execute only the first action of each four-hour plan and carry the resulting physical state into the next hourly origin;
2. physical-constraint feasibility and supply adequacy must be reported as separate quantities.

## Considered approaches

### A. Frozen-checkpoint chronological state-carry evaluation — selected

Reuse the existing validation-selected checkpoints. Start every method from the same first 2019 state, execute one action per hour through the canonical physical settlement operator, append the realized load/exogenous information and settled device state to the 24-hour histories, and carry SOC and previous CHP output to the next origin.

This isolates evaluation fairness from training changes, preserves all existing checkpoint hashes, and can be completed in minutes rather than hours.

### B. Independent-window first-step evaluation — rejected as final evidence

Evaluate every origin using the materialized SOC and previous CHP stored in that origin. This is useful as a diagnostic and has already been completed, but it does not reproduce the chronological state carry used by RSC-PF.

### C. Retrain every baseline with closed-loop roll-ins — deferred

Generate model-owned roll-in histories during training and retrain all baselines. This would answer a different research question and substantially expand the experiment budget. It is not required to determine whether the already frozen methods remain competitive under a common deployment rollout.

## Frozen inputs and models

- Evaluation chronology: all 8,709 hourly origins in the predeclared 2019 `selection_full.npz` artifact.
- RSC-PF reference: existing v4.6 `rsc_pf_joint.npz` rollout and frozen `J_joint.pt` checkpoint.
- External checkpoints: the five validation-selected seeds for iTransformer-PTO, DecisionFocused-Online, and DigitalTwins-Policy.
- Normalization: fitted from the v4.6 training split only.
- Sealed test data: prohibited.
- Model training, checkpoint selection, architecture, losses, and paper text: unchanged.

## Common chronological data flow

At the first 2019 origin, construct a closed-loop state from the first window's 24-hour load, exogenous, device-output, and activity histories together with its initial SOC and previous CHP output.

For each hourly origin:

1. build the method input from the current carried state and the origin's causal future context;
2. produce a four-hour forecast and/or dispatch plan;
3. for PTO methods, solve one four-hour LP using the current carried SOC and previous CHP output;
4. execute only the first planned action through `settle_first_step_v4` against realized first-hour electric/cooling/heating demand and PV/WT output;
5. record the settled action, shortage, cost, carbon, objective, and residual families;
6. append the settled observable dispatch and activity indicators, append the newly revealed load/exogenous observation, and carry the resulting SOC and CHP output into the next origin.

The input timestamps must be strictly consecutive at one-hour intervals. Any gap, non-finite tensor, failed LP, state-hash mismatch, or sealed-test-like path fails the run closed.

## Method-specific behavior

- **RSC-PF:** use the existing chronological v4.6 rollout as the reference. Recompute only the common summary fields from its saved arrays.
- **iTransformer-PTO:** precompute its load forecasts from causal load/exogenous histories, but solve its LP sequentially because SOC and previous CHP output are carried.
- **DecisionFocused-Online:** use its frozen decision-focused forecast checkpoint and solve its declared exact LP sequentially at inference.
- **DigitalTwins-Policy:** run the policy sequentially because its dispatch depends on current SOC and previous CHP output; it must make zero inference-time LP calls.

Although all methods receive the same available state container, a baseline may ignore fields that are not part of its declared architecture. This is an architectural distinction, not an evaluation-protocol mismatch.

## Metric definitions

Report the following separately:

- forecast MAE, RMSE, and WAPE by task and horizon plus the macro task-by-horizon mean;
- settled first-step operating cost, physical carbon, penalized objective, and shortage energy;
- **physical-constraint feasibility rate:** fraction of settled rows whose balance, capacity, conversion, SOC, ramp, exclusivity, renewable-accounting, and finite residuals all satisfy the formal tolerance;
- **no-shortage rate:** fraction of settled rows with zero electric, cooling, and heating shortage within tolerance;
- decision regret versus a perfect-information LP solved from that method's carried state;
- inference latency and optimizer accounting.

Inference LP calls and oracle-only evaluation LP calls must be recorded in separate fields. The oracle solver is never counted as part of deployable inference.

## Outputs and audit

Each method/seed writes a closed-loop NPZ artifact and JSON receipt containing checkpoint hash, input artifact hash, timestamp hash, metric definitions, optimizer role, inference LP calls, oracle LP calls, residual maxima, and `test_set_accessed: false`.

An aggregate comparison manifest must verify:

- exactly 8,709 aligned 2019 origins for every row;
- unchanged checkpoint hashes;
- five complete seeds for every external method;
- zero failed inference LP solves;
- zero non-finite values;
- explicit separation of physical feasibility and no-shortage rate;
- zero test-set access.

The existing independent-window receipts remain labeled as diagnostics and are not overwritten or used as the final paper table.

## Success criterion and next gate

This task succeeds when the common closed-loop manifest passes all audit checks and a review table compares the methods under the unified metric definitions. No performance threshold is imposed during execution.

If RSC-PF retains a material decision-quality advantage, freeze the main experiment and proceed to named ablations and statistical testing. If the advantage disappears, stop before ablations and diagnose the specific metric or state trajectory responsible. Manuscript modification remains out of scope for this task.

