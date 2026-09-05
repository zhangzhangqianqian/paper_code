# RSC-PF Matched Closed-Loop Evaluation Design

## Goal

Produce a matched 2019 closed-loop **protocol-validation diagnostic** for the rejected/unauthorized RSC-PF v4.6-b pilot checkpoint and the three frozen external baselines without retraining any model or accessing the sealed 2020 evaluation split.

This run validates the evaluation machinery and provides directional evidence only. The v4.6-b pilot receipt has `authorized_gate1: false` because its electricity-WAPE ratio is `1.0353263112927777`, above the predeclared `1.02` threshold. The checkpoint may therefore be used for numerical reconstruction and protocol diagnosis, but it is not an accepted model candidate and cannot authorize Gate 2. This is not the final paper comparison because the current RSC-PF artifact contains one seed while each external method contains five seeds, 2019 is the formal selection year rather than the held-out evaluation year, and the current roster omits the preregistered internal comparators needed to isolate the contribution of joint training.

The evaluation must correct two remaining ambiguities:

1. all methods must execute only the first action of each four-hour plan and carry the resulting physical state into the next hourly origin;
2. physical-constraint feasibility and supply adequacy must be reported as separate quantities.

## Considered approaches

### A. Frozen-checkpoint chronological state-carry evaluation — selected

Reuse the existing validation-selected checkpoints. Start every method from the same first 2019 state, execute one action per hour through the canonical physical settlement operator, append the realized load/exogenous information and settled device state to the 24-hour histories, and carry SOC and previous CHP output to the next origin.

This isolates evaluation fairness from training changes and preserves all existing checkpoint hashes. Runtime is not assumed in advance: an audited smoke run provides the projection before the full diagnostic is allowed. It freezes the evaluation protocol, not the final experimental result.

### B. Independent-window first-step evaluation — rejected as final evidence

Evaluate every origin using the materialized SOC and previous CHP stored in that origin. This is useful as a diagnostic and has already been completed, but it does not enforce one fully carried state consistently across model inputs, settlement, and the next origin.

### C. Retrain every baseline with closed-loop roll-ins — deferred

Generate model-owned roll-in histories during training and retrain all baselines. This would answer a different research question and substantially expand the experiment budget. It is not required to determine whether the already frozen methods remain competitive under a common deployment rollout.

## Frozen inputs and models

- Evaluation chronology: all 8,709 hourly origins in the predeclared 2019 `selection_full.npz` artifact.
- RSC-PF diagnostic reference: the existing single-seed (2026) v4.6 `rsc_pf_joint.npz` rollout and frozen `J_joint.pt` checkpoint, explicitly labeled `gate1_authorized: false` and `use: protocol_diagnostic_only`.
- External checkpoints: the five validation-selected seeds for iTransformer-PTO, DecisionFocused-Online, and DigitalTwins-Policy.
- Normalization: fitted from the v4.6 training split only.
- Formal split boundary: 2015–2018 training, 2019 selection/protocol validation, 2020 sealed evaluation, and 2021 excluded.
- Sealed 2020 evaluation data: prohibited until the matched protocol is sound, the RSC-PF candidate satisfies the unchanged predeclared Gate 1 criteria, the complete five-seed/internal/external roster is frozen, and Gate 2 explicitly authorizes access.
- Model training, checkpoint selection, architecture, losses, and paper text: unchanged.

## Common chronological data flow

At the first 2019 origin, construct a closed-loop state from the first window's 24-hour load, exogenous, device-output, and activity histories together with its initial SOC and previous CHP output.

Each origin is split into two structurally disjoint objects before any model code runs. `CausalOriginInput` contains only deployment-available histories, future PV/WT forecasts, prices/carbon context, the timestamp, and the carried SOC/previous-CHP state. `RealizedOriginLabels` contains the future true four-task targets, realized PV/WT, and the newly revealed observation required to advance the state. An action provider receives only `CausalOriginInput`; only the runner may pass `RealizedOriginLabels` to settlement, metric calculation, and the Perfect-Information-MPC reference. Dummy target tensors needed by a third-party inference wrapper must be created inside that wrapper and must never be populated from realized labels.

For each hourly origin:

1. split the materialized row into causal input and realized labels, then build the method input only from the causal object; in particular, the scheduler-context SOC and previous-CHP values must come from the carried state rather than the materialized origin row;
2. produce a four-hour forecast and/or dispatch plan;
3. for PTO methods, solve one four-hour LP using the current carried SOC and previous CHP output;
4. execute only the first planned action through `settle_first_step_v4` against realized first-hour electric/cooling/heating demand and PV/WT output;
5. record both the planned action and settled action, the physical correction applied by settlement, shortage, cost, carbon, objective, and residual families;
6. append the settled observable dispatch and activity indicators, append the newly revealed load/exogenous observation, and carry the resulting SOC and CHP output into the next origin.

Separately, run one deterministic Perfect-Information-MPC reference trajectory from the same initial state. At each origin it receives the realized four-hour electric/cooling/heating demand and PV/WT availability, solves one four-hour LP, executes only its first action through the same settlement function, and carries its own state. This reference is not a deployable competitor and is not duplicated for every model seed.

The input timestamps must be strictly consecutive at one-hour intervals. Any gap, non-finite tensor, failed LP, state-hash mismatch, or sealed-test-like path fails the run closed.

## Method-specific behavior

- **RSC-PF:** use the existing v4.6 checkpoint as the single-seed diagnostic reference and rerun it through the common chronological evaluator so latency has the same scope as the baselines. It makes zero inference-time LP calls. Before the matched run, reconstruct the checkpoint and reproduce the legacy `rsc_pf_joint.npz` through the unchanged legacy evaluator. That replay validates checkpoint loading only. The matched evaluator deliberately replaces the legacy materialized scheduler-context SOC with the actual carried SOC, so only the first origin is required to match the legacy rollout; later matched outputs are expected to differ and are treated as the corrected protocol result.
- **iTransformer-PTO:** precompute its load forecasts from causal load/exogenous histories, but solve one exact LP per origin sequentially because SOC and previous CHP output are carried. The full PTO method therefore uses an online optimizer at inference even though the neural forecaster itself does not call one.
- **DecisionFocused-Online:** use its frozen decision-focused forecast checkpoint and solve one exact LP per origin sequentially at inference.
- **DigitalTwins-Policy:** run the policy sequentially because its dispatch depends on current SOC and previous CHP output; it must make zero inference-time LP calls.

Although all methods receive the same available state container, a baseline may ignore fields that are not part of its declared architecture. This is an architectural distinction, not an evaluation-protocol mismatch.

## Metric definitions

The fourth prediction task is the normalized station-side gas-consumption auxiliary prior. It is not a rigid terminal gas demand. Its MAE/RMSE/WAPE may be reported, but it is excluded from the electricity/cooling/heating balance equations, shortage energy, and no-shortage rate. Tables and figures must label it as a gas prior rather than an end-user gas load. Changing only the predicted gas channel must leave an LP dispatch plan unchanged.

Report the following separately:

- forecast MAE, RMSE, and WAPE by task and horizon plus the macro task-by-horizon mean;
- for cooling and heating, truth-active MAE/RMSE/WAPE and normalized truth-inactive leakage, both by horizon and overall. Active means the realized target exceeds the frozen `1e-9` threshold. Inactive leakage is the mean absolute predicted value on truth-inactive points divided by that carrier's mean absolute truth on active points in the 2015–2018 training split; the denominator is frozen before 2019 is opened. Report P95 normalized leakage as well;
- settled first-step operating cost, physical carbon, penalized objective, and electricity/cooling/heating shortage energy;
- **planned-action physical feasibility:** residual families evaluated before realized-data settlement against the method's own declared scheduling demand and renewable forecast (nominal forecast plus risk adjustment for RSC-PF, forecast output for PTO methods, and policy forecast for DigitalTwins-Policy);
- **settled-action physical feasibility:** residual families evaluated after canonical first-step settlement against realized demand and renewable output;
- **recourse adjustment:** the first-step L1 distance between planned and settled dispatch, plus that distance divided by total realized electric/cooling/heating demand with an epsilon guard; report mean, median, and P95 so post-settlement feasibility cannot conceal a large correction;
- **no-shortage rate:** fraction of settled rows with zero electric, cooling, and heating shortage within tolerance;
- cumulative raw objective gap versus the separately rolled Perfect-Information-MPC reference;
- initial/final SOC and battery throughput, where throughput is the time-step-weighted sum of settled charge plus discharge power. The primary comparison uses the raw cumulative realized objective. A supplementary terminal-stock valuation sensitivity reports an adjusted objective using the frozen final-hour grid marginal value, but it is not described as an executed restoration schedule;
- inference latency and optimizer accounting.

Use the frozen tolerances consistently: a row is physically feasible only when every absolute physical residual is at most `1e-6`; a row is shortage-free only when every carrier shortage is at most `1e-8`. Do not compare RSC-PF's architecture-specific regime-head F1 with baselines that have no regime head. The separate RSC-PF legacy-replay check uses `numpy.allclose` with `atol=1e-5` and `rtol=1e-6` for saved floating-point arrays, plus exact timestamp and state-hash equality. In the corrected matched run, the first origin must meet the same equality check, while later origins are compared only under the new shared protocol.

For the supplementary terminal-stock valuation, let `E0 = initial_soc * bess_energy_capacity`, `ET = final_soc * bess_energy_capacity`, `eta = sqrt(roundtrip_efficiency)`, and `m = final_grid_price + final_carbon_price * grid_emission_factor`. The signed adjustment is `(E0-ET)/eta*m` when `ET < E0`, `-(ET-E0)*eta*m` when `ET > E0`, and zero otherwise. The sensitivity value is `J_adjusted = J_raw + adjustment`. Invalid capacity, efficiency, or non-finite inputs fail closed.

Inference LP calls and reference-only LP calls must be recorded in separate fields. The Perfect-Information-MPC solver is never counted as part of deployable inference. Both iTransformer-PTO and DecisionFocused-Online must record one online LP call per origin; RSC-PF and DigitalTwins-Policy must record zero. The shared Perfect-Information-MPC reference records exactly one reference LP call per origin. Because a four-hour rolling MPC is not a globally optimal year-long controller, the signed difference is called an objective gap rather than guaranteed non-negative regret.

Latency is a secondary engineering diagnostic, measured in chronological batch-one execution on the same machine. It includes the neural forward pass, the online LP when declared by the complete method, and canonical physical settlement. It excludes artifact loading, metric aggregation, and the separate Perfect-Information-MPC reference. Freeze and record device, dtype, method execution order, batch size, PyTorch intra/inter-op thread counts, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, solver method/options, host hardware, and Python/PyTorch/SciPy/HiGHS versions. If CUDA is used, synchronize immediately before and after timing. The first 100 chronological origins are executed normally but excluded from latency aggregation as warm-up. Report median and P95 milliseconds per origin and total runtime; do not treat these measurements as a final publication-grade speed claim.

## Outputs and audit

Each method/seed writes a closed-loop NPZ artifact and JSON receipt containing checkpoint hash, input artifact hash, timestamp hash, frozen training-only thermal scales, metric definitions, optimizer role, inference LP calls, planned and settled residual maxima, recourse-adjustment statistics, raw cumulative objective, terminal-stock valuation sensitivity, initial/final SOC, battery throughput, latency controls/statistics, `test_set_accessed: false`, and `evaluation_year_accessed: false`. A separate reference artifact records the Perfect-Information-MPC trajectory and its reference-only LP calls.

The RSC-PF provenance directory also contains a legacy-replay receipt. It must show full equivalence to the saved rollout and separately record that the legacy evaluator used materialized scheduler-context SOC. The matched receipt must show that carried SOC was used and must never claim full equivalence to the legacy rollout.

An aggregate comparison manifest must verify:

- exactly 8,709 aligned 2019 origins for every row;
- unchanged checkpoint hashes;
- one complete diagnostic seed for RSC-PF and five complete seeds for every external method, with the imbalance stated explicitly and no significance test across this pilot table;
- zero failed inference LP solves;
- zero non-finite values;
- explicit separation of planned feasibility, settled feasibility, recourse adjustment, and no-shortage rate;
- correct online-optimizer roles for all four complete methods;
- exactly one 8,709-origin Perfect-Information-MPC reference trajectory from the common initial state;
- explicit station-side gas-prior semantics;
- structural confirmation that deployable providers received no realized future labels;
- preserved v4.6-b status `gate1_authorized: false` and `formal_candidate: false`;
- zero 2020 evaluation access.

The existing independent-window receipts remain labeled as diagnostics and are not overwritten or used as the final paper table.

## Success criterion and next gate

This task succeeds when the 2019 common closed-loop manifest passes all audit checks and a diagnostic review table compares the methods under the unified metric definitions. No new performance threshold is invented during execution, and this task does not produce a final paper table or prove the contribution of joint training.

If the protocol is sound, freeze the evaluator but do not authorize Gate 2. First decide whether to repair/retrain the RSC-PF candidate under the existing preregistered Gate 1 threshold; do not relax that threshold after seeing 2019 results. Then complete five RSC-PF seeds, restore the complete preregistered internal/external baseline roster, freeze the same five seed identities for every stochastic method, and rerun Gate 1. Only a passing Gate 1 can authorize the one-time Gate 2 evaluation on 2020. The already trained external checkpoints may be reused when their hashes, split boundaries, and selection rules pass audit; they are not retrained merely for symmetry. Only matched held-out results may support the final main table and inferential statistics. If the 2019 diagnostic reveals a protocol failure or removes the apparent decision advantage, stop and diagnose it. Ablations and manuscript modification remain out of scope for this task.
