# RSC-PF Formal v4.1 Gate 0 Evidence Closure Design

## 1. Purpose

This design closes the twelve currently failing Gate 0 checks without starting Gate 1, neural-network training, baseline training, ablations, or 2020 evaluation. Gate 0 remains an authorization boundary: it may prepare and validate protocols, data lineage, small disposable probes, and resource estimates, but it must not create performance results.

The latest independent audit already confirms that the existing Gate 0 run is structurally clean and contains no forbidden downstream artifacts. Its only blocking condition is the absence of eleven physical receipt files that satisfy twelve checks because `DATA_ACCESS_RECEIPT.json` is used by both `data_access` and `no_evaluation_access`.

## 2. Fixed Scientific Boundary

- Training years: 2015–2018.
- Model-selection year: 2019.
- Locked evaluation year: 2020; no Gate 0 code may read, hash, materialize, or inspect its archive members.
- Excluded year: 2021.
- Lookback and horizon: 24 historical hours and 4 forecast/dispatch hours.
- Forecast tasks: electricity, cooling, heating, and station-side aggregate gas prior.
- The gas series is auxiliary standardized context, not a rigid terminal gas-demand balance.
- PV and WT use the frozen persistence rule already adopted by formal v4.1.
- Device representation: 17 continuous controls plus 6 activity indicators; final physical output has 21 dimensions and excludes slack/dump variables.
- State encoder: DS-TCN.
- Capacity multiplier: the certified result is currently 2.7. A clean lineage rerun must reproduce it; otherwise execution stops for audit rather than silently changing the frozen benchmark.
- Gate 0 performs no formal Stage P, Stage S, or Stage J training.

## 3. Chosen Architecture

Three approaches were considered:

1. **Manual receipt creation.** Rejected because it can produce formally valid JSON without proving that the claimed checks were executed.
2. **Self-generating preflight.** Rejected because combining evidence production with authorization validation makes it difficult to detect hidden data reads or accidental training.
3. **Staged evidence producers followed by a pure validator.** Selected. Dedicated producers create typed, immutable receipts in a fresh staging run. A separate fresh Gate 0 run imports those receipts and only validates them. Any failed producer or validator leaves `authorized_gate1=false` and prevents downstream execution.

The evidence path is therefore:

```text
fixed source + frozen contracts
        |
        v
fresh Gate 0 evidence staging root
  - benchmark/capacity/data lineage
  - protocol probes
  - source and dependency gates
  - resource projection
        |
        v
fresh pure-validation Gate 0 root
        |
        v
independent audit
        |
        +-- pass: Gate 1 is eligible but not started
        +-- fail: stop with exact blocking receipt
```

## 4. Evidence Integrity Rules

Every receipt must:

- use an exact formal-v4.1 schema version;
- be written atomically and refuse overwrite;
- include hashes of every upstream artifact on which its claims depend;
- record the source commit and protocol identifier;
- include `test_set_accessed=false` where data access is possible;
- be validated by content-aware typed validation, not by schema-name checks alone;
- distinguish a disposable Gate 0 probe from a production checkpoint, overlay, or performance result;
- remain inside its run root so a failed run cannot contaminate a later run.

The final preflight is read-only with respect to scientific evidence. It may copy already-frozen receipts into a new audit root, calculate hashes, and emit the authorization report, but it may not repair, regenerate, or reinterpret failed evidence.

## 5. Receipt Contracts

### 5.1 Trajectory receipt

`protocol/TRAJECTORY_RECEIPT.json` binds the selected capacity receipt, generated benchmark, train and selection archives, trajectory identifier, trajectory hash, settlement rule, and physical audit. It must prove:

- no future-label reads;
- positive solved and settled hour counts;
- maximum balance, conversion, SOC-recursion, CHP-ramp, renewable-availability, and SOC-bound violations no greater than `1e-6`;
- the exact formal-v4.1 causal realized-settlement rule;
- hashes matching the materialized dataset.

### 5.2 C-ref receipt

`C_REF_RECEIPT.json` is fitted from all and only the 2015–2018 training windows using the frozen perfect-information LP objective. It records the sample count, objective implementation hash, train archive hash, benchmark/capacity hash, median with floor 1, and first-step weights `[0.5, 1/6, 1/6, 1/6]`. This is an objective normalization artifact, not model training.

### 5.3 Teacher-alignment receipt

`protocol/TEACHER_ALIGNMENT_RECEIPT.json` is a 100-window train-only mechanism probe. It verifies timestamps, current state, information-set alignment, LP feasibility, and output shape. Because Stage P does not exist before Gate 1, it must explicitly record `probe_only=true` and `production_overlay_deferred_until_stage_p=true`; it must not invent a Stage P checkpoint hash. Production same-information overlays remain forbidden until an actual Stage P checkpoint exists.

### 5.4 Curriculum receipt

`protocol/CURRICULUM_RECEIPT.json` evaluates the fixed early, middle, and late weight schedule without updating parameters. It proves finite nonnegative weights, the exact four-step aggregation weights, gas forecast weight 0.25, rigid-task weights 1.0, and a strictly positive decision-loss weight from the first Stage J epoch.

### 5.5 Gradient-boundary receipt

`protocol/GRADIENT_RECEIPT.json` uses disposable identically initialized models and one fixed train-only mini-batch. It must prove:

- identical initial model hashes;
- finite nonzero scheduler gradients for both joint and decoupled paths;
- finite nonzero decision-loss gradient reaching the RSC-PF forecaster;
- zero decision-loss gradient reaching the Decoupled-RSC-PF forecaster;
- `probe_only=true` and no checkpoint creation.

### 5.6 Method-adapter receipt

`protocol/METHOD_ADAPTER_RECEIPT.json` covers the complete nine-method matrix:

- RSC-PF;
- Decoupled-RSC-PF;
- Direct-Policy;
- Scheme2R-PTO;
- State-Conditioned-PTO;
- Official iTransformer-PTO;
- Differentiable-LP;
- Perfect-Information-MPC;
- Seasonal-Naive-PTO.

It records role, deployability, forecast availability, expected forecast and dispatch shapes, online-optimizer call policy, and executable probe results. Gate 0 validates interfaces, not accuracy. Perfect-Information-MPC must be marked nondeployable, and Direct-Policy must mark forecast metrics as not applicable.

### 5.7 Official iTransformer receipt

`protocol/ITRANSFORMER_SOURCE_RECEIPT.json` binds the local THUML source checkout to its repository URL, commit, license file, imported-file hashes, and `model.iTransformer.Model` backbone. The existing local checkout may be used only if its commit and hashes verify. Gate 0 must not silently download or substitute another implementation.

### 5.8 Differentiable-LP gate

`protocol/DIFFERENTIABLE_LP_GATE.json` is produced only by the isolated differentiable-optimization environment. It uses the current run-root benchmark and fixed train-only materialization, not a legacy benchmark. Eligibility requires DPP compliance, 100/100 finite probes, objective parity relative gap at most `1e-4`, physical residual at most `1e-6`, finite nonzero gradients, a successful native probe, at least 20% disk and memory margin, and projected runtime no greater than 24 hours.

### 5.9 Data-access and archive receipts

`protocol/DATA_ACCESS_RECEIPT.json` records separate train and selection access events. It must show 2015–2018 for training, 2019 for selection, an explicit denied 2020 request, and `test_set_accessed=false` without reading 2020 contents.

`protocol/ARCHIVE_ACCESS_RECEIPT.json` records each ZIP container hash plus exact member names and member hashes for accessed years. No 2020 or 2021 member may appear. The canonical loader emits these events at the actual read boundary so the receipt cannot be reconstructed from assumptions after the fact.

### 5.10 Resource projection

`protocol/RESOURCE_PROJECTION.json` benchmarks 500 representative train-only operations and reports method-level throughput, memory, disk, and projected wall time. The validator checks every required method and the worst-case eligible path, with at least 20% free memory/disk and a projected runtime no greater than 24 hours.

## 6. Data-Lineage Repair

The current capacity root records 2015–2019 in one base read. Although capacity selection itself used only 2015–2018, that mixed receipt is not strong enough for the final access claim. The final staging run therefore performs distinct canonical reads for:

- the 2015–2018 training split used by benchmark scaling, capacity certification, C-ref, and Gate 0 probes;
- the 2019 selection split used only for frozen selection materialization.

The loader receives an optional audit sink. With no sink, all legacy callers behave exactly as before. With a sink, every container/member read emits a structured event before data are returned.

## 7. Failure Semantics

- A failed producer writes a diagnostic receipt where safe, but never an authorization marker.
- A hash mismatch, split violation, 2020 access, failed physics residual, missing official source, ineligible Differentiable-LP gate, or resource overrun stops the orchestrator immediately.
- If the clean capacity rerun selects anything other than 2.7, execution stops for scientific review.
- Missing isolated-environment dependencies are reported as a blocking condition; the main PyTorch environment is never modified.
- Previous failed and partial run roots are preserved for audit and are never reused.

## 8. Completion Criteria

Gate 0 evidence closure is complete only when all of the following are true:

1. all targeted and full regression tests pass;
2. a fresh evidence staging root contains all eleven immutable receipt files;
3. a separate fresh preflight root reports all twelve checks as passed;
4. `GATE0_AUTHORIZATION.json` states `authorized_gate1=true`;
5. the independent auditor reports `verified=true`, `authorized_gate1=true`, and no forbidden artifacts;
6. no Stage P/S/J checkpoint, evaluation result, baseline result, ablation result, or 2020-derived artifact exists in either root;
7. execution stops before Gate 1.

