# RSC-PF Formal-v4.4 Real Pilot Executor Design

## 1. Objective and Scope

Complete the missing production executor behind `run_pilot_v44` so the frozen
formal-v4.4 Pilot performs genuine data materialization, offline LP teacher
generation, P0/P1/S/J training, full-chronology 2019 rolling evaluation, and
independent artifact-based authorization.

This change implements the already frozen formal-v4.4 scientific protocol. It
does not change the model architecture, gas semantics, candidate grid, Pilot
thresholds, data years, method identities, or Gate transition rules. It does
not start Gate 1 automatically.

## 2. Version and Evidence Boundary

Formal-v4.2 remains reusable engineering infrastructure, not formal-v4.4
scientific evidence. Generic utilities for standard-IES trajectory generation,
same-information LP solving, physical decoding, and causal state advancement
may be reused only through formal-v4.4 adapters with explicit input validation
and implementation hashes.

Formal-v4.3 remains a failed diagnostic branch. Its Pilot receipts and any
constant authorization fields are never imported or copied into formal-v4.4.

The existing formal-v4.4 Gate 0 receipt was generated from a formal-v4.2 source
manifest whose recorded Git commit predates the real executor. After the
executor is implemented and tested, a fresh formal-v4.4 source manifest and a
new run ID are required. Gate 0 must be rerun as a short engineering preflight
before the real Pilot is started.

## 3. Selected Architecture

Use a modular real executor rather than a monolithic wrapper or a subprocess
call to the old Pilot. The executor is an orchestration layer over five focused
components:

1. causal Pilot data preparation;
2. immutable same-information LP teacher generation;
3. matched P0/P1/S/J training;
4. full-chronology 2019 rolling evaluation;
5. auditable artifact and resume management.

The public callable has the interface expected by `run_pilot_v44`:

```python
def execute_real_pilot_v44(
    *,
    train_data: str | Path,
    selection_data: str | Path,
    benchmark: str | Path,
    capacity_receipt: str | Path,
    split: Mapping[str, np.ndarray],
    contract: FormalV44Contract,
    artifact_root: str | Path,
) -> Mapping[str, Any]:
    ...
```

The command-line Pilot entry point constructs this executor explicitly and
passes a run-scoped `pilot/` artifact directory. Tests may inject a small
deterministic executor, but the production command must never accept a missing
or synthetic executor.

## 4. Data Preparation

The executor reads the explicit Gate-0-bound base archives only:

- training years: 2015--2018;
- Pilot selection chronology: 2019;
- sealed evaluation year: 2020, never opened before Gate 2.

All normalization parameters, thermal transition priors, class weights, and
active-magnitude statistics are fitted on the complete 2015--2018 training
split. The fixed Gate 0 indices select exactly 4,096 Pilot training windows;
subsampling must not refit any statistic.

The standard-IES simulator produces continuous, chronologically settled device
trajectories. Each materialized window contains:

- 24 hours of four-task load history and 12 exogenous features;
- 24 hours of 17 continuous device-state features;
- 24 hours of six derived device activity indicators;
- four-hour renewable forecasts and realized renewable values;
- four-hour electricity price, gas price, and carbon-weight context;
- current BESS SOC and previous-hour CHP output;
- four-hour supervised load targets.

Training and early-stopping windows are time-disjoint after a purge of at least
28 hours. The complete 2019 chronology is retained in timestamp order for Pilot
evaluation. Simultaneously positive cooling and heating targets fail closed.

## 5. Same-Information LP Teacher

Stage P1 predictions are generated for the selected 4,096 training windows.
The offline LP teacher receives only information available to the deployed
pipeline: P1 load forecasts, renewable forecasts, prices and carbon weight,
current SOC, previous CHP output, and the frozen standard-IES parameters.

Future realized electricity, cooling, or heating targets are not teacher
inputs. They remain supervised labels and later rolling-settlement observations.
Gas remains a supervised station-side auxiliary prior and is not introduced as
a rigid terminal balance.

Teacher output has shape `[4096, 4, 21]`. Each row records solver status,
objective value, shortage, state hash, forecast checkpoint hash, benchmark
hash, capacity hash, timestamp, and input hash. A non-finite or failed LP row
aborts the Pilot. Teacher artifacts are written incrementally and may resume
only when every lineage hash matches.

## 6. Training Execution

Training follows the frozen one-seed Pilot budget and stage order:

1. **P0:** train the common four-task Scheme2R parent.
2. **P1:** attach and train the residual thermal gate and magnitude residuals.
3. **Continuous control:** train the matched continuous-head mechanism control
   from the identical P0 parent and with matched forecast optimizer steps.
4. **S:** freeze the P1 forecast path and train the scheduling proxy by
   21-dimensional decoded-schedule imitation.
5. **J branches:** clone the same P1 and S parents, then train RSC-PF Joint and
   Fair Decoupled with identical batches, epoch budgets, and optimizer-step
   counts.

RSC-PF decision loss must update the Scheme2R base, residual gate, magnitude
residuals, and scheduler. Fair Decoupled decision loss must have gradient norm
at most `1e-12` in all forecast components while its supervised forecast loss
still updates them. The transition-prior-only diagnostic requires no trainable
forecast residual.

Each stage saves its checkpoint, optimizer state, epoch losses, learning rates,
optimizer-step count, stopping decision, seed, parent hashes, and termination
state before the next stage starts.

## 7. Full-Chronology 2019 Rolling Evaluation

The evaluator processes all valid 2019 origins chronologically. At every hour
it:

1. forms inputs from the currently available 24-hour history;
2. produces a four-hour load forecast and schedule;
3. settles only the first control interval against realized rigid demand and
   renewable availability;
4. recomputes physical balance, shortage, capacity, SOC, and CHP-ramp
   residuals;
5. carries the settled dispatch, activity state, SOC, and CHP output into the
   next origin.

The evaluator stores raw prediction, target, regime probability, prior
probability, regime label, planned dispatch, settled dispatch, shortage,
physical residual, state, and timestamp arrays. Pilot comparison metrics are
computed from these stored arrays with uniform full-chronology weights. The
transition/active stress view is secondary and cannot authorize a model by
itself.

## 8. Artifact Layout and Resumption

The run directory remains immutable and run-scoped:

```text
<run>/
  gate0/
  pilot/
    data/
    teacher/
    stages/P0/
    stages/P1/
    stages/continuous_control/
    stages/S/
    stages/J_joint/
    stages/J_decoupled/
    rollout/
    PILOT_ARRAYS.npz
    PILOT_RECEIPT.json
    PILOT_AUDIT.json
  PILOT_TRANSITION.json
```

Long-running stages use atomic temporary files followed by rename. A rerun may
resume a completed stage only if contract, source manifest, data, split,
normalization, benchmark, capacity, implementation, parent-checkpoint, and seed
hashes all match. Any mismatch fails closed and requires a new run ID. Existing
valid artifacts are never overwritten.

## 9. Authorization and Independent Audit

`run_pilot_v44` receives measured executor outputs and writes the canonical
Pilot arrays and receipt. No comparison, gradient, physical-feasibility, or
success value may be a constant placeholder.

The independent audit runs in a fresh process and reconstructs all
authorization-critical values from persisted arrays and metadata. It verifies:

- the five frozen Pilot rows are present;
- the accessed years are exactly 2015--2019;
- 2020 was not opened;
- all lineage hashes match;
- all metrics and objectives are finite;
- leakage, active accuracy, unaffected-task, overall-score, regime,
  decision, gradient-boundary, and physical-residual criteria agree with the
  frozen contract;
- the saved authorization boolean equals the independently recomputed result.

Any missing artifact, hash mismatch, incomplete chronology, duplicate origin,
non-finite value, failed LP row, or criterion disagreement produces
`authorized_gate1=false`.

## 10. Testing Strategy

Testing proceeds from small deterministic fixtures to the real bounded Pilot:

1. unit tests for version-bound input loading, causal materialization, purge,
   teacher inputs, resume hashes, and atomic writes;
2. integration tests that execute P0/P1/S/J and a short chronological rollout
   on deterministic small data;
3. gradient tests for each forecast component and the scheduler in Joint and
   Fair Decoupled modes;
4. adversarial tests for future-target teacher leakage, 2020 access, hash
   mismatch, incomplete LP output, duplicate chronology, hard-coded success,
   and corrupted resume artifacts;
5. all formal-v4.4 and protected formal-v4.2 regression tests;
6. a fresh Gate 0 run tied to the final implementation commit;
7. the real 4,096-window Pilot followed by the independent audit.

The process stops after Pilot and audit regardless of outcome. Gate 1 starts
only after the user reviews an independently verified
`authorized_gate1=true` transition.

## 11. Success Definition

The upgrade is complete when the production Pilot command, without injected
test fixtures, can generate causal v4.4 inputs, solve and persist the full
same-information teacher, train all five frozen Pilot rows, roll the complete
2019 chronology, write independently recomputable artifacts, and produce a
fail-closed Pilot decision under a fresh, implementation-bound Gate 0 lineage.

Completion of the executor does not imply that the scientific Pilot passes.
The measured Pilot outcome remains an empirical result.
