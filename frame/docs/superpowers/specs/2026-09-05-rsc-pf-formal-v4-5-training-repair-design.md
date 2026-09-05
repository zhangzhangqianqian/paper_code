# RSC-PF Formal-v4.5 Training-Protocol Repair Design

## 1. Purpose

Formal-v4.4 produced a valid negative Pilot result: joint training substantially
improved realized dispatch objective and shortage while allowing the forecast
path to drift away from its supervised meaning, especially for cooling and
heating. Formal-v4.5 repairs the training and model-selection protocol without
changing the RSC-PF architecture, task semantics, physical decoder, data years,
or Gate-1 scientific thresholds.

The completed formal-v4.4 run and its immutable evidence remain valid and are
never overwritten. Formal-v4.5 uses new configuration, source-manifest, run,
receipt, and artifact namespaces.

## 2. Considered Approaches

Three approaches were considered:

1. **Training-protocol repair (selected):** use the existing early-stop split,
   honor the frozen J-stage parameter-group learning rates, normalize and
   schedule the joint losses, anchor forecasts to the P1 parent, and select a
   checkpoint subject to forecast guardrails.
2. **Immediate architecture redesign (deferred):** add hard thermal masks or
   replace the residual-gated head. This is more invasive and would make it
   impossible to determine whether the v4.4 failure came from optimization or
   architecture.
3. **Relax Gate-1 thresholds (rejected):** this would admit a model whose
   forecast outputs no longer retain the claimed four-task meaning.

The selected approach is the smallest scientifically interpretable correction.

## 3. Frozen Scientific Boundary

Formal-v4.5 preserves all of the following:

- 24-hour causal history and four-hour forecast/scheduling horizon;
- electricity, cooling, heating, and station-side gas-prior forecast targets;
- residual-gated Scheme2R forecast architecture;
- 15-dimensional continuous control representation and 21-dimensional decoded
  dispatch output;
- no future binary unit-commitment decisions;
- differentiable physics-feasible decoder and rolling first-step settlement;
- training years 2015--2018, Pilot selection year 2019, sealed evaluation year
  2020, and exclusion of 2021;
- fixed Gate-1 thresholds and the requirement that decision loss update the
  forecast path only in RSC-PF Joint.

No baseline matrix, ablation matrix, or large multi-seed experiment starts as
part of this repair.

## 4. Data and Validation Use

The already materialized, purged `early_stop` partition becomes an actual
validation and checkpoint-selection set. It is drawn only from 2015--2018 and
remains chronologically disjoint from the training windows. The 2019 Pilot
chronology is evaluated only after the training recipe and checkpoint have been
selected. The 2020 archive remains inaccessible.

Every stage receives explicit `train` and `early_stop` loaders. Normalization,
thermal priors, and teacher lineage continue to be fitted or generated from
allowed training information only. Validation must never update parameters.

## 5. Stage-Specific Early Stopping

The generic epoch loop is extended to evaluate a stage-specific validation
criterion after every epoch, retain the best finite checkpoint in memory, and
restore it before producing the stage receipt.

- **P0 and P1:** minimize the same supervised forecast objective used by that
  stage on `early_stop`.
- **Continuous control:** use the matched supervised forecast objective.
- **S:** minimize normalized teacher-imitation loss on `early_stop` using
  same-information teacher labels generated from the frozen P1 parent.
- **J:** choose the lowest validation decision objective among epochs satisfying
  all forecast guardrails in Section 7.

Training runs for at least `minimum_epochs`, stops after
`early_stopping_patience` non-improving epochs, and never exceeds the fixed
maximum epoch budget. Receipts record the best epoch, final executed epoch,
training history, validation history, selection metric, stopping reason,
selected checkpoint hash, and last checkpoint hash.

## 6. Correct J-Stage Optimization

Formal-v4.5 uses separate optimizer parameter groups:

- Scheme2R base: `j_forecaster_lr`;
- residual regime gate and thermal magnitude heads: `j_head_lr`;
- scheduling proxy: `j_scheduler_lr`.

Empty groups are omitted; no group is silently merged into another learning
rate. The receipt records the effective learning rate and parameter count for
each group. RSC-PF Joint updates all three groups. Fair Decoupled keeps the
forecast parent fixed and updates only the scheduler, with the forecast-to-
dispatch edge detached. This preserves its role as the matched independently
trained forecast-plus-neural-scheduler comparator.

## 7. Joint Loss, Anchoring, and Guardrails

Before J training, the frozen P1+S parent is evaluated once on the training and
early-stop partitions. Its forecast, imitation, and decision losses become
detached normalization constants, each lower-bounded by `1e-8`. Training terms
use constants measured on the training partition; validation terms use the
corresponding early-stop constants. This prevents the much larger raw decision
objective from numerically overwhelming the forecast loss.

The Joint objective is

\[
\mathcal L_J(e)=
\lambda_f\widetilde{\mathcal L}_{forecast}
+\lambda_i(e)\widetilde{\mathcal L}_{imitation}
+\lambda_d(e)\widetilde{\mathcal L}_{decision}
+\lambda_a\mathcal L_{anchor}.
\]

`L_anchor` is a task-normalized smooth-L1 distance between the current physical
four-task forecast and the detached P1-parent forecast for the same input. It
protects forecast semantics but does not freeze the forecast path; decision
gradients must remain finite and non-zero in the Scheme2R base, regime gate,
thermal magnitude head, and scheduler.

The fixed Pilot curriculum uses `lambda_f=1.0` and `lambda_a=0.5` throughout.
Across the first five J epochs, `lambda_d` increases linearly from `0.05` to
`0.50`, while `lambda_i` decreases linearly from `1.00` to `0.25`; both remain
at their final values afterward. These values and the ramp length are declared
in the formal-v4.5 contract before the diagnostic run and are included in all
lineage hashes.

An epoch is eligible for J checkpoint selection only when, on `early_stop`:

- electricity and gas WAPE do not exceed the frozen P1-parent values by more
  than the existing unaffected-task tolerance;
- active cooling and heating WAPE do not exceed the P1-parent values by more
  than the existing active-task tolerance;
- the four-task score does not exceed the P1-parent score by more than the
  existing overall tolerance;
- macro regime F1 is finite and no more than `0.02` below the P1-parent value;
- inactive cooling and heating leakage are finite and each no more than `5%`
  above its P1-parent value.

Among eligible epochs, the checkpoint with the lowest normalized realized
decision objective is selected. If no epoch is eligible, the J candidate fails
closed; it is not replaced silently by the parent and cannot authorize Gate 1.

## 8. Diagnostic Execution

Implementation is verified first with deterministic fixtures and a short
real-data diagnostic run. The diagnostic uses only the 2015--2018 train and
early-stop partitions and does not inspect 2019 for hyperparameter selection.
It must demonstrate:

1. early stopping selects and restores a measured best checkpoint;
2. all J parameter groups use their declared learning rates;
3. normalized curriculum terms and anchor loss are finite;
4. Joint decision gradients reach every intended forecast component;
5. Fair Decoupled decision gradients into all forecast components are zero;
6. forecast guardrails reject a deliberately decision-dominated checkpoint.

Only after these checks pass is one new formal-v4.5 Pilot run allowed on the
full 2019 chronology. A failed diagnostic or Pilot stops the process before any
large baseline or ablation experiment.

## 9. Artifacts and Audit

Formal-v4.5 adds immutable artifacts for:

- train and early-stop loader hashes;
- validation histories and best-epoch decisions;
- parent loss normalization constants;
- curriculum weights per epoch;
- anchor-loss history;
- optimizer parameter-group names, counts, and learning rates;
- forecast-guardrail values and eligibility per epoch;
- selected and terminal checkpoint hashes.

The independent audit recomputes checkpoint eligibility and selection from
persisted validation arrays. A missing validation split, unrecorded learning
rate, non-finite term, selection mismatch, 2019-based tuning, or evaluation-year
access fails closed.

## 10. Test and Completion Criteria

Tests cover stage-specific early stopping, patience/minimum-epoch behavior,
best-state restoration, optimizer-group separation, loss normalization,
curriculum monotonicity, anchor gradients, Joint/Decoupled gradient boundaries,
guardrail selection, no-eligible-epoch failure, immutable artifacts, and
independent audit reconstruction.

The repair is complete only when targeted formal-v4.5 tests pass, protected
formal-v4.4 regression tests still pass, a fresh source manifest and Gate 0 are
bound to the final implementation, and the bounded diagnostic succeeds. This
engineering completion does not predetermine the scientific Pilot result.
