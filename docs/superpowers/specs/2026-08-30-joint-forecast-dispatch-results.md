# Joint forecast–dispatch results evidence specification

This specification defines the evidence required before any results are
reported in the Chinese or English manuscript.

## Required evidence

- one connected forecast-to-dispatch forward graph and one optimizer update;
- nonzero finite decision-only gradient into Scheme2R and the scheduling proxy;
- explicit forecast MAE/RMSE/WAPE by task and horizon;
- realized first-hour shortage, surplus, operating cost, physical carbon, and
  oracle regret under the deterministic recourse rule;
- canonical 21-variable dispatch MAE and physical-feasibility diagnostics;
- model-generated closed-loop state transitions with exactly zero online LP
  calls;
- one training-only 50/50 LP-history/model-history roll-in refresh;
- validation-only model selection and a sealed-test receipt;
- descriptive ablation names (`No-Device-State`, `No-Decision-Loss`,
  `Frozen-Forecaster`, `No-Gas-Prior`, and the diagnostic-only
  `Direct-Dispatch-Diagnostic-Only`);
- paired 168-hour moving-block uncertainty with FDR-adjusted comparisons.

## Interpretation rule

The final conclusion must be selected from the audited evidence:

1. joint training improved dispatch while preserving forecast accuracy;
2. joint training traded a bounded forecast degradation for improved dispatch;
3. joint training did not outperform the frozen PTO baseline under the fixed
   protocol.

No conclusion is valid before `audit_manifest.json` has `status=pass`. The
completed run has `status=pass`, with 500/500 resource-gate LP solves, causal
2015–2019 training data, 2020 validation, and a sealed 2021 test. The
from-scratch joint candidates lowered validation decision regret but failed
the predeclared rigid-task forecast noninferiority gate. Warm-start candidates
passed the forecast and gradient gates, but their paired 168-hour bootstrap
for regret relative to from-scratch was −1.83 to 8.89; consequently the
frozen PTO baseline was selected and evaluated on the sealed test. The
evidence-dependent conclusion is therefore:

> Joint training did not outperform the frozen PTO baseline under the fixed
> protocol.

This wording does not erase the diagnostic result that the joint candidates
had substantially lower validation regret; it records that the improvement
was not admissible under the simultaneous forecast-and-dispatch acceptance
rule. Compact recomputable tables are stored under
`reports/joint_forecast_dispatch_v1/evaluation_tables/`. The five required
ablation names are registered, but no ablation values are claimed because
those runs were not executed under the complete protocol.

Later publication work—submission-grade figures, bilingual manuscript
synchronization, typography, and final Word rendering—is explicitly outside
this implementation scope.
