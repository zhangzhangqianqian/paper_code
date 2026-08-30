# Joint forecast–dispatch experiment

This directory is reserved for the reviewed end-to-end experiment defined by
`configs/joint_forecast_dispatch_contract_v1.json`.

The implementation uses a single connected graph: a state-conditioned Scheme2R
forecaster emits an explicit four-task, four-hour bottleneck, a 15-control
scheduling proxy consumes that bottleneck and the fixed physical context, and
the existing differentiable decoder produces the canonical 21 dispatch
variables. The primary variants are `joint_from_scratch` and the frozen PTO
baseline; warm start is conditional on the predeclared validation rule.

Resource preflight is complete: 500/500 representative offline HiGHS LP solves
passed, with projected p95 full-generation time below the 24-hour gate. The
causal 2015–2019 training and 2020 validation splits contain 43,767 and 8,780
windows, respectively; 2021 was kept sealed until the validation selection
receipt was frozen. Smoke training, five-seed validation, selection, and the
sealed 2021 test have now completed. The independent audit manifest reports
`status=pass`.

The predeclared noninferiority rule did not select a joint checkpoint. The
from-scratch joint candidates reduced validation decision regret (mean 51.54
versus 372.29 for frozen PTO) but exceeded the rigid-task forecast gate. The
warm-start candidates passed the rigid-task forecast and gradient gates, but
the paired 168-hour bootstrap for warm-start minus from-scratch regret was
−1.83 to 8.89, so the protocol retained the frozen PTO baseline. The sealed
test therefore evaluates that frozen baseline only; its receipt reports 8,756
windows, zero online LP calls, and forecast WAPE of 13.10%, 50.63%, 57.64%,
and 29.44% for electricity, cooling, heating, and gas, respectively.

The evidence tables are in `evaluation_tables/`. The required descriptive
ablation names are registered in `ablations/ablation_registry.csv`, but no
ablation results are claimed because those runs were not executed under the
same full protocol. This is an evidence report, not a manuscript edit; the
Chinese and English Word files and all figures remain untouched.
