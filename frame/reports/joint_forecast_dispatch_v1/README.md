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
passed, with projected p95 full-generation time below the 24-hour gate. Formal
data generation, smoke training, validation selection, and sealed 2021 testing
remain intentionally gated until their receipts are produced. This file does
not claim a scientific result and does not modify either manuscript.
