# Complete-v1 Gate 1 Implementation Report

## Implemented

- Added `src/joint_dispatch/complete_formal_gate1.py` as the real Gate 1
  orchestrator.
- Added `scripts/run_rsc_pf_complete_formal_gate1_real.py` as the explicit
  operator command.
- Added strict Gate 0 authorization, source benchmark/capacity matching,
  train-only normalization, and 2019-only selection loading.
- Added the frozen 37-row matrix: seven stochastic methods over seeds
  2026--2030 plus Seasonal-Naive-PTO and Perfect-Information-MPC references.
- Added reserved 2019-only RSC-PF decision-weight and Differentiable-LP
  learning-rate search trials; their selected values are recorded in
  `GATE1_SEARCH.json` and reused for the final matrix.
- Added complete-v1 row envelopes with checkpoint, rollout, metric, and source
  hashes. The legacy v4.2 receipts remain available for their own validator.
- Added an independent complete-formal audit before writing the Gate 1
  transition.
- Added smoke-mode protection: smoke outputs 37 wiring rows but always writes
  `synthetic=true`, `paper_result=false`, and `authorized_gate2=false`.

## Verification

The following checks pass:

```text
57 passed (Gate0/contract/complete-Gate1 regression selection)
py_compile passed
git diff --check passed
```

A smoke execution produced 37 row receipts in about 10 seconds and correctly
refused authorization. It did not access 2020 or 2021.

## Full-run status

The full 37-row, 8,709-origin Gate 1 run has not been started by this change.
It remains an operator-started multi-hour experiment. Only its passing,
non-synthetic transition may authorize Gate 2.

## Task 6 recovery preflight (2026-09-07)

The failed run
`reports/rsc_pf_complete_formal/complete_formal_gate1_20260906_b` was
inspected read-only against the frozen contract, Gate 0 transition, and data
lineage.  No file in that run was overwritten or repaired in place.

The validated recovery classification is:

| Candidate | State | Recovery action |
| --- | --- | --- |
| RSC-PF / 1 | reusable-complete | copy checked row and reuse evaluation |
| RSC-PF / 1.5 | reusable-complete | copy checked row and reuse evaluation |
| RSC-PF / 2 | reusable-complete | copy checked row and reuse evaluation |
| RSC-PF / 3 | reusable-complete | copy checked row and reuse evaluation |
| Differentiable-LP / 1e-5 | reusable-checkpoint | copy checkpoint, restore model, rerun evaluation |
| Differentiable-LP / 3e-5 | retrain-required | train and evaluate |
| Differentiable-LP / 1e-4 | retrain-required | train and evaluate |
| Differentiable-LP / 3e-4 | retrain-required | train and evaluate |

The failed receipt hash is
`a3cfaf880eb0eded39078eb3753ba876c376f38e7645424b3cf7639c501dc089`.
The source-tree inventory contained 58 files and 82,564,654 bytes; the
canonical inventory SHA-256 was
`90ecf00d6925472e862ff21f59392d375bf11d9cd574549d47782723c2981703` both
before and after the preflight, confirming source immutability.

Verification completed:

```text
10 recovery tests passed
67 focused regression tests passed
py_compile passed
git diff --check passed (only expected LF-to-CRLF notices)
```

The continuation command is intentionally recorded but was not started during
this preflight:

```powershell
cd 'D:\Paper\github_work\paper-code-formal-v42-gate0\frame'
& 'D:\Paper\envs\rsc_pf_diffopt_v4\python.exe' scripts/run_rsc_pf_complete_formal_gate1_real.py --contract configs\rsc_pf_complete_formal_v1.json --gate0-transition reports\rsc_pf_complete_formal\complete_formal_gate0_20260906_i\gate0\GATE0_TRANSITION.json --source-run reports\joint_forecast_dispatch_formal_v4_2\formal_v4_2_20260905_j --output-root reports\rsc_pf_complete_formal --run-id complete_formal_gate1_recovered_20260907_a --resume-from reports\rsc_pf_complete_formal\complete_formal_gate1_20260906_b
```
