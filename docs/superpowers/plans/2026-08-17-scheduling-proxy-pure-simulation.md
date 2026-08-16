# Pure-Simulation Scheduling Proxy Implementation Plan

> **For Codex:** Execute this plan only for the scheduling-proxy subtask. Preserve the frozen Scheme2R forecasting code, the existing SciPy/HiGHS LP, all completed reports, and all manuscript files.

**Goal:** Add a CPU-friendly neural scheduling proxy trained exclusively on synthetic four-hour IES scenarios labelled by the existing exact HiGHS dispatch model, together with a frozen-Scheme2R inference adapter, physics-aware evaluation, tests, and a deterministic smoke pipeline.

**Architecture:** The existing LP remains the authoritative teacher and optional safety fallback. New modules generate pure synthetic scenarios, serialize exact LP labels, normalize train-only inputs/outputs, train a residual MLP to predict the complete 21-variable dispatch trajectory, calculate differentiable physical/economic penalties, adapt Scheme2R-shaped forecasts, and report raw-proxy metrics separately from fallback-assisted metrics.

**Tech Stack:** Python 3.9, NumPy, SciPy/HiGHS, PyYAML, PyTorch CPU, pytest/unittest-compatible tests.

---

## Frozen interfaces and boundaries

- Repository: `D:\Paper\github_work\paper-code`
- Benchmark: `D:\Paper\standard_ies_benchmark_v1.yaml`
- Exact teacher: `frame/src/scheduling/dispatch_lp.py`
- Dispatch label order: `dispatch_lp.VARIABLES` (`21` variables).
- Horizon: `H=4`.
- Proxy input order (`10` features): electricity, cooling, heating, gas prior, PV available, WT available, grid price, gas price, carbon price, initial SOC.
- Proxy output: `[batch, 4, 21]`.
- Gas is an auxiliary station-side fuel prior. It is not a fourth rigid balance equation.
- Do not edit Scheme2R, forecasting training/evaluation, existing scheduling LP equations, completed report directories, `new_paper`, or manuscript files.
- Smoke artifacts must be written below ignored `frame/reports/scheduling_proxy_v1/smoke/`.

## Authorized files

Create:

- `frame/configs/scheduling_proxy_contract_v1.json`
- `frame/src/scheduling/proxy_contract.py`
- `frame/src/scheduling/synthetic_scenarios.py`
- `frame/src/scheduling/proxy_dataset.py`
- `frame/src/scheduling/proxy_model.py`
- `frame/src/scheduling/proxy_physics.py`
- `frame/src/scheduling/proxy_training.py`
- `frame/src/scheduling/proxy_adapter.py`
- `frame/src/scheduling/proxy_evaluation.py`
- `frame/scripts/run_scheduling_proxy_pipeline.py`
- `frame/tests/test_scheduling_proxy_contract.py`
- `frame/tests/test_synthetic_scheduling_scenarios.py`
- `frame/tests/test_scheduling_proxy_dataset.py`
- `frame/tests/test_scheduling_proxy_model.py`
- `frame/tests/test_scheduling_proxy_physics.py`
- `frame/tests/test_scheduling_proxy_training.py`
- `frame/tests/test_scheduling_proxy_adapter.py`
- `frame/tests/test_scheduling_proxy_pipeline.py`

Modify only when required for public imports/documentation:

- `frame/src/scheduling/__init__.py`
- `frame/README.md`

## Task 1: Freeze and validate the proxy contract

- [ ] Write failing contract tests for schema version, `H=4`, exact feature order, exact `VARIABLES` label order, smoke/formal split sizes and seeds, gas-prior semantics, loss weights, training defaults, fallback tolerance, and benchmark SHA-256 validation.
- [ ] Add `scheduling_proxy_contract_v1.json` with smoke sizes `64/16/16`, formal sizes `8192/2048/2048`, seeds `2026/2027/2028`, width `128`, two residual blocks, dropout `0.1`, CPU device, early stopping, and explicit loss weights.
- [ ] Implement immutable contract dataclasses plus strict load/validate functions in `proxy_contract.py`.
- [ ] Reject duplicated feature/label names, wrong task order, invalid seeds/sizes, negative loss weights, non-positive scales, and a benchmark hash mismatch.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_contract.py -q`.

## Task 2: Generate deterministic pure-simulation scenarios

- [ ] Write failing tests for deterministic generation, split-ID disjointness, fixed shapes, finite values, physical bounds, and absence of observed timestamps/windows.
- [ ] Implement `SyntheticScenarioBatch` and deterministic split-specific generation in `synthetic_scenarios.py`.
- [ ] Anchor load ranges to benchmark `training_statistics` and device capacities, but synthesize every four-hour profile from latent regimes and bounded stochastic trajectories rather than copying data rows.
- [ ] Generate e/c/h demand, PV/WT availability, price schedules, initial SOC, scenario IDs, and generator metadata.
- [ ] Make train/validation/test RNG streams independent; include an explicit generator version.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_synthetic_scheduling_scenarios.py -q`.

## Task 3: Build exact-label datasets without leakage

- [ ] Write failing tests for successful HiGHS labels, `[N,4,21]` dispatch shape, exact label order, non-finite/failed-solve rejection counts, separate gas-prior RNG, gas-prior masking, and train-only normalization.
- [ ] Implement exact label generation by constructing `DispatchInputs` and calling `solve_dispatch_lp` for every accepted scenario.
- [ ] Derive teacher gas purchase as `g_chp + g_gb`, then create the noisy/masked auxiliary gas prior only after the exact label exists and with a separate deterministic RNG stream.
- [ ] Save compressed `train.npz`, `validation.npz`, `test.npz` and a manifest containing benchmark hash, contract hash, seeds, counts, rejected counts, feature/label orders, and source type `pure_simulation`.
- [ ] Implement `ProxyNormalizationStats`, fit only on train, save/load with validation, and a PyTorch `Dataset` returning input, dispatch target, teacher cost/carbon and masks.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_dataset.py -q`.

## Task 4: Implement the residual scheduling proxy

- [ ] Write failing tests for input validation, deterministic forward pass, `[B,4,21]` output, finite values, normalized output bounds, parameter count, and CPU checkpoint round-trip.
- [ ] Implement a flattened-horizon residual MLP: input projection (`Linear/GELU/LayerNorm/Dropout`), two residual MLP blocks, output projection and sigmoid in normalized label space.
- [ ] Keep architecture configuration explicit and serializable; do not embed benchmark-specific values in the module.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_model.py -q`.

## Task 5: Implement differentiable physics, cost and carbon terms

- [ ] Write numerical tests showing zero/near-zero residuals for exact LP labels and positive residuals after controlled perturbations.
- [ ] Implement vectorized PyTorch calculations for electricity/cooling/heating balance, CHP/boiler/chiller conversion, BESS SOC recursion/terminal deviation, operating cost and carbon emissions.
- [ ] Implement the weighted loss returning both total loss and named components: dispatch, balance, conversion, SOC, cost, carbon and masked gas-prior consistency.
- [ ] Keep these calculations consistent with `dispatch_lp.py`; do not alter the teacher equations.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_physics.py -q`.

## Task 6: Implement CPU training, checkpointing and inference

- [ ] Write failing tests for one-epoch training, validation-only early stopping, best-checkpoint reload, history persistence, deterministic seed behavior and no test-set use during selection.
- [ ] Implement CPU DataLoaders, AdamW, gradient clipping, weighted physics-aware loss, early stopping and atomic best-model saving in `proxy_training.py`.
- [ ] Save `best_model.pt`, `normalization_stats.npz`, `history.json`, and training metadata including contract/benchmark hashes.
- [ ] Ensure the test split is evaluated only after the best validation checkpoint is restored.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_training.py -q`.

## Task 7: Implement Scheme2R adapter and safe fallback

- [ ] Write failing adapter tests for `[N,4,4]` task order, `[N,4,2]` renewable order, optional price/SOC inputs, non-finite rejection, negative-forecast clipping counters, no future truth fields, and gas-prior/no-prior modes.
- [ ] Implement `proxy_adapter.py` to construct the exact `[N,4,10]` interface and apply saved train-only normalization.
- [ ] Implement raw-proxy feasibility evaluation before any fallback.
- [ ] Implement optional exact-LP fallback for scenarios exceeding the contract tolerance; return raw output, safe output, fallback mask and reason.
- [ ] Never overwrite or relabel raw feasibility with fallback feasibility.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_adapter.py -q`.

## Task 8: Implement evaluation and deterministic orchestration

- [ ] Write failing end-to-end tests that invoke the pipeline with smoke sizes and verify every required artifact.
- [ ] Implement metrics for physical/normalized dispatch MAE/RMSE, cost regret, carbon error, balance/conversion/SOC residuals, raw feasible rate, fallback rate, safe success rate, and median/P95 CPU latency.
- [ ] Evaluate both gas-prior and no-prior modes on the same synthetic test scenarios.
- [ ] Implement CLI phases `generate`, `train`, `evaluate`, and `smoke`; default smoke path is `frame/reports/scheduling_proxy_v1/smoke`.
- [ ] Make `smoke` perform generation, label solving, training, checkpoint reload, adapter inference, raw/fallback evaluation and manifest emission.
- [ ] Required artifacts: split NPZ files, dataset manifest, normalization stats, best checkpoint, history, test metrics, test predictions and smoke manifest.
- [ ] Run: `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_pipeline.py -q`.
- [ ] Run smoke: `D:\anaconda\envs\pytorch\python.exe frame/scripts/run_scheduling_proxy_pipeline.py --mode smoke --benchmark D:\Paper\standard_ies_benchmark_v1.yaml --contract frame/configs/scheduling_proxy_contract_v1.json --output-dir frame/reports/scheduling_proxy_v1/smoke`.

## Task 9: Preserve compatibility and document the new path

- [ ] Export only stable public types/functions from `frame/src/scheduling/__init__.py` if needed.
- [ ] Add a concise README section that distinguishes: frozen Scheme2R forecasting, exact LP teacher/oracle, pure-simulation proxy training, raw proxy output, and optional exact fallback.
- [ ] Explicitly state that the smoke result is pipeline validation rather than a formal paper result.
- [ ] Do not modify manuscript files or existing result directories.

## Task 10: Final verification

- [ ] Run targeted proxy suite:
  `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_scheduling_proxy_contract.py frame/tests/test_synthetic_scheduling_scenarios.py frame/tests/test_scheduling_proxy_dataset.py frame/tests/test_scheduling_proxy_model.py frame/tests/test_scheduling_proxy_physics.py frame/tests/test_scheduling_proxy_training.py frame/tests/test_scheduling_proxy_adapter.py frame/tests/test_scheduling_proxy_pipeline.py -q`.
- [ ] Run existing scheduling regression suite:
  `D:\anaconda\envs\pytorch\python.exe -m pytest frame/tests/test_dispatch_lp.py frame/tests/test_highs_solver.py frame/tests/test_scheduling_contracts.py frame/tests/test_scheduling_forecast_adapter_v2.py frame/tests/test_scheduling_metrics_v2.py frame/tests/test_scheduling_no_leakage_v2.py -q`.
- [ ] Run `git diff --check` and verify the changed-file set is restricted to the authorized files.
- [ ] Verify ignored smoke artifacts exist locally and are absent from `git status --short`.
- [ ] Record exact commands, pass counts, smoke artifact path, output shapes, and any residual caveats in the implementation report.

## Acceptance criteria

- All proxy-training samples are synthetic and split metadata is auditable.
- All teacher labels originate from successful exact LP solves.
- Train-only normalization and validation-only model selection are enforced.
- Proxy output shape is exactly `[N,4,21]` in `dispatch_lp.VARIABLES` order.
- Gas prior is auxiliary/maskable and never a rigid demand balance.
- Raw proxy and exact-fallback metrics are reported separately.
- CPU smoke pipeline completes and reloads the best checkpoint.
- Targeted tests and existing scheduling regressions pass.
- No forecasting, formal-result, or manuscript files are changed.
