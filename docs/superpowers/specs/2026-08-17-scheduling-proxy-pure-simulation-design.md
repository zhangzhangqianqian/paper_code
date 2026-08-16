# Pure-Simulation Scheduling Proxy Design

**Date:** 2026-08-17  
**Status:** approved from the conversation and ready for implementation  
**Scope:** only the pure-simulation scheduling dataset, learned scheduling proxy, Scheme2R adapter, tests, and smoke validation

## 1. Goal

Extend the frozen Scheme2R predict-then-optimize work with a separately trained scheduling proxy. Scheme2R remains trained on the real Kitakyushu forecasting protocol. The new proxy is trained only on synthetic standard-IES scenarios labelled by the existing SciPy/HiGHS linear program. At inference, four-hour Scheme2R forecasts are adapted to the proxy interface and mapped to device dispatch trajectories.

This is a two-stage learned pipeline, not end-to-end decision-focused training:

```text
Kitakyushu history -> frozen Scheme2R -> four-hour forecasts
                                           |
                                           v
pure synthetic scenarios -> HiGHS labels -> scheduling proxy -> dispatch, cost, carbon
```

The exact optimizer remains the offline teacher, Oracle benchmark, and optional safety fallback. It is not removed or described as a trained neural network.

## 2. Settled research boundary

### 2.1 Preserved work

- Do not retrain or alter Scheme2R or the completed five-model forecasting experiments.
- Do not alter the existing R/S dual-track scheduling results or Stage 10.14 evidence.
- Reuse `frame/src/scheduling/dispatch_lp.py` as the authoritative label generator.
- Reuse `D:/Paper/standard_ies_benchmark_v1.yaml` as the frozen topology, capacities, efficiencies, prices, and emission factors.

### 2.2 New work

- Generate a synthetic scheduling dataset without copying observed Kitakyushu windows into proxy training.
- Solve every accepted synthetic scenario with HiGHS and store optimal dispatch labels, cost, carbon, residuals, and solver metadata.
- Train a lightweight PyTorch proxy on those labels.
- Adapt frozen Scheme2R predictions to the proxy's units, order, and scale.
- Evaluate imitation error, cost regret, carbon error, constraint violation, fallback rate, and CPU latency.

### 2.3 Excluded work

- No mixing of real and synthetic samples in proxy training.
- No end-to-end gradient from scheduling loss into Scheme2R.
- No use of 2021 observations to tune scenario ranges, normalization, model selection, or thresholds.
- No change to the LP topology, integer commitment decisions, or unsupported devices.
- No claim that the proxy replaces the exact solver when its raw output is infeasible.
- No manuscript editing in this task.

## 3. Gas semantics

The fourth Scheme2R channel is station-side aggregate equipment gas consumption, not terminal residential gas demand. The physical LP therefore continues to balance only electricity, cooling, and heating demands. Gas purchase is an LP decision:

```text
gas_purchase = g_chp + g_gb
```

For proxy training, a synthetic `gas_prior` is an auxiliary, noisy estimate of the teacher solution's gas purchase. It is generated only after the exact dispatch label is available, then independently perturbed and occasionally masked. It is never imposed as a fourth rigid demand balance. This mirrors the deployment interface, in which Scheme2R's gas forecast is a prior for station-side fuel consumption.

The dataset records whether the prior is available. Core metrics are reported both with and without the gas prior so that the paper can show whether the fourth forecast channel contributes useful scheduling information instead of leaking the target.

## 4. Synthetic scenario contract

### 4.1 Scenario inputs

Each scenario contains a four-hour horizon (`H=4`) and the following arrays:

| Field | Shape | Meaning |
|---|---:|---|
| `demand` | `[H,3]` | electricity, cooling, heating demand |
| `gas_prior` | `[H,1]` | noisy station-side fuel prior; not a balance constraint |
| `renewable` | `[H,2]` | PV and WT available power |
| `prices` | `[H,3]` | grid price, gas price, carbon price |
| `initial_soc` | `[1]` | BESS initial state of charge |
| `gas_prior_mask` | `[H,1]` | one when the prior is supplied, zero when masked |

The proxy input is the concatenation of the step-wise fields plus initial SOC broadcast over the four steps, giving `[H,10]` in this fixed order:

```text
electricity, cooling, heating, gas_prior,
pv_available, wt_available,
grid_price, gas_price, carbon_price,
initial_soc
```

### 4.2 Pure-simulation rule

Scenario values are generated from deterministic stochastic processes whose ranges are anchored only to the frozen benchmark capacities and training-period statistics already stored in the benchmark YAML. No observed time window is copied. Correlated, smooth four-hour trajectories are produced from latent operating regimes and bounded autoregressive perturbations:

- electricity: base load plus activity and thermal effects;
- cooling and heating: season/regime-conditioned and negatively coupled in normal regimes, with shoulder-period overlap allowed;
- PV: bounded daylight-shaped availability with cloud attenuation;
- WT: bounded correlated availability;
- prices: scenario-level grid, gas, and carbon multipliers broadcast over the horizon;
- initial SOC: sampled from the valid interior interval `[0.2,0.8]`.

All synthetic demands remain within the frozen standard-IES operating envelope. Stress cases close to capacity limits are intentionally retained, but scenarios solved with non-optimal status or non-finite outputs are rejected and counted in the manifest.

### 4.3 Splits and reproducibility

Formal defaults are independent random streams and disjoint scenario identifiers:

| Split | Seed | Samples |
|---|---:|---:|
| train | 2026 | 8192 |
| validation | 2027 | 2048 |
| test | 2028 | 2048 |

Smoke defaults are `64/16/16`. Every artifact stores the generator version, seed, sample count, benchmark SHA-256, feature order, label order, and rejection count. Normalization statistics are fit on the synthetic training split only.

## 5. Teacher labels

For every accepted scenario, `solve_dispatch_lp(DispatchInputs)` provides the authoritative optimal label. The label order is exactly `dispatch_lp.VARIABLES`, currently 21 variables, stored as `[H,21]`. Additional labels are:

- LP objective;
- operating cost without carbon-price double counting;
- carbon emissions from grid electricity and purchased gas;
- maximum electricity/cooling/heating balance residual;
- simultaneous BESS charge/discharge diagnostic;
- solver status and message.

The saved dataset uses compressed NPZ arrays and a JSON manifest. Failed scenarios are never silently replaced with zeros.

The synthetic gas prior is constructed from `g_chp + g_gb` with bounded multiplicative and additive perturbations. Its noise realization uses a separate RNG stream from load generation. A configured fraction is masked to prevent a trivial identity mapping and to support a no-prior ablation.

## 6. Proxy model

### 6.1 Architecture

The CPU-oriented proxy is a horizon-aware residual MLP:

1. train-only standardization of the `[B,H,10]` input;
2. flattening of the four-hour context;
3. `Linear -> GELU -> LayerNorm -> Dropout` input block;
4. two residual MLP blocks of width 128;
5. linear output projected to `[B,H,21]`;
6. sigmoid in normalized label space to enforce non-negativity and training-range upper bounds.

Default configuration: hidden width 128, two residual blocks, dropout 0.1. This architecture is intentionally smaller than the forecasting model and is appropriate for the fixed four-hour LP mapping. It introduces no Transformer or recurrent state that the short horizon does not justify.

### 6.2 Normalization

- Input mean and scale are fitted on the training split only.
- Each dispatch variable uses a positive physical or train-only empirical scale.
- Capacity-bounded variables use frozen device capacities.
- renewable curtailment variables use the corresponding frozen renewable capacity.
- slack and heat-dump variables use robust train-only positive scales with a floor.
- Normalization metadata is saved and required for inference.

### 6.3 Multi-component objective

The proxy is trained with a weighted objective:

```text
L_total = L_dispatch
        + lambda_balance * L_balance
        + lambda_conversion * L_conversion
        + lambda_soc * L_soc
        + lambda_cost * L_cost
        + lambda_carbon * L_carbon
        + lambda_gas * L_gas_prior
```

- `L_dispatch`: Smooth L1 on normalized teacher dispatch variables;
- `L_balance`: electricity, cooling, and heating equality residuals;
- `L_conversion`: CHP, boiler, electric-chiller, and absorption-chiller conversion residuals;
- `L_soc`: BESS state recursion and terminal-SOC residuals;
- `L_cost`: relative error against teacher operating cost;
- `L_carbon`: relative error against teacher carbon emissions;
- `L_gas_prior`: masked consistency between proxy gas purchase and the supplied gas prior.

All weights are explicit in the contract and selected only from synthetic validation data. Scheme2R forecast loss is not included because the two models are trained separately.

## 7. Feasibility and safety

Two outputs must be distinguished:

1. **raw proxy output**: used to report the proxy's own constraint violation and feasibility rate;
2. **safe output**: if a configured tolerance is exceeded, the same scenario is solved by the exact LP and the fallback is recorded.

The safe mode may be used by a deployment demonstration, but raw and fallback metrics must be reported separately. Exact fallback results must not be presented as evidence that the neural proxy itself is feasible.

No ad-hoc clipping may overwrite raw outputs before feasibility metrics are calculated. Non-negativity and normalization bounds built into the network are part of the model and are reported as such.

## 8. Scheme2R interface

The adapter accepts:

- load predictions `[N,4,4]` ordered electricity, cooling, heating, gas;
- renewable predictions `[N,4,2]` ordered PV, WT;
- frozen price/carbon settings or an explicitly supplied schedule;
- initial SOC `[N]` or a fixed default.

It clips only physically impossible negative forecasts, records the count, maps the first three channels to LP demand, maps the fourth channel to `gas_prior`, and converts all fields with the saved proxy statistics. The adapter rejects wrong task order, shapes, non-finite values, or missing normalization metadata.

No true future load, weather, renewable realization, cost, carbon, or optimal dispatch label enters this inference path.

## 9. Artifacts and commands

The pipeline produces:

```text
dataset/{train,validation,test}.npz
dataset/manifest.json
normalization_stats.npz
best_model.pt
history.json
metrics_test.json
predictions_test.npz
smoke_manifest.json
```

One orchestration script supports a deterministic smoke run and separate generate/train/evaluate commands support formal work. The smoke run must complete on CPU without Kitakyushu files because it operates only on the frozen benchmark and synthetic data.

## 10. Evaluation

Required test-set metrics:

- normalized and physical-unit dispatch MAE/RMSE;
- teacher objective gap and cost regret;
- carbon absolute and relative error;
- raw balance/conversion/SOC residuals;
- raw feasible-scenario rate;
- exact-fallback rate and safe-mode success rate;
- median and 95th-percentile CPU inference latency;
- exact HiGHS latency on the same scenarios;
- gas-prior versus no-prior ablation.

Formal claims require multiple training seeds. The implementation task only needs deterministic unit tests and a complete smoke artifact; it does not overwrite or claim new formal paper conclusions.

## 11. File boundaries

New modules are separated by responsibility:

- synthetic scenario generation;
- exact-label dataset creation and serialization;
- proxy normalization and dataset loading;
- proxy model;
- differentiable physics/cost/carbon losses;
- Scheme2R adapter and feasibility/fallback evaluation;
- command-line orchestration;
- tests and fixed contract.

Existing forecasting modules are read-only. Existing LP behavior remains backward compatible. Package exports and README may be extended, but existing names and defaults must not change.

## 12. Acceptance criteria

The change is accepted only if:

1. all generated training samples are synthetic and manifests prove split/seed separation;
2. all stored teacher labels come from successful exact LP solves;
3. the proxy trains on CPU and reloads its best checkpoint;
4. a Scheme2R-shaped tensor passes through the adapter and proxy with `[N,4,21]` output;
5. raw feasibility metrics and exact-fallback metrics are separate;
6. tests cover determinism, shapes, no leakage, label validity, loss finiteness, checkpoint reload, and end-to-end smoke execution;
7. existing scheduling and forecasting tests remain green;
8. only the files authorized for this subtask are changed.
