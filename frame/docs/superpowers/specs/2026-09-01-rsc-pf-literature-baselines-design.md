# RSC-PF Literature Baselines and Formal Comparison Design

## Objective

Build a defensible external-baseline comparison for RSC-PF before any new ablation experiment. The work must first identify published forecast–scheduling methods, then freeze a reproducible set of compatible baselines, implement or adapt them under one causal rolling protocol, and finally compare them with the already-frozen RSC-PF model.

## Current Boundary

The existing v2 experiment has already evaluated `Seasonal-Naive-PTO`, `Scheme2R-PTO`, `From-Scratch-Joint`, `Warm-Start-Joint`, and `Oracle-LP`. These are retained as internal references:

- `Seasonal-Naive-PTO` and `Scheme2R-PTO` are internal predict-then-optimize controls.
- `From-Scratch-Joint` is an initialization/training control, not an external literature baseline.
- `Warm-Start-Joint` is the proposed RSC-PF implementation.
- `Oracle-LP` is a reference with realized future demand and must not enter method rankings.

This design adds external literature baselines without changing the frozen RSC-PF architecture, existing v2 checkpoints, or sealed test receipts.

## Baseline Taxonomy

The search must fill exactly three external slots:

1. `External-Forecast-PTO`: a published strong forecasting method followed by the same exact rolling LP used by the internal PTO methods.
2. `External-Decision-Focused`: a published decision-focused or differentiable-optimization method in which decision loss influences the predictor or decision layer.
3. `External-Direct-Policy`: a published end-to-end learned policy or forecast-to-decision network that produces continuous dispatch decisions without requiring a separately trained load predictor at inference.

Each slot must contain one exact paper and one exact implementation identity before training begins. A slot remains unfilled if no paper passes the compatibility gate; the pipeline then stops rather than inventing a weak or misleading baseline.

## Literature Search Protocol

Search years 2016–2026 across CrossRef and arXiv first, then Semantic Scholar; use ScienceDirect/Scopus only when primary sources are insufficient. Execute the following query families with energy/power variants:

1. `(joint forecasting scheduling OR integrated forecasting dispatch) AND (energy OR power OR integrated energy system)`
2. `(decision-focused learning OR task-based forecasting OR predict-and-optimize) AND (economic dispatch OR energy scheduling)`
3. `(differentiable optimization OR differentiable convex optimization OR OptNet OR cvxpylayers) AND (energy dispatch OR unit scheduling)`
4. `(end-to-end learning OR learning to optimize OR neural policy) AND (multi-energy dispatch OR energy management)`
5. `(predict-then-optimize OR smart predict-then-optimize OR SPO+) AND (electricity OR energy scheduling)`

Deduplicate by normalized DOI, then by normalized title plus first author with Jaccard similarity at least 0.90. Every retained candidate must have a primary paper, DOI or stable preprint identifier, accessible method details, venue/year, code status, license status, and a short statement of what is actually coupled.

## Screening and Ranking

Fatal exclusion criteria are:

- future realized demand or renewable output is used as an inference input;
- the method only addresses binary unit commitment and has no continuous-dispatch formulation that preserves its core method;
- neither code nor equations are sufficient for a faithful implementation;
- the method has no forecasting-to-decision coupling despite being described as end-to-end;
- evaluation cannot be mapped to a four-hour rolling continuous dispatch without changing the method's central contribution;
- software license forbids the planned use.

Candidates that pass fatal screening receive a 100-point score:

- forecast–decision coupling relevance: 25;
- compatibility with continuous rolling dispatch: 20;
- reproducibility from code/equations: 20;
- fairness under the RSC-PF information set: 15;
- venue and citation evidence: 10;
- recency: 5;
- projected compute/resource fit: 5.

The highest-scoring candidate in each slot is selected only if it scores at least 70. Ties are broken by reproducibility, then coupling relevance, then recency. The complete score table remains part of the evidence package.

## Reproduction Levels

Every selected method receives one immutable label:

- `official_code_exact`: official code can be run with only data/interface adaptation;
- `faithful_reimplementation`: published equations and training objective are reproduced without changing the core algorithm;
- `principled_adaptation`: the coupling mechanism is preserved but the device topology/output layer is adapted to the shared continuous IES environment.

The manuscript must use “adapted implementation” whenever the label is `principled_adaptation`; it must not claim exact reproduction.

## Fair Comparison Contract

All trainable external methods use:

- training years 2015–2019, validation year 2020, and sealed test year 2021;
- 24-hour causal history and four-hour forecast/dispatch horizon;
- the same four forecast targets where the method exposes forecasts;
- the same available causal inputs: historical loads, exogenous variables, device histories/statuses, PV/WT forecasts, prices, carbon context, SOC, and previous CHP state;
- five seeds: 2026–2030;
- validation-only checkpoint selection and one sealed test evaluation;
- the same Standard-IES topology, capacities, rolling settlement, and physical metric implementation;
- the same compute receipt format, including offline and online exact-optimizer calls.

A baseline may ignore inputs its published architecture does not consume, but it may not receive information unavailable to RSC-PF. Method-specific optimizer and loss settings follow the source paper when reproducible; all deviations are recorded.

`External-Forecast-PTO` uses the shared exact LP after forecasting. `External-Decision-Focused` preserves its published gradient-coupling mechanism and may use a differentiable optimization layer. `External-Direct-Policy` uses the shared 15-control-to-21-dispatch physical interface when its native output cannot represent the Standard-IES topology; this is disclosed as an adaptation and creates a strong feasibility-controlled comparison.

## Evaluation

Forecast metrics are MAE, RMSE, and WAPE by task and horizon when a method exposes a forecast bottleneck. Decision metrics are realized normalized operating cost, physical carbon emissions, shortage, feasibility rate, decision regret relative to Oracle, runtime, parameter count, peak memory, and offline/online exact-optimizer calls.

Primary comparisons are RSC-PF versus each of the three selected external baselines. Use paired contiguous moving-block bootstrap with 168-hour blocks and 2,000 replicates. Apply Benjamini–Hochberg correction within each endpoint family at α=0.05. Report effect size, 95% interval, raw p value, adjusted p value, and seed-level dispersion. If the interval crosses zero or adjusted p exceeds 0.05, use “no statistically reliable difference was detected.”

## Output and Freeze Gates

The search stage produces a deduplicated literature table, evidence cards, selection scores, implementation/license status, and a baseline freeze receipt. The implementation stage cannot begin until all three slots pass. Validation pilots cannot access test artifacts. Formal test evaluation cannot begin until the external-baseline validation freeze receipt exists.

The final comparison report must distinguish:

- external literature baselines;
- internal controls;
- the proposed RSC-PF method;
- Oracle reference;
- exact reproductions, faithful reimplementations, and principled adaptations.

## Out of Scope

- RSC-PF architecture changes;
- new ablation experiments;
- changing the sealed v2 main-method receipts;
- inserting final performance claims into Chinese or English manuscripts before external comparisons pass;
- presenting an empty baseline slot as evidence that no related method exists.
