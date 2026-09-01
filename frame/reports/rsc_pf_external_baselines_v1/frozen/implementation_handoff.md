# RSC-PF external-baseline implementation handoff

This handoff is authorized only when the selection audit reports `complete`.

## forecast_pto: iTransformer-PTO
- Citation: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (2023).
- Primary source: https://arxiv.org/abs/2310.06625
- Identifier: 10.48550/arxiv.2310.06625
- Code/license route: https://github.com/thuml/iTransformer; `MIT`.
- Reproduction level: `official_code_exact`.
- Permitted inputs: `causal history + available context`.
- Output contract: `forecast vector -> Standard-IES rolling LP`.
- Optimizer role: `none at inference`.
- Implementation class: `external.iTransformer-PTO`.
- Adaptation boundary: retain the published learning objective; adapt only tensor/head and device mapping
- Primary-paper anchors: Abstract; Sec. 3, inverted tokenization; official README.
- Forecast representation: continuous multi-step forecast.
- Decision layer: none; output type: forecast vector.
- Loss and gradient path: supervised forecasting loss; forecast loss only.
- Topology: published method with adaptation to the Standard-IES contract.
- Required adaptation: map the published output to the 24-to-4 Standard-IES contract while retaining the method's core objective.
- Forbidden substitution: do not use RSC-PF test data, internal controls, or an unreported exact optimizer in the deployed path.

## decision_focused: DecisionFocused-Online
- Citation: Decision focused online learning for real time energy aware scheduling of interconnected data centers with photovoltaic generation and battery storage (2026).
- Primary source: https://doi.org/10.1038/s41598-026-67967-z
- Identifier: 10.1038/s41598-026-67967-z
- Code/license route: equation-level reproduction; `CC BY-NC-ND 4.0`.
- Reproduction level: `faithful_reimplementation`.
- Permitted inputs: `causal history + available context`.
- Output contract: `forecast vector -> exact-optimizer dispatch`.
- Optimizer role: `exact optimizer at inference`.
- Implementation class: `external.DecisionFocused-Online`.
- Adaptation boundary: retain the published learning objective; adapt only tensor/head and device mapping
- Primary-paper anchors: Abstract; Definitions and problem formulation; Decision-focused loss and surrogate gradient; Online rolling execution.
- Forecast representation: continuous multi-step forecast.
- Decision layer: exact optimizer; output type: forecast vector.
- Loss and gradient path: decision-focused downstream scheduling loss; decision loss to prediction parameters.
- Topology: published method with adaptation to the Standard-IES contract.
- Required adaptation: map the published output to the 24-to-4 Standard-IES contract while retaining the method's core objective.
- Forbidden substitution: do not use RSC-PF test data, internal controls, or an unreported exact optimizer in the deployed path.

## direct_policy: DigitalTwins-Policy
- Citation: Digital Twins based Day-ahead Integrated Energy System Scheduling under Load and Renewable Energy Uncertainties (2021).
- Primary source: https://arxiv.org/abs/2109.14423
- Identifier: arxiv:2109.14423
- Code/license route: equation-level reproduction; `equations_only`.
- Reproduction level: `faithful_reimplementation`.
- Permitted inputs: `causal history + available context`.
- Output contract: `continuous dispatch`.
- Optimizer role: `none at inference`.
- Implementation class: `external.DigitalTwins-Policy`.
- Adaptation boundary: retain the published learning objective; adapt only tensor/head and device mapping
- Primary-paper anchors: Abstract; Sec. 3, deep-learning embedded scheduling; Sec. 3.2, constraint enforcement; Sec. 4 case studies.
- Forecast representation: optional latent forecast/state.
- Decision layer: neural policy head; output type: continuous dispatch.
- Loss and gradient path: operating-cost plus physical-constraint penalties; decision loss to prediction parameters.
- Topology: published method with adaptation to the Standard-IES contract.
- Required adaptation: map the published output to the 24-to-4 Standard-IES contract while retaining the method's core objective.
- Forbidden substitution: do not use RSC-PF test data, internal controls, or an unreported exact optimizer in the deployed path.
