# External Baseline Source Freeze Design

## Purpose

Create an auditable source snapshot for the three frozen RSC-PF external comparison methods before any adapter or training code is added. This is a provenance gate, not an experiment run.

## Scope

The design covers only:

1. Loading `configs/rsc_pf_external_baselines_v1.json`.
2. Checking the required provenance fields for `iTransformer-PTO`, `DecisionFocused-Online`, and `DigitalTwins-Policy`.
3. Recording source URLs, method identifiers, source type, license/evidence route, retrieval time, and SHA-256 evidence hashes.
4. Writing an immutable, machine-readable receipt under `reports/rsc_pf_external_baselines_v1/implementation/sources`.

It does not modify the RSC-PF model, access the sealed test split, download or execute training data, or claim that an unverified equation/optimizer detail is reproducible.

## Design

The source-preparation function receives a registry path and output root and returns a summary containing the registry hash, per-method receipt paths, and a boolean `implementation_ready`. A method is ready only when it has a stable identifier, primary source URL, a code URL or explicit `equations_only` route, a non-empty license route, a retrieval timestamp, and a valid SHA-256 evidence hash. The existing frozen registry remains the source of method identity; this task does not change method selection.

The generated receipt records:

- registry schema and SHA-256;
- method ID, slot, paper identifier, primary URL, code URL if present;
- reproduction level and license/evidence route;
- source retrieval timestamp and evidence SHA-256;
- `test_set_accessed: false` and `validation_only: true`;
- a per-method pass/fail reason.

The implementation configuration freezes the canonical 24-hour lookback, 4-hour horizon, four-task order, 12 exogenous features, 21 dispatch outputs, 6 status features, five seeds, train-only normalization, optimizer settings, and the separate implementation output root.

## Error handling and stop conditions

The script fails closed on missing provenance, malformed hashes, registry/method mismatch, test-set paths, or a source route that cannot be represented honestly. It must not invent paper equations, optimizer coefficients, or licenses. A failed method is reported as blocked and cannot advance to later implementation tasks.

## Verification

Tests will validate required fields, hash format, registry consistency, test-set isolation, deterministic receipt generation, and the exact canonical configuration. The focused test suite must pass before Task 2 begins.

## Output contract

All generated files are confined to `reports/rsc_pf_external_baselines_v1/implementation/sources`. Existing RSC-PF reports and user changes remain untouched. The source receipt is the input artifact for the later data/model implementation tasks.
