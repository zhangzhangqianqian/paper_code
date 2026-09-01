# RSC-PF External Baseline Reproduction and Validation Implementation Plan

> **For agentic workers:** Use the repository's implementation-plan workflow. Each task is intentionally small, testable, and bounded; do not start the large experiment until the validation gates in Task 7 pass.

**Goal:** Implement and validation-test the three frozen external comparison methods for RSC-PF without changing the frozen RSC-PF graph, leaking future observations, or accessing the sealed 2021 test split.

**Architecture:** Use three distinct behavioral adapters with a shared data and evaluation contract:

- `iTransformer-PTO` / `forecast_pto`: forecast-only iTransformer adapter followed by the same offline rolling LP used by the PTO protocol.
- `DecisionFocused-Online` / `decision_focused`: decision-loss-trained predictor using the verified published downstream-optimization/surrogate-gradient path and its declared optimizer role.
- `DigitalTwins-Policy` / `direct_policy`: direct continuous dispatch policy followed by the canonical physical decoder, with zero exact optimizer calls at inference.

The adapters share only the causal sample ordering, train-only normalization, physical units, five seeds, rolling horizon, metrics, and feasibility checks. They must not import or silently reuse the RSC-PF forecasting graph.

**Tech stack:** Python 3.9+, PyTorch, NumPy, SciPy LP, JSON/JSONL/CSV, pytest, the official MIT iTransformer source, and the repository's Standard-IES benchmark/parameter ledger.

## Global constraints

- Treat `configs/rsc_pf_external_baselines_v1.json` as the frozen method registry; keep the three external slots separate from the internal RSC-PF, PTO, and ablation slots.
- Preserve the canonical dimensions and order: 24-hour history, 4-hour horizon, task order `[electricity, cooling, heating, gas]`, 12 exogenous features, 21 continuous dispatch outputs, and 6 status features.
- Future observations are labels only. No future load, renewable realization, device trajectory, or oracle quantity may enter an external model's inference features.
- Use the same Standard-IES ledger, energy-equivalent units, rolling execution rule, physical decoder, and 21-dimensional output ordering as the frozen protocol.
- Exact LP calls are allowed only for the PTO/evaluation path and for the decision-focused method when the published method explicitly declares a downstream optimizer. Direct policy inference must make zero exact LP calls.
- Use seeds `[2026, 2027, 2028, 2029, 2030]`, train-only normalization, batch size 256, learning rate `1e-3`, weight decay `1e-4`, gradient clipping 1, and the existing 20–30 epoch/patience-5 validation protocol unless the frozen source method requires a documented exception.
- Keep the 2021 test split sealed. No test artifact may be produced in source receipts, calibration, validation, or checkpoint selection.
- Write all new outputs below `reports/rsc_pf_external_baselines_v1/implementation`; never overwrite existing outputs without an explicit resume flag.
- Do not edit the frozen RSC-PF core, manuscript, figures, or the existing formal-protocol configuration as part of this implementation.
- Do not invent equations, optimizer details, or coefficients from an abstract. If the full paper/source does not establish a required training operation, stop that method at the evidence gate and record the blocker rather than creating a placeholder implementation.

## File map

Create only the following implementation-owned files (plus generated reports below the implementation output root):

- `configs/rsc_pf_external_baseline_implementation_v1.json`
- `src/joint_dispatch/external_baseline_data.py`
- `src/joint_dispatch/external_baselines.py`
- `src/joint_dispatch/external_baseline_losses.py`
- `src/joint_dispatch/external_baseline_training.py`
- `scripts/prepare_rsc_pf_external_baseline_sources.py`
- `scripts/run_rsc_pf_external_baselines.py`
- `scripts/evaluate_rsc_pf_external_baselines.py`
- `scripts/freeze_rsc_pf_external_validation.py`
- `tests/test_rsc_pf_external_baseline_data.py`
- `tests/test_rsc_pf_external_baselines.py`
- `tests/test_rsc_pf_external_baseline_losses.py`
- `tests/test_rsc_pf_external_baseline_training.py`
- `tests/test_rsc_pf_external_baseline_runner.py`

Do not add generated checkpoints or cached data to version control.

## Task 1 — Verify source snapshots and freeze the implementation configuration

**Files:** create the implementation config, source-preparation script, and `tests/test_rsc_pf_external_baseline_sources.py`.

**Interfaces:**

```python
def prepare_sources(registry_path: Path, output_root: Path) -> dict: ...
def load_external_implementation_config(path: Path) -> dict: ...
```

**TDD steps:**

1. Write a failing test requiring every registry method to have an identifier, primary URL, source/code URL (or an explicit `equations_only` marker), license, retrieval timestamp, and SHA-256 receipt.
2. Run that focused test and confirm it fails before implementation.
3. Add `rsc-pf-external-implementation-v1` config with the frozen dimensions, seeds, optimizer settings, source root, `validation_only: true`, `test_set_accessed: false`, and separate implementation output root.
4. Implement lawful source verification and immutable receipts. Pin the official iTransformer repository to its recorded revision and license; retain a paper/equation receipt for the other two methods.
5. Run:

   ```powershell
   python scripts/prepare_rsc_pf_external_baseline_sources.py --registry configs/rsc_pf_external_baselines_v1.json --output-root reports/rsc_pf_external_baselines_v1/implementation/sources
   python -m pytest tests/test_rsc_pf_external_baseline_sources.py -q
   ```

6. Commit as `chore: freeze external baseline implementation sources`.

**Stop gate:** if any source cannot be verified or lawfully retrieved, do not proceed with that adapter; write an evidence blocker and leave the other methods isolated.

## Task 2 — Build fair causal data adapters

**Files:** create `src/joint_dispatch/external_baseline_data.py` and `tests/test_rsc_pf_external_baseline_data.py`.

Read `src/joint_dispatch/data.py` and `src/joint_dispatch/pto.py` first; reuse their split ordering and shape checks rather than creating a second convention.

**Common batch contract:**

```python
@dataclass(frozen=True)
class ExternalBaselineBatch:
    load_history: Tensor                 # [B, 24, 4]
    exog_history: Tensor                 # [B, 24, 12]
    device_history: Tensor               # [B, 24, 21]
    device_status: Tensor                # [B, 24, 6]
    scheduler_context: Tensor             # [B, 4, 6]
    previous_chp: Tensor                  # [B, 1]
    forecast_target: Tensor               # [B, 4, 4], label only
    teacher_dispatch: Tensor              # [B, 4, 21], label only
    oracle_first_step_objective: Tensor   # [B]
    split: Literal["train", "validation"]
```

**Interfaces:**

```python
def load_external_batches(split_path: Path, *, batch_size: int, shuffle: bool, seed: int): ...
def build_causal_error_history(split: JointWindowSplit) -> np.ndarray: ...
def assert_external_batch_causal(batch: ExternalBaselineBatch) -> None: ...
def fit_external_normalization(train: JointWindowSplit) -> ExternalNormalization: ...
```

**TDD steps:**

1. Add failing shape/order tests for every tensor and a test that a future label mutation cannot change the feature batch.
2. Add a chronology test proving device history and status history end at the origin and that future truth is kept in labels.
3. Implement train-only normalization and causal error-history construction using only pre-origin rows. Keep the `history_source` and split in the receipt.
4. Run the focused test file; then run the existing data-contract tests.
5. Commit as `feat: add causal external-baseline data adapters`.

**Stop gate:** any future-to-feature dependency or mismatch with the canonical 24/4/21/6 contract blocks all adapters.

## Task 3 — Implement isolated method wrappers

**Files:** create `src/joint_dispatch/external_baselines.py` and `tests/test_rsc_pf_external_baselines.py`.

**Interfaces:**

```python
def build_external_baseline(method_id: str, config: Mapping[str, Any]) -> nn.Module: ...
class ExternalForecastPTO(nn.Module): ...
class DecisionFocusedOnline(nn.Module): ...
class DigitalTwinsPolicy(nn.Module): ...
```

Each wrapper returns a typed output containing forecast/policy tensors, physical dispatch when applicable, and exact-LP-call metadata. All outputs must be finite and satisfy the canonical shape/order checks.

**TDD steps:**

1. Add a test that the iTransformer adapter produces `[2, 4, 4]` from causal history/exogenous inputs.
2. Monkeypatch `solve_dispatch_lp` and assert the direct-policy wrapper makes zero exact LP calls.
3. Assert the decision-focused wrapper exposes a forecast tensor with `requires_grad=True` and a documented downstream-optimizer role.
4. Implement the iTransformer adapter from the pinned official source, adapting only input/output projection and the shared normalization contract; use the existing offline rolling-LP helper for its PTO path.
5. Implement the DecisionFocused-Online wrapper only with the verified full-paper/source surrogate or implicit/SPSA gradient path. If that source detail is unavailable, raise a clear evidence error rather than guessing.
6. Implement Digital Twins as a standalone causal DNN from history/error summaries and scheduler context to continuous policy controls, then call the canonical physical decoder. Do not import `StateConditionedScheme2R`.
7. Run focused tests and commit as `feat: add isolated external baseline wrappers`.

**Stop gate:** no wrapper may call the sealed test split, use future truth, silently import RSC-PF, or claim a gradient path not present in the source evidence.

## Task 4 — Add losses and gradient audits

**Files:** create `src/joint_dispatch/external_baseline_losses.py` and `tests/test_rsc_pf_external_baseline_losses.py`.

**Interfaces:**

```python
def forecast_loss(prediction: Tensor, target: Tensor, normalization: ExternalNormalization) -> Tensor: ...
def decision_focused_loss(output: ExternalBaselineOutput, batch: ExternalBaselineBatch, parameters: Mapping[str, Any]) -> Tensor: ...
def policy_imitation_loss(dispatch: Tensor, teacher_dispatch: Tensor) -> Tensor: ...
def audit_external_gradients(loss: Tensor, module: nn.Module) -> dict[str, Any]: ...
```

**TDD steps:**

1. Test finite weighted normalized-Huber/MAE forecast loss and finite backward gradients.
2. Test the exact published decision-focused surrogate path against a tiny differentiable fixture; reject missing or fabricated coefficients.
3. Test direct-policy imitation and physical feasibility penalties.
4. Test that the PTO LP is treated as a non-differentiable evaluation/teacher operation and does not falsely claim a gradient into the forecast adapter.
5. Implement gradient receipts (parameter names, finite norms, zero/nonzero status, loss components) for every run.
6. Run focused tests and commit as `feat: add external baseline losses and gradient audits`.

**Stop gate:** a decision-focused method cannot be marked implemented if its decision loss does not produce the documented gradient path.

## Task 5 — Build resumable calibration and validation training ✅

**Files:** create `src/joint_dispatch/external_baseline_training.py`, `scripts/run_rsc_pf_external_baselines.py`, and `tests/test_rsc_pf_external_baseline_training.py`.

**Interfaces:**

```python
def run_external_calibration(method_id: str, seed: int, config: Mapping[str, Any]) -> Path: ...
def run_external_validation(method_id: str, seed: int, config: Mapping[str, Any]) -> Path: ...
def load_external_checkpoint(path: Path, *, expected_method: str, expected_seed: int) -> dict[str, Any]: ...
```

**TDD steps:**

1. Add tests rejecting an unknown method/seed, a wrong registry hash, a checkpoint from another method, and any path containing test data.
2. Add an 8-window smoke fixture and assert deterministic resume from a saved checkpoint.
3. Implement calibration for seed 2026 using validation only; it is a feasibility and loss-scale check, not a reported five-seed result.
4. Implement formal validation for all five frozen seeds with 20–30 epochs, patience 5, train-only normalization, and method-specific loss terms.
5. Save `last`, `best`, epoch history, gradient receipts, source/config hashes, split manifest, and method/seed provenance. Require an explicit `--resume` to reuse an output directory.
6. Enforce stage gates: calibration must complete before formal validation; all five seed receipts must exist before freeze.
7. Run smoke commands:

   ```powershell
   python scripts/run_rsc_pf_external_baselines.py --stage calibration --method iTransformer-PTO --seed 2026 --smoke-limit 8
   python scripts/run_rsc_pf_external_baselines.py --stage calibration --method DecisionFocused-Online --seed 2026 --smoke-limit 8
   python scripts/run_rsc_pf_external_baselines.py --stage calibration --method DigitalTwins-Policy --seed 2026 --smoke-limit 8
   ```

8. After calibration receipts pass, formal runs are:

   ```powershell
   python scripts/run_rsc_pf_external_baselines.py --stage validation --method iTransformer-PTO --all-seeds
   python scripts/run_rsc_pf_external_baselines.py --stage validation --method DecisionFocused-Online --all-seeds
   python scripts/run_rsc_pf_external_baselines.py --stage validation --method DigitalTwins-Policy --all-seeds
   ```

9. Run focused tests and commit as `feat: add resumable external baseline training`.

**Stop gate:** do not start all-seed training when calibration has non-finite losses, failed feasibility checks, or unresolved source evidence.

## Task 6 — Evaluate with common metrics and resource accounting ✅

**Files:** create `scripts/evaluate_rsc_pf_external_baselines.py` and `tests/test_rsc_pf_external_baseline_runner.py`.

**Interfaces:**

```python
def evaluate_external_validation(method_id: str, seed: int) -> dict[str, Any]: ...
def write_external_validation_manifest(output_root: Path) -> Path: ...
```

**TDD steps:**

1. Test that all methods receive the same validation windows and metric definitions.
2. Test LP accounting: PTO has one offline LP solve per window; direct policy has zero exact LP calls; the decision-focused method reports exactly the optimizer role authorized by its source.
3. Implement forecast MAE/RMSE/WAPE, first-step and cumulative cost/carbon/regret versus the offline oracle, feasibility/slack/SOC violations, runtime, and finite-output flags.
4. Aggregate five seeds with bootstrap confidence intervals and preserve per-window/per-seed raw values.
5. Add resource gates (including the 500-window LP benchmark and p95 latency target) without changing the paper's model-selection rule.
6. Run evaluation and commit as `feat: add common external-baseline evaluation`.

**Stop gate:** incomplete LP accounting, missing seed manifests, or a test-set read invalidates the comparison and prevents freeze.

## Task 7 — Freeze validation and hand off to the paper/results phase

**Files:** create `scripts/freeze_rsc_pf_external_validation.py` and extend `tests/test_rsc_pf_external_baseline_runner.py` with freeze checks.

**Interface:**

```python
def freeze_external_validation(output_root: Path, registry_path: Path) -> dict[str, Any]: ...
```

**TDD steps:**

1. Add a fixture that fails when any of the three methods or any of the five seed receipts is missing.
2. Add a complete fixture and assert that the freeze receipt contains source/config/artifact hashes, metric definitions, LP-role accounting, feasibility flags, and `test_set_accessed: false`.
3. Require finite metrics, complete receipts, and all stage gates before freezing.
4. Write `validation_freeze_receipt.json`, a machine-readable summary, and `implementation_to_test_handoff.md` under `reports/rsc_pf_external_baselines_v1/implementation`.
5. Run:

   ```powershell
   python scripts/freeze_rsc_pf_external_validation.py --registry configs/rsc_pf_external_baselines_v1.json --output-root reports/rsc_pf_external_baselines_v1/implementation
   python -m pytest tests/test_rsc_pf_external_baseline_sources.py tests/test_rsc_pf_external_baseline_data.py tests/test_rsc_pf_external_baselines.py tests/test_rsc_pf_external_baseline_losses.py tests/test_rsc_pf_external_baseline_training.py tests/test_rsc_pf_external_baseline_runner.py --basetemp D:/Paper/pytest_tmp_rsc_pf_external_impl -p no:cacheprovider -q
   ```

6. Commit as `docs: freeze external baseline validation handoff`.

**Stop gate:** this task must not run or expose the sealed 2021 test split. The resulting handoff is the input to the later results-table/figure/manuscript phase, not permission to rewrite RSC-PF or select a favorable seed.

## Plan self-review checklist

- **Requirements coverage:** each frozen external method has a source gate, causal data adapter, isolated model wrapper, method-appropriate loss, resumable five-seed validation, common evaluation, and a freeze receipt.
- **Contract coverage:** 24-hour history, 4-hour horizon, four task order, 12 exogenous features, 21 dispatch outputs, 6 status features, five seeds, and Standard-IES units are explicit and tested.
- **Leakage coverage:** future truth is labels only; train-only normalization; no test artifacts; chronology and path-level checks are required.
- **Behavioral separation:** PTO, decision-focused, and direct-policy inference have distinct optimizer roles and explicit LP-call accounting.
- **No placeholders:** the plan contains no “implement later” step. Missing published equations are an evidence stop condition, not an invitation to guess.
- **Execution order:** source verification → data → wrappers → losses → calibration/validation → evaluation → freeze. Large runs start only after focused tests and calibration gates pass.

## Stop rules after execution

Stop and report the exact blocker if source evidence is insufficient, a shape/order contract fails, future data enters features, a gradient receipt is non-finite or undocumented, a direct policy makes exact LP calls, or any test-set access is detected. Do not silently substitute another method or modify the frozen RSC-PF model.

## Mandatory self-review after every task

Before starting the next task, perform and record a self-review for the task just completed:

1. Run the task's focused tests and the relevant existing regression tests.
2. Inspect the diff and confirm that only implementation-owned files and the task's output directory changed.
3. Check the task's stated interfaces, tensor shapes, feature chronology, optimizer-role accounting, and provenance fields.
4. Search generated code and receipts for test-set paths, future-label features, non-finite values, placeholder claims, and undocumented substitutions.
5. Write a task receipt containing the test command, result, changed-file list, contract checks, and any residual risks.
6. Advance only when the receipt is passing; otherwise apply the stop rules and report the blocker.

The same self-review applies to the calibration run, each formal seed group, common evaluation, and final freeze. A passing test alone is not sufficient to advance.
