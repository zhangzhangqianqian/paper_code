# RSC-PF External Baseline v4.6 Protocol Adaptation Implementation Plan

> **For agentic workers:** Execute this plan inline with a test checkpoint after each task. Do not start five-seed training until the short calibration gate passes for all three methods.

**Goal:** Adapt the three frozen external baselines to the current formal v4.6 data contract and validate them without mixing the obsolete v1 data protocol.

**Architecture:** Keep the RSC-PF core unchanged. Add an external-only v4.6 window adapter using 17-dimensional historical device outputs plus six status features, add the missing PTO interface, then run short calibration before any long validation training.

**Tech Stack:** Python, NumPy, PyTorch, SciPy LP, pytest, JSON/NPZ receipts.

## Global Constraints

- Use the current formal v4.6 train/early-stop/2019 selection artifacts only.
- Keep the sealed test split unopened and write `test_set_accessed: false` in every receipt.
- Preserve four forecast tasks and the canonical 21-dimensional future dispatch order.
- Do not change the RSC-PF model, frozen formal contract, or manuscript in this work.
- Report iTransformer as an official-source adaptation, not an untouched official reproduction.
- Do not start five-seed training until all short calibration gates pass.

---

### Task 1: Record and validate the v4.6 external data adapter

**Files:**
- Create: `src/joint_dispatch/external_v46_data.py`
- Modify: `src/joint_dispatch/external_baseline_training.py`
- Modify: `scripts/evaluate_rsc_pf_external_baselines.py`
- Test: `tests/test_rsc_pf_external_v46_data.py`

**Interfaces:**
- `load_external_v46_split(path: Path, split_name: Literal["train","validation","pilot"]) -> ExternalV46Split`
- `build_external_v46_teacher(split: ExternalV46Split, parameters: Mapping[str, Any]) -> np.ndarray`
- `fit_external_v46_normalization(train: ExternalV46Split) -> ExternalV46Normalization`

- [ ] Write tests for the exact `[24,4]`, `[24,12]`, `[24,17]`, `[24,6]`, `[4,4]`, `[4,21]` shapes.
- [ ] Add a causality test that mutating future targets does not change history/context tensors.
- [ ] Implement NPZ loading from the current formal v4.6 materialized artifacts and construct the six-column scheduler context.
- [ ] Generate missing teacher dispatch labels with the same offline LP parameters used by the formal v4.6 protocol; keep them label-only.
- [ ] Fit normalization from train only and record source path/hash and split role.
- [ ] Run the focused data tests and the existing external baseline contract tests.
- [ ] Commit: `feat: adapt external baselines to formal v4.6 data contract`.

### Task 2: Add and test the common PTO interface

**Files:**
- Create: `src/joint_dispatch/pto.py`
- Test: `tests/test_joint_dispatch_pto.py`

**Interfaces:**
- `PTOForecasts(method_id: str, prediction: np.ndarray, target: np.ndarray)`
- `PTODispatchCache(dispatch: np.ndarray, success: np.ndarray, messages: tuple[str, ...], offline_exact_lp_calls: int)`
- `solve_pto_windows(forecasts: PTOForecasts, split: ExternalV46Split, parameters: Mapping[str, Any]) -> PTODispatchCache`
- `seasonal_naive_forecasts(load_history: np.ndarray, season_length: int = 24) -> np.ndarray`

- [ ] Write tests for finite forecast shapes, one LP call per window, and failure propagation.
- [ ] Implement the LP bridge using the canonical dispatch order and current prices/weights/SOC.
- [ ] Reject test-role splits and non-finite inputs.
- [ ] Ensure no LP call is made by direct policy inference.
- [ ] Run the PTO tests and import checks for `src.joint_dispatch` lazy exports.
- [ ] Commit: `feat: add canonical PTO bridge for external baselines`.

### Task 3: Run short calibration only

**Files:**
- Modify: `configs/rsc_pf_external_baseline_implementation_v1.json`
- Modify: `scripts/run_rsc_pf_external_baselines.py`
- Modify: `scripts/evaluate_rsc_pf_external_baselines.py`
- Test: `tests/test_rsc_pf_external_baseline_runner.py`

- [ ] Point the implementation config to the v4.6 train and early-stop artifacts, with explicit `pilot_path` for 2019 selection.
- [ ] Use a small deterministic subset (no more than 16 train and 16 early-stop windows), one seed, and at most two epochs.
- [ ] Run calibration for iTransformer-PTO, DecisionFocused-Online, and DigitalTwins-Policy.
- [ ] Verify finite losses, causal feature hashes, physical feasibility, LP-call roles, and no test path access.
- [ ] If any method fails, write a blocker receipt and stop before all-seed training.
- [ ] Commit: `test: pass external baseline v4.6 calibration gate` only if all three pass.

### Task 4: Start formal external validation only after calibration passes

**Files:**
- Modify: `scripts/run_rsc_pf_external_baselines.py`
- Modify: `scripts/evaluate_rsc_pf_external_baselines.py`
- Create: `scripts/run_rsc_pf_external_v46_pilot.py`
- Test: `tests/test_rsc_pf_external_v46_pilot.py`

- [ ] Run each approved method for seeds 2026–2030 with the frozen 20–30 epoch budget on train/early-stop only.
- [ ] Select checkpoints using early-stop loss only.
- [ ] Evaluate 2019 selection_full with forecast MAE/RMSE/WAPE, operating cost, physical carbon, penalized objective, first-step regret, shortage, feasibility, latency, and exact LP calls.
- [ ] Write one receipt per method/seed and an aggregate manifest with adaptation disclosures.
- [ ] Stop if any receipt is incomplete or any test access flag is not false.
- [ ] Commit: `feat: run formal external baseline validation under v4.6`.

### Task 5: Review external comparison before manuscript integration

**Files:**
- Create: `reports/rsc_pf_external_baselines_v46/EXTERNAL_COMPARISON_REVIEW.md`
- Create: `reports/rsc_pf_external_baselines_v46/external_validation_manifest.json`

- [ ] Compare RSC-PF, Fair Decoupled, and the three external baselines under identical 2019 selection windows.
- [ ] Separate forecast quality, dispatch quality, optimizer role, and deployment latency; do not rank by a single metric.
- [ ] State clearly which methods are true end-to-end, which are decision-focused with an optimizer, and which are PTO.
- [ ] Do not modify the paper until this review confirms protocol and provenance completeness.
