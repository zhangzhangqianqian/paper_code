# RSC-PF External Literature Baselines Implementation Plan

> **Audit status (2026-09-01):** Retained as the overall comparison roadmap, not as the first executable plan. Tasks 4–11 depend on exact papers that are not yet frozen. Execute `docs/superpowers/plans/2026-09-01-rsc-pf-external-baseline-selection.md` first; generate a source-specific reproduction/training plan only after its Task 5 audit authorizes handoff.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select, reproduce, and formally compare three published forecast–scheduling baselines against the frozen RSC-PF model under one causal rolling Standard-IES protocol before any ablation experiment begins.

**Architecture:** The pipeline is fail-closed and has three phases. Phase A performs multi-source literature discovery, DOI deduplication, compatibility scoring, and immutable selection of exactly one external Forecast-PTO, one decision-focused/differentiable-optimization, and one direct end-to-end policy baseline. Phase B adapts the selected algorithms behind a shared causal batch/output interface without changing their coupling mechanism or the frozen RSC-PF implementation. Phase C performs validation-only model selection, seals five-seed checkpoints, evaluates the sealed 2021 test once, and reports paired uncertainty plus optimizer/runtime accounting.

**Tech Stack:** Python 3.9, PyTorch, NumPy, pandas, pytest, JSON/CSV/BibTeX, CrossRef/arXiv/Semantic Scholar metadata, existing `JointWindowSplit`, Standard-IES LP/decoder, moving-block bootstrap, Benjamini–Hochberg FDR correction.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-09-01-rsc-pf-literature-baselines-design.md`.
- Do not modify the frozen RSC-PF architecture, `configs/joint_forecast_dispatch_contract_v2.json`, its ten joint checkpoints, or any existing v2 validation/test receipt.
- Do not run ablation experiments in this plan.
- External literature baselines, internal controls, RSC-PF, and Oracle must remain separate categories in every contract, table, and figure.
- The external slots are exactly `External-Forecast-PTO`, `External-Decision-Focused`, and `External-Direct-Policy`.
- Each external slot must resolve to one primary paper with a DOI or stable preprint identifier, a reproducible algorithm definition, a recorded software license, and a compatibility score of at least 70/100.
- A missing slot causes `selection_status=failed`; no substitute may be invented from an internal training variant.
- Search years are 2016–2026. Search CrossRef and arXiv first, Semantic Scholar second, and Scopus/ScienceDirect only when T1/T2 evidence is insufficient.
- Deduplicate by normalized DOI; when DOI is absent, require identical first-author surname and normalized-title Jaccard similarity of at least 0.90.
- Data splits remain 2015–2019 train, 2020 validation, and sealed 2021 test with a 24-hour causal history and four-hour horizon.
- Trainable external methods use seeds 2026, 2027, 2028, 2029, and 2030; no best-seed selection is allowed.
- Checkpoint and hyperparameter selection use validation only. Test artifacts stay inaccessible until the external validation freeze receipt exists.
- All methods receive the same available causal information. A source architecture may ignore fields it does not consume, but it may not receive future realized information.
- All methods are settled with the same Standard-IES topology, physical parameters, rolling state transitions, realized-demand recourse, and metric functions.
- `External-Forecast-PTO` must use the same exact LP as the internal PTO controls after producing its forecasts.
- `External-Decision-Focused` must preserve the selected paper's decision-gradient mechanism.
- `External-Direct-Policy` must preserve the selected paper's direct policy principle and use the shared 15-control-to-21-dispatch physical interface when topology adaptation is required.
- Report both actual experiment-time optimizer calls and deployment-required optimizer calls. Cached PTO execution does not justify claiming zero optimizer dependence at deployment.
- Primary paired comparisons are RSC-PF versus each external baseline. Use 168-hour moving blocks, 2,000 bootstrap replicates, α=0.05, and Benjamini–Hochberg correction within each endpoint family.
- If a 95% interval crosses zero or the adjusted p value exceeds 0.05, write “no statistically reliable difference was detected.”
- Use lawful full text and code only; record source URL, access route, code repository, commit/tag, and license.
- Existing user changes in the dirty worktree are out of scope and must be preserved.

---

## File Structure

- Create `configs/rsc_pf_external_baseline_search_v1.json`: immutable query families, search sources, years, scoring weights, fatal exclusions, and required slots.
- Create `configs/rsc_pf_external_baselines_v1.json`: exact selected papers and implementation identities; generated only after literature screening passes.
- Create `src/joint_dispatch/external_registry.py`: strict dataclasses and loaders for search candidates and the frozen registry.
- Create `src/joint_dispatch/external_protocol.py`: shared external-baseline batch, output, accounting, and checkpoint interfaces.
- Create `src/joint_dispatch/external_forecast_pto.py`: adapter from the selected literature forecaster to the shared exact LP.
- Create `src/joint_dispatch/external_decision_focused.py`: selected decision-focused/differentiable-optimization reproduction.
- Create `src/joint_dispatch/external_direct_policy.py`: selected direct-policy reproduction using the common physical dispatch environment.
- Create `src/joint_dispatch/external_evaluation.py`: common rolling evaluation, deployment optimizer accounting, paired statistics, and category-safe summaries.
- Create `scripts/normalize_rsc_pf_baseline_search.py`: normalize multi-source search exports, deduplicate, score, and generate evidence cards.
- Create `scripts/freeze_rsc_pf_external_baselines.py`: fail-closed selection and registry freeze.
- Create `scripts/run_rsc_pf_external_baselines.py`: smoke, validation, freeze, and sealed-test runner.
- Create `scripts/summarize_rsc_pf_external_comparison.py`: combine existing v2 receipts and new external receipts without rewriting either source.
- Create `scripts/audit_rsc_pf_external_baselines.py`: provenance, fairness, leakage, receipt, and completeness audit.
- Create `tests/test_rsc_pf_external_registry.py`.
- Create `tests/test_rsc_pf_external_protocol.py`.
- Create `tests/test_rsc_pf_external_forecast_pto.py`.
- Create `tests/test_rsc_pf_external_decision_focused.py`.
- Create `tests/test_rsc_pf_external_direct_policy.py`.
- Create `tests/test_rsc_pf_external_runner.py`.
- Create `tests/test_rsc_pf_external_evaluation.py`.
- Generate under `reports/rsc_pf_external_baselines_v1/`: literature evidence, validation/test receipts, freeze receipts, comparison tables, statistics, figures, and audit reports.
- Reuse without modification: `src/external_models.py` and existing DLinear/MMoE-lite/SOFTS forecasting artifacts where their provenance passes the new registry and validation gates.

---

### Task 1: Freeze the Literature Search and Registry Schemas

**Files:**
- Create: `configs/rsc_pf_external_baseline_search_v1.json`
- Create: `src/joint_dispatch/external_registry.py`
- Create: `tests/test_rsc_pf_external_registry.py`
- Read: `docs/superpowers/specs/2026-09-01-rsc-pf-literature-baselines-design.md`
- Read: `configs/joint_forecast_dispatch_contract_v2.json`

**Interfaces:**
- Produces `SearchProtocol`, `LiteratureCandidate`, `ExternalBaselineSpec`, and `ExternalBaselineRegistry` frozen dataclasses.
- Produces `load_search_protocol(path: str | Path) -> SearchProtocol`.
- Produces `load_external_registry(path: str | Path) -> ExternalBaselineRegistry`.
- `ExternalBaselineSpec.slot` is one of `forecast_pto`, `decision_focused`, or `direct_policy`.
- `ExternalBaselineSpec.reproduction_level` is one of `official_code_exact`, `faithful_reimplementation`, or `principled_adaptation`.

- [ ] **Step 1: Write strict schema tests**

```python
def test_search_protocol_freezes_sources_queries_and_scores(frame_root: Path) -> None:
    protocol = load_search_protocol(frame_root / "configs/rsc_pf_external_baseline_search_v1.json")
    assert protocol.year_start == 2016 and protocol.year_end == 2026
    assert protocol.required_slots == ("forecast_pto", "decision_focused", "direct_policy")
    assert protocol.sources_t1 == ("crossref", "arxiv")
    assert protocol.sources_t2 == ("semantic_scholar",)
    assert sum(protocol.score_weights.values()) == 100
    assert protocol.minimum_score == 70


def test_registry_rejects_internal_variants_as_external(tmp_path: Path) -> None:
    path = write_registry_fixture(tmp_path, method_id="From-Scratch-Joint", slot="decision_focused")
    with pytest.raises(ValueError, match="internal method cannot fill an external slot"):
        load_external_registry(path)
```

- [ ] **Step 2: Run the tests and verify missing schema code fails**

Run:

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -q
```

Expected: FAIL because the protocol, dataclasses, loader, and fixture helper do not exist.

- [ ] **Step 3: Create the exact search protocol JSON**

The JSON must store the five approved query families verbatim, T1/T2/T3 routing, 2016–2026 bounds, DOI/title-author deduplication rules, six fatal exclusions, score weights `{coupling:25, continuous_dispatch:20, reproducibility:20, information_fairness:15, venue_citation:10, recency:5, resource_fit:5}`, minimum score 70, and three required slots.

- [ ] **Step 4: Implement strict loaders**

Reject unknown fields, missing DOI/stable ID, missing license status, scores outside `[0,100]`, total weights other than 100, duplicate normalized DOI, duplicate slots, internal v2 method names, and a registry that does not contain exactly three selected specs.

- [ ] **Step 5: Run registry tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the search contract**

```powershell
git add configs/rsc_pf_external_baseline_search_v1.json src/joint_dispatch/external_registry.py tests/test_rsc_pf_external_registry.py
git commit -m "feat: define external baseline search contract"
```

---

### Task 2: Execute Multi-Source Discovery and Build Candidate Evidence

**Files:**
- Create: `scripts/normalize_rsc_pf_baseline_search.py`
- Modify: `tests/test_rsc_pf_external_registry.py`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/raw/crossref.json`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/raw/arxiv.json`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/raw/semantic_scholar.json`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/deduplicated_candidates.csv`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/search_receipt.json`

**Interfaces:**
- Consumes search exports with `source`, `title`, `authors`, `year`, `doi`, `stable_id`, `venue`, `abstract`, `citation_count`, `url`, and `code_url`.
- Produces `normalize_candidate(record: Mapping[str, Any]) -> LiteratureCandidate`.
- Produces `deduplicate_candidates(candidates: Sequence[LiteratureCandidate]) -> tuple[LiteratureCandidate, ...]`.
- Produces `write_search_receipt(...)` containing queries, sources attempted, failures, raw counts, deduplicated count, and file hashes.

- [ ] **Step 1: Write DOI and title-author deduplication tests**

```python
def test_dedup_prefers_publisher_record_for_same_doi() -> None:
    merged = deduplicate_candidates((crossref_record(), arxiv_duplicate()))
    assert len(merged) == 1
    assert merged[0].source == "crossref"


def test_dedup_uses_title_jaccard_and_first_author_without_doi() -> None:
    merged = deduplicate_candidates((no_doi_record("Zhang"), no_doi_variant("Zhang")))
    assert len(merged) == 1
```

- [ ] **Step 2: Run the tests and verify normalization code is missing**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -k "dedup" -q
```

Expected: FAIL because normalization and deduplication functions do not exist.

- [ ] **Step 3: Run the five query families through T1 sources**

Use CrossRef and arXiv for every query family. Save each complete returned record with the exact query and retrieval timestamp. If either T1 source fails, record the error and continue with the other source.

- [ ] **Step 4: Escalate to Semantic Scholar when slot coverage is insufficient**

Escalation occurs when a slot has fewer than ten unique candidates after T1 screening. Save Semantic Scholar exports separately. Use Scopus or ScienceDirect only if a slot still has fewer than five candidates, and record that escalation in the receipt.

- [ ] **Step 5: Implement normalization and deterministic deduplication**

Normalize DOI to lowercase without `https://doi.org/`. For DOI-less records, normalize title punctuation/stopwords and require first-author surname equality plus Jaccard `>=0.90`. Prefer DOI+volume+pages completeness, publisher over preprint, then higher citation count.

- [ ] **Step 6: Generate and validate the search receipt**

```powershell
& $Py scripts\normalize_rsc_pf_baseline_search.py --protocol configs\rsc_pf_external_baseline_search_v1.json --raw-root reports\rsc_pf_external_baselines_v1\literature\raw --output-root reports\rsc_pf_external_baselines_v1\literature
```

Expected: a nonempty deduplicated CSV, no duplicate DOI, source-attempt records for all five query families, and explicit failures for any unavailable provider.

- [ ] **Step 7: Commit search tooling and immutable raw metadata**

```powershell
git add scripts/normalize_rsc_pf_baseline_search.py tests/test_rsc_pf_external_registry.py reports/rsc_pf_external_baselines_v1/literature
git commit -m "data: collect joint forecast dispatch baseline candidates"
```

---

### Task 3: Screen Full Methods and Freeze Exactly Three External Baselines

**Files:**
- Create: `scripts/freeze_rsc_pf_external_baselines.py`
- Modify: `src/joint_dispatch/external_registry.py`
- Modify: `tests/test_rsc_pf_external_registry.py`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/screening_scores.csv`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/evidence_cards.md`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/baseline_references.bib`
- Generate: `configs/rsc_pf_external_baselines_v1.json`
- Generate: `reports/rsc_pf_external_baselines_v1/frozen/literature_selection_receipt.json`

**Interfaces:**
- Produces `score_candidate(candidate, evidence) -> CandidateScore` with seven fixed components totaling 100.
- Produces `select_external_slots(scores: Sequence[CandidateScore]) -> ExternalBaselineRegistry`.
- Produces one exact implementation class path, reproduction level, code commit/tag, license, input mode, output mode, optimizer dependence, and adaptation disclosure per selected method.

- [ ] **Step 1: Write fail-closed selection tests**

```python
def test_selection_requires_one_method_per_external_slot() -> None:
    scores = scored_candidates_without_direct_policy()
    with pytest.raises(ValueError, match="unfilled external slot: direct_policy"):
        select_external_slots(scores)


def test_selection_uses_score_then_reproducibility_then_recency() -> None:
    selected = select_external_slots(tied_scored_candidates())
    assert selected.by_slot("decision_focused").method_id == "higher_reproducibility"
```

- [ ] **Step 2: Run the tests and verify selection functions are missing**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -k "selection" -q
```

Expected: FAIL because scoring and slot selection do not exist.

- [ ] **Step 3: Obtain lawful method/code evidence for high-ranked candidates**

For the top five candidates per slot, record full method equations, training objective, inference flow, output type, continuous/binary decision boundary, code URL, repository tag/commit, license, and every required adaptation. A paper is fatally excluded if any design exclusion applies.

- [ ] **Step 4: Implement deterministic scoring and selection**

Calculate each component from explicit evidence fields. Select the highest total score `>=70` per slot; break ties by reproducibility score, coupling score, then publication year. The script must fail if fewer than three slots pass.

- [ ] **Step 5: Generate evidence cards and exact registry**

```powershell
& $Py scripts\freeze_rsc_pf_external_baselines.py --protocol configs\rsc_pf_external_baseline_search_v1.json --candidates reports\rsc_pf_external_baselines_v1\literature\deduplicated_candidates.csv --scores reports\rsc_pf_external_baselines_v1\literature\screening_scores.csv --registry configs\rsc_pf_external_baselines_v1.json --receipt reports\rsc_pf_external_baselines_v1\frozen\literature_selection_receipt.json
```

Expected: exactly three external entries, all scores at least 70, no fatal exclusions, complete DOI/stable IDs, code/license evidence, and SHA-256 hashes for the protocol, scores, registry, and evidence cards.

- [ ] **Step 6: Audit manuscript labels before implementation**

Require every `principled_adaptation` entry to have a publication label ending in `-Adapted`; prohibit “exact reproduction” unless `reproduction_level=official_code_exact`.

- [ ] **Step 7: Run all registry tests and commit the frozen selection**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -q
git add configs/rsc_pf_external_baselines_v1.json scripts/freeze_rsc_pf_external_baselines.py src/joint_dispatch/external_registry.py tests/test_rsc_pf_external_registry.py reports/rsc_pf_external_baselines_v1/literature reports/rsc_pf_external_baselines_v1/frozen/literature_selection_receipt.json
git commit -m "docs: freeze published external baseline selection"
```

---

### Task 4: Build the Shared Causal Baseline Runtime Contract

**Files:**
- Create: `src/joint_dispatch/external_protocol.py`
- Create: `tests/test_rsc_pf_external_protocol.py`
- Read: `src/joint_dispatch/data.py`
- Read: `src/joint_dispatch/rollout.py`
- Read: `src/joint_dispatch/evaluation.py`

**Interfaces:**
- Produces `ExternalBaselineBatch` with normalized causal histories plus raw scheduler context, previous CHP, targets, teacher dispatch, and Oracle first-step objective.
- Produces `ExternalBaselineOutput(forecast, dispatch, control_logits, training_optimizer_calls, inference_optimizer_calls, deployment_required_optimizer_calls)`.
- Defines abstract `ExternalBaselineAdapter.fit_epoch(batch)`, `forward(batch)`, `state_dict()`, and `load_state_dict(...)`.
- Produces `build_external_batch(raw_split, normalized_split, indices) -> ExternalBaselineBatch`.
- Produces `validate_information_boundary(spec, batch) -> None`.

- [ ] **Step 1: Write exact batch/output contract tests**

```python
def test_external_batch_matches_frozen_24_to_4_protocol(joint_split_fixture) -> None:
    batch = build_external_batch(*joint_split_fixture, indices=np.arange(3))
    assert batch.load_history.shape == (3, 24, 4)
    assert batch.exog_history.shape == (3, 24, 12)
    assert batch.device_history.shape == (3, 24, 21)
    assert batch.device_status.shape == (3, 24, 6)
    assert batch.scheduler_context.shape == (3, 4, 6)


def test_output_separates_execution_and_deployment_optimizer_calls() -> None:
    output = output_fixture(inference_optimizer_calls=0, deployment_required_optimizer_calls=8)
    assert output.inference_optimizer_calls == 0
    assert output.deployment_required_optimizer_calls == 8
```

- [ ] **Step 2: Run tests and verify the protocol is missing**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_protocol.py -q
```

Expected: FAIL because the batch, output, and adapter interfaces do not exist.

- [ ] **Step 3: Implement immutable tensor contracts and validators**

Validate all exact dimensions, finite values, binary historical status, SOC in `[0,1]`, target separation, and chronological origin order. Reject future target tensors as model inputs and record each adapter's consumed field names.

- [ ] **Step 4: Add a paired future-perturbation leakage test**

Perturb all true values after an origin while holding histories and approved forecasts constant; the adapter input and output at that origin must remain unchanged. Oracle output must change and is tested separately as a reference.

- [ ] **Step 5: Run protocol and existing data/rollout tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_protocol.py tests\test_joint_dispatch_data.py tests\test_joint_dispatch_rollout.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit the common runtime**

```powershell
git add src/joint_dispatch/external_protocol.py tests/test_rsc_pf_external_protocol.py
git commit -m "feat: add causal external baseline runtime"
```

---

### Task 5: Implement the Selected Literature Forecast-PTO Baseline

**Files:**
- Create: `src/joint_dispatch/external_forecast_pto.py`
- Create: `tests/test_rsc_pf_external_forecast_pto.py`
- Modify only if selected and provenance-compatible: `src/external_models.py`
- Read: `src/joint_dispatch/pto.py`
- Read: `scripts/train_external_baseline.py`

**Interfaces:**
- Consumes the registry entry for slot `forecast_pto` and a model factory registered by its immutable `implementation_class`.
- Produces `ExternalForecastPTOAdapter`, four-task forecasts `[B,4,4]`, LP dispatch `[B,4,21]`, and deployment optimizer calls equal to the number of forecast origins.
- Reuses `solve_pto_windows(...)` and the exact Standard-IES parameters.

- [ ] **Step 1: Write forecast-PTO adapter tests**

```python
def test_forecast_pto_uses_selected_literature_forecaster_and_shared_lp(registry, batch) -> None:
    adapter = ExternalForecastPTOAdapter.from_registry(registry.by_slot("forecast_pto"))
    output = adapter.forward(batch)
    assert output.forecast.shape == (len(batch), 4, 4)
    assert output.dispatch.shape == (len(batch), 4, 21)
    assert output.deployment_required_optimizer_calls == len(batch)


def test_forecast_pto_never_uses_joint_finetuned_scheme2r_checkpoint(adapter) -> None:
    assert "Warm-Start-Joint" not in adapter.provenance.checkpoint_source
```

- [ ] **Step 2: Run tests and verify the adapter is missing**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_forecast_pto.py -q
```

Expected: FAIL because the selected forecaster adapter does not exist.

- [ ] **Step 3: Reuse a selected DLinear/MMoE-lite/SOFTS implementation only when registry identity matches**

Load the chosen registry entry, map its approved causal input mode to `ExternalBaselineBatch`, and reuse the existing forecasting class when class/equations/provenance agree. Otherwise implement the exact selected architecture in this module. Never choose among candidates using test WAPE.

- [ ] **Step 4: Train five forecast-only seeds on the shared split**

Use forecast loss only, v2 training normalization, maximum 30 epochs, validation early stopping, and seeds 2026–2030. Save one receipt per seed with paper ID, code commit, registry hash, split hashes, hyperparameters, and `test_set_accessed=false`.

- [ ] **Step 5: Generate validation LP caches and optimizer accounting**

Produce four-hour forecasts for every validation origin and solve the same exact LP. Record actual cache-generation calls and `deployment_required_optimizer_calls=number_of_origins` even though cached evaluation performs zero new calls.

- [ ] **Step 6: Run adapter/PTO/LP regression tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_forecast_pto.py tests\test_joint_dispatch_pto.py tests\test_dispatch_lp.py -q
```

Expected: all tests pass with zero LP failures.

- [ ] **Step 7: Commit the Forecast-PTO implementation**

```powershell
git add src/joint_dispatch/external_forecast_pto.py tests/test_rsc_pf_external_forecast_pto.py
git commit -m "feat: add published forecast PTO baseline"
```

---

### Task 6: Implement the Selected Decision-Focused Baseline

**Files:**
- Create: `src/joint_dispatch/external_decision_focused.py`
- Create: `tests/test_rsc_pf_external_decision_focused.py`
- Read: `src/joint_dispatch/losses.py`
- Read: `src/joint_dispatch/training.py`
- Read: `src/scheduling/proxy_decoder.py`

**Interfaces:**
- Consumes the frozen `decision_focused` registry entry.
- Produces `ExternalDecisionFocusedAdapter` with the selected paper's forecast/decision coupling and exact recorded reproduction level.
- Produces a decision-derived scalar whose gradient reaches every trainable upstream predictor parameter declared by the selected method.

- [ ] **Step 1: Write coupling and gradient tests**

```python
def test_decision_focused_loss_updates_predictor_when_forecast_weight_zero(adapter, batch) -> None:
    adapter.zero_grad()
    loss = adapter.loss(batch, forecast_weight=0.0, decision_weight=1.0)
    loss.backward()
    assert finite_nonzero_grad_norm(adapter.predictor_parameters()) > 0.0


def test_decision_focused_adapter_declares_solver_role(adapter) -> None:
    assert adapter.provenance.reproduction_level in {
        "official_code_exact", "faithful_reimplementation", "principled_adaptation"
    }
    assert adapter.optimizer_accounting.training_calls >= 0
```

- [ ] **Step 2: Run tests and verify the selected coupling is absent**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_decision_focused.py -q
```

Expected: FAIL because the selected decision-focused adapter is not implemented.

- [ ] **Step 3: Implement the frozen paper's decision-gradient mechanism**

Follow the registry's implementation class and evidence card. Preserve whether the source uses an exact differentiable optimization layer, surrogate decision loss, SPO-style loss, or implicit differentiation. Map only the data and Standard-IES decision dimensions; record every deviation in the adapter provenance.

- [ ] **Step 4: Add gradient and numerical consistency audits**

Verify decision-only gradients by autograd and centered finite difference away from piecewise kinks. If an exact solver is nondifferentiable, implement the source paper's documented surrogate rather than silently substituting RSC-PF loss.

- [ ] **Step 5: Train a bounded smoke seed and enforce the resource gate**

Run seed 2026 for two epochs on 512 training and 128 validation windows. Require finite loss, nonzero predictor/decision gradients, no leakage, and projected five-seed runtime below 24 hours. Stop formal execution if the resource gate fails.

- [ ] **Step 6: Run decision-focused tests and commit**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_decision_focused.py tests\test_joint_dispatch_losses.py tests\test_joint_dispatch_training.py -q
git add src/joint_dispatch/external_decision_focused.py tests/test_rsc_pf_external_decision_focused.py
git commit -m "feat: add published decision focused baseline"
```

---

### Task 7: Implement the Selected Direct End-to-End Policy Baseline

**Files:**
- Create: `src/joint_dispatch/external_direct_policy.py`
- Create: `tests/test_rsc_pf_external_direct_policy.py`
- Read: `src/scheduling/proxy_decoder.py`
- Read: `src/joint_dispatch/model.py`

**Interfaces:**
- Consumes the frozen `direct_policy` registry entry and `ExternalBaselineBatch`.
- Produces `ExternalDirectPolicyAdapter` and continuous control logits `[B,15]` decoded to `[B,4,21]` when the selected paper requires topology adaptation.
- Does not instantiate `StateConditionedScheme2R` or copy its router/state-fusion modules unless the selected primary paper independently defines the same component.

- [ ] **Step 1: Write architecture-independence and output tests**

```python
def test_direct_policy_is_not_rsc_pf_with_a_new_name(adapter, batch) -> None:
    module_names = {type(module).__name__ for module in adapter.modules()}
    assert "StateConditionedScheme2R" not in module_names
    output = adapter.forward(batch)
    assert output.control_logits.shape == (len(batch), 15)
    assert output.dispatch.shape == (len(batch), 4, 21)


def test_direct_policy_requires_no_deployment_optimizer(adapter, batch) -> None:
    output = adapter.forward(batch)
    assert output.deployment_required_optimizer_calls == 0
```

- [ ] **Step 2: Run tests and verify the direct policy is absent**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_direct_policy.py -q
```

Expected: FAIL because the selected direct-policy adapter does not exist.

- [ ] **Step 3: Implement the selected policy backbone and published loss**

Preserve the source paper's temporal encoder, decision head, and training objective. Provide the same causal input availability; record which fields the source architecture actually consumes. If the native output cannot represent Standard-IES, expose 15 continuous horizon controls and use the shared decoder, marking the run `principled_adaptation`.

- [ ] **Step 4: Add feasibility and leakage tests**

Require finite outputs, exact tensor dimensions, decoder balance/conversion/SOC feasibility, and invariance to future-target perturbation. Count shared decoder use explicitly; do not call LP during inference.

- [ ] **Step 5: Run a bounded smoke seed and resource gate**

Use the same smoke sample counts and two-epoch seed 2026 protocol as Task 6. Require projected formal runtime below 24 hours and record parameter count/peak memory.

- [ ] **Step 6: Run tests and commit**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_direct_policy.py tests\test_scheduling_proxy_decoder.py tests\test_scheduling_proxy_physics.py -q
git add src/joint_dispatch/external_direct_policy.py tests/test_rsc_pf_external_direct_policy.py
git commit -m "feat: add published direct dispatch policy baseline"
```

---

### Task 8: Build One Runner with Validation Freeze and Sealed Test Boundaries

**Files:**
- Create: `scripts/run_rsc_pf_external_baselines.py`
- Create: `tests/test_rsc_pf_external_runner.py`
- Modify: `src/joint_dispatch/external_protocol.py`
- Generate: `reports/rsc_pf_external_baselines_v1/validation/<method>/seed_<seed>/`
- Generate: `reports/rsc_pf_external_baselines_v1/frozen/validation_freeze_receipt.json`
- Generate: `reports/rsc_pf_external_baselines_v1/test/<method>/seed_<seed>/`
- Generate: `reports/rsc_pf_external_baselines_v1/test/test_manifest.json`

**Interfaces:**
- CLI stages are `smoke`, `validation`, `freeze-validation`, and `sealed-test`.
- Produces one best checkpoint and receipt for each trainable method/seed.
- `freeze-validation` hashes all 15 external checkpoints/caches and records `test_set_accessed=false`.
- `sealed-test` refuses to run without the complete validation freeze receipt.

- [ ] **Step 1: Write runner boundary tests**

```python
def test_sealed_test_refuses_to_run_before_validation_freeze(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="external validation is not frozen"):
        run_sealed_test(contract_fixture(tmp_path))


def test_validation_matrix_contains_three_methods_and_five_seeds(registry) -> None:
    matrix = build_run_matrix(registry)
    assert {(row.method_id, row.seed) for row in matrix} == {
        (method.method_id, seed) for method in registry.methods for seed in (2026, 2027, 2028, 2029, 2030)
    }
```

- [ ] **Step 2: Run tests and verify the formal runner is missing**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_runner.py -q
```

Expected: FAIL because run-matrix and sealed-stage functions do not exist.

- [ ] **Step 3: Implement resumable validation training**

Use maximum 30 epochs, minimum 20 epochs when the selected method remains finite, patience 5, batch size 256, gradient clipping 1.0, and each source paper's frozen learning-rate settings. Save `last_checkpoint.pt` atomically and resume only when registry, data, normalization, method, and seed hashes match.

- [ ] **Step 4: Select checkpoints using validation only**

Use the primary decision score declared in the registry. Require finite forecast/dispatch, complete feasibility reporting, and resource receipt. Never compare five seeds to select one; retain every seed's own validation-best checkpoint.

- [ ] **Step 5: Freeze validation artifacts**

```powershell
& $Py scripts\run_rsc_pf_external_baselines.py --registry configs\rsc_pf_external_baselines_v1.json --stage validation --all-methods --all-seeds --resume
& $Py scripts\run_rsc_pf_external_baselines.py --registry configs\rsc_pf_external_baselines_v1.json --stage freeze-validation
```

Expected: 15 complete external validation receipts, hashes for every checkpoint/cache, no test access, and an immutable freeze receipt.

- [ ] **Step 6: Run the sealed 2021 test exactly once**

```powershell
& $Py scripts\run_rsc_pf_external_baselines.py --registry configs\rsc_pf_external_baselines_v1.json --stage sealed-test --all-methods --all-seeds
```

Expected: 15 complete test receipts; no training, tuning, or checkpoint selection occurs during this stage.

- [ ] **Step 7: Run runner tests and commit**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_runner.py tests\test_joint_dispatch_formal_training.py tests\test_joint_dispatch_runner_v2.py -q
git add scripts/run_rsc_pf_external_baselines.py src/joint_dispatch/external_protocol.py tests/test_rsc_pf_external_runner.py
git commit -m "feat: add sealed external baseline runner"
```

---

### Task 9: Recompute Fair Rolling Metrics and Paired Statistics

**Files:**
- Create: `src/joint_dispatch/external_evaluation.py`
- Create: `tests/test_rsc_pf_external_evaluation.py`
- Create: `scripts/summarize_rsc_pf_external_comparison.py`
- Read without modification: `reports/joint_forecast_dispatch_v2/test/`
- Generate: `reports/rsc_pf_external_baselines_v1/comparison/method_summary.csv`
- Generate: `reports/rsc_pf_external_baselines_v1/comparison/paired_effects.csv`
- Generate: `reports/rsc_pf_external_baselines_v1/comparison/optimizer_accounting.csv`
- Generate: `reports/rsc_pf_external_baselines_v1/comparison/RESULTS_SUMMARY.md`

**Interfaces:**
- Consumes existing v2 RSC-PF/internal receipts and new external test receipts by path; never rewrites them.
- Produces `evaluate_external_rollout(...)`, `paired_external_comparison(...)`, and `summarize_method_categories(...)`.
- Uses existing `paired_moving_block_bootstrap` and `benjamini_hochberg` from `src/joint_dispatch/evaluation.py`.

- [ ] **Step 1: Write metric and category tests**

```python
def test_summary_keeps_external_internal_proposed_and_oracle_separate(receipt_fixture) -> None:
    summary = summarize_method_categories(receipt_fixture)
    assert set(summary) == {"proposed", "external", "internal_control", "oracle_reference"}
    assert summary["proposed"] == ["Warm-Start-Joint"]
    assert "Oracle-LP" not in summary["external"]


def test_cached_pto_reports_deployment_optimizer_dependence() -> None:
    row = optimizer_row(cached_pto_receipt())
    assert row["experiment_time_calls"] == 0
    assert row["deployment_required_calls"] > 0
```

- [ ] **Step 2: Run tests and verify comparison functions are missing**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_evaluation.py -q
```

Expected: FAIL because external evaluation and category-safe summaries do not exist.

- [ ] **Step 3: Recompute all methods under one rolling evaluator**

Use identical realized-demand recourse, SOC/previous-CHP transitions, cost/carbon parameters, shortage, feasibility, and Oracle regret. Reject a receipt whose topology, data, registry, or evaluator hash differs.

- [ ] **Step 4: Compute paired uncertainty**

For each external method versus RSC-PF and each primary endpoint, use paired 168-hour moving blocks with 2,000 replicates. Apply Benjamini–Hochberg within the cost, regret, shortage, feasibility, carbon, and runtime families. Store mean difference, relative difference, 95% interval, raw p, adjusted p, and rejection flag.

- [ ] **Step 5: Write evidence-constrained result language**

Generate sentences from statistical flags. Use “lower/higher on average” without “significant” when adjusted p exceeds 0.05; use the required no-reliable-difference sentence when the interval crosses zero. Never call Oracle a competing method.

- [ ] **Step 6: Generate the comparison package**

```powershell
& $Py scripts\summarize_rsc_pf_external_comparison.py --main-root reports\joint_forecast_dispatch_v2 --external-root reports\rsc_pf_external_baselines_v1 --output-root reports\rsc_pf_external_baselines_v1\comparison
```

Expected: complete rows for RSC-PF, three external baselines, internal controls, and Oracle; all categories and optimizer counts are explicit.

- [ ] **Step 7: Run evaluation tests and commit**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_evaluation.py tests\test_joint_dispatch_evaluation.py -q
git add src/joint_dispatch/external_evaluation.py scripts/summarize_rsc_pf_external_comparison.py tests/test_rsc_pf_external_evaluation.py reports/rsc_pf_external_baselines_v1/comparison
git commit -m "feat: compare RSC-PF with published baselines"
```

---

### Task 10: Audit Provenance, Fairness, Completeness, and Manuscript Claims

**Files:**
- Create: `scripts/audit_rsc_pf_external_baselines.py`
- Modify: `tests/test_rsc_pf_external_runner.py`
- Modify: `tests/test_rsc_pf_external_evaluation.py`
- Generate: `reports/rsc_pf_external_baselines_v1/audit/external_baseline_audit.json`
- Generate: `reports/rsc_pf_external_baselines_v1/audit/external_baseline_audit.md`

**Interfaces:**
- Produces `audit_external_baselines(project_root: Path, registry_path: Path, output_root: Path) -> dict[str, Any]`.
- Mandatory gates are `literature`, `license`, `registry`, `data`, `information`, `implementation`, `validation_freeze`, `sealed_test`, `metrics`, `statistics`, and `claims`.
- Final status is `complete` only when every gate passes.

- [ ] **Step 1: Write fail-closed audit tests**

```python
def test_audit_fails_when_external_slot_is_missing(audit_fixture) -> None:
    audit_fixture.remove_slot("direct_policy")
    result = audit_external_baselines(**audit_fixture.args)
    assert result["status"] == "failed"
    assert "registry" in result["failed_gates"]


def test_audit_rejects_exact_reproduction_claim_for_adaptation(audit_fixture) -> None:
    audit_fixture.set_claim("principled_adaptation", "exact reproduction")
    result = audit_external_baselines(**audit_fixture.args)
    assert "claims" in result["failed_gates"]
```

- [ ] **Step 2: Run audit tests and verify the audit is missing**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_runner.py tests\test_rsc_pf_external_evaluation.py -k "audit" -q
```

Expected: FAIL because the final audit does not exist.

- [ ] **Step 3: Implement source and license verification**

Verify DOI/stable ID, bibliographic metadata, code URL, commit/tag, license, reproduction level, evidence-card hash, and publication label for every selected method. Fail if a repository changed without an updated registry receipt.

- [ ] **Step 4: Implement data/fairness/leakage verification**

Require identical split hashes, normalization fitted on train, causal histories, approved field consumption, no future truth, identical topology/evaluator hashes, five seeds, validation-only selection, and complete deployment optimizer accounting.

- [ ] **Step 5: Implement claim verification**

Scan generated summaries for forbidden phrases: external superiority without an adjusted test, Oracle described as a competitor, adaptation described as exact reproduction, cached PTO described as optimizer-free deployment, or internal controls described as published baselines.

- [ ] **Step 6: Run the final audit**

```powershell
& $Py scripts\audit_rsc_pf_external_baselines.py --project-root D:\Paper\github_work\paper-code\frame --registry configs\rsc_pf_external_baselines_v1.json --output-root reports\rsc_pf_external_baselines_v1
```

Expected: `status=complete`, no failed gates, three external slots, 15 external test receipts, and an explicit statement that ablations remain unexecuted.

- [ ] **Step 7: Commit the audit tooling and evidence**

```powershell
git add scripts/audit_rsc_pf_external_baselines.py tests/test_rsc_pf_external_runner.py tests/test_rsc_pf_external_evaluation.py reports/rsc_pf_external_baselines_v1/audit
git commit -m "test: audit published baseline comparison"
```

---

### Task 11: Run Full Regression and Freeze the Comparison Package

**Files:**
- Verify all new external-baseline files and tests.
- Verify existing frozen RSC-PF tests and receipts remain unchanged.
- Generate: `reports/rsc_pf_external_baselines_v1/frozen/comparison_freeze_receipt.json`

**Interfaces:**
- Consumes the complete external-baseline audit and the existing architecture-freeze receipt.
- Produces a final comparison freeze receipt with code, registry, data, checkpoint, result, statistics, and audit hashes.

- [ ] **Step 1: Run the complete relevant regression suite**

```powershell
& $Py -m pytest `
  tests\test_rsc_pf_external_registry.py `
  tests\test_rsc_pf_external_protocol.py `
  tests\test_rsc_pf_external_forecast_pto.py `
  tests\test_rsc_pf_external_decision_focused.py `
  tests\test_rsc_pf_external_direct_policy.py `
  tests\test_rsc_pf_external_runner.py `
  tests\test_rsc_pf_external_evaluation.py `
  tests\test_rsc_pf_architecture_freeze.py `
  tests\test_joint_dispatch_data.py `
  tests\test_joint_dispatch_model.py `
  tests\test_joint_dispatch_losses.py `
  tests\test_joint_dispatch_training.py `
  tests\test_joint_dispatch_rollout.py `
  tests\test_joint_dispatch_evaluation.py `
  tests\test_joint_dispatch_runner_v2.py `
  tests\test_scheduling_proxy_decoder.py `
  tests\test_scheduling_proxy_physics.py `
  tests\test_dispatch_lp.py `
  --basetemp D:\Paper\pytest_tmp_rsc_pf_external `
  -p no:cacheprovider `
  -q
```

Expected: 100% pass. Record the exact test count and duration rather than copying a previous count.

- [ ] **Step 2: Re-run architecture freeze audit**

```powershell
& $Py scripts\audit_rsc_pf_architecture_freeze.py --project-root D:\Paper\github_work\paper-code\frame --freeze-spec configs\rsc_pf_architecture_freeze_v1.json --output-dir reports\joint_forecast_dispatch_v2\architecture_freeze --regression-verified
```

Expected: `status=frozen` and all original source/checkpoint hashes unchanged.

- [ ] **Step 3: Re-run external comparison audit**

```powershell
& $Py scripts\audit_rsc_pf_external_baselines.py --project-root D:\Paper\github_work\paper-code\frame --registry configs\rsc_pf_external_baselines_v1.json --output-root reports\rsc_pf_external_baselines_v1
```

Expected: `status=complete` and all eleven gates pass.

- [ ] **Step 4: Write an immutable comparison freeze receipt**

Hash the selected literature registry, search/evidence receipts, all 15 validation and test receipts, three method implementations, data/normalization, statistics tables, result summary, and both audits. Store `test_set_accessed=true`, `ablation_experiments_complete=false`, and `authorized_for_manuscript_external_comparison=true`.

- [ ] **Step 5: Commit the comparison freeze receipt**

```powershell
git add reports/rsc_pf_external_baselines_v1/frozen/comparison_freeze_receipt.json
git commit -m "docs: freeze RSC-PF external baseline evidence"
```

---

## Plan Self-Review Record

### Spec Coverage

- Multi-source literature discovery, T1→T2→T3 routing, DOI/title-author deduplication, scoring, and lawful evidence are covered by Tasks 1–3.
- Exactly three external categories are enforced before implementation; internal controls and Oracle cannot occupy those slots.
- Existing forecasting models may be reused only through frozen paper/provenance identity, covered by Task 5.
- Decision-focused gradient coupling and direct-policy independence are separately implemented and tested in Tasks 6–7.
- Common causal inputs, Standard-IES physics, rolling settlement, seeds, validation selection, and sealed test boundaries are covered by Tasks 4 and 8.
- Experiment-time versus deployment-required optimizer accounting is explicit in Tasks 4, 5, and 9.
- Paired moving-block bootstrap, BH correction, effect sizes, uncertainty, and evidence-constrained prose are covered by Task 9.
- Provenance, license, fairness, leakage, receipt completeness, and claim boundaries are covered by Tasks 10–11.
- Ablations and manuscript insertion are excluded until the external comparison freezes.

### Design Decisions Confirmed During Review

1. The existing five v2 methods remain internal references; they are not relabeled as published external baselines.
2. The plan does not preselect papers from memory. Exact identities are an auditable output of the literature-selection gate, and downstream adapters resolve immutable class paths from that registry.
3. A slot with no compatible paper fails the pipeline rather than lowering scientific criteria.
4. Forecast-PTO receives the same exact LP; decision-focused preserves its source gradient mechanism; direct policy remains architecturally independent from Scheme2R.
5. Sharing the physical decoder for a topology-adapted direct policy is disclosed and creates a strong feasibility-controlled baseline, not an exact reproduction claim.
6. Cached PTO results still report deployment optimizer dependence.
7. Test-set comparison starts only after external validation freeze; no external method is selected or tuned using 2021 results.

### Type and Interface Consistency

- External slots: `forecast_pto`, `decision_focused`, `direct_policy`.
- Batch histories: `[B,24,4]`, `[B,24,12]`, `[B,24,21]`, `[B,24,6]`.
- Forecast output when exposed: `[B,4,4]`.
- Compact direct-policy controls when adapted: `[B,15]`.
- Physical dispatch: `[B,4,21]`.
- Trainable-method seeds: `(2026, 2027, 2028, 2029, 2030)`.
- Aggregate final audit: `audit_external_baselines(project_root: Path, registry_path: Path, output_root: Path) -> dict[str, Any]`.

### Stop Rules

- Stop after Task 3 if any external slot is unfilled, below 70, unlicensed, or not reproducible.
- Stop after Tasks 6–7 if decision coupling, leakage, finite output, feasibility, or the 24-hour resource projection fails.
- Stop before sealed test if any validation receipt or hash is incomplete.
- Stop before manuscript claims if the final audit is not `complete`.
