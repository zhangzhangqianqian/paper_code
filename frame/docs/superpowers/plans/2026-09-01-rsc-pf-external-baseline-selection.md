# RSC-PF External Baseline Discovery and Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify, verify, score, and immutably freeze three scientifically defensible published baselines for the later RSC-PF comparison without accessing sealed 2021 test results or starting new training.

**Architecture:** This is the executable first phase of the external-comparison program. Multi-source search creates normalized candidates; primary-paper and official-code evidence drives fatal screening and slot-specific scoring; a fail-closed audit freezes exactly one Forecast-PTO, one decision-focused, and one direct joint forecast-dispatch method. The exact papers, equations, repositories, licenses, input/output contracts, and adaptation boundaries produced here are mandatory inputs to a separate implementation-and-training plan.

**Tech Stack:** Python 3.9, dataclasses, JSON/JSONL/CSV/BibTeX, pytest, CrossRef, arXiv, Semantic Scholar, Scopus/ScienceDirect fallback, publisher/preprint full text, official code repositories, SHA-256 receipts.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-09-01-rsc-pf-literature-baselines-design.md`.
- This plan performs literature discovery and selection only. It must not implement a selected neural model, train a baseline, run an LP comparison, access 2021 test results, run ablations, or edit either manuscript.
- The frozen RSC-PF architecture, `configs/joint_forecast_dispatch_contract_v2.json`, existing checkpoints, validation/test receipts, and result summaries are read-only.
- Existing methods remain correctly categorized: `Warm-Start-Joint` is RSC-PF; `Seasonal-Naive-PTO`, `Scheme2R-PTO`, and `From-Scratch-Joint` are internal controls; `Oracle-LP` is a reference. None may fill an external slot.
- The external slots are exactly `forecast_pto`, `decision_focused`, and `direct_policy`.
- `forecast_pto` means a published forecasting model followed later by the same exact rolling LP as the internal PTO controls.
- `decision_focused` means a published method whose decision loss changes upstream prediction or decision parameters through a differentiable optimizer, implicit gradient, SPO-style loss, or documented decision surrogate. It may still require an optimizer at inference.
- `direct_policy` means one jointly trained network that maps causal histories/context through an explicit or latent forecast representation to dispatch decisions without an exact optimizer at inference. It may expose an intermediate forecast head; “direct” does not mean “forecast-free.”
- If a candidate could occupy both joint slots, classify it as `direct_policy` when its deployed forward pass directly produces dispatch without an exact optimizer; otherwise classify it as `decision_focused`.
- Search years are 2016–2026. Search CrossRef and arXiv first, Semantic Scholar second, and Scopus/ScienceDirect only when earlier tiers do not provide adequate slot coverage.
- Deduplicate by normalized DOI. When DOI is absent, require identical normalized first-author surname and normalized-title token Jaccard similarity of at least 0.90.
- Every selected method must have a DOI or stable preprint identifier, a primary paper, sufficient equations or lawful code for reproduction, an identified code license or a code-independent equation-level reproduction route, and no fatal exclusion.
- A method cannot pass on total score alone. It must score at least 70/100 overall, at least 18/25 for slot fit, at least 14/20 for reproducibility, and at least 10/15 for information fairness.
- A missing slot causes `selection_status=failed`. Do not fill it with an internal variant, a forecast-only method relabeled as joint, or a method whose central idea would be destroyed by adaptation.
- Candidate ranking must not use any RSC-PF test metric or sealed 2021 data.
- Use lawful paper and code access. Record canonical URL, access route, retrieval date, repository URL, exact commit/tag when available, and SPDX license or `equations_only` status.
- Existing user changes in the dirty worktree are out of scope and must be preserved.

---

## File Structure

- Create `configs/rsc_pf_external_baseline_search_v1.json`: search years, sources, query families, slot definitions, fatal exclusions, score thresholds, and tie-break rules.
- Create `src/joint_dispatch/external_registry.py`: strict dataclasses and JSON loaders for the protocol, screened evidence, scores, and frozen registry.
- Create `src/joint_dispatch/literature_candidates.py`: metadata normalization, DOI/title-author deduplication, source merging, and deterministic candidate IDs.
- Create `scripts/normalize_rsc_pf_baseline_search.py`: normalize raw exports and generate the deduplicated table plus receipt.
- Create `scripts/screen_rsc_pf_external_candidates.py`: validate primary-paper/code evidence and apply fatal exclusions.
- Create `scripts/freeze_rsc_pf_external_baselines.py`: score, resolve overlaps, freeze exactly three methods, and write hashes.
- Create `scripts/audit_rsc_pf_external_selection.py`: fail-closed audit of provenance, license, category identity, test isolation, and implementation readiness.
- Create `tests/test_rsc_pf_external_registry.py`.
- Create `tests/test_rsc_pf_literature_candidates.py`.
- Create `tests/test_rsc_pf_external_selection.py`.
- Generate only under `reports/rsc_pf_external_baselines_v1/literature/` and `reports/rsc_pf_external_baselines_v1/frozen/`.
- Generate `reports/rsc_pf_external_baselines_v1/frozen/implementation_handoff.md` for the next plan.

---

### Task 1: Freeze Search, Evidence, and Registry Schemas

**Files:**
- Create: `configs/rsc_pf_external_baseline_search_v1.json`
- Create: `src/joint_dispatch/external_registry.py`
- Create: `tests/test_rsc_pf_external_registry.py`
- Read: `docs/superpowers/specs/2026-09-01-rsc-pf-literature-baselines-design.md`
- Read: `src/joint_dispatch/formal_protocol.py`

**Interfaces:**
- Produce frozen dataclasses `SearchProtocol`, `CandidateEvidence`, `CandidateScore`, `ExternalBaselineSpec`, and `ExternalBaselineRegistry`.
- Produce `load_search_protocol(path: str | Path) -> SearchProtocol`.
- Produce `load_candidate_evidence(path: str | Path) -> tuple[CandidateEvidence, ...]`.
- Produce `load_external_registry(path: str | Path) -> ExternalBaselineRegistry`.
- Slots are `forecast_pto`, `decision_focused`, and `direct_policy`.
- Reproduction levels are `official_code_exact`, `faithful_reimplementation`, and `principled_adaptation`.

- [ ] **Step 1: Write strict protocol-loading tests**

```python
def test_search_protocol_has_exact_slots_thresholds_and_sources(tmp_path: Path) -> None:
    payload = {
        "schema_version": "rsc-pf-external-search-v1",
        "year_start": 2016,
        "year_end": 2026,
        "sources": {
            "tier_1": ["crossref", "arxiv"],
            "tier_2": ["semantic_scholar"],
            "tier_3": ["scopus", "sciencedirect"],
        },
        "required_slots": ["forecast_pto", "decision_focused", "direct_policy"],
        "query_families": ["q1", "q2", "q3", "q4", "q5"],
        "fatal_exclusions": ["future_truth", "binary_only", "insufficient_definition", "no_coupling", "destructive_adaptation", "license_forbidden"],
        "score_weights": {"slot_fit": 25, "dispatch_compatibility": 20, "reproducibility": 20, "information_fairness": 15, "source_quality": 10, "recency": 5, "resource_fit": 5},
        "minimum_total": 70,
        "minimum_components": {"slot_fit": 18, "reproducibility": 14, "information_fairness": 10},
        "tie_break": ["reproducibility", "slot_fit", "publication_year"],
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    protocol = load_search_protocol(path)
    assert protocol.required_slots == ("forecast_pto", "decision_focused", "direct_policy")
    assert sum(protocol.score_weights.values()) == 100
    assert protocol.minimum_components["reproducibility"] == 14
```

- [ ] **Step 2: Write registry rejection tests**

```python
@pytest.mark.parametrize("method_id", ["Warm-Start-Joint", "From-Scratch-Joint", "Scheme2R-PTO", "Oracle-LP"])
def test_registry_rejects_existing_internal_or_reference_method(tmp_path: Path, method_id: str) -> None:
    payload = valid_registry_payload()
    payload["methods"][0]["method_id"] = method_id
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot fill an external slot"):
        load_external_registry(path)
```

Define `valid_registry_payload()` in the same test file with all three slots, nonempty DOI/stable ID, source URL, code/license route, score components, reproduction level, implementation inputs, and SHA-256 fields.

- [ ] **Step 3: Run tests and verify failure**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -q
```

Expected: FAIL because dataclasses and loaders do not exist.

- [ ] **Step 4: Create the exact protocol JSON**

Store the five approved query families verbatim, the three behavioral slot definitions, source routing, years, deduplication rules, fatal exclusions, score weights, component floors, total threshold, and tie-break rules. Reject unknown fields.

- [ ] **Step 5: Implement strict dataclasses and loaders**

Reject unknown fields, duplicate slots, invalid identifiers, missing primary-paper URL, missing license route, invalid score sums, unsupported reproduction labels, internal method names, and any supposedly complete registry lacking exactly three slots.

- [ ] **Step 6: Run tests**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add configs/rsc_pf_external_baseline_search_v1.json src/joint_dispatch/external_registry.py tests/test_rsc_pf_external_registry.py
git commit -m "feat: define external baseline selection contract"
```

---

### Task 2: Retrieve, Normalize, and Deduplicate Candidates

**Files:**
- Create: `src/joint_dispatch/literature_candidates.py`
- Create: `scripts/normalize_rsc_pf_baseline_search.py`
- Create: `tests/test_rsc_pf_literature_candidates.py`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/raw/<source>/<query_id>.json`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/deduplicated_candidates.csv`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/search_receipt.json`

**Interfaces:**
- Produce `LiteratureCandidate` with candidate ID, source IDs, title, authors, year, DOI/stable ID, venue, abstract, citation count, URLs, source list, and matched query IDs.
- Produce `normalize_candidate(record: Mapping[str, Any], source: str, query_id: str) -> LiteratureCandidate`.
- Produce `deduplicate_candidates(candidates: Sequence[LiteratureCandidate]) -> tuple[LiteratureCandidate, ...]`.
- Produce `write_search_receipt(protocol, attempts, candidates, output_path) -> Path`.

- [ ] **Step 1: Write DOI normalization and merge tests**

```python
def test_same_doi_merges_crossref_and_arxiv_records() -> None:
    records = (
        normalize_candidate({"title": "Decision Focused Learning", "authors": ["A Zhang"], "year": 2022, "doi": "https://doi.org/10.1000/ABC", "url": "https://publisher.example/paper"}, "crossref", "q2"),
        normalize_candidate({"title": "Decision-Focused Learning", "authors": ["A. Zhang"], "year": 2022, "doi": "10.1000/abc", "url": "https://arxiv.org/abs/2201.00001"}, "arxiv", "q2"),
    )
    merged = deduplicate_candidates(records)
    assert len(merged) == 1
    assert merged[0].doi == "10.1000/abc"
    assert set(merged[0].sources) == {"crossref", "arxiv"}
```

- [ ] **Step 2: Write DOI-less boundary tests**

```python
def test_doi_less_merge_requires_same_first_author_and_jaccard_threshold() -> None:
    left = normalize_candidate({"title": "End to End Energy Dispatch with Neural Policies", "authors": ["Li Wang"], "year": 2023, "stable_id": "arxiv:2301.1", "url": "https://arxiv.org/abs/2301.1"}, "arxiv", "q4")
    same = normalize_candidate({"title": "End-to-End Energy Dispatch Using Neural Policies", "authors": ["Li Wang"], "year": 2023, "stable_id": "openalex:W1", "url": "https://example.org/W1"}, "semantic_scholar", "q4")
    other_author = normalize_candidate({"title": left.title, "authors": ["Chen Liu"], "year": 2023, "stable_id": "openalex:W2", "url": "https://example.org/W2"}, "semantic_scholar", "q4")
    assert len(deduplicate_candidates((left, same))) == 1
    assert len(deduplicate_candidates((left, other_author))) == 2
```

- [ ] **Step 3: Run tests and verify failure**

```powershell
& $Py -m pytest tests\test_rsc_pf_literature_candidates.py -q
```

Expected: FAIL because normalization and deduplication are absent.

- [ ] **Step 4: Implement deterministic normalization**

Normalize DOI to lowercase without resolver prefix. Normalize title using Unicode NFKC, lowercase, punctuation removal, whitespace collapse, and an explicit English stopword set. Use `candidate_id=doi:<doi>`; otherwise use the first 16 characters of SHA-256 over normalized title and first author. Preserve alternate source URLs.

- [ ] **Step 5: Search all query families through tier 1**

Save complete CrossRef and arXiv exports with query text, retrieval timestamp, status, and pagination. Raw files contain returned records plus a metadata envelope; do not manually rewrite records.

- [ ] **Step 6: Escalate coverage deterministically**

Assign provisional slots using protocol keywords. Query Semantic Scholar when any slot has fewer than ten unique candidates. Query Scopus or ScienceDirect when any slot still has fewer than five. Record unavailable credentials or provider failures explicitly.

- [ ] **Step 7: Generate table and receipt**

```powershell
& $Py scripts\normalize_rsc_pf_baseline_search.py --protocol configs\rsc_pf_external_baseline_search_v1.json --raw-root reports\rsc_pf_external_baselines_v1\literature\raw --output-root reports\rsc_pf_external_baselines_v1\literature
```

Expected: no duplicate DOI, deterministic IDs, all query/source attempts recorded, explicit escalation status, counts, and raw-export hashes.

- [ ] **Step 8: Run tests and commit**

```powershell
& $Py -m pytest tests\test_rsc_pf_literature_candidates.py tests\test_rsc_pf_external_registry.py -q
git add src/joint_dispatch/literature_candidates.py scripts/normalize_rsc_pf_baseline_search.py tests/test_rsc_pf_literature_candidates.py reports/rsc_pf_external_baselines_v1/literature/raw reports/rsc_pf_external_baselines_v1/literature/deduplicated_candidates.csv reports/rsc_pf_external_baselines_v1/literature/search_receipt.json
git commit -m "data: collect external forecast dispatch candidates"
```

---

### Task 3: Build Primary-Source Evidence and Apply Fatal Screening

**Files:**
- Create: `scripts/screen_rsc_pf_external_candidates.py`
- Modify: `src/joint_dispatch/external_registry.py`
- Modify: `tests/test_rsc_pf_external_registry.py`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/screening_evidence.jsonl`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/excluded_candidates.csv`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/evidence_cards.md`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/baseline_references.bib`

**Interfaces:**
- Produce `validate_candidate_evidence(evidence: CandidateEvidence) -> tuple[str, ...]`.
- Produce `classify_slot(evidence: CandidateEvidence) -> str` using deployment behavior and gradient coupling.
- Each evidence record stores exact paper anchors for inputs, prediction representation, decision layer, loss, gradient path, inference optimizer, output type, topology, and adaptation.

- [ ] **Step 1: Write fatal-exclusion tests**

```python
def test_future_truth_and_missing_reproduction_route_are_fatal() -> None:
    evidence = complete_evidence(
        uses_future_truth_at_inference=True,
        equations_sufficient=False,
        official_code_url=None,
    )
    assert validate_candidate_evidence(evidence) == ("future_truth", "insufficient_definition")
```

- [ ] **Step 2: Write behavioral classification tests**

```python
def test_joint_slots_are_classified_by_deployed_forward_path() -> None:
    direct = complete_evidence(deployment_produces_dispatch=True, deployment_exact_optimizer=False, gradient_coupling="joint_network")
    focused = complete_evidence(deployment_produces_dispatch=False, deployment_exact_optimizer=True, gradient_coupling="implicit_optimization")
    assert classify_slot(direct) == "direct_policy"
    assert classify_slot(focused) == "decision_focused"
```

Define `complete_evidence(**overrides)` in the same test file with every field populated.

- [ ] **Step 3: Run tests and verify failure**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -k "fatal or classified" -q
```

Expected: FAIL because validators are absent.

- [ ] **Step 4: Implement fatal screening and classification**

Apply all six fatal exclusions. Forecast-only methods enter only `forecast_pto`. Methods with decision gradients and an inference optimizer enter `decision_focused`. A unified trained forward network directly emitting continuous decisions without an exact inference optimizer enters `direct_policy`, even if it exposes forecasts.

- [ ] **Step 5: Read and anchor the highest-ranked five candidates per provisional slot, or every candidate when fewer than five remain**

Inspect the primary paper and official repository. Record section/page/equation anchors, input availability, gradient relation, objective, inference path, continuous/binary output, solver role, topology, code revision, and license. Do not infer coupling from abstract wording.

- [ ] **Step 6: Generate evidence artifacts**

```powershell
& $Py scripts\screen_rsc_pf_external_candidates.py --protocol configs\rsc_pf_external_baseline_search_v1.json --candidates reports\rsc_pf_external_baselines_v1\literature\deduplicated_candidates.csv --evidence reports\rsc_pf_external_baselines_v1\literature\screening_evidence.jsonl --output-root reports\rsc_pf_external_baselines_v1\literature
```

Expected: every scored candidate has a primary-source anchor; every exclusion has exact codes; every retained joint candidate has verified gradient/inference behavior; every code route has license status.

- [ ] **Step 7: Run tests and commit**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py -q
git add scripts/screen_rsc_pf_external_candidates.py src/joint_dispatch/external_registry.py tests/test_rsc_pf_external_registry.py reports/rsc_pf_external_baselines_v1/literature/screening_evidence.jsonl reports/rsc_pf_external_baselines_v1/literature/excluded_candidates.csv reports/rsc_pf_external_baselines_v1/literature/evidence_cards.md reports/rsc_pf_external_baselines_v1/literature/baseline_references.bib
git commit -m "docs: verify external baseline evidence"
```

---

### Task 4: Score and Freeze Exactly Three Methods

**Files:**
- Create: `scripts/freeze_rsc_pf_external_baselines.py`
- Create: `tests/test_rsc_pf_external_selection.py`
- Modify: `src/joint_dispatch/external_registry.py`
- Generate: `reports/rsc_pf_external_baselines_v1/literature/screening_scores.csv`
- Generate: `configs/rsc_pf_external_baselines_v1.json`
- Generate: `reports/rsc_pf_external_baselines_v1/frozen/literature_selection_receipt.json`

**Interfaces:**
- Produce `score_candidate(evidence: CandidateEvidence, protocol: SearchProtocol) -> CandidateScore`.
- Produce `select_external_slots(scores: Sequence[CandidateScore]) -> ExternalBaselineRegistry`.
- `CandidateScore.total: int` is the exact component sum and `CandidateScore.eligible: bool` is true only when there are no fatal exclusions and all total/component floors pass.
- Produce exact method ID, paper ID, reproduction route, input/output contract, optimizer role, and adaptation boundary per slot.

- [ ] **Step 1: Write component-floor tests**

```python
def test_high_total_cannot_hide_weak_reproducibility() -> None:
    score = CandidateScore(candidate_id="c1", slot="decision_focused", slot_fit=25, dispatch_compatibility=20, reproducibility=10, information_fairness=15, source_quality=10, recency=5, resource_fit=5, fatal_exclusions=())
    assert score.total == 90
    assert score.eligible is False
```

- [ ] **Step 2: Write fail-closed and overlap tests**

```python
def test_selection_fails_when_a_slot_is_unfilled() -> None:
    scores = (
        eligible_score("forecast", "forecast_pto", 82),
        eligible_score("focused", "decision_focused", 80),
    )
    with pytest.raises(ValueError, match="unfilled external slot: direct_policy"):
        select_external_slots(scores)


def test_same_candidate_cannot_fill_two_slots() -> None:
    with pytest.raises(ValueError, match="candidate cannot fill multiple slots"):
        select_external_slots(duplicate_candidate_across_slots())
```

Define `eligible_score` and `duplicate_candidate_across_slots` in the same test file with complete score objects.

- [ ] **Step 3: Run tests and verify failure**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_selection.py -q
```

Expected: FAIL because scoring and selection are absent.

- [ ] **Step 4: Implement slot-specific scoring**

Use the common 100-point envelope but define slot fit separately: forecasting relevance for `forecast_pto`; verified decision-gradient mechanism for `decision_focused`; unified forward network plus optimizer-free dispatch inference for `direct_policy`. Enforce component floors before total ranking and behavioral overlap resolution.

- [ ] **Step 5: Generate and verify the registry**

```powershell
& $Py scripts\freeze_rsc_pf_external_baselines.py --protocol configs\rsc_pf_external_baseline_search_v1.json --evidence reports\rsc_pf_external_baselines_v1\literature\screening_evidence.jsonl --scores reports\rsc_pf_external_baselines_v1\literature\screening_scores.csv --registry configs\rsc_pf_external_baselines_v1.json --receipt reports\rsc_pf_external_baselines_v1\frozen\literature_selection_receipt.json
```

Expected: exactly three different papers; all thresholds pass; no fatal exclusions; complete source anchors, license route, reproduction label, inference optimizer status, adaptation disclosure, and evidence hashes.

- [ ] **Step 6: Run tests and commit**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py tests\test_rsc_pf_literature_candidates.py tests\test_rsc_pf_external_selection.py -q
git add configs/rsc_pf_external_baselines_v1.json scripts/freeze_rsc_pf_external_baselines.py src/joint_dispatch/external_registry.py tests/test_rsc_pf_external_selection.py reports/rsc_pf_external_baselines_v1/literature/screening_scores.csv reports/rsc_pf_external_baselines_v1/frozen/literature_selection_receipt.json
git commit -m "docs: freeze published external baselines"
```

---

### Task 5: Audit Selection and Produce the Implementation Handoff

**Files:**
- Create: `scripts/audit_rsc_pf_external_selection.py`
- Modify: `tests/test_rsc_pf_external_selection.py`
- Generate: `reports/rsc_pf_external_baselines_v1/frozen/selection_audit.json`
- Generate: `reports/rsc_pf_external_baselines_v1/frozen/selection_audit.md`
- Generate: `reports/rsc_pf_external_baselines_v1/frozen/implementation_handoff.md`

**Interfaces:**
- Produce `audit_external_selection(project_root: Path, protocol_path: Path, registry_path: Path, report_root: Path) -> dict[str, Any]`.
- Gates are `search_coverage`, `deduplication`, `primary_sources`, `behavioral_classification`, `fatal_exclusions`, `score_thresholds`, `license`, `registry_integrity`, `test_isolation`, and `implementation_readiness`.
- Final status is `complete` only when all gates pass and the handoff has one complete section per slot.

- [ ] **Step 1: Write fail-closed audit tests**

```python
def test_audit_fails_on_missing_source_anchor(tmp_path: Path) -> None:
    fixture = write_complete_selection_fixture(tmp_path)
    evidence = read_jsonl(fixture.evidence_path)
    evidence[0]["primary_source_anchors"] = []
    write_jsonl(fixture.evidence_path, evidence)
    result = audit_external_selection(tmp_path, fixture.protocol_path, fixture.registry_path, fixture.report_root)
    assert result["status"] == "failed"
    assert "primary_sources" in result["failed_gates"]


def test_audit_fails_if_test_results_influenced_selection(tmp_path: Path) -> None:
    fixture = write_complete_selection_fixture(tmp_path)
    receipt = json.loads(fixture.selection_receipt.read_text(encoding="utf-8"))
    receipt["test_set_accessed"] = True
    fixture.selection_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    result = audit_external_selection(tmp_path, fixture.protocol_path, fixture.registry_path, fixture.report_root)
    assert "test_isolation" in result["failed_gates"]
```

Implement `SelectionFixture`, `write_complete_selection_fixture`, `read_jsonl`, and `write_jsonl` in the same test file with real temporary files for all ten gates.

- [ ] **Step 2: Run tests and verify failure**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_selection.py -k "audit" -q
```

Expected: FAIL because the audit is absent.

- [ ] **Step 3: Implement all audit gates**

Verify source hashes, query coverage, DOI uniqueness, evidence anchors, behavioral slot identity, exclusions, component floors, license/reproduction consistency, three unique papers, `test_set_accessed=false`, and complete implementation inputs. Any failure prevents handoff authorization.

- [ ] **Step 4: Generate the method-specific handoff**

For each method record: citation; source URLs; code revision/license or equations-only route; permitted input tensors; forecast representation; decision/optimizer layer; loss equations and gradients; source output dimensions; required 24→4 Standard-IES mapping; operations that must remain unchanged; allowed adaptations; forbidden substitutions; expected optimizer calls; reproduction label; and technical risks. Do not leave unresolved markers; an unresolved item fails `implementation_readiness`.

- [ ] **Step 5: Run the audit**

```powershell
& $Py scripts\audit_rsc_pf_external_selection.py --project-root D:\Paper\github_work\paper-code\frame --protocol configs\rsc_pf_external_baseline_search_v1.json --registry configs\rsc_pf_external_baselines_v1.json --report-root reports\rsc_pf_external_baselines_v1
```

Expected: `status=complete`, ten passing gates, three unique slots, `test_set_accessed=false`, and `authorized_for_implementation_plan=true`.

- [ ] **Step 6: Run phase-one regression**

```powershell
& $Py -m pytest tests\test_rsc_pf_external_registry.py tests\test_rsc_pf_literature_candidates.py tests\test_rsc_pf_external_selection.py --basetemp D:\Paper\pytest_tmp_rsc_pf_selection -p no:cacheprovider -q
```

Expected: 100% pass; record exact count and duration.

- [ ] **Step 7: Commit audit and handoff**

```powershell
git add scripts/audit_rsc_pf_external_selection.py tests/test_rsc_pf_external_selection.py reports/rsc_pf_external_baselines_v1/frozen/selection_audit.json reports/rsc_pf_external_baselines_v1/frozen/selection_audit.md reports/rsc_pf_external_baselines_v1/frozen/implementation_handoff.md
git commit -m "docs: audit external baseline selection"
```

---

## Post-Gate Roadmap

This plan intentionally stops after selection. When Task 5 reports `authorized_for_implementation_plan=true`, create a second implementation plan from the frozen registry and `implementation_handoff.md`. It will define exact classes, losses, solver interfaces, hyperparameters, source-specific tests, five-seed validation runs, and validation freeze. After those runs pass, create a third plan for sealed 2021 testing, paired statistics, comparison audit, and result-package freeze. Ablations remain outside all three plans until the external comparison is complete.

## Plan Self-Review Record

### Findings Corrected

1. The previous 846-line plan mixed literature selection, unknown-method reproduction, and sealed comparison. This executable plan covers one independently testable subsystem.
2. Implementation details cannot be exact before papers are selected. Hidden placeholders are removed; a complete handoff is required before the second plan.
3. Previously undefined test helpers must now be defined in their test files with complete payloads.
4. `decision_focused` and `direct_policy` are behaviorally disjoint. A direct joint network may retain a forecast head.
5. Component floors prevent weak coupling or irreproducibility from passing through unrelated score points.
6. Test isolation is audited before implementation.

### Spec Coverage

- Tasks 1–3 cover search, source routing, deduplication, lawful evidence, fatal exclusions, licenses, coupling, and adaptation evidence.
- Task 4 enforces three external categories, component floors, unique papers, and immutable selection.
- Task 5 enforces provenance, category identity, test isolation, implementation readiness, and fail-closed authorization.
- RSC-PF code and sealed results remain untouched.
- Exact reproduction details are deferred only until exact papers are frozen.

### Stop Rules

- Stop after Task 2 if source attempts or query receipts are incomplete.
- Stop after Task 3 if a slot lacks primary-source evidence and a lawful reproduction route.
- Stop after Task 4 if a slot is unfilled, a component floor fails, a paper fills multiple slots, or a fatal exclusion exists.
- Stop after Task 5 unless all ten gates pass and every handoff field is resolved.
- Do not generate the implementation/training plan, access sealed test artifacts, or run a baseline experiment before Task 5 authorization.
