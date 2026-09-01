from __future__ import annotations

from pathlib import Path

import pytest

from src.joint_dispatch.external_registry import (
    CandidateEvidence, CandidateScore, REQUIRED_SLOTS, SearchProtocol, score_candidate,
    select_external_slots,
)
from tests.test_rsc_pf_external_registry import complete_evidence


def protocol() -> SearchProtocol:
    return SearchProtocol(
        schema_version="rsc-pf-external-search-v1", year_start=2016, year_end=2026,
        sources_t1=("crossref", "arxiv"), sources_t2=("semantic_scholar",), sources_t3=("scopus", "sciencedirect"),
        required_slots=REQUIRED_SLOTS, query_families=(), fatal_exclusions=(),
        score_weights={"slot_fit": 25, "dispatch_compatibility": 20, "reproducibility": 20, "information_fairness": 15, "source_quality": 10, "recency": 5, "resource_fit": 5},
        minimum_total=70, minimum_components={"slot_fit": 18, "reproducibility": 14, "information_fairness": 10},
        tie_break=("reproducibility", "slot_fit", "publication_year"),
    )


def eligible_score(candidate_id: str, slot: str, total_hint: int = 82) -> CandidateScore:
    return CandidateScore(candidate_id, slot, 2024, 25, 20, 15, 15, 8, 4, 5, ())


def test_high_total_cannot_hide_weak_reproducibility() -> None:
    score = CandidateScore("c1", "decision_focused", 2024, 25, 20, 10, 15, 10, 5, 5, ())
    assert score.total == 90
    assert score.eligible is False


def test_selection_fails_when_a_slot_is_unfilled() -> None:
    scores = (eligible_score("forecast", "forecast_pto"), eligible_score("focused", "decision_focused"))
    with pytest.raises(ValueError, match="unfilled external slot: direct_policy"):
        select_external_slots(scores)


def test_same_candidate_cannot_fill_two_slots() -> None:
    scores = tuple(eligible_score("same", slot) for slot in REQUIRED_SLOTS)
    with pytest.raises(ValueError, match="candidate cannot fill multiple slots"):
        select_external_slots(scores)


def test_score_candidate_uses_slot_specific_envelope() -> None:
    forecast = complete_evidence(proposed_slot="forecast_pto", paper_title="iTransformer Forecasting", official_code_url="https://github.com/example/iTransformer")
    score = score_candidate(forecast, protocol())
    assert score.slot == "forecast_pto"
    assert score.slot_fit == 25
    assert score.reproducibility == 20
    assert score.eligible
