from __future__ import annotations

from src.joint_dispatch.literature_candidates import deduplicate_candidates, normalize_candidate


def test_same_doi_merges_crossref_and_arxiv_records() -> None:
    records = (
        normalize_candidate({"title": "Decision Focused Learning", "authors": ["A Zhang"], "year": 2022, "doi": "https://doi.org/10.1000/ABC", "url": "https://publisher.example/paper"}, "crossref", "q2"),
        normalize_candidate({"title": "Decision-Focused Learning", "authors": ["A. Zhang"], "year": 2022, "doi": "10.1000/abc", "url": "https://arxiv.org/abs/2201.00001"}, "arxiv", "q2"),
    )
    merged = deduplicate_candidates(records)
    assert len(merged) == 1
    assert merged[0].doi == "10.1000/abc"
    assert set(merged[0].sources) == {"crossref", "arxiv"}


def test_doi_less_merge_requires_same_first_author_and_jaccard_threshold() -> None:
    left = normalize_candidate({"title": "End to End Energy Dispatch with Neural Policies", "authors": ["Li Wang"], "year": 2023, "stable_id": "arxiv:2301.1", "url": "https://arxiv.org/abs/2301.1"}, "arxiv", "q4")
    same = normalize_candidate({"title": "End-to-End Energy Dispatch Using Neural Policies", "authors": ["Li Wang"], "year": 2023, "stable_id": "openalex:W1", "url": "https://example.org/W1"}, "semantic_scholar", "q4")
    other_author = normalize_candidate({"title": left.title, "authors": ["Chen Liu"], "year": 2023, "stable_id": "openalex:W2", "url": "https://example.org/W2"}, "semantic_scholar", "q4")
    assert len(deduplicate_candidates((left, same))) == 1
    assert len(deduplicate_candidates((left, other_author))) == 2


def test_duplicate_merge_prefers_complete_publisher_metadata() -> None:
    preprint = normalize_candidate({"title": "Energy Dispatch", "authors": ["Li Wang"], "year": 2023, "doi": "10.1000/x", "url": "https://arxiv.org/abs/2301.1"}, "arxiv", "q1")
    publisher = normalize_candidate({"title": "Energy Dispatch", "authors": ["Li Wang"], "year": 2023, "doi": "10.1000/x", "url": "https://publisher.example/paper", "venue": "Energy", "volume": "10", "pages": "1-10", "abstract": "Abstract", "citation_count": 5}, "crossref", "q1")
    merged = deduplicate_candidates((preprint, publisher))
    assert merged[0].primary_url == "https://publisher.example/paper"
    assert merged[0].volume == "10" and merged[0].pages == "1-10"


def test_doi_record_merges_with_doi_less_preprint_by_title_and_author() -> None:
    preprint = normalize_candidate({"title": "Decision Focused Forecasting for Energy Storage", "authors": ["Egon Persak"], "year": 2024, "stable_id": "arxiv:2405.14719", "url": "https://arxiv.org/abs/2405.14719"}, "arxiv", "q2")
    publisher = normalize_candidate({"title": "Decision-Focused Forecasting for Energy Storage", "authors": ["Egon Persak"], "year": 2024, "doi": "10.48550/arXiv.2405.14719", "url": "https://doi.org/10.48550/arXiv.2405.14719"}, "crossref", "q2")
    merged = deduplicate_candidates((preprint, publisher))
    assert len(merged) == 1
    assert merged[0].doi == "10.48550/arxiv.2405.14719"
