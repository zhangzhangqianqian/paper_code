from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src.joint_dispatch.external_registry import (
    CandidateEvidence, CandidateScore, REQUIRED_SLOTS, SearchProtocol, score_candidate,
    select_external_slots,
)
from scripts.audit_rsc_pf_external_selection import audit_external_selection
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


def _copy_selection_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    project = Path(__file__).parents[1]
    root = tmp_path / "project"
    shutil.copytree(project / "reports" / "rsc_pf_external_baselines_v1", root / "reports" / "rsc_pf_external_baselines_v1")
    (root / "configs").mkdir(parents=True)
    shutil.copy(project / "configs" / "rsc_pf_external_baseline_search_v1.json", root / "configs" / "rsc_pf_external_baseline_search_v1.json")
    shutil.copy(project / "configs" / "rsc_pf_external_baselines_v1.json", root / "configs" / "rsc_pf_external_baselines_v1.json")
    return root, root / "configs" / "rsc_pf_external_baseline_search_v1.json", root / "configs" / "rsc_pf_external_baselines_v1.json", root / "reports" / "rsc_pf_external_baselines_v1"


def test_audit_current_selection_is_complete() -> None:
    root = Path(__file__).parents[1]
    result = audit_external_selection(root, root / "configs" / "rsc_pf_external_baseline_search_v1.json", root / "configs" / "rsc_pf_external_baselines_v1.json", root / "reports" / "rsc_pf_external_baselines_v1")
    assert result["status"] == "complete"
    assert result["authorized_for_implementation_plan"] is True


def test_audit_fails_on_missing_source_anchor(tmp_path: Path) -> None:
    root, protocol_path, registry_path, report_root = _copy_selection_fixture(tmp_path)
    evidence_path = report_root / "literature" / "screening_evidence.jsonl"
    evidence = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()]
    selected_id = json.loads(registry_path.read_text(encoding="utf-8"))["methods"][0]["candidate_id"]
    next(item for item in evidence if item["candidate_id"] == selected_id)["primary_source_anchors"] = []
    evidence_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in evidence) + "\n", encoding="utf-8")
    result = audit_external_selection(root, protocol_path, registry_path, report_root)
    assert result["status"] == "failed"
    assert "primary_sources" in result["failed_gates"]


def test_audit_fails_if_test_results_influenced_selection(tmp_path: Path) -> None:
    root, protocol_path, registry_path, report_root = _copy_selection_fixture(tmp_path)
    receipt_path = report_root / "frozen" / "literature_selection_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["test_set_accessed"] = True
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    result = audit_external_selection(root, protocol_path, registry_path, report_root)
    assert result["status"] == "failed"
    assert "test_isolation" in result["failed_gates"]
