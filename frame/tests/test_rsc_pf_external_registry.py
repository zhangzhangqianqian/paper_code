from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.external_registry import (
    CandidateEvidence, INTERNAL_METHODS, REQUIRED_SLOTS, classify_slot, load_external_registry,
    load_search_protocol, validate_candidate_evidence,
)


def valid_registry_payload() -> dict:
    methods = []
    for index, slot in enumerate(REQUIRED_SLOTS):
        methods.append({
            "slot": slot, "candidate_id": f"candidate-{index}", "method_id": f"Published-{index}",
            "paper_title": f"Published method {index}", "paper_year": 2022, "doi": f"10.1000/method{index}",
            "stable_id": None, "primary_source_url": "https://example.org/paper", "official_code_url": "https://github.com/example/repo",
            "license_route": "MIT", "reproduction_level": "faithful_reimplementation", "score_total": 82,
            "input_mode": "causal", "output_mode": "dispatch", "optimizer_role": "none",
            "implementation_class": f"src.example.Method{index}", "adaptation_disclosure": "none",
            "evidence_sha256": "a" * 64, "implementation_ready": True,
        })
    return {"schema_version": "rsc-pf-external-registry-v1", "methods": methods}


def test_search_protocol_has_exact_slots_thresholds_and_sources() -> None:
    protocol = load_search_protocol(Path(__file__).parents[1] / "configs" / "rsc_pf_external_baseline_search_v1.json")
    assert protocol.year_start == 2016 and protocol.year_end == 2026
    assert protocol.required_slots == ("forecast_pto", "decision_focused", "direct_policy")
    assert protocol.sources_t1 == ("crossref", "arxiv")
    assert sum(protocol.score_weights.values()) == 100
    assert protocol.minimum_total == 70


@pytest.mark.parametrize("method_id", sorted(INTERNAL_METHODS))
def test_registry_rejects_existing_internal_or_reference_method(tmp_path: Path, method_id: str) -> None:
    payload = valid_registry_payload()
    payload["methods"][0]["method_id"] = method_id
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="internal method cannot fill an external slot"):
        load_external_registry(path)


def test_registry_requires_all_three_slots(tmp_path: Path) -> None:
    payload = valid_registry_payload()
    payload["methods"] = payload["methods"][:2]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="missing external slots|exactly three"):
        load_external_registry(path)


def complete_evidence(**overrides: object) -> CandidateEvidence:
    values: dict[str, object] = {
        "candidate_id": "candidate-evidence", "paper_title": "Published method", "paper_year": 2024,
        "doi": "10.1000/evidence", "stable_id": None, "primary_source_url": "https://example.org/paper",
        "primary_source_anchors": ("Sec. 3, Eq. (4)",), "official_code_url": None,
        "license_route": "equations_only", "proposed_slot": "decision_focused",
        "uses_future_truth_at_inference": False, "binary_only_without_continuous_form": False,
        "equations_sufficient": True, "has_forecast_decision_coupling": True,
        "preserves_core_under_adaptation": True, "deployment_produces_dispatch": False,
        "deployment_exact_optimizer": True, "gradient_coupling": "implicit_optimization",
        "input_fields": ("history", "context"), "forecast_representation": "continuous forecast",
        "decision_layer": "differentiable optimizer", "loss_description": "decision loss",
        "gradient_path": "decision loss to predictor", "output_type": "continuous dispatch",
        "topology_description": "forecast plus optimizer", "adaptation_disclosure": "head adaptation only",
    }
    values.update(overrides)
    return CandidateEvidence(**values)


def test_future_truth_and_missing_reproduction_route_are_fatal() -> None:
    evidence = complete_evidence(uses_future_truth_at_inference=True, equations_sufficient=False, official_code_url=None)
    assert validate_candidate_evidence(evidence) == ("future_truth", "insufficient_definition")


def test_joint_slots_are_classified_by_deployed_forward_path() -> None:
    direct = complete_evidence(deployment_produces_dispatch=True, deployment_exact_optimizer=False, gradient_coupling="joint_network")
    focused = complete_evidence(deployment_produces_dispatch=False, deployment_exact_optimizer=True, gradient_coupling="implicit_optimization")
    assert classify_slot(direct) == "direct_policy"
    assert classify_slot(focused) == "decision_focused"
