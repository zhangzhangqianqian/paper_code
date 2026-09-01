"""Strict contracts for discovering and freezing external literature baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SEARCH_SCHEMA = "rsc-pf-external-search-v1"
REGISTRY_SCHEMA = "rsc-pf-external-registry-v1"
REQUIRED_SLOTS = ("forecast_pto", "decision_focused", "direct_policy")
REPRODUCTION_LEVELS = ("official_code_exact", "faithful_reimplementation", "principled_adaptation")
INTERNAL_METHODS = {"Warm-Start-Joint", "From-Scratch-Joint", "Seasonal-Naive-PTO", "Scheme2R-PTO", "Oracle-LP"}
SCORE_FIELDS = (
    "slot_fit", "dispatch_compatibility", "reproducibility", "information_fairness",
    "source_quality", "recency", "resource_fit",
)
SCORE_MAXIMA = {
    "slot_fit": 25, "dispatch_compatibility": 20, "reproducibility": 20,
    "information_fairness": 15, "source_quality": 10, "recency": 5, "resource_fit": 5,
}


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an array")
    return value


def _strict_keys(value: Mapping[str, Any], required: set[str], field: str) -> None:
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{field} has unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{field} is missing fields: {missing}")


def _text(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _text(value, field)


def _integer(value: Any, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if float(value) != result or (positive and result <= 0):
        raise ValueError(f"{field} must be {'a positive ' if positive else 'an '}integer")
    return result


def _boolean(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _string_tuple(value: Any, field: str, *, nonempty: bool = False) -> tuple[str, ...]:
    items = tuple(_text(item, f"{field}[]") for item in _sequence(value, field))
    if nonempty and not items:
        raise ValueError(f"{field} must not be empty")
    return items


def _mapping_proxy(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value[key] for key in value})


@dataclass(frozen=True)
class QueryFamily:
    query_id: str
    query: str


@dataclass(frozen=True)
class SearchProtocol:
    schema_version: str
    year_start: int
    year_end: int
    sources_t1: tuple[str, ...]
    sources_t2: tuple[str, ...]
    sources_t3: tuple[str, ...]
    required_slots: tuple[str, ...]
    query_families: tuple[QueryFamily, ...]
    fatal_exclusions: tuple[str, ...]
    score_weights: Mapping[str, int]
    minimum_total: int
    minimum_components: Mapping[str, int]
    tie_break: tuple[str, ...]


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_id: str
    paper_title: str
    paper_year: int
    doi: str | None
    stable_id: str | None
    primary_source_url: str
    primary_source_anchors: tuple[str, ...]
    official_code_url: str | None
    license_route: str
    proposed_slot: str
    uses_future_truth_at_inference: bool
    binary_only_without_continuous_form: bool
    equations_sufficient: bool
    has_forecast_decision_coupling: bool
    preserves_core_under_adaptation: bool
    deployment_produces_dispatch: bool
    deployment_exact_optimizer: bool
    gradient_coupling: str
    input_fields: tuple[str, ...]
    forecast_representation: str
    decision_layer: str
    loss_description: str
    gradient_path: str
    output_type: str
    topology_description: str
    adaptation_disclosure: str


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    slot: str
    publication_year: int
    slot_fit: int
    dispatch_compatibility: int
    reproducibility: int
    information_fairness: int
    source_quality: int
    recency: int
    resource_fit: int
    fatal_exclusions: tuple[str, ...]

    @property
    def total(self) -> int:
        return sum(int(getattr(self, field)) for field in SCORE_FIELDS)

    @property
    def eligible(self) -> bool:
        return (
            not self.fatal_exclusions
            and self.total >= 70
            and self.slot_fit >= 18
            and self.reproducibility >= 14
            and self.information_fairness >= 10
        )


@dataclass(frozen=True)
class ExternalBaselineSpec:
    slot: str
    candidate_id: str
    method_id: str
    paper_title: str
    paper_year: int
    doi: str | None
    stable_id: str | None
    primary_source_url: str
    official_code_url: str | None
    license_route: str
    reproduction_level: str
    score_total: int
    input_mode: str
    output_mode: str
    optimizer_role: str
    implementation_class: str
    adaptation_disclosure: str
    evidence_sha256: str
    implementation_ready: bool


@dataclass(frozen=True)
class ExternalBaselineRegistry:
    schema_version: str
    methods: tuple[ExternalBaselineSpec, ...]
    source_path: Path | None = None

    def by_slot(self, slot: str) -> ExternalBaselineSpec:
        for method in self.methods:
            if method.slot == slot:
                return method
        raise KeyError(f"unfilled external slot: {slot}")


def _load_json(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON contract {source}") from exc
    return _mapping(payload, str(source))


def load_search_protocol(path: str | Path) -> SearchProtocol:
    payload = _load_json(path)
    required = {
        "schema_version", "year_start", "year_end", "sources", "required_slots",
        "query_families", "fatal_exclusions", "score_weights", "minimum_total",
        "minimum_components", "tie_break",
    }
    _strict_keys(payload, required, "search protocol")
    if payload["schema_version"] != SEARCH_SCHEMA:
        raise ValueError(f"schema_version must equal {SEARCH_SCHEMA!r}")
    year_start = _integer(payload["year_start"], "year_start", positive=True)
    year_end = _integer(payload["year_end"], "year_end", positive=True)
    if year_end < year_start:
        raise ValueError("year_end must not precede year_start")
    sources = _mapping(payload["sources"], "sources")
    _strict_keys(sources, {"tier_1", "tier_2", "tier_3"}, "sources")
    t1 = _string_tuple(sources["tier_1"], "sources.tier_1", nonempty=True)
    t2 = _string_tuple(sources["tier_2"], "sources.tier_2")
    t3 = _string_tuple(sources["tier_3"], "sources.tier_3")
    slots = _string_tuple(payload["required_slots"], "required_slots", nonempty=True)
    if slots != REQUIRED_SLOTS:
        raise ValueError(f"required_slots must equal {REQUIRED_SLOTS!r}")
    families: list[QueryFamily] = []
    seen_query_ids: set[str] = set()
    for index, raw in enumerate(_sequence(payload["query_families"], "query_families")):
        item = _mapping(raw, f"query_families[{index}]")
        _strict_keys(item, {"id", "query"}, f"query_families[{index}]")
        query_id = _text(item["id"], f"query_families[{index}].id")
        if query_id in seen_query_ids:
            raise ValueError(f"duplicate query family: {query_id}")
        seen_query_ids.add(query_id)
        families.append(QueryFamily(query_id, _text(item["query"], f"query_families[{index}].query")))
    if len(families) != 5:
        raise ValueError("exactly five query families are required")
    exclusions = _string_tuple(payload["fatal_exclusions"], "fatal_exclusions", nonempty=True)
    if set(exclusions) != {"future_truth", "binary_only", "insufficient_definition", "no_coupling", "destructive_adaptation", "license_forbidden"}:
        raise ValueError("fatal_exclusions do not match the frozen six exclusions")
    raw_weights = _mapping(payload["score_weights"], "score_weights")
    _strict_keys(raw_weights, set(SCORE_MAXIMA), "score_weights")
    weights = {field: _integer(raw_weights[field], f"score_weights.{field}") for field in SCORE_FIELDS}
    if weights != SCORE_MAXIMA or sum(weights.values()) != 100:
        raise ValueError("score_weights must match the frozen 100-point envelope")
    minimum_total = _integer(payload["minimum_total"], "minimum_total", positive=True)
    if minimum_total != 70:
        raise ValueError("minimum_total must equal 70")
    raw_min = _mapping(payload["minimum_components"], "minimum_components")
    _strict_keys(raw_min, {"slot_fit", "reproducibility", "information_fairness"}, "minimum_components")
    minimum_components = {key: _integer(raw_min[key], f"minimum_components.{key}") for key in raw_min}
    if minimum_components != {"slot_fit": 18, "reproducibility": 14, "information_fairness": 10}:
        raise ValueError("minimum_components do not match the frozen floors")
    tie_break = _string_tuple(payload["tie_break"], "tie_break", nonempty=True)
    if tie_break != ("reproducibility", "slot_fit", "publication_year"):
        raise ValueError("tie_break does not match the frozen order")
    return SearchProtocol(
        schema_version=SEARCH_SCHEMA, year_start=year_start, year_end=year_end,
        sources_t1=t1, sources_t2=t2, sources_t3=t3, required_slots=slots,
        query_families=tuple(families), fatal_exclusions=exclusions,
        score_weights=MappingProxyType(weights), minimum_total=minimum_total,
        minimum_components=MappingProxyType(minimum_components), tie_break=tie_break,
    )


def _score_from_payload(raw: Mapping[str, Any], index: int) -> CandidateScore:
    required = {"candidate_id", "slot", "publication_year", *SCORE_FIELDS, "fatal_exclusions"}
    _strict_keys(raw, required, f"scores[{index}]")
    slot = _text(raw["slot"], f"scores[{index}].slot")
    if slot not in REQUIRED_SLOTS:
        raise ValueError(f"scores[{index}].slot is not an external slot")
    values = {field: _integer(raw[field], f"scores[{index}].{field}") for field in SCORE_FIELDS}
    for field, value in values.items():
        if value < 0 or value > SCORE_MAXIMA[field]:
            raise ValueError(f"scores[{index}].{field} is outside its allowed range")
    return CandidateScore(
        candidate_id=_text(raw["candidate_id"], f"scores[{index}].candidate_id"), slot=slot,
        publication_year=_integer(raw["publication_year"], f"scores[{index}].publication_year", positive=True),
        **values, fatal_exclusions=_string_tuple(raw["fatal_exclusions"], f"scores[{index}].fatal_exclusions"),
    )


def load_candidate_evidence(path: str | Path) -> tuple[CandidateEvidence, ...]:
    source = Path(path)
    try:
        lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError as exc:
        raise ValueError(f"cannot read candidate evidence {source}") from exc
    evidence: list[CandidateEvidence] = []
    seen: set[str] = set()
    required = {
        "candidate_id", "paper_title", "paper_year", "doi", "stable_id", "primary_source_url",
        "primary_source_anchors", "official_code_url", "license_route", "proposed_slot",
        "uses_future_truth_at_inference", "binary_only_without_continuous_form", "equations_sufficient",
        "has_forecast_decision_coupling", "preserves_core_under_adaptation", "deployment_produces_dispatch",
        "deployment_exact_optimizer", "gradient_coupling", "input_fields", "forecast_representation",
        "decision_layer", "loss_description", "gradient_path", "output_type", "topology_description",
        "adaptation_disclosure",
    }
    for index, line in enumerate(lines):
        raw = _mapping(json.loads(line), f"evidence[{index}]")
        _strict_keys(raw, required, f"evidence[{index}]")
        candidate_id = _text(raw["candidate_id"], f"evidence[{index}].candidate_id")
        if candidate_id in seen:
            raise ValueError(f"duplicate evidence candidate: {candidate_id}")
        seen.add(candidate_id)
        proposed_slot = _text(raw["proposed_slot"], f"evidence[{index}].proposed_slot")
        if proposed_slot not in REQUIRED_SLOTS:
            raise ValueError(f"evidence[{index}].proposed_slot is not an external slot")
        doi = _optional_text(raw["doi"], f"evidence[{index}].doi")
        stable_id = _optional_text(raw["stable_id"], f"evidence[{index}].stable_id")
        if not doi and not stable_id:
            raise ValueError(f"evidence[{index}] requires doi or stable_id")
        evidence.append(CandidateEvidence(
            candidate_id=candidate_id, paper_title=_text(raw["paper_title"], f"evidence[{index}].paper_title"),
            paper_year=_integer(raw["paper_year"], f"evidence[{index}].paper_year", positive=True),
            doi=doi, stable_id=stable_id,
            primary_source_url=_text(raw["primary_source_url"], f"evidence[{index}].primary_source_url"),
            primary_source_anchors=_string_tuple(raw["primary_source_anchors"], f"evidence[{index}].primary_source_anchors"),
            official_code_url=_optional_text(raw["official_code_url"], f"evidence[{index}].official_code_url"),
            license_route=_text(raw["license_route"], f"evidence[{index}].license_route"), proposed_slot=proposed_slot,
            uses_future_truth_at_inference=_boolean(raw["uses_future_truth_at_inference"], f"evidence[{index}].uses_future_truth_at_inference"),
            binary_only_without_continuous_form=_boolean(raw["binary_only_without_continuous_form"], f"evidence[{index}].binary_only_without_continuous_form"),
            equations_sufficient=_boolean(raw["equations_sufficient"], f"evidence[{index}].equations_sufficient"),
            has_forecast_decision_coupling=_boolean(raw["has_forecast_decision_coupling"], f"evidence[{index}].has_forecast_decision_coupling"),
            preserves_core_under_adaptation=_boolean(raw["preserves_core_under_adaptation"], f"evidence[{index}].preserves_core_under_adaptation"),
            deployment_produces_dispatch=_boolean(raw["deployment_produces_dispatch"], f"evidence[{index}].deployment_produces_dispatch"),
            deployment_exact_optimizer=_boolean(raw["deployment_exact_optimizer"], f"evidence[{index}].deployment_exact_optimizer"),
            gradient_coupling=_text(raw["gradient_coupling"], f"evidence[{index}].gradient_coupling"),
            input_fields=_string_tuple(raw["input_fields"], f"evidence[{index}].input_fields"),
            forecast_representation=_text(raw["forecast_representation"], f"evidence[{index}].forecast_representation"),
            decision_layer=_text(raw["decision_layer"], f"evidence[{index}].decision_layer"),
            loss_description=_text(raw["loss_description"], f"evidence[{index}].loss_description"),
            gradient_path=_text(raw["gradient_path"], f"evidence[{index}].gradient_path"),
            output_type=_text(raw["output_type"], f"evidence[{index}].output_type"),
            topology_description=_text(raw["topology_description"], f"evidence[{index}].topology_description"),
            adaptation_disclosure=_text(raw["adaptation_disclosure"], f"evidence[{index}].adaptation_disclosure"),
        ))
    return tuple(evidence)


def load_external_registry(path: str | Path) -> ExternalBaselineRegistry:
    source = Path(path)
    payload = _load_json(source)
    _strict_keys(payload, {"schema_version", "methods"}, "external registry")
    if payload["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError(f"schema_version must equal {REGISTRY_SCHEMA!r}")
    methods: list[ExternalBaselineSpec] = []
    seen_slots: set[str] = set()
    seen_candidates: set[str] = set()
    required = {
        "slot", "candidate_id", "method_id", "paper_title", "paper_year", "doi", "stable_id",
        "primary_source_url", "official_code_url", "license_route", "reproduction_level", "score_total",
        "input_mode", "output_mode", "optimizer_role", "implementation_class", "adaptation_disclosure",
        "evidence_sha256", "implementation_ready",
    }
    for index, raw_value in enumerate(_sequence(payload["methods"], "methods")):
        raw = _mapping(raw_value, f"methods[{index}]")
        _strict_keys(raw, required, f"methods[{index}]")
        slot = _text(raw["slot"], f"methods[{index}].slot")
        if slot not in REQUIRED_SLOTS:
            raise ValueError(f"methods[{index}].slot is not an external slot")
        if slot in seen_slots:
            raise ValueError(f"duplicate external slot: {slot}")
        seen_slots.add(slot)
        candidate_id = _text(raw["candidate_id"], f"methods[{index}].candidate_id")
        if candidate_id in seen_candidates:
            raise ValueError(f"candidate cannot fill multiple slots: {candidate_id}")
        seen_candidates.add(candidate_id)
        method_id = _text(raw["method_id"], f"methods[{index}].method_id")
        if method_id in INTERNAL_METHODS:
            raise ValueError("internal method cannot fill an external slot")
        if raw["reproduction_level"] not in REPRODUCTION_LEVELS:
            raise ValueError(f"methods[{index}].reproduction_level is unsupported")
        doi = _optional_text(raw["doi"], f"methods[{index}].doi")
        stable_id = _optional_text(raw["stable_id"], f"methods[{index}].stable_id")
        if not doi and not stable_id:
            raise ValueError(f"methods[{index}] requires doi or stable_id")
        if not _boolean(raw["implementation_ready"], f"methods[{index}].implementation_ready"):
            raise ValueError("registry cannot be frozen before implementation readiness is recorded")
        score = _integer(raw["score_total"], f"methods[{index}].score_total")
        if score < 70:
            raise ValueError("selected external baseline score must be at least 70")
        methods.append(ExternalBaselineSpec(
            slot=slot, candidate_id=candidate_id, method_id=method_id,
            paper_title=_text(raw["paper_title"], f"methods[{index}].paper_title"),
            paper_year=_integer(raw["paper_year"], f"methods[{index}].paper_year", positive=True), doi=doi, stable_id=stable_id,
            primary_source_url=_text(raw["primary_source_url"], f"methods[{index}].primary_source_url"),
            official_code_url=_optional_text(raw["official_code_url"], f"methods[{index}].official_code_url"),
            license_route=_text(raw["license_route"], f"methods[{index}].license_route"),
            reproduction_level=str(raw["reproduction_level"]), score_total=score,
            input_mode=_text(raw["input_mode"], f"methods[{index}].input_mode"), output_mode=_text(raw["output_mode"], f"methods[{index}].output_mode"),
            optimizer_role=_text(raw["optimizer_role"], f"methods[{index}].optimizer_role"),
            implementation_class=_text(raw["implementation_class"], f"methods[{index}].implementation_class"),
            adaptation_disclosure=_text(raw["adaptation_disclosure"], f"methods[{index}].adaptation_disclosure"),
            evidence_sha256=_text(raw["evidence_sha256"], f"methods[{index}].evidence_sha256"),
            implementation_ready=True,
        ))
    if tuple(method.slot for method in methods) and set(seen_slots) != set(REQUIRED_SLOTS):
        missing = sorted(set(REQUIRED_SLOTS) - seen_slots)
        raise ValueError(f"registry missing external slots: {missing}")
    if len(methods) != 3:
        raise ValueError("external registry must contain exactly three methods")
    return ExternalBaselineRegistry(REGISTRY_SCHEMA, tuple(methods), source)


__all__ = [
    "CandidateEvidence", "CandidateScore", "ExternalBaselineRegistry", "ExternalBaselineSpec",
    "INTERNAL_METHODS", "QueryFamily", "REPRODUCTION_LEVELS", "REQUIRED_SLOTS", "REGISTRY_SCHEMA",
    "SEARCH_SCHEMA", "SearchProtocol", "load_candidate_evidence", "load_external_registry", "load_search_protocol",
]
