"""Fail-closed audit for the external-baseline selection package."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_registry import (  # noqa: E402
    classify_slot,
    load_candidate_evidence,
    load_external_registry,
    load_search_protocol,
    validate_candidate_evidence,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(gates: dict[str, dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    gates[name] = {"status": "pass" if passed else "fail", "detail": detail}


def _handoff(registry: Any, evidence_by_id: dict[str, Any]) -> str:
    lines = ["# RSC-PF external-baseline implementation handoff", "",
             "This handoff is authorized only when the selection audit reports `complete`.", ""]
    for method in registry.methods:
        evidence = evidence_by_id.get(method.candidate_id)
        lines.extend([
            f"## {method.slot}: {method.method_id}",
            f"- Citation: {method.paper_title} ({method.paper_year}).",
            f"- Primary source: {method.primary_source_url}",
            f"- Identifier: {method.doi or method.stable_id}",
            f"- Code/license route: {method.official_code_url or 'equation-level reproduction'}; `{method.license_route}`.",
            f"- Reproduction level: `{method.reproduction_level}`.",
            f"- Permitted inputs: `{method.input_mode}`.",
            f"- Output contract: `{method.output_mode}`.",
            f"- Optimizer role: `{method.optimizer_role}`.",
            f"- Implementation class: `{method.implementation_class}`.",
            f"- Adaptation boundary: {method.adaptation_disclosure}",
        ])
        if evidence is not None:
            lines.extend([
                f"- Primary-paper anchors: {'; '.join(evidence.primary_source_anchors)}.",
                f"- Forecast representation: {evidence.forecast_representation}.",
                f"- Decision layer: {evidence.decision_layer}; output type: {evidence.output_type}.",
                f"- Loss and gradient path: {evidence.loss_description}; {evidence.gradient_path}.",
                f"- Topology: {evidence.topology_description}.",
                f"- Required adaptation: map the published output to the 24-to-4 Standard-IES contract while retaining the method's core objective.",
                f"- Forbidden substitution: do not use RSC-PF test data, internal controls, or an unreported exact optimizer in the deployed path.",
            ])
        lines.append("")
    return "\n".join(lines)


def audit_external_selection(project_root: Path, protocol_path: Path, registry_path: Path,
                             report_root: Path) -> dict[str, Any]:
    protocol = load_search_protocol(protocol_path)
    registry = load_external_registry(registry_path)
    literature = report_root / "literature"
    frozen = report_root / "frozen"
    evidence_path = literature / "screening_evidence.jsonl"
    evidence = load_candidate_evidence(evidence_path)
    evidence_by_id = {item.candidate_id: item for item in evidence}
    receipt_path = frozen / "literature_selection_receipt.json"
    search_receipt_path = literature / "search_receipt.json"
    candidates_path = literature / "deduplicated_candidates.csv"
    scores_path = literature / "screening_scores.csv"
    gates: dict[str, dict[str, Any]] = {}

    search_ok = search_receipt_path.exists()
    if search_ok:
        search = _read_json(search_receipt_path)
        attempts = search.get("attempts", [])
        search_ok = search.get("status") == "complete" and len(search.get("query_families", [])) == 5 and all(
            item.get("status") == "ok" for item in attempts
        )
        search_detail = f"receipt complete; {len(attempts)} tier-1 attempts recorded"
    else:
        search_detail = "search receipt is missing"
    _gate(gates, "search_coverage", search_ok, search_detail)

    doi_values: list[str] = []
    candidate_ok = candidates_path.exists()
    if candidate_ok:
        with candidates_path.open(newline="", encoding="utf-8") as handle:
            candidate_rows = list(csv.DictReader(handle))
        doi_values = [row["doi"].strip().lower() for row in candidate_rows if row.get("doi", "").strip()]
        candidate_ok = len(doi_values) == len(set(doi_values)) and len(candidate_rows) > 0
    _gate(gates, "deduplication", candidate_ok, f"{len(doi_values)} DOI-bearing rows; duplicate DOI count={len(doi_values) - len(set(doi_values))}")

    selected_evidence = [evidence_by_id.get(method.candidate_id) for method in registry.methods]
    source_ok = all(item is not None and item.primary_source_url and item.primary_source_anchors for item in selected_evidence)
    _gate(gates, "primary_sources", source_ok, "all frozen methods have a primary URL and at least one anchor" if source_ok else "missing primary URL/anchor")

    behavior_ok = all(item is not None and classify_slot(item) == method.slot for item, method in zip(selected_evidence, registry.methods))
    _gate(gates, "behavioral_classification", behavior_ok, "deployed forward-path classes match all frozen slots" if behavior_ok else "slot behavior does not match evidence")

    fatal_ok = all(item is not None and not validate_candidate_evidence(item) for item in selected_evidence)
    _gate(gates, "fatal_exclusions", fatal_ok, "no frozen method carries a fatal exclusion" if fatal_ok else "a frozen method carries an exclusion")

    score_lookup: dict[str, dict[str, str]] = {}
    if scores_path.exists():
        with scores_path.open(newline="", encoding="utf-8") as handle:
            score_lookup = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    threshold_ok = True
    for method in registry.methods:
        row = score_lookup.get(method.candidate_id)
        if row is None or row.get("eligible") != "True" or int(row.get("total", "0")) < 70:
            threshold_ok = False
            continue
        threshold_ok = threshold_ok and int(row["slot_fit"]) >= 18 and int(row["reproducibility"]) >= 14 and int(row["information_fairness"]) >= 10
    _gate(gates, "score_thresholds", threshold_ok, "all frozen methods meet total and component floors" if threshold_ok else "score or component floor failed")

    license_ok = all(method.license_route and method.license_route.lower() not in {"forbidden", "license_forbidden", "no_lawful_route"} and method.reproduction_level for method in registry.methods)
    _gate(gates, "license", license_ok, "each method has a lawful code or equation-level reproduction route" if license_ok else "missing or forbidden reproduction route")

    registry_ok = len(registry.methods) == 3 and len({method.candidate_id for method in registry.methods}) == 3 and {method.slot for method in registry.methods} == set(protocol.required_slots)
    _gate(gates, "registry_integrity", registry_ok, "three distinct papers fill the three required slots" if registry_ok else "registry is incomplete or duplicated")

    isolation_ok = receipt_path.exists() and _read_json(receipt_path).get("test_set_accessed") is False
    _gate(gates, "test_isolation", isolation_ok, "selection receipt records test_set_accessed=false" if isolation_ok else "test-set isolation receipt failed")

    readiness_ok = all(method.implementation_ready and method.implementation_class and method.adaptation_disclosure and method.input_mode and method.output_mode for method in registry.methods)
    _gate(gates, "implementation_readiness", readiness_ok, "all method-specific handoff fields are resolved" if readiness_ok else "implementation handoff has unresolved fields")

    failed = [name for name, value in gates.items() if value["status"] != "pass"]
    status = "complete" if not failed else "failed"
    frozen.mkdir(parents=True, exist_ok=True)
    handoff_path = frozen / "implementation_handoff.md"
    handoff_path.write_text(_handoff(registry, evidence_by_id), encoding="utf-8")
    result: dict[str, Any] = {
        "schema_version": "rsc-pf-external-selection-audit-v1", "status": status,
        "failed_gates": failed, "gates": gates,
        "authorized_for_implementation_plan": status == "complete",
        "registry_methods": [method.method_id for method in registry.methods],
        "project_root": str(project_root), "handoff": str(handoff_path),
    }
    (frozen / "selection_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# RSC-PF external-baseline selection audit", "", f"Status: **{status}**", "",
          "| Gate | Status | Detail |", "|---|---|---|"]
    md.extend(f"| {name} | {value['status']} | {value['detail']} |" for name, value in gates.items())
    md.extend(["", f"Authorized for implementation plan: **{result['authorized_for_implementation_plan']}**", "",
               "Frozen methods: " + ", ".join(result["registry_methods"])])
    (frozen / "selection_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--report-root", required=True)
    args = parser.parse_args()
    result = audit_external_selection(Path(args.project_root), Path(args.protocol), Path(args.registry), Path(args.report_root))
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
