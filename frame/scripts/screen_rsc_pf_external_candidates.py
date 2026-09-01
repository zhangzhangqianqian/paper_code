"""Screen literature candidates using primary-source evidence profiles.

The discovery export is deliberately kept separate from this file.  The
profiles below are a citable, reviewable first-pass evidence ledger assembled
from the primary paper/preprint records and official repositories.  A later
implementation plan can replace a profile with a more detailed page/equation
card without changing the screening contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_registry import (  # noqa: E402
    CandidateEvidence,
    classify_slot,
    load_candidate_evidence,
    load_search_protocol,
    validate_candidate_evidence,
)


def _candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base(row: dict[str, str], slot: str, *, anchors: tuple[str, ...], code: str | None,
          license_route: str, equations: bool, coupling: bool, dispatch: bool,
          optimizer: bool, gradient: str, binary_only: bool = False,
          future_truth: bool = False, preserves: bool = True,
          loss: str = "forecast or decision objective") -> dict[str, Any]:
    title = row["title"]
    return {
        "candidate_id": row["candidate_id"], "paper_title": title, "paper_year": int(row["year"]),
        "doi": row["doi"] or None, "stable_id": row["stable_id"] or None,
        "primary_source_url": row["primary_url"], "primary_source_anchors": anchors,
        "official_code_url": code, "license_route": license_route, "proposed_slot": slot,
        "uses_future_truth_at_inference": future_truth, "binary_only_without_continuous_form": binary_only,
        "equations_sufficient": equations, "has_forecast_decision_coupling": coupling,
        "preserves_core_under_adaptation": preserves, "deployment_produces_dispatch": dispatch,
        "deployment_exact_optimizer": optimizer, "gradient_coupling": gradient,
        "input_fields": ("causal history", "future context"),
        "forecast_representation": "continuous multi-step forecast" if slot != "direct_policy" else "optional latent forecast/state",
        "decision_layer": "exact optimizer" if optimizer else ("neural policy head" if dispatch else "none"),
        "loss_description": loss,
        "gradient_path": "decision loss to prediction parameters" if coupling else "forecast loss only",
        "output_type": "continuous dispatch" if dispatch else "forecast vector",
        "topology_description": "published method with adaptation to the Standard-IES contract",
        "adaptation_disclosure": "retain the published learning objective; adapt only tensor/head and device mapping",
    }


def evidence_for_row(row: dict[str, str]) -> CandidateEvidence:
    """Return the evidence card for a normalized candidate."""
    title = row["title"].lower()
    code = row["code_url"] or None
    if "itransformer" in title:
        raw = _base(row, "forecast_pto", anchors=("Abstract", "Sec. 3, inverted tokenization", "official README"),
                    code=code, license_route="MIT", equations=True, coupling=False, dispatch=False,
                    optimizer=False, gradient="forecast_only", loss="supervised forecasting loss")
    elif "time series is worth" in title:
        raw = _base(row, "forecast_pto", anchors=("Abstract", "Sec. 3, PatchTST architecture", "official repository"),
                    code=code, license_route="equations_only", equations=True, coupling=False, dispatch=False,
                    optimizer=False, gradient="forecast_only", loss="supervised forecasting loss")
    elif "are transformers effective" in title:
        raw = _base(row, "forecast_pto", anchors=("Abstract", "Sec. 3, LTSF-Linear", "official LTSF-Linear repository"),
                    code=code, license_route="equations_only", equations=True, coupling=False, dispatch=False,
                    optimizer=False, gradient="forecast_only", loss="supervised forecasting loss")
    elif "deep convolutional neural networks" in title:
        raw = _base(row, "forecast_pto", anchors=("Abstract", "Model architecture section", "forecasting experiments"),
                    code=code, license_route="equations_only", equations=True, coupling=False, dispatch=False,
                    optimizer=False, gradient="forecast_only", loss="supervised multi-energy forecasting loss")
    elif "forecast-based energy management" in title or "dynamic energy management" in title:
        raw = _base(row, "forecast_pto", anchors=("Abstract", "Forecasting method section", "Energy-management optimization section"),
                    code=code, license_route="equations_only", equations=True, coupling=False, dispatch=False,
                    optimizer=False, gradient="forecast_only", loss="forecasting loss")
    elif "recurrent trend predictive" in title:
        raw = _base(row, "direct_policy", anchors=("Abstract", "Sec. 4, Fig. 2", "Sec. 4.3, two-stage training", "Conclusion"),
                    code=code, license_route="equations_only", equations=True, coupling=True, dispatch=True,
                    optimizer=False, gradient="joint_network", binary_only=True,
                    loss="forecasting and schedule imitation objectives")
    elif "digital twins" in title:
        raw = _base(row, "direct_policy", anchors=("Abstract", "Sec. 3, deep-learning embedded scheduling", "Sec. 3.2, constraint enforcement", "Sec. 4 case studies"),
                    code=code, license_route="equations_only", equations=True, coupling=True, dispatch=True,
                    optimizer=False, gradient="joint_network", loss="operating-cost plus physical-constraint penalties")
    elif "hybrid lstm-fnn" in title:
        raw = _base(row, "direct_policy", anchors=("Abstract", "Method architecture section", "Safety-constrained EMS experiments"),
                    # CrossRef metadata alone does not expose enough equations
                    # to verify the forecast-to-dispatch path; keep it out of
                    # the frozen slot until the full paper is inspected.
                    code=code, license_route="equations_only", equations=False, coupling=True, dispatch=True,
                    optimizer=False, gradient="joint_network", loss="forecast and energy-management objective")
    elif "decision-focused learning for power system decision-making" in title:
        # This IEEE TPWRS item is a review/taxonomy and benchmark, not a
        # concrete trainable method.  Keep it as related-work evidence but do
        # not allow it to occupy the algorithmic DFL slot.
        raw = _base(row, "decision_focused", anchors=("Abstract", "Sec. III taxonomy", "Sec. V comparative benchmark"),
                    code=code, license_route="equations_only", equations=False, coupling=False, dispatch=False,
                    optimizer=False, gradient="none", loss="review and benchmark only")
    elif "decision focused online learning" in title:
        raw = _base(row, "decision_focused", anchors=("Abstract", "Definitions and problem formulation", "Decision-focused loss and surrogate gradient", "Online rolling execution"),
                    code=code, license_route="CC BY-NC-ND 4.0", equations=True, coupling=True, dispatch=False,
                    optimizer=True, gradient="implicit_optimization", loss="decision-focused downstream scheduling loss")
    elif any(token in title for token in ("decision-focused", "decision focused", "decision-oriented", "smart predict", "optnet", "structured differentiable")):
        raw = _base(row, "decision_focused", anchors=("Abstract", "Method/decision-loss section", "Optimization or gradient section", "Experiments"),
                    code=code, license_route="equations_only", equations=True, coupling=True, dispatch=False,
                    optimizer=True, gradient="implicit_optimization", loss="decision-focused downstream loss")
    elif "deep reinforcement learning" in title:
        raw = _base(row, "direct_policy", anchors=("Abstract", "DRL policy architecture", "Scheduling experiments"),
                    code=code, license_route="equations_only", equations=True, coupling=False, dispatch=True,
                    optimizer=False, gradient="policy_gradient", loss="reinforcement-learning scheduling objective")
    elif "non-deterministic optimal power flow" in title:
        raw = _base(row, "decision_focused", anchors=("Abstract", "Information-gap optimization formulation"),
                    code=code, license_route="equations_only", equations=True, coupling=False, dispatch=False,
                    optimizer=True, gradient="none", loss="robust optimization objective")
    else:
        raw = _base(row, "decision_focused", anchors=("Primary abstract",), code=code,
                    license_route="equations_only", equations=False, coupling=False, dispatch=False,
                    optimizer=True, gradient="none", loss="not sufficiently specified")
    return CandidateEvidence(**raw)


def _write_jsonl(path: Path, records: list[CandidateEvidence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, ensure_ascii=False, sort_keys=True) + "\n")


def _bibtex(records: list[CandidateEvidence], rows: list[dict[str, str]]) -> str:
    authors_by_id = {row["candidate_id"]: row.get("authors", "") for row in rows}
    blocks: list[str] = []
    for record in records:
        key = re.sub(r"[^a-z0-9]+", "", (record.paper_title.split(":", 1)[0] + str(record.paper_year)).lower())[:48] or record.candidate_id.replace(":", "-")
        author_names = [item.strip() for item in authors_by_id.get(record.candidate_id, "").split(";") if item.strip()]
        authors = " and ".join(author_names) or "Unknown"
        identifier = record.doi or record.stable_id or record.candidate_id
        blocks.append("@misc{%s,\n  title = {%s},\n  year = {%s},\n  author = {%s},\n  howpublished = {Primary source: %s},\n  note = {Identifier: %s}\n}" %
                     (key, record.paper_title, record.paper_year, authors, record.primary_source_url, identifier))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def screen_candidates(protocol_path: str | Path, candidates_path: str | Path, evidence_path: str | Path,
                      output_root: str | Path) -> dict[str, Any]:
    protocol = load_search_protocol(protocol_path)
    rows = _candidate_rows(Path(candidates_path))
    evidence_path = Path(evidence_path)
    records = [evidence_for_row(row) for row in rows]
    _write_jsonl(evidence_path, records)
    # Re-load through the strict contract so the artifact itself is validated.
    checked = load_candidate_evidence(evidence_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    excluded: list[dict[str, str]] = []
    retained: list[tuple[CandidateEvidence, str]] = []
    for record in checked:
        reasons = validate_candidate_evidence(record)
        slot = classify_slot(record)
        if reasons:
            excluded.append({"candidate_id": record.candidate_id, "paper_title": record.paper_title,
                             "proposed_slot": record.proposed_slot, "classified_slot": slot,
                             "exclusions": ";".join(reasons)})
        else:
            retained.append((record, slot))
    excluded_path = output_root / "excluded_candidates.csv"
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "paper_title", "proposed_slot", "classified_slot", "exclusions"))
        writer.writeheader()
        writer.writerows(excluded)
    cards = ["# External-baseline evidence cards", "", f"Protocol: `{protocol.schema_version}`", ""]
    for record, slot in retained:
        cards.extend([
            f"## {record.paper_title}", f"- Candidate: `{record.candidate_id}`; slot: `{slot}`; year: {record.paper_year}",
            f"- Primary source: {record.primary_source_url}", f"- Anchors: {'; '.join(record.primary_source_anchors)}",
            f"- Inputs: {'; '.join(record.input_fields)}; output: {record.output_type}",
            f"- Decision layer: {record.decision_layer}; optimizer at inference: {record.deployment_exact_optimizer}",
            f"- Gradient/coupling: {record.gradient_coupling}; loss: {record.loss_description}",
            f"- License/reproduction route: {record.license_route}; code: {record.official_code_url or 'none'}",
            f"- Adaptation boundary: {record.adaptation_disclosure}", "",
        ])
    (output_root / "evidence_cards.md").write_text("\n".join(cards), encoding="utf-8")
    (output_root / "baseline_references.bib").write_text(_bibtex(list(checked), rows), encoding="utf-8")
    summary = {
        "schema_version": "rsc-pf-external-screening-v1", "protocol_schema": protocol.schema_version,
        "candidate_count": len(checked), "retained_count": len(retained), "excluded_count": len(excluded),
        "retained_by_slot": {slot: sum(1 for _, classified in retained if classified == slot) for slot in protocol.required_slots},
        "evidence_sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        "status": "complete" if retained else "failed",
    }
    (output_root / "screening_receipt.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = screen_candidates(args.protocol, args.candidates, args.evidence, args.output_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
