"""Score screened candidates and freeze one external method per slot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_registry import (  # noqa: E402
    CandidateEvidence,
    CandidateScore,
    ExternalBaselineRegistry,
    load_candidate_evidence,
    load_search_protocol,
    score_candidate,
    select_external_slots,
)


def _score_row(score: CandidateScore) -> dict[str, object]:
    row = asdict(score)
    row["total"] = score.total
    row["eligible"] = score.eligible
    row["fatal_exclusions"] = ";".join(score.fatal_exclusions)
    return row


def _registry_payload(registry: ExternalBaselineRegistry) -> dict[str, object]:
    return {
        "schema_version": registry.schema_version,
        "methods": [asdict(method) for method in registry.methods],
    }


def freeze_external_baselines(protocol_path: str | Path, evidence_path: str | Path, scores_path: str | Path,
                              registry_path: str | Path, receipt_path: str | Path) -> dict[str, object]:
    protocol = load_search_protocol(protocol_path)
    evidence = load_candidate_evidence(evidence_path)
    evidence_by_id = {item.candidate_id: item for item in evidence}
    scores = [score_candidate(item, protocol) for item in evidence]
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["candidate_id", "slot", "publication_year", "slot_fit", "dispatch_compatibility",
                  "reproducibility", "information_fairness", "source_quality", "recency", "resource_fit",
                  "fatal_exclusions", "total", "eligible"]
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_score_row(score) for score in scores)
    registry = select_external_slots(scores, evidence_by_id)
    registry_path = Path(registry_path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_payload = _registry_payload(registry)
    registry_path.write_text(json.dumps(registry_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # Round-trip through the strict loader before declaring the freeze usable.
    from src.joint_dispatch.external_registry import load_external_registry
    checked = load_external_registry(registry_path)
    receipt = {
        "schema_version": "rsc-pf-external-selection-receipt-v1",
        "protocol_schema": protocol.schema_version,
        "selection_status": "complete",
        "test_set_accessed": False,
        "candidate_count": len(evidence),
        "eligible_count": sum(score.eligible for score in scores),
        "methods": [{"slot": method.slot, "method_id": method.method_id, "candidate_id": method.candidate_id,
                     "score_total": method.score_total, "doi": method.doi, "stable_id": method.stable_id}
                    for method in checked.methods],
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
    }
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    result = freeze_external_baselines(args.protocol, args.evidence, args.scores, args.registry, args.receipt)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
