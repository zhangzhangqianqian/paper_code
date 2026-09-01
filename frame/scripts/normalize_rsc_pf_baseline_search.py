"""Normalize raw multi-source literature exports and write a deterministic receipt."""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_registry import load_search_protocol
from src.joint_dispatch.literature_candidates import LiteratureCandidate, deduplicate_candidates, normalize_candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _raw_records(path: Path) -> tuple[str, str, list[Mapping[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return path.parent.name, path.stem, [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        raise ValueError(f"raw export must be an object or list: {path}")
    source = str(payload.get("source") or path.parent.name)
    query_id = str(payload.get("query_id") or path.stem)
    records = payload.get("records") or payload.get("results") or []
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, list):
        raise ValueError(f"records must be a list: {path}")
    return source, query_id, [item for item in records if isinstance(item, Mapping)]


def _candidate_row(candidate: LiteratureCandidate) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id, "title": candidate.title, "authors": "; ".join(candidate.authors),
        "first_author_surname": candidate.first_author_surname, "year": candidate.year, "doi": candidate.doi or "",
        "stable_id": candidate.stable_id or "", "venue": candidate.venue, "abstract": candidate.abstract,
        "citation_count": candidate.citation_count, "primary_url": candidate.primary_url, "code_url": candidate.code_url or "",
        "volume": candidate.volume or "", "pages": candidate.pages or "", "sources": ";".join(candidate.sources),
        "matched_query_ids": ";".join(candidate.matched_query_ids), "source_records": ";".join(candidate.source_records),
    }


def normalize_search_exports(protocol_path: str | Path, raw_root: str | Path, output_root: str | Path) -> Path:
    protocol = load_search_protocol(protocol_path)
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates: list[LiteratureCandidate] = []
    attempts: list[dict[str, Any]] = []
    raw_files = sorted(raw_root.rglob("*.json")) if raw_root.exists() else []
    for path in raw_files:
        source = path.parent.name
        query_id = path.stem
        try:
            source, query_id, records = _raw_records(path)
            normalized = [normalize_candidate(record, source, query_id) for record in records]
            candidates.extend(normalized)
            attempts.append({"source": source, "query_id": query_id, "path": str(path), "status": "ok", "records": len(records), "sha256": _sha256(path)})
        except Exception as exc:  # retain provider/file failure in the receipt
            attempts.append({"source": source, "query_id": query_id, "path": str(path), "status": "failed", "error": f"{type(exc).__name__}: {exc}", "sha256": _sha256(path)})
    deduplicated = deduplicate_candidates(candidates)
    table = output_root / "deduplicated_candidates.csv"
    fieldnames = list(_candidate_row(deduplicated[0]).keys()) if deduplicated else list(_candidate_row(LiteratureCandidate("", "", "", ("",), "", 2000, None, "x", "", "", 0, "", None, None, None, (), (), ())).keys())
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_candidate_row(candidate) for candidate in deduplicated)
    receipt = {
        "schema_version": "rsc-pf-external-search-receipt-v1", "protocol_schema": protocol.schema_version,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "query_families": [item.query_id for item in protocol.query_families],
        "sources_attempted": {source: any(item["source"] == source for item in attempts) for source in (*protocol.sources_t1, *protocol.sources_t2, *protocol.sources_t3)},
        "attempts": attempts, "raw_record_count": len(candidates), "deduplicated_count": len(deduplicated),
        "duplicate_doi_count": len(candidates) - len(deduplicated), "table_sha256": _sha256(table),
        "status": "complete" if deduplicated and attempts else "failed",
    }
    receipt_path = output_root / "search_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    if receipt["status"] != "complete":
        raise RuntimeError("no normalized literature candidates were produced")
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    normalize_search_exports(args.protocol, args.raw_root, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
