"""Deterministic candidate normalization and cross-source deduplication."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence


_STOPWORDS = {"a", "an", "the", "in", "of", "for", "on", "to", "and", "with", "by", "et", "al", "using"}
_DOI_RE = re.compile(r"(?:https?://)?(?:dx\.)?doi\.org/|^doi:", flags=re.IGNORECASE)


@dataclass(frozen=True)
class LiteratureCandidate:
    candidate_id: str
    title: str
    normalized_title: str
    authors: tuple[str, ...]
    first_author_surname: str
    year: int
    doi: str | None
    stable_id: str | None
    venue: str
    abstract: str
    citation_count: int
    primary_url: str
    code_url: str | None
    volume: str | None
    pages: str | None
    sources: tuple[str, ...]
    matched_query_ids: tuple[str, ...]
    source_records: tuple[str, ...]


def normalize_doi(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    doi = _DOI_RE.sub("", str(value).strip()).strip().rstrip(".,;)")
    return doi.lower() or None


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = [token for token in text.split() if token not in _STOPWORDS]
    return " ".join(tokens)


def _surname(author: str) -> str:
    text = re.sub(r"[^\w\s,.-]", "", str(author or "")).strip()
    if not text:
        return ""
    if "," in text:
        first = text.split(",", 1)[0].strip()
        return first.lower().rstrip(".")
    tokens = [token.rstrip(".") for token in text.split() if token.rstrip(".")]
    if len(tokens) == 1:
        return tokens[0].lower()
    for token in reversed(tokens):
        if len(token) > 1:
            return token.lower()
    return tokens[-1].lower()


def _authors(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        items = []
        for item in value:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("author") or item.get("family")
                if item.get("given") and item.get("family"):
                    name = f"{item['given']} {item['family']}"
                items.append(str(name or ""))
            else:
                items.append(str(item))
    else:
        items = []
    result = tuple(item.strip() for item in items if item and item.strip())
    if not result:
        raise ValueError("authors must contain at least one author")
    return result


def _year(value: Any) -> int:
    try:
        year = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("year must be an integer") from exc
    if year < 1900 or year > 2100:
        raise ValueError("year is outside the supported range")
    return year


def normalize_candidate(record: Mapping[str, Any], source: str, query_id: str) -> LiteratureCandidate:
    if not isinstance(record, Mapping):
        raise ValueError("candidate record must be an object")
    title = str(record.get("title") or record.get("paper_title") or "").strip()
    if not title:
        raise ValueError("candidate title is required")
    authors = _authors(record.get("authors") or record.get("author"))
    doi = normalize_doi(record.get("doi") or record.get("DOI"))
    stable_id = record.get("stable_id") or record.get("arxiv_id") or record.get("paper_id") or record.get("id")
    stable_id = str(stable_id).strip() if stable_id is not None and str(stable_id).strip() else None
    if not doi and not stable_id:
        raise ValueError("candidate requires doi or stable_id")
    primary_url = str(record.get("primary_url") or record.get("url") or record.get("link") or "").strip()
    if not primary_url:
        raise ValueError("candidate primary_url is required")
    citation = record.get("citation_count") or record.get("cited_by_count") or 0
    try:
        citation_count = max(0, int(citation))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("citation_count must be an integer") from exc
    normalized = normalize_title(title)
    key = f"doi:{doi}" if doi else f"title:{normalized}|author:{_surname(authors[0])}"
    candidate_id = str(record.get("candidate_id") or ("doi:" + doi if doi else "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]))
    return LiteratureCandidate(
        candidate_id=candidate_id, title=title, normalized_title=normalized, authors=authors,
        first_author_surname=_surname(authors[0]), year=_year(record.get("year") or record.get("publication_year")),
        doi=doi, stable_id=stable_id, venue=str(record.get("venue") or record.get("journal") or "").strip(),
        abstract=str(record.get("abstract") or "").strip(), citation_count=citation_count,
        primary_url=primary_url, code_url=(str(record["code_url"]).strip() if record.get("code_url") else None),
        volume=(str(record["volume"]).strip() if record.get("volume") else None),
        pages=(str(record["pages"]).strip() if record.get("pages") else None),
        sources=(str(source),), matched_query_ids=(str(query_id),), source_records=(str(source),),
    )


def _jaccard(left: str, right: str) -> float:
    a, b = set(left.split()), set(right.split())
    return len(a & b) / len(a | b) if a or b else 0.0


def _same_doi(left: LiteratureCandidate, right: LiteratureCandidate) -> bool:
    return bool(left.doi and right.doi and left.doi == right.doi)


def _doi_less_duplicate(left: LiteratureCandidate, right: LiteratureCandidate) -> bool:
    # DOI is the primary key when both records have one.  If only one source
    # exposes a DOI, the prescribed title/first-author fallback is still
    # needed to merge the publisher record with its DOI-less preprint record.
    return not (left.doi and right.doi) and left.first_author_surname == right.first_author_surname and _jaccard(left.normalized_title, right.normalized_title) >= 0.90


def _completeness(candidate: LiteratureCandidate) -> tuple[int, int, int, int]:
    return (int(bool(candidate.doi)), int(bool(candidate.volume)), int(bool(candidate.pages)), int(bool(candidate.abstract)))


def _preferred(left: LiteratureCandidate, right: LiteratureCandidate) -> LiteratureCandidate:
    source_rank = {"crossref": 5, "sciencedirect": 4, "scopus": 4, "semantic_scholar": 3, "arxiv": 2}
    left_key = (_completeness(left), source_rank.get(left.sources[0], 1), left.citation_count)
    right_key = (_completeness(right), source_rank.get(right.sources[0], 1), right.citation_count)
    return left if left_key >= right_key else right


def _merge(left: LiteratureCandidate, right: LiteratureCandidate) -> LiteratureCandidate:
    preferred = _preferred(left, right)
    sources = tuple(dict.fromkeys(left.sources + right.sources))
    query_ids = tuple(dict.fromkeys(left.matched_query_ids + right.matched_query_ids))
    source_records = tuple(dict.fromkeys(left.source_records + right.source_records))
    return replace(
        preferred, sources=sources, matched_query_ids=query_ids, source_records=source_records,
        code_url=preferred.code_url or left.code_url or right.code_url,
        abstract=preferred.abstract or left.abstract or right.abstract,
        venue=preferred.venue or left.venue or right.venue,
        volume=preferred.volume or left.volume or right.volume,
        pages=preferred.pages or left.pages or right.pages,
    )


def deduplicate_candidates(candidates: Sequence[LiteratureCandidate]) -> tuple[LiteratureCandidate, ...]:
    merged: list[LiteratureCandidate] = []
    for candidate in candidates:
        match_index = None
        for index, existing in enumerate(merged):
            if _same_doi(candidate, existing) or _doi_less_duplicate(candidate, existing):
                match_index = index
                break
        if match_index is None:
            merged.append(candidate)
        else:
            merged[match_index] = _merge(merged[match_index], candidate)
    return tuple(sorted(merged, key=lambda item: (item.year, item.candidate_id), reverse=True))


__all__ = ["LiteratureCandidate", "deduplicate_candidates", "normalize_candidate", "normalize_doi", "normalize_title"]
