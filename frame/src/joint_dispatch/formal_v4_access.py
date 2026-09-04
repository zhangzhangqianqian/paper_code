"""Split-aware data access logging for formal-v4.1."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
import zipfile


TRAIN_YEARS = (2015, 2016, 2017, 2018)
SPLIT_YEARS = {"train": TRAIN_YEARS, "selection": (2019,), "evaluation": (2020,)}


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AccessEvent:
    split: str
    years: tuple[int, ...]
    purpose: str
    path: str
    path_sha256: str
    caller: str
    decision: str
    container_sha256: str = ""
    member: str = ""
    member_sha256: str = ""
    materialized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"split": self.split, "years": list(self.years), "purpose": self.purpose, "path": self.path, "path_sha256": self.path_sha256, "caller": self.caller, "decision": self.decision, "container_sha256": self.container_sha256, "member": self.member, "member_sha256": self.member_sha256, "materialized": self.materialized}


@dataclass
class DataAccessReceipt:
    events: list[AccessEvent] = field(default_factory=list)
    schema_version: str = "formal-v4.1-data-access-v1"

    @property
    def test_set_accessed(self) -> bool:
        return any(event.decision == "allow" and (2020 in event.years or event.split == "evaluation") for event in self.events)

    @property
    def blocked_count(self) -> int:
        return sum(event.decision == "deny" for event in self.events)

    def to_payload(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "events": [event.to_dict() for event in self.events], "test_set_accessed": self.test_set_accessed, "blocked_count": self.blocked_count}

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite access receipt: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")


class FormalV4AccessController:
    """The only approved boundary for formal-v4 dataset reads."""

    def __init__(self, *, allow_evaluation: bool = False, receipt: DataAccessReceipt | None = None) -> None:
        self.allow_evaluation = bool(allow_evaluation)
        self.receipt = receipt or DataAccessReceipt()

    def request(self, path: str | Path, *, split: str, purpose: str, caller: str, years: Iterable[int] | None = None, materialize: bool = False) -> Path:
        if split not in SPLIT_YEARS:
            raise ValueError("unknown formal-v4 split")
        requested_years = tuple(sorted(set(int(value) for value in (years if years is not None else SPLIT_YEARS[split]))))
        expected = set(SPLIT_YEARS[split])
        path_obj = Path(path).resolve()
        path_hash = _hash_file(path_obj) if path_obj.is_file() else ""
        allowed = set(requested_years).issubset(expected) and (split != "evaluation" or self.allow_evaluation)
        event = AccessEvent(split, requested_years, purpose, str(path_obj), path_hash, caller, "allow" if allowed else "deny", materialized=bool(materialize and allowed))
        self.receipt.events.append(event)
        if not allowed:
            raise PermissionError(f"formal-v4 access denied for {split} years {requested_years}")
        if not path_obj.exists():
            raise FileNotFoundError(path_obj)
        return path_obj

    def request_archive_member(self, archive: str | Path, member: str, *, split: str, purpose: str, caller: str, years: Iterable[int] | None = None, materialize: bool = False) -> bytes:
        archive_path = self.request(archive, split=split, purpose=purpose, caller=caller, years=years, materialize=False)
        container_hash = _hash_file(archive_path)
        with zipfile.ZipFile(archive_path) as handle:
            try:
                info = handle.getinfo(member)
            except KeyError as exc:
                raise FileNotFoundError(member) from exc
            data = handle.read(info)
        member_hash = hashlib.sha256(data).hexdigest()
        self.receipt.events[-1] = AccessEvent(self.receipt.events[-1].split, self.receipt.events[-1].years, purpose, str(archive_path), self.receipt.events[-1].path_sha256, caller, self.receipt.events[-1].decision, container_hash, member, member_hash, bool(materialize))
        if materialize and split == "evaluation" and not self.allow_evaluation:
            raise PermissionError("evaluation archive members cannot be materialized before Gate 3")
        return data


def scan_runtime_access(paths: Iterable[str | Path], *, allowlisted_paths: Iterable[str | Path] = ()) -> dict[str, Any]:
    """Static guard against direct dataset opens outside the canonical loader."""

    allow = {Path(value).resolve() for value in allowlisted_paths}
    findings: list[dict[str, Any]] = []
    patterns = ("read_kitakyushu_canonical", "pd.read_csv", "np.load(", "zipfile.ZipFile", "Path.open(", "open(")
    for raw in paths:
        path = Path(raw).resolve()
        if path in allow or not path.is_file() or path.suffix != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern in line for pattern in patterns):
                findings.append({"path": str(path), "line": line_number, "text": line.strip()})
    return {"status": "pass" if not findings else "fail", "findings": findings, "allowlisted_paths": [str(value) for value in sorted(allow)]}


__all__ = ["AccessEvent", "DataAccessReceipt", "FormalV4AccessController", "SPLIT_YEARS", "scan_runtime_access"]
