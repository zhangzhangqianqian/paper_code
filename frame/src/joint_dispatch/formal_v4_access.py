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

    @staticmethod
    def _requested_years(split: str, years: Iterable[int] | None) -> tuple[int, ...]:
        if split not in SPLIT_YEARS:
            raise ValueError("unknown formal-v4 split")
        return tuple(sorted(set(int(value) for value in (years if years is not None else SPLIT_YEARS[split]))))

    def request_years(self, split: str, years: Iterable[int]) -> tuple[int, ...]:
        """Check a split/year request before a canonical loader opens a member."""

        requested_years = self._requested_years(split, years)
        expected = set(SPLIT_YEARS[split])
        allowed = set(requested_years).issubset(expected) and (split != "evaluation" or self.allow_evaluation)
        if not allowed:
            raise PermissionError(f"formal-v4 access denied for {split} years {requested_years}")
        return requested_years

    def guard_archive_member(
        self,
        archive: str | Path,
        member: str,
        *,
        split: str,
        purpose: str,
        caller: str,
        years: Iterable[int] | None = None,
    ) -> None:
        """Deny forbidden members before their bytes are read.

        The guard intentionally does not hash or open the member.  Allowed
        reads are recorded by :meth:`record_archive_event` after the canonical
        loader has obtained the exact member bytes.
        """

        requested_years = self._requested_years(split, years)
        expected = set(SPLIT_YEARS[split])
        path_obj = Path(archive).resolve()
        allowed = set(requested_years).issubset(expected) and (split != "evaluation" or self.allow_evaluation)
        if not allowed:
            self.receipt.events.append(
                AccessEvent(split, requested_years, purpose, str(path_obj), "", caller, "deny", member=str(member))
            )
            raise PermissionError(f"formal-v4 access denied for {split} years {requested_years}")

    def record_archive_event(
        self,
        archive: str | Path,
        member: str,
        *,
        split: str,
        purpose: str,
        caller: str,
        years: Iterable[int] | None = None,
        container_sha256: str,
        member_sha256: str,
        materialize: bool = False,
    ) -> None:
        """Record hashes for bytes that were actually returned by a loader."""

        requested_years = self.request_years(split, years if years is not None else ())
        path_obj = Path(archive).resolve()
        actual_container_hash = _hash_file(path_obj)
        if str(container_sha256) != actual_container_hash:
            raise ValueError("archive container hash does not match the bytes that were read")
        if len(str(member_sha256)) != 64:
            raise ValueError("archive member hash must be a SHA-256 digest")
        self.receipt.events.append(
            AccessEvent(
                split, requested_years, purpose, str(path_obj), actual_container_hash, caller,
                "allow", str(container_sha256), str(member), str(member_sha256), bool(materialize),
            )
        )

    def build_data_access_receipt(self) -> dict[str, Any]:
        return self.receipt.to_payload()

    def build_archive_access_receipt(self) -> dict[str, Any]:
        events = []
        for event in self.receipt.events:
            if event.decision != "allow" or not event.member:
                continue
            events.append({
                "split": event.split,
                "years": list(event.years),
                "container_path": event.path,
                "container_sha256": event.container_sha256,
                "member_name": event.member,
                "member_sha256": event.member_sha256,
            })
        return {
            "schema_version": "formal-v4.1-archive-access-v1",
            "events": events,
            "test_set_accessed": self.receipt.test_set_accessed,
        }

    def request(self, path: str | Path, *, split: str, purpose: str, caller: str, years: Iterable[int] | None = None, materialize: bool = False) -> Path:
        requested_years = self._requested_years(split, years)
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
    # Generic ``open`` calls are also used to hash receipts and source files;
    # flag only data-loading primitives here.  The canonical loader and this
    # controller are the reviewed access boundary and can be allowlisted by
    # the Gate 0 caller.
    patterns = ("pd.read_csv(", "pd.read_parquet(", "np.load(", "zipfile.ZipFile(", "tarfile.open(")
    for raw in paths:
        path = Path(raw).resolve()
        if path in allow or not path.is_file() or path.suffix != ".py" or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_number, line in enumerate(text.splitlines(), start=1):
            # Dynamic NPZ artifact readers are not raw-dataset bypasses.  A
            # literal NumPy path, in contrast, is an auditable direct read.
            np_literal = "np.load(" in line and any(token in line.split("np.load(", 1)[1].lstrip()[:1] for token in ("'", '"'))
            flagged = any(pattern in line for pattern in patterns if pattern != "np.load(") or np_literal
            if flagged:
                findings.append({"path": str(path), "line": line_number, "text": line.strip()})
    return {"status": "pass" if not findings else "fail", "findings": findings, "allowlisted_paths": [str(value) for value in sorted(allow)]}


__all__ = ["AccessEvent", "DataAccessReceipt", "FormalV4AccessController", "SPLIT_YEARS", "scan_runtime_access"]
