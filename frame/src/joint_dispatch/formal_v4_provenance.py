"""Hash-bound provenance contracts for the repaired formal-v4.1 run.

The module is deliberately small and side-effect free: it hashes source bytes,
validates the Git state of a declared runtime closure, and validates the
registry that excludes pre-repair trials.  It does not write receipts or make
Gate 0 decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


SOURCE_MANIFEST_SCHEMA = "rsc-pf-source-manifest-v4.1"
INVALID_RUN_REGISTRY_SCHEMA = "rsc-pf-invalid-run-registry-v4"
INVALID_TRIAL_ROOTS = (
    "formal_v4_20260903",
    "formal_v4_20260903_retry1",
    "formal_v4_20260903_retry2",
    "formal_v4_20260903_retry3",
    "formal_v4_20260903_retry3_gate1d",
    "formal_v4_20260903_retry3_gate1e",
    "formal_v4_20260903_retry3_gate1f",
    "formal_v4_20260903_retry3_gate1_g",
    "formal_v4_20260903_retry3_gate1_h",
    "formal_v4_20260903_retry3_gate1_i",
    "formal_v4_20260903_retry3_gate1_final",
    "formal_v4_20260903_retry3_gate2",
    "gate1_smoke",
    "joint_forecast_dispatch_formal_v4_retry_gate1",
)


@dataclass(frozen=True)
class SourceFileEntry:
    path: str
    sha256: str
    size_bytes: int

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class FormalV4SourceManifest:
    git_commit: str
    entries: tuple[SourceFileEntry, ...]
    schema_version: str = SOURCE_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_MANIFEST_SCHEMA:
            raise ValueError(f"source manifest schema must equal {SOURCE_MANIFEST_SCHEMA!r}")
        if not self.git_commit.strip():
            raise ValueError("source manifest git_commit must be non-empty")
        paths = tuple(entry.path for entry in self.entries)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("source manifest entries must be unique and sorted")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "git_commit": self.git_commit,
            "entries": [entry.to_payload() for entry in self.entries],
        }

    @property
    def identity_sha256(self) -> str:
        encoded = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_root(repo_root: Path) -> Path | None:
    result = _git(repo_root, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _canonical_relative(path: str | Path, repo_root: Path) -> tuple[str, Path]:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"source path escapes repository root: {path}") from exc
    normalized = relative.as_posix()
    if not normalized or normalized == ".":
        raise ValueError("source path must name a file")
    if any(token in INVALID_TRIAL_ROOTS for token in relative.parts):
        raise ValueError(f"invalid pre-repair trial is not a source input: {normalized}")
    return normalized, resolved


def _validate_git_state(repo_root: Path, relative: str, expected_commit: str) -> None:
    git_root = _git_root(repo_root)
    if git_root is None:
        return
    if git_root != repo_root.resolve():
        raise ValueError(f"repo_root is not the Git root: {repo_root}")
    head = _git(repo_root, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise ValueError("source manifest Git commit does not match HEAD")
    tracked = _git(repo_root, "ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode != 0 or tracked.stdout.strip() != relative:
        raise ValueError(f"source is untracked: {relative}")
    status = _git(repo_root, "status", "--porcelain=v1", "--untracked-files=all", "--", relative)
    if status.returncode != 0:
        raise ValueError(f"could not inspect Git status for source: {relative}")
    if status.stdout.strip():
        raise ValueError(f"source is dirty: {relative}")


def build_source_manifest(
    repo_root: str | Path,
    paths: Iterable[str | Path],
    git_commit: str,
) -> FormalV4SourceManifest:
    """Build a deterministic manifest from raw source bytes.

    In a Git checkout the supplied commit must be HEAD and every path must be
    tracked and clean.  A non-Git temporary directory is accepted for isolated
    unit tests, while still enforcing path, file, duplicate and hash rules.
    """

    root = Path(repo_root).resolve()
    if not git_commit or not git_commit.strip():
        raise ValueError("git_commit must be non-empty")
    normalized: dict[str, Path] = {}
    for raw in paths:
        relative, resolved = _canonical_relative(raw, root)
        if relative in normalized:
            raise ValueError(f"duplicate source path: {relative}")
        if not resolved.is_file():
            raise ValueError(f"source file does not exist: {relative}")
        _validate_git_state(root, relative, git_commit)
        normalized[relative] = resolved
    entries = tuple(
        SourceFileEntry(path=relative, sha256=_sha256_file(path), size_bytes=path.stat().st_size)
        for relative, path in sorted(normalized.items())
    )
    return FormalV4SourceManifest(git_commit=git_commit.strip(), entries=entries)


def _manifest_from_payload(value: FormalV4SourceManifest | Mapping[str, Any]) -> FormalV4SourceManifest:
    if isinstance(value, FormalV4SourceManifest):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("source manifest must be an object")
    expected = {"schema_version", "git_commit", "entries"}
    if set(value) != expected:
        raise ValueError("source manifest has unknown or missing fields")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("source manifest entries must be an array")
    entries: list[SourceFileEntry] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"source manifest entry {index} is invalid")
        path = str(raw["path"])
        digest = str(raw["sha256"])
        try:
            size = int(raw["size_bytes"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source manifest entry {index} size is invalid") from exc
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest.lower()) or size < 0:
            raise ValueError(f"source manifest entry {index} hash or size is invalid")
        entries.append(SourceFileEntry(path, digest.lower(), size))
    return FormalV4SourceManifest(str(value["git_commit"]), tuple(entries), str(value["schema_version"]))


def validate_source_manifest(
    manifest: FormalV4SourceManifest | Mapping[str, Any],
    repo_root: str | Path,
) -> FormalV4SourceManifest:
    """Rehash and validate every manifest entry against the current checkout."""

    root = Path(repo_root).resolve()
    resolved_manifest = _manifest_from_payload(manifest)
    for entry in resolved_manifest.entries:
        relative, path = _canonical_relative(entry.path, root)
        if relative != entry.path:
            raise ValueError(f"source path is not normalized: {entry.path}")
        if not path.is_file():
            raise ValueError(f"source file does not exist: {entry.path}")
        if path.stat().st_size != entry.size_bytes or _sha256_file(path) != entry.sha256:
            raise ValueError(f"source hash mismatch: {entry.path}")
        _validate_git_state(root, entry.path, resolved_manifest.git_commit)
    return resolved_manifest


def validate_invalid_run_registry(value: Mapping[str, Any]) -> None:
    """Validate the immutable list of pre-repair roots excluded from results."""

    if not isinstance(value, Mapping):
        raise ValueError("invalid-run registry must be an object")
    if set(value) != {"schema_version", "status", "audit_date", "runs"}:
        raise ValueError("invalid-run registry has unknown or missing fields")
    if value["schema_version"] != INVALID_RUN_REGISTRY_SCHEMA or value["status"] != "frozen":
        raise ValueError("invalid-run registry schema or status is invalid")
    runs = value["runs"]
    if not isinstance(runs, list) or not runs:
        raise ValueError("invalid-run registry runs must be non-empty")
    seen: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping) or set(run) != {"run_root", "status", "reasons", "allowed_for_formal_results"}:
            raise ValueError(f"invalid-run registry entry {index} is invalid")
        root = str(run["run_root"])
        if root in seen or root not in INVALID_TRIAL_ROOTS:
            raise ValueError(f"invalid-run registry root is not frozen: {root}")
        seen.add(root)
        if run["status"] != "invalid_pre_repair_trial" or run["allowed_for_formal_results"] is not False:
            raise ValueError(f"invalid-run registry entry {root} is admissible")
        if not isinstance(run["reasons"], list) or not run["reasons"] or not all(isinstance(item, str) and item.strip() for item in run["reasons"]):
            raise ValueError(f"invalid-run registry reasons are invalid: {root}")
    missing = set(INVALID_TRIAL_ROOTS) - seen
    if missing:
        raise ValueError(f"invalid-run registry is missing roots: {sorted(missing)}")


def load_invalid_run_registry(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    validate_invalid_run_registry(payload)
    return payload


__all__ = [
    "FormalV4SourceManifest", "INVALID_RUN_REGISTRY_SCHEMA", "INVALID_TRIAL_ROOTS",
    "SOURCE_MANIFEST_SCHEMA", "SourceFileEntry", "build_source_manifest",
    "load_invalid_run_registry", "validate_invalid_run_registry", "validate_source_manifest",
]
