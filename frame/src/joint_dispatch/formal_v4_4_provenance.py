"""Version-bound provenance for the formal-v4.4 execution chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

from .formal_v4_4_artifacts import canonical_sha256, write_json_once


SOURCE_MANIFEST_SCHEMA = "formal-v4.4-source-manifest-v1"
ALLOWLISTED_UNTRACKED = ("third_party/iTransformer_source/",)
DEFAULT_SOURCE_PATHS = (
    "configs/joint_forecast_dispatch_formal_v4_4.json",
    "scripts/run_rsc_pf_formal_v4_4_gate0.py",
    "scripts/run_rsc_pf_formal_v4_4_pilot.py",
    "scripts/audit_rsc_pf_formal_v4_4_pilot.py",
    "scripts/build_rsc_pf_formal_v4_4_source_manifest.py",
    "src/joint_dispatch/formal_v4_4_artifacts.py",
    "src/joint_dispatch/formal_v4_4_contract.py",
    "src/joint_dispatch/formal_v4_4_gate0.py",
    "src/joint_dispatch/formal_v4_4_loss.py",
    "src/joint_dispatch/formal_v4_4_metrics.py",
    "src/joint_dispatch/formal_v4_4_model.py",
    "src/joint_dispatch/formal_v4_4_pilot.py",
    "src/joint_dispatch/formal_v4_4_pilot_data.py",
    "src/joint_dispatch/formal_v4_4_pilot_gate.py",
    "src/joint_dispatch/formal_v4_4_regime.py",
    "src/joint_dispatch/formal_v4_4_training.py",
    "tests/test_joint_dispatch_formal_v4_4_provenance.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )


def _git_head(root: Path) -> str:
    result = _git(root, "rev-parse", "HEAD")
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("repository HEAD is unavailable")
    return result.stdout.strip()


def _relative(path: str | Path, root: Path) -> tuple[str, Path]:
    resolved = (Path(path) if Path(path).is_absolute() else root / Path(path)).resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"source path escapes repository root: {path}") from exc
    if not relative or relative == "." or not resolved.is_file():
        raise ValueError(f"source file does not exist: {relative}")
    return relative, resolved


def _tracked_clean(root: Path, relative: str, expected_head: str) -> None:
    head = _git_head(root)
    if head != expected_head:
        raise ValueError("source manifest Git commit does not match HEAD")
    tracked = _git(root, "ls-files", "--error-unmatch", "--", relative)
    if tracked.returncode != 0 or tracked.stdout.strip() != relative:
        raise ValueError(f"v4.4 source is untracked: {relative}")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", relative)
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError(f"v4.4 source is dirty: {relative}")


def _worktree_status(root: Path) -> tuple[bool, list[str]]:
    result = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if result.returncode != 0:
        raise ValueError("unable to inspect Git worktree")
    entries = [line for line in result.stdout.splitlines() if line.strip()]
    unexpected: list[str] = []
    for line in entries:
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if not any(path.replace("\\", "/").startswith(prefix) for prefix in ALLOWLISTED_UNTRACKED):
            unexpected.append(line)
    return bool(entries), unexpected


def build_source_manifest(
    *, repo_root: str | Path, run_id: str, contract_sha256: str,
    required_paths: Sequence[str | Path] = DEFAULT_SOURCE_PATHS,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    if len(contract_sha256) != 64:
        raise ValueError("contract_sha256 must be a SHA-256 digest")
    head = _git_head(root)
    dirty, unexpected = _worktree_status(root)
    if unexpected:
        raise ValueError(f"unexpected worktree changes: {unexpected[0]}")
    files: dict[str, dict[str, Any]] = {}
    for raw in required_paths:
        relative, path = _relative(raw, root)
        if relative in files:
            raise ValueError(f"duplicate v4.4 source path: {relative}")
        _tracked_clean(root, relative, head)
        files[relative] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    payload: dict[str, Any] = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "run_id": run_id,
        "git_commit": head,
        "git_dirty": dirty,
        "allowlisted_untracked": list(ALLOWLISTED_UNTRACKED),
        "contract_sha256": contract_sha256,
        "evaluation_year_accessed": False,
        "files": dict(sorted(files.items())),
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def validate_source_manifest_payload(payload: Mapping[str, Any], *, repo_root: str | Path, expected_run_id: str | None = None, expected_contract_sha256: str | None = None) -> dict[str, Any]:
    if payload.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise ValueError("source manifest schema mismatch")
    if expected_run_id is not None and payload.get("run_id") != expected_run_id:
        raise ValueError("source manifest run_id mismatch")
    if expected_contract_sha256 is not None and payload.get("contract_sha256") != expected_contract_sha256:
        raise ValueError("source manifest contract mismatch")
    if payload.get("evaluation_year_accessed") is not False:
        raise ValueError("source manifest evaluation access flag is invalid")
    identity = payload.get("identity_sha256")
    unsigned = dict(payload); unsigned.pop("identity_sha256", None)
    if identity != canonical_sha256(unsigned):
        raise ValueError("source manifest identity hash mismatch")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("source manifest files are missing")
    root = Path(repo_root).resolve()
    head = _git_head(root)
    if payload.get("git_commit") != head:
        raise ValueError("source manifest Git commit does not match HEAD")
    for raw, entry in files.items():
        relative, path = _relative(str(raw), root)
        if relative != str(raw) or not isinstance(entry, Mapping):
            raise ValueError("source manifest file entry is invalid")
        if entry.get("sha256") != _sha256_file(path) or int(entry.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError(f"source hash mismatch: {relative}")
        _tracked_clean(root, relative, head)
    return dict(payload)


def write_source_manifest(path: str | Path, manifest: Mapping[str, Any]) -> str:
    return write_json_once(path, manifest)


__all__ = [
    "ALLOWLISTED_UNTRACKED", "DEFAULT_SOURCE_PATHS", "SOURCE_MANIFEST_SCHEMA",
    "build_source_manifest", "validate_source_manifest_payload", "write_source_manifest",
]
