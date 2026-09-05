"""Version-bound provenance for the formal-v4.6 execution chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .formal_v4_4_artifacts import canonical_sha256, write_json_once


SOURCE_MANIFEST_SCHEMA_V46 = "formal-v4.6-source-manifest-v1"
V46_SOURCE_PATHS = (
    "configs/joint_forecast_dispatch_formal_v4_6.json",
    "scripts/build_rsc_pf_formal_v4_6_source_manifest.py",
    "scripts/run_rsc_pf_formal_v4_6_diagnostic.py",
    "scripts/run_rsc_pf_formal_v4_6_pilot.py",
    "src/joint_dispatch/formal_v4_6_artifacts.py",
    "src/joint_dispatch/formal_v4_6_contract.py",
    "src/joint_dispatch/formal_v4_6_loss.py",
    "src/joint_dispatch/formal_v4_6_metrics.py",
    "src/joint_dispatch/formal_v4_6_model.py",
    "src/joint_dispatch/formal_v4_6_pilot_executor.py",
    "src/joint_dispatch/formal_v4_6_pilot_gate.py",
    "src/joint_dispatch/formal_v4_6_provenance.py",
    "src/joint_dispatch/formal_v4_6_risk.py",
    "src/joint_dispatch/formal_v4_6_rollout.py",
    "src/joint_dispatch/formal_v4_6_training.py",
    "src/joint_dispatch/formal_v4_5_artifacts.py",
    "src/joint_dispatch/formal_v4_5_contract.py",
    "src/joint_dispatch/formal_v4_5_loss.py",
    "src/joint_dispatch/formal_v4_5_pilot_data.py",
    "src/joint_dispatch/formal_v4_5_pilot_executor.py",
    "src/joint_dispatch/formal_v4_5_training.py",
    "src/joint_dispatch/formal_v4_4_artifacts.py",
    "src/joint_dispatch/formal_v4_4_contract.py",
    "src/joint_dispatch/formal_v4_4_loss.py",
    "src/joint_dispatch/formal_v4_4_metrics.py",
    "src/joint_dispatch/formal_v4_4_model.py",
    "src/joint_dispatch/formal_v4_4_pilot_data.py",
    "src/joint_dispatch/formal_v4_4_pilot_executor.py",
    "src/joint_dispatch/formal_v4_4_pilot_materializer.py",
    "src/joint_dispatch/formal_v4_4_provenance.py",
    "src/joint_dispatch/formal_v4_4_regime.py",
    "src/joint_dispatch/formal_v4_4_rollout.py",
    "src/joint_dispatch/formal_v4_4_teacher.py",
    "src/joint_dispatch/formal_v4_4_training.py",
    "src/joint_dispatch/formal_v4_2_data.py",
    "src/joint_dispatch/formal_v4_2_rollout.py",
    "src/joint_dispatch/formal_v4_data.py",
    "src/joint_dispatch/formal_v4_history.py",
    "src/joint_dispatch/formal_v4_models.py",
    "src/joint_dispatch/formal_v4_objective.py",
    "src/joint_dispatch/formal_v4_recourse.py",
    "src/scheduling/proxy_physics.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def _git_head(root: Path) -> str:
    result = _git(root, "rev-parse", "HEAD")
    if result.returncode != 0 or not result.stdout.strip():
        raise ValueError("repository HEAD is unavailable")
    return result.stdout.strip()


def _normal_paths(paths: Sequence[str]) -> tuple[str, ...]:
    result = tuple(sorted({Path(value).as_posix() for value in paths}))
    if not result or any(not value or value.startswith("../") or "/../" in value for value in result):
        raise ValueError("source manifest paths must be non-empty repository-relative paths")
    return result


def _entries(repo_root: Path, source_paths: Sequence[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for relative in _normal_paths(source_paths):
        path = (repo_root / relative).resolve()
        if path.parent != repo_root.resolve() and repo_root.resolve() not in path.parents:
            raise ValueError("source manifest path escapes repository")
        if not path.is_file():
            raise FileNotFoundError(path)
        tracked = _git(repo_root, "ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            raise ValueError(f"source manifest file is not tracked: {relative}")
        dirty = _git(repo_root, "status", "--porcelain", "--", relative)
        if dirty.stdout.strip():
            raise ValueError(f"source manifest file is dirty: {relative}")
        entries.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256_file(path)})
    return entries


def build_source_manifest_v46(
    repo_root: str | Path,
    run_id: str,
    contract_sha256: str,
    source_paths: Sequence[str] = V46_SOURCE_PATHS,
) -> Mapping[str, Any]:
    root = Path(repo_root).resolve()
    if len(str(contract_sha256)) != 64:
        raise ValueError("contract_sha256 must be a SHA-256 digest")
    entries = _entries(root, source_paths)
    payload: dict[str, Any] = {
        "schema": SOURCE_MANIFEST_SCHEMA_V46,
        "run_id": str(run_id),
        "contract_sha256": str(contract_sha256),
        "git_commit": _git_head(root),
        "source_paths": [entry["path"] for entry in entries],
        "entries": entries,
    }
    payload["entry_count"] = len(entries)
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def _validate_manifest_entries_against_git_v46(payload: Mapping[str, Any], repo_root: Path) -> Mapping[str, Any]:
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("source manifest entries are missing")
    source_paths = _normal_paths([str(value) for value in payload.get("source_paths", ())])
    actual_paths = tuple(str(row.get("path")) for row in entries if isinstance(row, Mapping))
    if actual_paths != source_paths:
        raise ValueError("source manifest entries are not sorted or complete")
    expected_identity = payload.get("identity_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "identity_sha256"}
    if expected_identity != canonical_sha256(unsigned):
        raise ValueError("source manifest identity hash mismatch")
    expected_head = _git_head(repo_root)
    if payload.get("git_commit") != expected_head:
        raise ValueError("source manifest Git commit does not match HEAD")
    checked = _entries(repo_root, source_paths)
    if checked != entries:
        raise ValueError("source manifest file hash or size mismatch")
    return dict(payload)


def validate_source_manifest_v46(
    payload: Mapping[str, Any],
    repo_root: str | Path,
    expected_run_id: str,
    expected_contract_sha256: str,
    expected_source_paths: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    if payload.get("schema") != SOURCE_MANIFEST_SCHEMA_V46:
        raise ValueError("formal-v4.6 source-manifest schema mismatch")
    if payload.get("run_id") != expected_run_id:
        raise ValueError("formal-v4.6 source-manifest run-id mismatch")
    if payload.get("contract_sha256") != expected_contract_sha256:
        raise ValueError("formal-v4.6 source-manifest contract mismatch")
    if expected_source_paths is not None and _normal_paths(expected_source_paths) != _normal_paths(payload.get("source_paths", ())):
        raise ValueError("formal-v4.6 source-manifest closure mismatch")
    return _validate_manifest_entries_against_git_v46(payload, Path(repo_root).resolve())


def write_source_manifest_v46(path: str | Path, payload: Mapping[str, Any]) -> str:
    return write_json_once(path, payload)


__all__ = [
    "SOURCE_MANIFEST_SCHEMA_V46", "V46_SOURCE_PATHS",
    "build_source_manifest_v46", "validate_source_manifest_v46",
    "write_source_manifest_v46",
]
