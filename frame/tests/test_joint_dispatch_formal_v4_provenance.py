from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from src.joint_dispatch.formal_protocol_v4 import SCHEMA_VERSION_V41, load_formal_v4_spec
from src.joint_dispatch.formal_v4_provenance import (
    FormalV4SourceManifest,
    build_source_manifest,
    load_invalid_run_registry,
    validate_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG = ROOT / "configs" / "joint_forecast_dispatch_formal_v4.json"
V41_CONFIG = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_1.json"
REGISTRY = ROOT / "configs" / "joint_dispatch_invalid_runs_v4.json"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "formal-v4-test")
    _git(repo, "config", "user.email", "formal-v4-test@example.invalid")
    source = repo / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo, source, _git(repo, "rev-parse", "HEAD")


def test_manifest_rejects_changed_formal_source(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    manifest = build_source_manifest(tmp_path, [source], "test-commit")
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_source_manifest(manifest, tmp_path)


def test_manifest_rejects_missing_or_escaped_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        build_source_manifest(tmp_path, ["missing.py"], "test-commit")
    outside = tmp_path.parent / "outside-formal-v4-source.py"
    outside.write_text("outside\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="escapes repository root"):
            build_source_manifest(tmp_path, [outside], "test-commit")
    finally:
        outside.unlink()


def test_manifest_rejects_untracked_and_dirty_git_sources(tmp_path: Path) -> None:
    repo, source, commit = _git_repo(tmp_path)
    build_source_manifest(repo, [source], commit)
    source.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source is dirty"):
        build_source_manifest(repo, [source], commit)
    untracked = repo / "untracked.py"
    untracked.write_text("untracked\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source is untracked"):
        build_source_manifest(repo, [untracked], commit)


def test_manifest_is_sorted_unique_and_hash_valid(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("a\n", encoding="utf-8")
    b.write_text("b\n", encoding="utf-8")
    manifest = build_source_manifest(tmp_path, [b, a], "test-commit")
    assert isinstance(manifest, FormalV4SourceManifest)
    assert [entry.path for entry in manifest.entries] == ["a.py", "b.py"]
    assert validate_source_manifest(manifest.to_payload(), tmp_path).identity_sha256 == manifest.identity_sha256


def test_invalid_run_registry_contains_only_excluded_trials() -> None:
    payload = load_invalid_run_registry(REGISTRY)
    assert payload["status"] == "frozen"
    assert all(item["allowed_for_formal_results"] is False for item in payload["runs"])


def test_v41_config_is_additive_and_does_not_modify_legacy() -> None:
    legacy_before = LEGACY_CONFIG.read_bytes()
    spec = load_formal_v4_spec(V41_CONFIG)
    assert spec.schema_version == SCHEMA_VERSION_V41
    assert spec.source_manifest_required is True
    assert spec.invalid_run_registry == ROOT / "configs" / "joint_dispatch_invalid_runs_v4.json"
    assert spec.source_closure_file == ROOT / "configs" / "formal_v4_source_closure_v4_1.txt"
    assert LEGACY_CONFIG.read_bytes() == legacy_before
