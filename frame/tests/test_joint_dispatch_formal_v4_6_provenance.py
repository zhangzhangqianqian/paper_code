from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_6_provenance import (
    build_source_manifest_v46,
    validate_source_manifest_v46,
)


def make_git_repo(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(root), "config", "user.name", "v46-test"], check=True)
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-qm", "fixture"], check=True)
    return root


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def test_v46_manifest_binds_contract_head_and_every_declared_file(tmp_path: Path) -> None:
    repo = make_git_repo(tmp_path / "repo", {"a.py": "x = 1\n", "b.json": "{}\n"})
    payload = build_source_manifest_v46(
        repo_root=repo, run_id="v46-test", contract_sha256="a" * 64,
        source_paths=("a.py", "b.json"),
    )
    checked = validate_source_manifest_v46(payload, repo, "v46-test", "a" * 64)
    assert checked["entry_count"] == 2
    assert checked["git_commit"] == git_head(repo)


def test_v46_manifest_rejects_dirty_missing_or_post_hash_source(tmp_path: Path) -> None:
    repo = make_git_repo(tmp_path / "repo", {"a.py": "x = 1\n"})
    payload = build_source_manifest_v46(
        repo_root=repo, run_id="v46-test", contract_sha256="a" * 64,
        source_paths=("a.py",),
    )
    (repo / "a.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirty|changed|hash"):
        validate_source_manifest_v46(payload, repo, "v46-test", "a" * 64)
