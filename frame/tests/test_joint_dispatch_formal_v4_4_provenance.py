from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_4_artifacts import sha256_file
from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract
from src.joint_dispatch.formal_v4_4_provenance import (
    SOURCE_MANIFEST_SCHEMA,
    build_source_manifest,
    validate_source_manifest_payload,
    write_source_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "joint_forecast_dispatch_formal_v4_4.json"


def test_v44_manifest_records_current_commit_and_executor_hash(tmp_path: Path) -> None:
    contract = load_formal_v4_4_contract(CONTRACT)
    manifest = build_source_manifest(
        repo_root=ROOT,
        run_id="formal_v4_4_provenance_test",
        contract_sha256=contract.contract_sha256,
    )
    assert manifest["schema"] == SOURCE_MANIFEST_SCHEMA
    assert manifest["git_commit"]
    assert manifest["files"]["src/joint_dispatch/formal_v4_4_provenance.py"]["sha256"]
    assert manifest["evaluation_year_accessed"] is False
    path = tmp_path / "SOURCE_MANIFEST.json"
    digest = write_source_manifest(path, manifest)
    assert digest == sha256_file(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert validate_source_manifest_payload(loaded, repo_root=ROOT, expected_run_id="formal_v4_4_provenance_test")["identity_sha256"] == manifest["identity_sha256"]


def test_v44_manifest_rejects_identity_mutation() -> None:
    contract = load_formal_v4_4_contract(CONTRACT)
    manifest = build_source_manifest(repo_root=ROOT, run_id="formal_v4_4_provenance_test", contract_sha256=contract.contract_sha256)
    manifest["git_commit"] = "0" * 40
    with pytest.raises(ValueError, match="identity|Git commit"):
        validate_source_manifest_payload(manifest, repo_root=ROOT)


def test_v44_manifest_rejects_unexpected_tracked_change(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = load_formal_v4_4_contract(CONTRACT)
    monkeypatch.setattr(
        "src.joint_dispatch.formal_v4_4_provenance._worktree_status",
        lambda _root: (True, [" M src/joint_dispatch/formal_v4_4_model.py"]),
    )
    with pytest.raises(ValueError, match="unexpected worktree"):
        build_source_manifest(repo_root=ROOT, run_id="formal_v4_4_provenance_test", contract_sha256=contract.contract_sha256)
