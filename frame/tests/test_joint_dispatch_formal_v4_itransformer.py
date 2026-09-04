from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess

import pytest

from src.joint_dispatch import formal_v4_itransformer as itr
from src.joint_dispatch.formal_v4_itransformer import OfficialITransformerAdapter, validate_itransformer_receipt, verify_itransformer_source_files


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_adapter_requires_verified_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="source receipt"):
        OfficialITransformerAdapter(source_root=tmp_path, receipt_path=tmp_path / "missing.json")


def test_receipt_rejects_local_inverted_token_forecaster():
    with pytest.raises(ValueError, match="official THUML"):
        validate_itransformer_receipt({
            "schema_version": "formal-v4.1-itransformer-source-v1",
            "repository": "local", "backbone_class": "src.joint_dispatch.external_baselines.InvertedTokenForecaster",
            "reproduction_level": "official_backbone_adaptation", "commit": itr.OFFICIAL_COMMIT, "verified": True,
            "source_root": "frame/third_party/iTransformer_source", "license_file": "LICENSE", "license_sha256": "a" * 64,
        })


def test_official_receipt_identity_fields():
    payload = {
        "schema_version": "formal-v4.1-itransformer-source-v1",
        "repository": "https://github.com/thuml/iTransformer",
        "backbone_class": "model.iTransformer.Model",
        "reproduction_level": "official_backbone_adaptation",
        "commit": itr.OFFICIAL_COMMIT, "verified": True,
        "source_root": "frame/third_party/iTransformer_source",
        "license_file": "LICENSE", "license_sha256": "a" * 64,
    }
    validate_itransformer_receipt(payload)


def test_source_verification_binds_commit_files_and_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "itransformer"
    (source / "model").mkdir(parents=True)
    (source / "model" / "iTransformer.py").write_text("class Model: pass\n", encoding="utf-8")
    (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Gate 0 Test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(itr, "OFFICIAL_COMMIT", commit)
    receipt = {
        "schema_version": "formal-v4.1-itransformer-source-v1",
        "repository": itr.OFFICIAL_REPOSITORY,
        "backbone_class": itr.OFFICIAL_BACKBONE_CLASS,
        "reproduction_level": "official_backbone_adaptation",
        "commit": commit,
        "verified": True,
        "source_root": "frame/third_party/iTransformer_source",
        "imported_file_hashes": {"model/iTransformer.py": _hash(source / "model" / "iTransformer.py")},
        "license_file": "LICENSE",
        "license_sha256": _hash(source / "LICENSE"),
    }
    verify_itransformer_source_files(source, receipt, require_license=True)
    (source / "model" / "iTransformer.py").write_text("class Model: changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_itransformer_source_files(source, receipt, require_license=True)


def test_source_verification_rejects_changed_head_and_missing_license(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "itransformer"
    source.mkdir()
    (source / "LICENSE").write_text("MIT\n", encoding="utf-8")
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "test@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Gate 0 Test"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "fixture"], check=True, capture_output=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(itr, "OFFICIAL_COMMIT", commit)
    receipt = {
        "schema_version": "formal-v4.1-itransformer-source-v1", "repository": itr.OFFICIAL_REPOSITORY,
        "backbone_class": itr.OFFICIAL_BACKBONE_CLASS, "reproduction_level": "official_backbone_adaptation",
        "commit": commit, "verified": True, "source_root": "frame/third_party/iTransformer_source",
        "imported_file_hashes": {}, "license_file": "LICENSE",
        "license_sha256": _hash(source / "LICENSE"),
    }
    with pytest.raises(ValueError, match="imported source hashes"):
        verify_itransformer_source_files(source, receipt, require_license=True)
    (source / "extra.py").write_text("x=1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-m", "changed"], check=True, capture_output=True)
    receipt["imported_file_hashes"] = {"LICENSE": _hash(source / "LICENSE")}
    with pytest.raises(ValueError, match="Git HEAD"):
        verify_itransformer_source_files(source, receipt, require_license=True)
