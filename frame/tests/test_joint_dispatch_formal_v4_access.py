from __future__ import annotations

import hashlib
import zipfile

import pytest

from src.joint_dispatch.formal_v4_access import FormalV4AccessController, scan_runtime_access


def test_evaluation_access_is_denied_and_logged(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("x", encoding="utf-8")
    controller = FormalV4AccessController()
    with pytest.raises(PermissionError):
        controller.request(path, split="evaluation", purpose="gate0", caller="test", years=(2020,))
    assert controller.receipt.blocked_count == 1
    assert not controller.receipt.test_set_accessed


def test_train_access_and_archive_member_hashes(tmp_path):
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("2018.csv", "value")
    controller = FormalV4AccessController()
    data = controller.request_archive_member(archive, "2018.csv", split="train", purpose="base", caller="loader", years=(2018,))
    assert data == b"value"
    event = controller.receipt.events[-1]
    assert event.container_sha256 and event.member_sha256 and event.member == "2018.csv"


def test_static_access_scan_respects_loader_allowlist(tmp_path):
    source = tmp_path / "bad.py"
    source.write_text("import numpy as np\nnp.load('2020.npz')\n", encoding="utf-8")
    result = scan_runtime_access([source])
    assert result["status"] == "fail"
    assert result["findings"]


def test_guard_archive_member_denies_before_member_read(tmp_path):
    archive = tmp_path / "evaluation.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("2020.xlsx", "sealed")
    controller = FormalV4AccessController()
    with pytest.raises(PermissionError):
        controller.guard_archive_member(
            archive,
            "2020.xlsx",
            split="evaluation",
            purpose="gate0",
            caller="loader",
            years=(2020,),
        )
    event = controller.receipt.events[-1]
    assert event.decision == "deny"
    assert event.member == "2020.xlsx"
    assert event.member_sha256 == ""


def test_archive_receipt_contains_only_actual_allowed_member_reads(tmp_path):
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("2018.xlsx", "value")
    controller = FormalV4AccessController()
    controller.guard_archive_member(
        archive,
        "2018.xlsx",
        split="train",
        purpose="base",
        caller="loader",
        years=(2018,),
    )
    controller.record_archive_event(
        archive,
        "2018.xlsx",
        split="train",
        purpose="base",
        caller="loader",
        years=(2018,),
        container_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        member_sha256=hashlib.sha256(b"value").hexdigest(),
    )
    payload = controller.build_archive_access_receipt()
    assert payload["schema_version"] == "formal-v4.1-archive-access-v1"
    assert payload["events"][0]["member_name"] == "2018.xlsx"
