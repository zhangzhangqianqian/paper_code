from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_2_artifacts import (
    ArtifactStore,
    LineageError,
    MethodSeedKey,
    create_v42_run_root,
    write_once_json,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _complete_payload(contract_sha256: str = HASH_A) -> dict[str, object]:
    return {
        "contract_sha256": contract_sha256,
        "source_manifest_sha256": HASH_B,
        "data_sha256": HASH_C,
        "checkpoint_sha256": "d" * 64,
        "runtime_seconds": 1.25,
        "status": "complete",
    }


def test_write_once_is_idempotent_only_for_identical_payload(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    first = write_once_json(path, {"status": "pass"})
    second = write_once_json(path, {"status": "pass"})
    assert first == second
    with pytest.raises(FileExistsError, match="immutable artifact differs"):
        write_once_json(path, {"status": "fail"})


def test_run_root_rejects_contract_change(tmp_path: Path) -> None:
    root = create_v42_run_root(tmp_path, "formal_v4_2_test", HASH_A)
    assert (root / "protocol" / "RUN.json").is_file()
    assert create_v42_run_root(tmp_path, "formal_v4_2_test", HASH_A) == root
    with pytest.raises(FileExistsError, match="immutable artifact differs"):
        create_v42_run_root(tmp_path, "formal_v4_2_test", HASH_B)


def test_resume_accepts_only_hash_identical_completed_rows(tmp_path: Path) -> None:
    row = MethodSeedKey("RSC-PF", 2026)
    store = ArtifactStore(tmp_path, contract_sha256=HASH_A)
    store.complete(row, _complete_payload())
    assert store.completed_rows() == {row}
    with pytest.raises(LineageError, match="contract_sha256"):
        ArtifactStore(tmp_path, contract_sha256=HASH_B).completed_rows()


def test_failure_receipt_does_not_count_as_complete(tmp_path: Path) -> None:
    row = MethodSeedKey("Differentiable-LP", 2028)
    store = ArtifactStore(tmp_path, contract_sha256=HASH_A)
    store.failed(
        row,
        RuntimeError("solver failed"),
        {
            "source_manifest_sha256": HASH_B,
            "data_sha256": HASH_C,
        },
    )
    assert store.completed_rows() == set()
    payload = json.loads(store.failure_path(row).read_text(encoding="utf-8"))
    assert payload["exception_class"] == "RuntimeError"
    assert payload["status"] == "failed"


def test_complete_receipt_requires_full_lineage(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, contract_sha256=HASH_A)
    with pytest.raises(LineageError, match="checkpoint_sha256"):
        store.complete(MethodSeedKey("RSC-PF", 2026), {"status": "complete"})


def test_failure_receipt_also_requires_source_and_data_hashes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, contract_sha256=HASH_A)
    with pytest.raises(LineageError, match="source_manifest_sha256"):
        store.failed(MethodSeedKey("RSC-PF", 2026), RuntimeError("x"), {})
