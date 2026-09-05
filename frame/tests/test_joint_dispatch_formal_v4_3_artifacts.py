from __future__ import annotations

import pytest

from src.joint_dispatch.formal_v4_3_artifacts import (
    assert_shared_parent_v43,
    validate_v43_checkpoint,
    write_once_json,
)


def _lineage() -> dict[str, object]:
    return {
        "contract_version": "formal-v4.3",
        "contract_sha256": "a" * 64,
        "normalization_sha256": "b" * 64,
        "thermal_magnitude_receipt_sha256": "c" * 64,
        "parent_checkpoint_sha256": "d" * 64,
        "teacher_sha256": "e" * 64,
        "method_id": "RSC-PF",
        "seed": 2026,
    }


def test_v42_checkpoint_cannot_be_claimed_by_v43() -> None:
    lineage = _lineage()
    lineage["contract_version"] = "formal-v4.2"
    with pytest.raises(ValueError, match="formal-v4.3"):
        validate_v43_checkpoint(lineage)


def test_joint_and_decoupled_share_byte_identical_parents() -> None:
    joint = {"stage_p_parent_sha256": "a", "stage_s_parent_sha256": "b", "training_seed": 2026}
    decoupled = dict(joint)
    assert_shared_parent_v43(joint, decoupled)
    decoupled["stage_s_parent_sha256"] = "different"
    with pytest.raises(ValueError, match="stage_s_parent_sha256"):
        assert_shared_parent_v43(joint, decoupled)


def test_write_once_json_rejects_overwrite(tmp_path) -> None:
    path = write_once_json(tmp_path / "receipt.json", {"ok": True})
    assert path.exists()
    with pytest.raises(FileExistsError):
        write_once_json(path, {"ok": False})
