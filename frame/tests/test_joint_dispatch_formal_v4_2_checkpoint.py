from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.joint_dispatch.formal_v4_2_checkpoint import (
    clone_stage_s_branches,
    load_training_checkpoint,
    optimizer_step_value,
    save_training_checkpoint,
)
from src.joint_dispatch.formal_v4_2_artifacts import LineageError, sha256_file


LINEAGE = {
    "contract_sha256": "a" * 64,
    "source_manifest_sha256": "b" * 64,
    "data_sha256": "c" * 64,
}


def _train_one_batch(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = (model(torch.ones(4, 2)) ** 2).mean()
    loss.backward()
    optimizer.step()


def test_optimizer_step_increases_across_two_batches() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    _train_one_batch(model, optimizer)
    first = optimizer_step_value(optimizer)
    _train_one_batch(model, optimizer)
    second = optimizer_step_value(optimizer)
    assert first == 1 and second == 2


def test_checkpoint_round_trip_restores_model_optimizer_and_rng(tmp_path: Path) -> None:
    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    _train_one_batch(model, optimizer)
    path = tmp_path / "stage_s.pt"
    receipt = save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        early_stopping={"best": 1.0, "bad_epochs": 0},
        lineage=LINEAGE,
    )
    expected_next = (torch.rand(1).item(), np.random.rand(), random.random())
    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0e-3)
    load_training_checkpoint(path, model=restored_model, optimizer=restored_optimizer, expected_lineage=LINEAGE)
    actual_next = (torch.rand(1).item(), np.random.rand(), random.random())
    assert actual_next == pytest.approx(expected_next)
    assert receipt.model_sha256 == sha256_file(path)
    assert optimizer_step_value(restored_optimizer) == 1


def test_stage_s_clones_are_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "stage_s.pt"
    source.write_bytes(b"checkpoint-bytes")
    joint, decoupled = clone_stage_s_branches(source, tmp_path / "joint.pt", tmp_path / "decoupled.pt")
    assert sha256_file(joint) == sha256_file(decoupled) == sha256_file(source)


def test_stage_s_clone_rejects_changed_existing_destination(tmp_path: Path) -> None:
    source = tmp_path / "stage_s.pt"
    source.write_bytes(b"checkpoint-bytes")
    destination = tmp_path / "joint.pt"
    destination.write_bytes(b"different")
    with pytest.raises(FileExistsError, match="clone differs"):
        clone_stage_s_branches(source, destination, tmp_path / "decoupled.pt")


def test_checkpoint_requires_contract_data_and_source_lineage(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    with pytest.raises(LineageError, match="data_sha256"):
        save_training_checkpoint(
            tmp_path / "bad.pt",
            model=model,
            optimizer=optimizer,
            epoch=0,
            early_stopping={},
            lineage={"contract_sha256": "a" * 64, "source_manifest_sha256": "b" * 64},
        )
