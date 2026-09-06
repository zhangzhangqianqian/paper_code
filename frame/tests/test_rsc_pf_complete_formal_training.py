from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract, MethodSeedKey
from src.joint_dispatch.complete_formal_training import (
    TRAINABLE_METHOD_IDS,
    resume_row,
    train_one_step,
    train_row,
)


FRAME_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = FRAME_ROOT / "configs" / "rsc_pf_complete_formal_v1.json"


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)


def _tiny_data(method_parent: bool = True):
    payload = {
        "model": _TinyModel(),
        "train_batches": [
            {"x": torch.tensor([[1.0, 2.0]]), "y": torch.tensor([[0.5]])},
            {"x": torch.tensor([[2.0, 1.0]]), "y": torch.tensor([[0.25]])},
        ],
        "years": [2015, 2016],
        "max_epochs": 1,
        "source_sha256": "a" * 64,
        "preprocessing": "train-only-normalization-v1",
    }
    if method_parent:
        payload["parent_checkpoint_sha256"] = "b" * 64
    return payload


@pytest.fixture(scope="module")
def contract():
    return CompleteFormalContract.from_path(CONTRACT_PATH)


def test_joint_and_decoupled_have_different_forecaster_gradient_contract():
    joint = train_one_step("RSC-PF", _tiny_data(), seed=2026)
    decoupled = train_one_step("Decoupled-RSC-PF", _tiny_data(), seed=2026)
    assert joint.dispatch_to_forecaster_gradient_norm > 0.0
    assert decoupled.dispatch_to_forecaster_gradient_norm <= 1e-12
    assert joint.forecast_loss_applicable is True
    assert decoupled.forecast_loss_applicable is True


def test_direct_policy_and_pto_gradient_contracts():
    direct = train_one_step("Direct-Policy", _tiny_data(), seed=2028)
    pto = train_one_step("Scheme2R-PTO", _tiny_data(), seed=2028)
    assert direct.forecast_loss_applicable is False
    assert direct.dispatch_to_forecaster_gradient_norm > 0.0
    assert pto.forecast_loss_applicable is True
    assert pto.dispatch_to_forecaster_gradient_norm <= 1e-12
    assert pto.seed == 2028


def test_resume_rejects_contract_or_data_hash_change(tmp_path, contract):
    artifact = train_row(contract, MethodSeedKey("RSC-PF", 2026), _tiny_data(), tmp_path)
    assert artifact.complete is True
    assert artifact.training_exposures == 2
    assert artifact.parent_checkpoint_sha256 == "b" * 64
    with pytest.raises(ValueError, match="hash"):
        resume_row(tmp_path, "wrong-contract-hash")
    resumed = resume_row(tmp_path, contract.contract_sha256)
    assert resumed.checkpoint_sha256 == artifact.checkpoint_sha256
    with pytest.raises(ValueError, match="refusing to overwrite"):
        train_row(contract, MethodSeedKey("RSC-PF", 2026), _tiny_data(), tmp_path)


def test_parent_checkpoint_hash_is_shared_by_joint_pair(tmp_path, contract):
    parent = "c" * 64
    first = _tiny_data()
    second = _tiny_data()
    first["parent_checkpoint_sha256"] = parent
    second["parent_checkpoint_sha256"] = parent
    joint = train_row(contract, MethodSeedKey("RSC-PF", 2026), first, tmp_path / "joint")
    decoupled = train_row(contract, MethodSeedKey("Decoupled-RSC-PF", 2026), second, tmp_path / "decoupled")
    assert joint.parent_checkpoint_sha256 == decoupled.parent_checkpoint_sha256 == parent


def test_training_rejects_non_train_year_or_future_access(contract, tmp_path):
    broken = _tiny_data()
    broken["years"] = [2020]
    with pytest.raises(PermissionError, match="non-training year"):
        train_row(contract, MethodSeedKey("RSC-PF", 2026), broken, tmp_path / "year")
    broken = _tiny_data()
    broken["test_set_accessed"] = True
    with pytest.raises(PermissionError, match="test-set"):
        train_row(contract, MethodSeedKey("RSC-PF", 2026), broken, tmp_path / "test")


def test_all_stochastic_methods_have_frozen_seed_contract(contract):
    for method_id in TRAINABLE_METHOD_IDS:
        assert contract.method(method_id).stochastic is True

