from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.proxy_contract import contract_sha256, load_contract
from src.scheduling.proxy_model import (
    ProxyModelConfig,
    SchedulingProxy,
    load_model_checkpoint,
    save_model_checkpoint,
)


CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"
BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")


def test_proxy_forward_shape_bounds_and_eval_determinism():
    contract = load_contract(CONTRACT, BENCHMARK)
    model = SchedulingProxy(ProxyModelConfig.from_contract(contract))
    model.eval()
    inputs = torch.zeros(3, 4, 10)
    first = model(inputs)
    second = model(inputs)
    assert tuple(first.shape) == (3, 4, 21)
    assert torch.isfinite(first).all()
    assert torch.all((first >= 0) & (first <= 1))
    assert torch.equal(first, second)
    assert model.parameter_count > 10000


def test_proxy_checkpoint_round_trip(tmp_path):
    model = SchedulingProxy()
    model.eval()
    path = tmp_path / "best_model.pt"
    save_model_checkpoint(path, model, epoch=2, validation_loss=0.5, metadata={"selection_split": "validation"})
    restored, payload = load_model_checkpoint(path)
    assert payload["epoch"] == 2
    x = torch.randn(2, 4, 10)
    assert torch.allclose(model(x), restored(x))


def test_checkpoint_expected_contract_hash_is_validated(tmp_path):
    contract = load_contract(CONTRACT, BENCHMARK)
    model = SchedulingProxy(ProxyModelConfig.from_contract(contract))
    path = tmp_path / "best_model.pt"
    save_model_checkpoint(path, model, metadata={"contract_sha256": contract_sha256(contract)})
    load_model_checkpoint(path, expected_contract=contract)
    wrong_contract = replace(contract, benchmark_sha256="0" * 64)
    with __import__("pytest").raises(ValueError, match="contract SHA-256"):
        load_model_checkpoint(path, expected_contract=wrong_contract)
