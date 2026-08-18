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
    FeasibleSchedulingProxy,
    FeasibleSchedulingProxyConfig,
    ProxyModelConfig,
    SchedulingProxy,
    build_proxy_model,
    load_model_checkpoint,
    save_model_checkpoint,
)


CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"
V2_CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v2.json"
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


def test_v2_model_emits_controls_and_decodes_physical_dispatch():
    contract = load_contract(V2_CONTRACT, BENCHMARK)
    model = build_proxy_model(contract)
    assert isinstance(model, FeasibleSchedulingProxy)
    model.eval()
    normalized = torch.zeros(2, 4, 10)
    logits = model.forward_logits(normalized)
    assert logits.shape == (2, 15)
    physical = torch.zeros(2, 4, 10, dtype=torch.float64)
    physical[..., :3] = torch.tensor([100.0, 100.0, 100.0])
    physical[..., 4:6] = 100.0
    physical[..., 6:9] = 1.0
    physical[..., 9] = 0.5
    parameters = {
        "grid_import_capacity": 1917.0, "chp_electric_capacity": 447.3,
        "chp_heat_capacity": 575.1, "gas_boiler_capacity": 1148.2812,
        "electric_chiller_capacity": 869.115, "absorption_chiller_capacity": 869.115,
        "bess_power_capacity": 255.6, "bess_energy_capacity": 1022.4,
        "chp_electric_efficiency": 0.35, "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
        "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9,
        "chp_ramp_fraction": 0.5,
    }
    dispatch = model.predict_dispatch(normalized, physical, parameters)
    assert dispatch.shape == (2, 4, 21)
    assert dispatch.dtype == torch.float64


def test_v2_checkpoint_round_trip_and_cross_version_rejection(tmp_path):
    contract = load_contract(V2_CONTRACT, BENCHMARK)
    model = build_proxy_model(contract)
    path = tmp_path / "v2.pt"
    save_model_checkpoint(path, model, metadata={"contract_sha256": contract_sha256(contract)})
    restored, payload = load_model_checkpoint(path, expected_contract=contract)
    assert isinstance(restored, FeasibleSchedulingProxy)
    assert restored(torch.zeros(2, 4, 10)).shape == (2, 15)
    v1_contract = load_contract(CONTRACT, BENCHMARK)
    with __import__("pytest").raises(ValueError, match="configuration|family|SHA-256"):
        load_model_checkpoint(path, expected_contract=v1_contract)


def test_v2_checkpoint_requires_complete_decoder_metadata(tmp_path):
    contract = load_contract(V2_CONTRACT, BENCHMARK)
    model = build_proxy_model(contract)
    path = tmp_path / "v2.pt"
    save_model_checkpoint(path, model, metadata={"contract_sha256": contract_sha256(contract)})
    original = torch.load(path, map_location="cpu", weights_only=False)
    for field, value in (("model_family", None), ("decoder_schema_version", "wrong"), ("control_temperature", 0.5)):
        tampered = dict(original)
        tampered["metadata"] = dict(original["metadata"])
        if value is None:
            tampered["metadata"].pop(field)
        else:
            tampered["metadata"][field] = value
        torch.save(tampered, path)
        try:
            with __import__("pytest").raises(ValueError, match="family|schema|temperature"):
                load_model_checkpoint(path)
        finally:
            torch.save(original, path)
