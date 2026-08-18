from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.proxy_contract import load_contract
from src.scheduling.proxy_dataset import build_labeled_proxy_split
from src.scheduling.proxy_decoder import (
    CONTROL_DIM,
    DECISION_GROUPS,
    decode_feasible_controls,
    decode_feasible_dispatch,
    recover_teacher_controls,
)
from src.scheduling.proxy_adapter import evaluate_raw_feasibility
from src.scheduling.proxy_physics import balance_residuals, conversion_residuals, soc_residuals
from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios, load_benchmark


BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v2.json"


def _features(batch: int = 3) -> torch.Tensor:
    result = torch.zeros(batch, 4, 10, dtype=torch.float64)
    result[..., :3] = torch.tensor([100.0, 150.0, 125.0])
    result[..., 4] = 250.0
    result[..., 5] = 175.0
    result[..., 6:9] = 1.0
    result[..., 9] = 0.5
    return result


def test_decoder_constructs_float64_feasible_dispatch_and_gradients():
    parameters = load_benchmark(BENCHMARK)["values"]
    controls = torch.full((3, CONTROL_DIM), 0.5, dtype=torch.float64, requires_grad=True)
    features = _features()
    dispatch = decode_feasible_controls(controls, features, parameters)
    assert dispatch.shape == (3, 4, 21)
    assert dispatch.dtype == torch.float64
    assert torch.isfinite(dispatch).all()
    assert float(dispatch.detach().min()) >= -1.0e-12
    assert float(balance_residuals(dispatch, features).detach().abs().max()) <= 1.0e-6
    assert float(conversion_residuals(dispatch, parameters).detach().abs().max()) <= 1.0e-6
    soc = soc_residuals(dispatch, features, parameters)
    assert float(soc["state"].detach().abs().max()) <= 1.0e-6
    assert float(soc["terminal"].detach().abs().max()) <= 1.0e-6
    assert float(torch.minimum(dispatch[..., 14], dispatch[..., 15]).detach().max()) <= 1.0e-12
    dispatch.sum().backward()
    assert controls.grad is not None and torch.isfinite(controls.grad).all()


def test_decoder_logit_temperature_and_control_layout_are_versioned():
    parameters = load_benchmark(BENCHMARK)["values"]
    features = _features(2)
    logits = torch.zeros(2, CONTROL_DIM)
    decoded = decode_feasible_dispatch(logits, features, parameters)
    assert decoded.shape == (2, 4, 21)
    assert dict(DECISION_GROUPS) == {
        "cooling": (0, 4), "chp": (4, 8), "soc": (8, 11), "renewable_pv": (11, 15),
    }
    with pytest.raises(ValueError, match="controls"):
        decode_feasible_controls(torch.zeros(2, 14), features, parameters)
    bad = features.clone()
    bad[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        decode_feasible_dispatch(logits, bad, parameters)


def test_teacher_controls_reconstruct_exact_lp_dispatch():
    contract = load_contract(CONTRACT, BENCHMARK)
    split = build_labeled_proxy_split(
        generate_synthetic_scenarios(BENCHMARK, "train", 2026, 4), BENCHMARK, contract,
    )
    parameters = load_benchmark(BENCHMARK)["values"]
    features = torch.from_numpy(split.inputs)
    teacher = torch.from_numpy(split.dispatch)
    controls = recover_teacher_controls(teacher, features, parameters)
    reconstructed = decode_feasible_controls(controls, features, parameters)
    assert controls.shape == (split.n_samples, CONTROL_DIM)
    assert float(controls.detach().min()) >= -1.0e-10 and float(controls.detach().max()) <= 1.0 + 1.0e-10
    assert float((reconstructed - teacher).detach().abs().max()) <= 1.0e-6


def _random_features(batch: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260818)
    result = _features(batch)
    result[..., :3] = torch.rand((batch, 4, 3), generator=generator, dtype=torch.float64) * torch.tensor([450.0, 700.0, 500.0])
    result[..., 4:6] = torch.rand((batch, 4, 2), generator=generator, dtype=torch.float64) * 500.0
    result[..., 6:9] = 0.5 + torch.rand((batch, 4, 3), generator=generator, dtype=torch.float64)
    soc = 0.1 + 0.8 * torch.rand((batch, 1, 1), generator=generator, dtype=torch.float64)
    result[..., 9] = soc[:, 0, 0][:, None]
    return result


def _assert_decoder_feasible(dispatch: torch.Tensor, features: torch.Tensor, parameters: dict[str, float]) -> None:
    report = evaluate_raw_feasibility(
        dispatch.detach().cpu().numpy(),
        features.detach().cpu().numpy(),
        parameters,
        tolerance=1.0e-6,
        include_charge_discharge_overlap=True,
    )
    assert np.isfinite(dispatch.detach().cpu().numpy()).all()
    assert np.all(np.asarray(report["feasible_mask"], dtype=bool))
    for name in ("balance_residual", "conversion_residual", "renewable_residual", "soc_residual", "ramp_residual", "capacity_residual", "charge_discharge_overlap"):
        assert float(np.max(np.asarray(report[name]))) <= 1.0e-6


def test_decoder_randomized_inputs_are_feasible_and_finite():
    parameters = load_benchmark(BENCHMARK)["values"]
    features = _random_features(32)
    controls = 0.02 + 0.96 * torch.rand((32, CONTROL_DIM), generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    dispatch = decode_feasible_controls(controls, features, parameters)
    _assert_decoder_feasible(dispatch, features, parameters)


def test_decoder_zero_renewable_and_capacity_edges_are_supported():
    parameters = load_benchmark(BENCHMARK)["values"]
    zero_renewable = _features(4)
    zero_renewable[..., 4:6] = 0.0
    dispatch = decode_feasible_dispatch(torch.zeros(4, CONTROL_DIM), zero_renewable, parameters)
    _assert_decoder_feasible(dispatch, zero_renewable, parameters)
    assert torch.equal(dispatch[..., 1], torch.zeros_like(dispatch[..., 1]))
    assert torch.equal(dispatch[..., 2], torch.zeros_like(dispatch[..., 2]))
    assert torch.equal(dispatch[..., 3], torch.zeros_like(dispatch[..., 3]))
    assert torch.equal(dispatch[..., 4], torch.zeros_like(dispatch[..., 4]))

    edge = _features(2)
    edge[..., 0] = parameters["grid_import_capacity"]
    edge[..., 1] = parameters["electric_chiller_capacity"] + parameters["absorption_chiller_capacity"]
    edge[..., 2] = parameters["gas_boiler_capacity"]
    edge[..., 4:6] = 0.0
    edge_dispatch = decode_feasible_controls(torch.full((2, CONTROL_DIM), 0.5), edge, parameters)
    _assert_decoder_feasible(edge_dispatch, edge, parameters)


def test_decoder_overloaded_demands_use_explicit_shortage_slacks():
    parameters = load_benchmark(BENCHMARK)["values"]
    overloaded = _features(3)
    overloaded[..., 0] = parameters["grid_import_capacity"] + parameters["chp_electric_capacity"] + 500.0
    overloaded[..., 1] = parameters["electric_chiller_capacity"] + parameters["absorption_chiller_capacity"] + 300.0
    overloaded[..., 2] = parameters["gas_boiler_capacity"] + parameters["chp_heat_capacity"] + 300.0
    overloaded[..., 4:6] = 0.0
    dispatch = decode_feasible_controls(torch.full((3, CONTROL_DIM), 0.5), overloaded, parameters)
    _assert_decoder_feasible(dispatch, overloaded, parameters)
    assert bool((dispatch[..., 17] > 0.0).any())
    assert bool((dispatch[..., 19] > 0.0).any())
    assert bool((dispatch[..., 20] >= 0.0).all())


def test_decoder_rejects_invalid_physical_parameters():
    parameters = load_benchmark(BENCHMARK)["values"]
    features = _features(1)
    controls = torch.full((1, CONTROL_DIM), 0.5, dtype=torch.float64)
    for field, value in (("grid_import_capacity", 0.0), ("chp_electric_efficiency", -0.1), ("bess_energy_capacity", float("nan"))):
        invalid = dict(parameters)
        invalid[field] = value
        with pytest.raises(ValueError, match="decoder parameter"):
            decode_feasible_controls(controls, features, invalid)
    missing = dict(parameters)
    missing.pop("electric_chiller_cop")
    with pytest.raises(ValueError, match="missing"):
        decode_feasible_controls(controls, features, missing)


def test_teacher_controls_reconstruct_validation_split_without_test_access():
    contract = load_contract(CONTRACT, BENCHMARK)
    split = build_labeled_proxy_split(
        generate_synthetic_scenarios(BENCHMARK, "validation", 2027, 4), BENCHMARK, contract,
    )
    parameters = load_benchmark(BENCHMARK)["values"]
    features = torch.from_numpy(split.inputs)
    teacher = torch.from_numpy(split.dispatch)
    controls = recover_teacher_controls(teacher, features, parameters)
    reconstructed = decode_feasible_controls(controls, features, parameters)
    assert split.split == "validation"
    assert float((reconstructed - teacher).detach().abs().max()) <= 1.0e-6


def test_decoder_has_usable_gradients_for_each_decision_group():
    parameters = load_benchmark(BENCHMARK)["values"]
    features = _features(4)
    controls = torch.full((4, CONTROL_DIM), 0.37, dtype=torch.float64, requires_grad=True)
    dispatch = decode_feasible_controls(controls, features, parameters)
    group_outputs = {
        "cooling": dispatch[..., 11],
        "chp": dispatch[..., 7],
        "soc": dispatch[..., 16],
        "renewable_pv": dispatch[..., 1],
    }
    for name, output in group_outputs.items():
        gradient = torch.autograd.grad(output.sum(), controls, retain_graph=True)[0]
        start, end = DECISION_GROUPS[name]
        assert torch.isfinite(gradient[:, start:end]).all()
        assert float(gradient[:, start:end].abs().sum().detach()) > 1.0e-10
