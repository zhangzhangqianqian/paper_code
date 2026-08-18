from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.proxy_contract import load_contract
from src.scheduling.proxy_dataset import build_labeled_proxy_split, ProxyNormalizationStats
from src.scheduling.proxy_physics import (
    balance_residuals,
    conversion_residuals,
    feasible_proxy_loss,
    physics_terms,
    physics_aware_loss,
    soc_residuals,
)
from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios, load_benchmark


ROOT_BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"
V2_CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v2.json"


def test_exact_lp_labels_have_zero_physics_residuals_and_perturbation_is_positive():
    contract = load_contract(CONTRACT, ROOT_BENCHMARK)
    split = build_labeled_proxy_split(generate_synthetic_scenarios(ROOT_BENCHMARK, "train", 2026, 2), ROOT_BENCHMARK, contract)
    params = load_benchmark(ROOT_BENCHMARK)["values"]
    dispatch = torch.from_numpy(split.dispatch)
    features = torch.from_numpy(split.inputs)
    assert float(balance_residuals(dispatch, features).abs().max()) < 1e-6
    assert float(conversion_residuals(dispatch, params).abs().max()) < 1e-6
    assert float(soc_residuals(dispatch, features, params)["state"].abs().max()) < 1e-6
    perturbed = dispatch.clone()
    perturbed[..., 0] += 1.0
    assert float(balance_residuals(perturbed, features).abs().max()) > 0.1
    terms = physics_terms(dispatch, features, params)
    assert float(terms["balance"].detach()) < 1e-12
    assert float(terms["conversion"].detach()) < 1e-12
    assert float(terms["soc"].detach()) < 1e-12
    assert torch.isfinite(terms["balance_residual_scaled"]).all()
    assert torch.isfinite(terms["conversion_residual_scaled"]).all()
    assert torch.isfinite(terms["soc_state_residual_scaled"]).all()
    perturbed_terms = physics_terms(perturbed, features, params)
    assert float(perturbed_terms["balance"]) > 0.0


def test_physics_aware_loss_is_finite():
    contract = load_contract(CONTRACT, ROOT_BENCHMARK)
    split = build_labeled_proxy_split(generate_synthetic_scenarios(ROOT_BENCHMARK, "train", 2026, 2), ROOT_BENCHMARK, contract)
    stats = ProxyNormalizationStats.fit(split, benchmark=ROOT_BENCHMARK)
    prediction = torch.from_numpy(stats.normalize_dispatch(split.dispatch).astype(np.float32)).requires_grad_(True)
    target = prediction.detach().clone()
    total, components = physics_aware_loss(
        prediction, target, torch.from_numpy(split.inputs.astype(np.float32)), load_benchmark(ROOT_BENCHMARK)["values"],
        teacher_cost=torch.from_numpy(split.teacher_cost.astype(np.float32)), teacher_carbon=torch.from_numpy(split.teacher_carbon.astype(np.float32)),
        gas_prior_mask=torch.from_numpy(split.gas_prior_mask.astype(np.float32)), label_scale=torch.from_numpy(stats.label_scale.astype(np.float32)),
        loss_weights=contract.loss_weights,
    )
    assert torch.isfinite(total)
    assert all(torch.isfinite(value).all() for value in components.values() if torch.is_tensor(value))
    total.backward()


def test_feasible_v2_loss_is_zero_for_teacher_and_penalizes_coordinates():
    contract = load_contract(V2_CONTRACT, ROOT_BENCHMARK)
    split = build_labeled_proxy_split(generate_synthetic_scenarios(ROOT_BENCHMARK, "train", 2026, 2), ROOT_BENCHMARK, contract)
    params = load_benchmark(ROOT_BENCHMARK)["values"]
    dispatch = torch.from_numpy(split.dispatch)
    features = torch.from_numpy(split.inputs)
    total, components = feasible_proxy_loss(
        dispatch,
        dispatch,
        features,
        params,
        teacher_objective=torch.from_numpy(split.teacher_objective),
        teacher_carbon=torch.from_numpy(split.teacher_carbon),
        weights=contract.loss_weights,
    )
    assert float(total) <= 1.0e-12
    assert set(("coordinate", "renewable_split", "objective", "carbon", "slack", "total")) <= set(components)
    perturbed = dispatch.clone()
    perturbed[..., 11] += 1.0
    perturbed_total, perturbed_components = feasible_proxy_loss(
        perturbed, dispatch, features, params,
        teacher_objective=torch.from_numpy(split.teacher_objective),
        teacher_carbon=torch.from_numpy(split.teacher_carbon),
    )
    assert float(perturbed_total) > float(total)
    assert float(perturbed_components["coordinate"]) > float(components["coordinate"])
