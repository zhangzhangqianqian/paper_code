from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.proxy_adapter import Scheme2RProxyAdapter, adapt_scheme2r_inputs, evaluate_raw_feasibility
from src.scheduling.proxy_contract import load_contract
from src.scheduling.proxy_dataset import ProxyNormalizationStats, build_labeled_proxy_split
from src.scheduling.proxy_evaluation import dispatch_metrics, measure_exact_lp_latency, measure_proxy_latency
from src.scheduling.proxy_model import SchedulingProxy
from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios, load_benchmark


BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"


def _stats():
    c = load_contract(CONTRACT, BENCHMARK)
    split = build_labeled_proxy_split(generate_synthetic_scenarios(BENCHMARK, "train", 2026, 2), BENCHMARK, c)
    return ProxyNormalizationStats.fit(split, benchmark=BENCHMARK)


def test_adapter_maps_order_clips_negative_and_supports_no_prior():
    stats = _stats()
    loads = np.zeros((2, 4, 4), dtype=float)
    loads[..., 0] = -1.0
    loads[..., 1:] = 2.0
    renewables = np.ones((2, 4, 2), dtype=float)
    renewables[0, 0, 0] = -3.0
    output = adapt_scheme2r_inputs(loads, renewables, stats, benchmark=BENCHMARK)
    assert output.features.shape == (2, 4, 10)
    assert output.clipped_negative_count == 9
    no_prior = adapt_scheme2r_inputs(loads, renewables, stats, benchmark=BENCHMARK, use_gas_prior=False)
    assert np.all(no_prior.features[..., 3] == 0)
    assert np.all(no_prior.gas_prior_mask == 0)


def test_adapter_applies_mixed_gas_prior_mask_before_feature_assembly():
    stats = _stats()
    loads = np.zeros((1, 4, 4), dtype=float)
    loads[..., 3] = 7.0
    mask = np.asarray([[1.0, 0.0, 1.0, 0.0]])
    output = adapt_scheme2r_inputs(
        loads, np.zeros((1, 4, 2)), stats, gas_prior_mask=mask,
    )
    assert np.array_equal(output.gas_prior_mask, mask)
    assert np.array_equal(output.features[0, :, 3], np.asarray([7.0, 0.0, 7.0, 0.0]))


def test_adapter_rejects_wrong_task_order_and_nonfinite_values():
    stats = _stats()
    with pytest.raises(ValueError):
        adapt_scheme2r_inputs(np.zeros((1, 4, 4)), np.zeros((1, 4, 2)), stats, task_order=("gas", "cooling", "heating", "electricity"))
    with pytest.raises(ValueError):
        bad = np.zeros((1, 4, 4)); bad[0, 0, 0] = np.nan
        adapt_scheme2r_inputs(bad, np.zeros((1, 4, 2)), stats)


def test_full_lp_feasibility_checks_renewable_bounds_and_ramping():
    contract = load_contract(CONTRACT, BENCHMARK)
    split = build_labeled_proxy_split(generate_synthetic_scenarios(BENCHMARK, "train", 2026, 1), BENCHMARK, contract)
    parameters = load_benchmark(BENCHMARK)["values"]
    exact = evaluate_raw_feasibility(split.dispatch, split.inputs, parameters)
    assert bool(exact["feasible_mask"][0])
    perturbed = split.dispatch.copy()
    perturbed[0, 0, 1] += 1.0  # pv_use: renewable split equality
    renewable = evaluate_raw_feasibility(perturbed, split.inputs, parameters)
    assert not renewable["feasible_mask"][0] and renewable["renewable_residual"][0] > 0.0
    perturbed = split.dispatch.copy()
    perturbed[0, 0, 0] = parameters["grid_import_capacity"] + 1.0
    bounded = evaluate_raw_feasibility(perturbed, split.inputs, parameters)
    assert not bounded["feasible_mask"][0] and bounded["bound_residual"][0] > 0.0
    perturbed = split.dispatch.copy()
    perturbed[0, 1, 7] = perturbed[0, 0, 7] + parameters["chp_ramp_fraction"] * parameters["chp_electric_capacity"] + 1.0
    ramp = evaluate_raw_feasibility(perturbed, split.inputs, parameters)
    assert not ramp["feasible_mask"][0] and ramp["ramp_residual"][0] > 0.0


def test_adapter_rejects_time_varying_prices_for_scalar_lp_interface():
    stats = _stats()
    prices = np.ones((1, 4, 3), dtype=float)
    prices[0, 1, 0] = 2.0
    with pytest.raises(ValueError, match="time-varying price"):
        adapt_scheme2r_inputs(np.ones((1, 4, 4)), np.ones((1, 4, 2)), stats, prices=prices)


def test_adapter_rejects_invalid_normalization_stats_and_provenance():
    contract = load_contract(CONTRACT, BENCHMARK)
    stats = _stats()
    with pytest.raises(ValueError, match="shape"):
        Scheme2RProxyAdapter(replace(stats, input_mean=np.zeros(9)), BENCHMARK, contract)
    with pytest.raises(ValueError, match="strictly positive"):
        Scheme2RProxyAdapter(replace(stats, input_scale=np.zeros(10)), BENCHMARK, contract)
    with pytest.raises(ValueError, match="order"):
        Scheme2RProxyAdapter(replace(stats, label_order=("wrong",) + stats.label_order[1:]), BENCHMARK, contract)
    with pytest.raises(ValueError, match="provenance hashes"):
        Scheme2RProxyAdapter(replace(stats, contract_sha256=""), BENCHMARK, contract)
    with pytest.raises(ValueError, match="benchmark SHA-256"):
        Scheme2RProxyAdapter(replace(stats, benchmark_sha256="0" * 64), BENCHMARK, contract)
    with pytest.raises(ValueError, match="source/generator"):
        Scheme2RProxyAdapter(replace(stats, generator_version=""), BENCHMARK, contract)


def test_objective_gap_and_latency_metrics_are_explicit_and_nonnegative():
    contract = load_contract(CONTRACT, BENCHMARK)
    split = build_labeled_proxy_split(generate_synthetic_scenarios(BENCHMARK, "train", 2026, 2), BENCHMARK, contract)
    parameters = load_benchmark(BENCHMARK)["values"]
    metrics = dispatch_metrics(
        split.dispatch, split.dispatch, split.inputs, split.teacher_cost, split.teacher_carbon,
        parameters, teacher_objective=split.teacher_objective,
    )
    assert metrics["objective_gap_absolute_mean"] < 1e-8
    for key in ("objective_gap_mean", "objective_gap_absolute_mean", "objective_gap_relative_absolute_mean"):
        assert key in metrics
    stats = ProxyNormalizationStats.fit(split, benchmark=BENCHMARK)
    latency = measure_proxy_latency(SchedulingProxy(), stats.transform_inputs(split.inputs), repeats=2, warmup=1)
    for key in ("batch_size", "repeats", "proxy_batch_median_ms", "proxy_batch_p95_ms", "proxy_per_scenario_median_ms", "proxy_per_scenario_p95_ms"):
        assert key in latency
        assert latency[key] >= 0
    exact_latency = measure_exact_lp_latency(split.inputs, parameters, repeats=1, warmup=1)
    for key in ("exact_lp_per_scenario_median_ms", "exact_lp_per_scenario_p95_ms"):
        assert key in exact_latency
        assert exact_latency[key] >= 0
