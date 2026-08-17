from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.proxy_contract import load_contract
from src.scheduling.proxy_contract import LABEL_ORDER
from src.scheduling.proxy_dataset import (
    ProxyDataset,
    ProxyNormalizationStats,
    build_labeled_proxy_split,
    load_proxy_split,
    save_proxy_split,
)
from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios


BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"


def _split(name: str, seed: int, n: int):
    contract = load_contract(CONTRACT, BENCHMARK)
    scenarios = generate_synthetic_scenarios(BENCHMARK, name, seed, n)
    return build_labeled_proxy_split(scenarios, BENCHMARK, contract)


def test_exact_labels_and_gas_prior_contract(tmp_path):
    split = _split("train", 2026, 4)
    assert split.dispatch.shape == (4, 4, 21)
    assert split.inputs.shape == (4, 4, 10)
    assert split.rejected_count == 0
    assert np.isfinite(split.dispatch).all()
    assert np.all(split.teacher_status == "optimal")
    assert split.teacher_message.shape == (4,)
    assert np.all(split.teacher_balance_residual <= 1e-7)
    assert np.all(split.teacher_simultaneous_charge_discharge <= 1e-7)
    # The prior is generated after the label and is noisy/masked, never a rigid
    # fourth demand balance in the LP.
    gas_purchase = split.dispatch[:, :, 5] + split.dispatch[:, :, 6]
    assert not np.array_equal(split.inputs[:, :, 3], gas_purchase)
    assert set(np.unique(split.gas_prior_mask)).issubset({0.0, 1.0})
    path = tmp_path / "train.npz"
    save_proxy_split(split, path)
    reloaded = load_proxy_split(path, split="train")
    assert np.array_equal(reloaded.dispatch, split.dispatch)
    assert np.array_equal(reloaded.teacher_status, split.teacher_status)
    assert np.array_equal(reloaded.teacher_message, split.teacher_message)
    assert np.array_equal(reloaded.teacher_balance_residual, split.teacher_balance_residual)
    assert np.array_equal(reloaded.teacher_simultaneous_charge_discharge, split.teacher_simultaneous_charge_discharge)
    assert reloaded.benchmark_sha256 == split.benchmark_sha256
    assert reloaded.contract_sha256 == split.contract_sha256


def test_normalization_is_train_only_and_dataset_returns_contract_shapes():
    train = _split("train", 2026, 4)
    validation = _split("validation", 2027, 2)
    stats = ProxyNormalizationStats.fit(train, benchmark=BENCHMARK)
    altered = train.inputs.copy()
    altered[..., 0] += 10000.0
    altered_stats = ProxyNormalizationStats.fit(
        altered, train.dispatch, benchmark=BENCHMARK,
        benchmark_sha256=train.benchmark_sha256, contract_sha256=train.contract_sha256,
        fit_split="train", train_seed=train.seed,
        source_type=train.source_type, generator_version=train.generator_version,
    )
    assert not np.array_equal(stats.input_mean, altered_stats.input_mean)
    dataset = ProxyDataset(validation, stats)
    item = dataset[0]
    assert tuple(item["inputs"].shape) == (4, 10)
    assert tuple(item["target"].shape) == (4, 21)


def test_normalization_provenance_round_trip_and_wrong_hash_rejection(tmp_path):
    train = _split("train", 2026, 2)
    stats = ProxyNormalizationStats.fit(train, benchmark=BENCHMARK)
    path = tmp_path / "normalization_stats.npz"
    stats.save(path)
    loaded = ProxyNormalizationStats.load(
        path,
        expected_benchmark_sha256=train.benchmark_sha256,
        expected_contract_sha256=train.contract_sha256,
    )
    assert loaded.benchmark_sha256 == train.benchmark_sha256
    assert loaded.contract_sha256 == train.contract_sha256
    with np.testing.assert_raises(ValueError):
        ProxyNormalizationStats.load(path, expected_contract_sha256="0" * 64)


def test_bounded_label_scales_match_frozen_lp_upper_bounds():
    train = _split("train", 2026, 4)
    stats = ProxyNormalizationStats.fit(train, benchmark=BENCHMARK)
    benchmark_values = __import__("yaml").safe_load(BENCHMARK.read_text(encoding="utf-8"))["values"]
    expected = np.asarray([
        benchmark_values["grid_import_capacity"],
        benchmark_values["pv_capacity"], benchmark_values["pv_capacity"],
        benchmark_values["wt_capacity"], benchmark_values["wt_capacity"],
        benchmark_values["chp_electric_capacity"] / benchmark_values["chp_electric_efficiency"],
        benchmark_values["gas_boiler_capacity"] / benchmark_values["gas_boiler_efficiency"],
        benchmark_values["chp_electric_capacity"], benchmark_values["chp_heat_capacity"], benchmark_values["gas_boiler_capacity"],
        benchmark_values["electric_chiller_capacity"] / benchmark_values["electric_chiller_cop"], benchmark_values["electric_chiller_capacity"],
        benchmark_values["absorption_chiller_capacity"] / benchmark_values["absorption_chiller_cop"], benchmark_values["absorption_chiller_capacity"],
        benchmark_values["bess_power_capacity"], benchmark_values["bess_power_capacity"], benchmark_values["bess_energy_capacity"],
    ])
    assert LABEL_ORDER[14:17] == ("p_charge", "p_discharge", "soc")
    assert np.allclose(stats.label_scale[:17], expected, rtol=0.0, atol=1e-10)
    assert np.isclose(stats.label_scale[14], 255.6, rtol=0.0, atol=1e-10)
    altered = train.dispatch.copy()
    altered[..., 17:] += 100.0
    altered_stats = ProxyNormalizationStats.fit(
        train.inputs, altered, benchmark=BENCHMARK,
        benchmark_sha256=train.benchmark_sha256, contract_sha256=train.contract_sha256,
        fit_split="train", train_seed=train.seed, source_type=train.source_type,
        generator_version=train.generator_version,
    )
    assert np.array_equal(altered_stats.label_scale[:17], stats.label_scale[:17])
    assert np.all(altered_stats.label_scale[17:] > stats.label_scale[17:])


def test_normalization_rejects_validation_and_split_loader_fails_closed(tmp_path):
    train = _split("train", 2026, 2)
    validation = _split("validation", 2027, 2)
    with np.testing.assert_raises(ValueError):
        ProxyNormalizationStats.fit(validation, benchmark=BENCHMARK)
    path = tmp_path / "train.npz"
    save_proxy_split(train, path)
    with np.testing.assert_raises(ValueError):
        load_proxy_split(path, expected_split="validation")
    with np.testing.assert_raises(ValueError):
        load_proxy_split(path, expected_split="train", expected_benchmark_sha256="0" * 64)
    with np.load(path, allow_pickle=False) as payload:
        corrupted = {name: payload[name] for name in payload.files if name != "teacher_message"}
    corrupted_path = tmp_path / "missing_audit.npz"
    np.savez_compressed(corrupted_path, **corrupted)
    with np.testing.assert_raises(ValueError):
        load_proxy_split(corrupted_path, expected_split="train")
