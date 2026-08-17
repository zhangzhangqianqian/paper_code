from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.proxy_contract import load_contract
from src.scheduling.proxy_dataset import build_labeled_proxy_split
from src.scheduling.proxy_training import load_trained_proxy, train_proxy
from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios


BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"


def _split(name: str, seed: int, n: int):
    c = load_contract(CONTRACT, BENCHMARK)
    return build_labeled_proxy_split(generate_synthetic_scenarios(BENCHMARK, name, seed, n), BENCHMARK, c)


def test_one_epoch_training_selects_validation_and_reloads_checkpoint(tmp_path):
    contract = load_contract(CONTRACT, BENCHMARK)
    result = train_proxy(
        _split("train", 2026, 4),
        _split("validation", 2027, 2),
        tmp_path,
        contract=contract,
        benchmark=BENCHMARK,
        max_epochs=1,
        patience=1,
        batch_size=2,
    )
    assert result.best_epoch == 0
    assert result.checkpoint_path.exists()
    assert result.normalization_path.exists()
    assert result.history_path.exists()
    assert result.metadata["selection_split"] == "validation"
    assert result.metadata["test_split_used_for_selection"] is False
    restored, stats, payload = load_trained_proxy(tmp_path, contract)
    assert tuple(restored(torch.zeros(2, 4, 10)).shape) == (2, 4, 21)
    assert stats.label_scale.shape == (21,)
    history_path = tmp_path / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["contract_sha256"] = "0" * 64
    history_path.write_text(json.dumps(history), encoding="utf-8")
    with __import__("pytest").raises(ValueError, match="provenance"):
        load_trained_proxy(tmp_path, contract)


def test_repeated_training_is_deterministic_on_same_synthetic_splits(tmp_path):
    contract = load_contract(CONTRACT, BENCHMARK)
    first = train_proxy(
        _split("train", 2026, 4), _split("validation", 2027, 2), tmp_path / "first",
        contract=contract, benchmark=BENCHMARK, max_epochs=1, patience=1, batch_size=2,
    )
    second = train_proxy(
        _split("train", 2026, 4), _split("validation", 2027, 2), tmp_path / "second",
        contract=contract, benchmark=BENCHMARK, max_epochs=1, patience=1, batch_size=2,
    )
    assert first.history == second.history
    first_state = torch.load(first.checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
    second_state = torch.load(second.checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)


def test_load_trained_proxy_cross_checks_all_artifact_provenance(tmp_path):
    contract = load_contract(CONTRACT, BENCHMARK)
    train_proxy(
        _split("train", 2026, 4), _split("validation", 2027, 2), tmp_path,
        contract=contract, benchmark=BENCHMARK, max_epochs=1, patience=1, batch_size=2,
    )
    # All supported call forms are safe when the independently written
    # checkpoint, history and normalization artifacts agree.
    load_trained_proxy(tmp_path)
    load_trained_proxy(tmp_path, benchmark_path=BENCHMARK)
    load_trained_proxy(tmp_path, contract=contract)
    load_trained_proxy(tmp_path, contract=contract, benchmark_path=BENCHMARK)
    checkpoint_path = tmp_path / "best_model.pt"
    original = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def assert_tampered(field: str, value: object):
        tampered = dict(original)
        tampered["metadata"] = dict(original["metadata"])
        tampered["metadata"][field] = value
        torch.save(tampered, checkpoint_path)
        try:
            with __import__("pytest").raises(ValueError):
                load_trained_proxy(tmp_path)
        finally:
            torch.save(original, checkpoint_path)

    assert_tampered("benchmark_sha256", "0" * 64)
    assert_tampered("contract_sha256", "0" * 64)
    assert_tampered("feature_order", ["wrong"])
    assert_tampered("selection_split", "test")
    assert_tampered("seed", 9999)
    assert_tampered("train_scenario_id_digest", "0" * 64)
    assert_tampered("validation_scenario_id_digest", "0" * 64)
    checkpoint_bytes = checkpoint_path.read_bytes()
    history_path = tmp_path / "history.json"
    history_bytes = history_path.read_bytes()

    def assert_checkpoint_consistency_tampered(field: str, value: object):
        checkpoint_value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_value[field] = value
        torch.save(checkpoint_value, checkpoint_path)
        try:
            with __import__("pytest").raises(ValueError):
                load_trained_proxy(tmp_path)
        finally:
            checkpoint_path.write_bytes(checkpoint_bytes)

    assert_checkpoint_consistency_tampered("epoch", 999)
    assert_checkpoint_consistency_tampered("validation_loss", float(original["validation_loss"]) + 1.0)
    history_value = json.loads(history_bytes.decode("utf-8"))
    history_value["history"][0]["validation_total"] += 1.0
    history_path.write_text(json.dumps(history_value), encoding="utf-8")
    try:
        with __import__("pytest").raises(ValueError):
            load_trained_proxy(tmp_path)
    finally:
        history_path.write_bytes(history_bytes)
    incomplete = dict(original)
    incomplete["metadata"] = dict(original["metadata"])
    incomplete["metadata"].pop("label_order")
    torch.save(incomplete, checkpoint_path)
    try:
        with __import__("pytest").raises(ValueError, match="incomplete"):
            load_trained_proxy(tmp_path, contract=contract)
    finally:
        torch.save(original, checkpoint_path)

    stats_path = tmp_path / "normalization_stats.npz"
    original_stats_bytes = stats_path.read_bytes()

    def assert_stats_tampered(field: str, value: object):
        with np.load(stats_path, allow_pickle=False) as payload:
            tampered_stats = {name: payload[name] for name in payload.files}
        tampered_stats[field] = np.asarray(value)
        np.savez_compressed(stats_path, **tampered_stats)
        try:
            with __import__("pytest").raises(ValueError):
                load_trained_proxy(tmp_path)
        finally:
            stats_path.write_bytes(original_stats_bytes)

    assert_stats_tampered("selection_split", "test")
    assert_stats_tampered("validation_scenario_id_digest", "0" * 64)
