from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios


BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")


def test_generation_is_deterministic_and_disjoint():
    first = generate_synthetic_scenarios(BENCHMARK, "train", 2026, 8)
    second = generate_synthetic_scenarios(BENCHMARK, "train", 2026, 8)
    validation = generate_synthetic_scenarios(BENCHMARK, "validation", 2027, 8)
    assert np.array_equal(first.demand, second.demand)
    assert np.array_equal(first.scenario_ids, second.scenario_ids)
    assert set(first.scenario_ids).isdisjoint(validation.scenario_ids)
    assert first.demand.shape == (8, 4, 3)
    assert first.prices.shape == (8, 4, 3)
    assert first.gas_prior.shape == (8, 4)
    assert first.metadata["uses_observed_windows"] is False if "uses_observed_windows" in first.metadata else True


def test_generation_stays_within_operating_envelope():
    batch = generate_synthetic_scenarios(BENCHMARK, "test", 2028, 32)
    assert np.isfinite(batch.demand).all()
    assert np.isfinite(batch.prices).all()
    assert (batch.demand >= 0).all()
    assert ((batch.initial_soc >= 0.2) & (batch.initial_soc <= 0.8)).all()
    assert (batch.pv_available >= 0).all()
    assert (batch.wt_available >= 0).all()
