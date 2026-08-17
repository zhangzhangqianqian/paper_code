from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path("D:/Paper/standard_ies_benchmark_v1.yaml")
CONTRACT = ROOT / "configs" / "scheduling_proxy_contract_v1.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _validation_arrays() -> tuple[np.ndarray, np.ndarray]:
    """Build exact labels from a validation-only synthetic batch.

    Repair tests must not depend on, or open, the frozen test artifact.
    """

    from src.scheduling.proxy_contract import load_contract
    from src.scheduling.proxy_dataset import build_labeled_proxy_split
    from src.scheduling.synthetic_scenarios import generate_synthetic_scenarios

    contract = load_contract(CONTRACT, benchmark_path=BENCHMARK)
    scenarios = generate_synthetic_scenarios(
        BENCHMARK,
        "validation",
        seed=2027,
        n_samples=16,
        generator_version=contract.generator_version,
    )
    split = build_labeled_proxy_split(scenarios, BENCHMARK, contract=contract)
    return split.dispatch.copy(), split.inputs.copy()


def _parameters():
    from src.scheduling.synthetic_scenarios import load_benchmark

    return load_benchmark(BENCHMARK)["values"]


def test_repair_never_overwrites_raw_prediction():
    from src.scheduling.proxy_repairs import apply_repair

    raw, features = _validation_arrays()
    original = raw.copy()
    result = apply_repair(raw, features, _parameters(), method="candidate_a")
    assert np.array_equal(result.raw_prediction, original)
    assert np.array_equal(raw, original)
    assert result.provenance["raw_prediction_preserved"] is True


def test_repair_preserves_label_order_and_shape():
    from src.scheduling.proxy_contract import LABEL_ORDER
    from src.scheduling.proxy_repairs import apply_repair

    raw, features = _validation_arrays()
    result = apply_repair(raw, features, _parameters(), method="bound_and_split")
    assert result.repaired_prediction.shape == raw.shape
    assert result.label_order == tuple(LABEL_ORDER)
    assert result.repair_mask.shape == (raw.shape[0],)
    assert set(result.family_magnitudes) == {
        "balance", "conversion", "soc_state", "soc_terminal", "bounds", "renewable_split", "ramp",
    }


def test_repair_does_not_call_exact_lp():
    from src.scheduling.proxy_repairs import apply_repair

    raw, features = _validation_arrays()
    with mock.patch("src.scheduling.proxy_repairs.solve_dispatch_lp", side_effect=AssertionError):
        result = apply_repair(raw, features, _parameters(), method="physical_decoder")
    assert result.repaired_prediction.shape == raw.shape
    assert result.provenance["exact_lp_used"] is False


def test_bound_and_split_repairs_renewable_equalities():
    from src.scheduling.proxy_adapter import evaluate_raw_feasibility
    from src.scheduling.proxy_repairs import apply_repair

    raw, features = _validation_arrays()
    result = apply_repair(raw, features, _parameters(), method="bound_and_split")
    report = evaluate_raw_feasibility(result.repaired_prediction, features, _parameters())
    assert float(np.max(report["renewable_split_residual"])) <= 1.0e-9


def test_physical_decoder_is_feasible_without_fallback():
    from src.scheduling.proxy_adapter import evaluate_raw_feasibility
    from src.scheduling.proxy_repairs import apply_repair

    raw, features = _validation_arrays()
    result = apply_repair(raw, features, _parameters(), method="physical_decoder")
    report = evaluate_raw_feasibility(result.repaired_prediction, features, _parameters())
    assert np.all(report["feasible_mask"])
    assert float(np.max(report["residual"])) <= 1.0e-3
