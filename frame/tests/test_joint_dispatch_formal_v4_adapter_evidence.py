from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_gate0_evidence import FORMAL_V4_METHOD_IDS, validate_gate0_receipt
from src.joint_dispatch.formal_v4_method_adapter import FORMAL_V4_METHOD_CONTRACTS
from scripts.build_rsc_pf_formal_v4_adapter_evidence import build_method_adapter_receipt


_HASH = "a" * 64
_COMMIT = "b" * 40


def _write_fixture(root: Path) -> tuple[Path, Path]:
    benchmark = root / "benchmark.yaml"
    benchmark.write_text(
        "values:\n"
        "  grid_import_capacity: 100\n"
        "  chp_electric_capacity: 20\n"
        "  chp_heat_capacity: 30\n"
        "  gas_boiler_capacity: 40\n"
        "  electric_chiller_capacity: 30\n"
        "  absorption_chiller_capacity: 30\n"
        "  bess_power_capacity: 10\n"
        "  bess_energy_capacity: 40\n"
        "  chp_electric_efficiency: 0.35\n"
        "  chp_heat_efficiency: 0.45\n"
        "  gas_boiler_efficiency: 0.9\n"
        "  electric_chiller_cop: 3.5\n"
        "  absorption_chiller_cop: 0.75\n"
        "  bess_roundtrip_efficiency: 0.9\n"
        "  bess_throughput_cost: 0.01\n"
        "  grid_energy_price: 1.0\n"
        "  gas_energy_price: 0.6\n"
        "  grid_emission_factor: 0.5\n"
        "  gas_emission_factor: 0.25\n"
        "  carbon_price_default: 0.2\n"
        "  unserved_penalty: 100\n"
        "  surplus_penalty: 0\n"
        "  chp_ramp_fraction: 0.5\n",
        encoding="utf-8",
    )
    arrays = {
        "load_history": np.ones((1, 24, 4)),
        "exog_history": np.zeros((1, 24, 12)),
        "device_history": np.zeros((1, 24, 17)),
        "activity_history": np.zeros((1, 24, 6)),
        "renewable_forecast": np.ones((1, 4, 2)),
        "prices_and_weights": np.tile(np.array([[1.0, 0.6, 0.0]]), (1, 4, 1)),
        "initial_soc": np.array([[0.5]]),
        "previous_chp": np.array([[0.0]]),
        "forecast_target": np.ones((1, 4, 4)),
    }
    archive = root / "train.npz"
    np.savez(archive, **arrays)
    return benchmark, archive


def test_contract_registry_is_exact_and_explicit():
    assert tuple(item.method_id for item in FORMAL_V4_METHOD_CONTRACTS) == FORMAL_V4_METHOD_IDS
    assert FORMAL_V4_METHOD_CONTRACTS[0].online_optimizer_calls_per_window == 0
    assert FORMAL_V4_METHOD_CONTRACTS[6].role == "differentiable_optimizer"
    assert FORMAL_V4_METHOD_CONTRACTS[7].deployable is False


def test_adapter_receipt_records_executable_and_contract_only_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    benchmark, archive = _write_fixture(tmp_path)

    class FakeAdapter:
        def __init__(self, method_id):
            self.method_id = method_id

        def predict_and_dispatch(self, window, rolling_state):
            return {
                "forecast": np.ones((4, 4)),
                "dispatch": np.ones((4, 21)),
                "optimizer_calls": 0 if self.method_id in {"RSC-PF", "Decoupled-RSC-PF", "Direct-Policy"} else 1,
            }

    monkeypatch.setattr(
        "scripts.build_rsc_pf_formal_v4_adapter_evidence.build_formal_v4_method_adapter",
        lambda method_id, parameters: FakeAdapter(method_id),
    )
    payload = build_method_adapter_receipt(
        benchmark_path=benchmark,
        train_archive_path=archive,
        source_commit=_COMMIT,
    )
    assert payload["schema_version"] == "formal-v4.1-method-adapter-receipt-v1"
    assert payload["source_commit"] == _COMMIT
    assert len(payload["methods"]) == 9
    statuses = {row["method_id"]: row["probe_status"] for row in payload["methods"]}
    assert statuses["RSC-PF"] == "passed"
    assert statuses["Direct-Policy"] == "passed"
    assert statuses["Official iTransformer-PTO"] == "contract_only"
    assert statuses["Differentiable-LP"] == "contract_only"
    assert statuses["Perfect-Information-MPC"] == "contract_only"
    validate_gate0_receipt("method_adapter", payload, run_root=tmp_path)
