from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate_rsc_pf_common_validation import (
    aggregate_common_validation_table,
    evaluate_rsc_pf_validation,
)


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "joint_forecast_dispatch_contract_v2.json"


def test_common_evaluation_rejects_test_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        evaluate_rsc_pf_validation(
            2026,
            CONFIG,
            data_root=tmp_path / "test",
            output_root=tmp_path / "outputs",
        )


def test_common_evaluation_receipt_has_frozen_contract() -> None:
    output = ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation"
    receipt = evaluate_rsc_pf_validation(2026, CONFIG, output_root=output)
    assert receipt["test_set_accessed"] is False
    assert receipt["metrics"]["windows"] == 8780
    assert receipt["exact_lp_calls"] == 0
    assert receipt["optimizer_role"] == "none at inference"


def test_common_validation_table_has_four_methods_and_five_seeds(tmp_path: Path) -> None:
    manifest = {
        "entries": [
            {
                "method_id": method,
                "seed": seed,
                "metrics": {
                    "windows": 8780,
                    "forecast_mae_mean": 1.0,
                    "forecast_rmse_mean": 2.0,
                    "forecast_wape_mean": 0.1,
                    "operating_cost_mean": 3.0,
                    "physical_carbon_mean": 4.0,
                    "penalized_objective_mean": 4.5,
                    "regret_vs_oracle_mean": 5.0,
                    "shortage_mean": 6.0,
                    "feasibility_rate": 0.5,
                },
                "test_set_accessed": False,
                "exact_lp_calls": 0,
                "optimizer_role": "none at inference",
            }
            for method in ("RSC-PF", "iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy")
            for seed in (2026, 2027, 2028, 2029, 2030)
        ],
        "test_set_accessed": False,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    table = aggregate_common_validation_table(path, tmp_path)
    assert table.is_file()
    text = table.read_text(encoding="utf-8-sig")
    assert text.count("RSC-PF") == 1
    assert text.count("iTransformer-PTO") == 1
