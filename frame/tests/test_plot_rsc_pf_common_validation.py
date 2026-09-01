from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.plot_rsc_pf_common_validation import EXPECTED_METHODS, load_common_table


ROOT = Path(__file__).parents[1]
TABLE = ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation" / "common_validation_summary.csv"


def test_common_figure_table_is_complete_and_ordered() -> None:
    table = load_common_table(TABLE)
    assert table["method_id"].tolist() == list(EXPECTED_METHODS)
    assert table["seed_count"].astype(int).tolist() == [5, 5, 5, 5]
    assert table["windows_per_seed"].astype(int).tolist() == [8780] * 4
    assert table["test_set_accessed"].astype(str).str.lower().tolist() == ["false"] * 4


def test_common_figure_has_finite_means_and_standard_deviations() -> None:
    table = load_common_table(TABLE)
    columns = [
        "forecast_mae_mean", "forecast_mae_sd", "forecast_rmse_mean", "forecast_rmse_sd",
        "forecast_wape_mean", "forecast_wape_sd", "operating_cost_mean", "operating_cost_sd",
        "physical_carbon_mean", "physical_carbon_sd", "penalized_objective_mean", "penalized_objective_sd",
        "regret_vs_oracle_mean", "regret_vs_oracle_sd", "shortage_mean", "shortage_sd",
        "feasibility_rate_mean", "feasibility_rate_sd",
    ]
    assert np.isfinite(table[columns].to_numpy(dtype=float)).all()
    assert (table[[c for c in columns if c.endswith("_sd")]].to_numpy(dtype=float) >= 0.0).all()
