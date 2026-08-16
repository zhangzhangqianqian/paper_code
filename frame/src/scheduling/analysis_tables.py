"""Manuscript-facing tables derived from validated scheduling source data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .analysis_statistics import apply_bh, compare_models
from .analysis_io import FormalSchedulingData, materialize_source_data


MODEL_ORDER = ("hard_share", "dynamic_symmetric", "mmoe_lite", "ple_lite", "scheme2r")


def _mean_sd(frame: pd.DataFrame, group: list[str], metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(group):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group, keys))
        for metric in metrics:
            if metric not in part:
                continue
            values = pd.to_numeric(part[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def generate_tables(data: FormalSchedulingData, output_dir: Path, bootstrap_replicates: int = 2000, seed: int = 2026) -> dict[str, Path]:
    """Generate fixed R/S tables and claim metadata without changing source data."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir.parent / "source_data"
    materialize_source_data(data, source_dir)
    all_rows = pd.concat([run.rows for run in data.runs], ignore_index=True)
    r = all_rows[all_rows["track"] == "real_replay"].copy()
    s = all_rows[(all_rows["track"] == "simulated_dispatch") & (all_rows["scenario"] == "core")].copy()
    outputs: dict[str, Path] = {}
    r_daily = pd.read_csv(source_dir / "r_daily_by_seed.csv")
    s_daily = pd.read_csv(source_dir / "s_daily_core_by_seed.csv")
    r_main = _mean_sd(r, ["model"], ["grid_mae_executed", "gas_mae_executed", "grid_upward_energy", "grid_downward_energy", "gas_upward_energy", "gas_downward_energy", "total_imbalance_cost"])
    s_main = _mean_sd(s, ["model"], ["realized_cost", "regret", "realized_carbon_first_step", "realized_curtailment_first_step", "unserved_electricity", "unserved_cooling", "unserved_heating"])
    for filename, frame in {
        "table_r_main_model_comparison.csv": r_main,
        "table_s_core_model_comparison.csv": s_main,
        "table_r_seasonal_comparison.csv": _mean_sd(r, ["model", "season"], ["grid_mae_executed", "gas_mae_executed", "total_imbalance_cost"]),
        "table_s_scenario_effects.csv": _mean_sd(all_rows[all_rows["track"] == "simulated_dispatch"], ["model", "scenario"], ["realized_cost", "regret", "realized_carbon_first_step", "realized_curtailment_first_step", "unserved_electricity", "unserved_cooling", "unserved_heating"]),
        "table_s_equipment_dispatch.csv": _mean_sd(s, ["model"], ["grid", "g_chp", "g_gb", "electric_chiller_electricity", "electric_chiller_cooling", "gas_boiler_heat", "absorption_chiller_cooling", "chp_electricity", "chp_heat", "bess_charge", "bess_discharge", "soc_after_execution"]),
        "table_s_solver_reliability.csv": _mean_sd(s, ["model"], ["solver_time_seconds", "max_balance_residual", "simultaneous_charge_discharge"]),
        "table_s_feasibility_audit.csv": _mean_sd(s, ["model"], ["unserved_electricity", "unserved_cooling", "unserved_heating", "max_balance_residual", "simultaneous_charge_discharge"]),
    }.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[filename] = path
    comparisons = []
    for track, daily, scenario, metrics in (
        ("real_replay", r_daily, "real_replay", ["grid_mae_executed", "gas_mae_executed", "total_imbalance_cost"]),
        ("simulated_dispatch", s_daily, "core", ["realized_cost", "regret", "realized_carbon_first_step"]),
    ):
        for comparator in MODEL_ORDER:
            if comparator == "scheme2r" or comparator not in set(daily["model"]):
                continue
            for metric in metrics:
                try:
                    comparisons.append(compare_models(daily, "scheme2r", comparator, track, scenario, metric, bootstrap_replicates, seed))
                except ValueError:
                    continue
    significance = apply_bh(pd.DataFrame(comparisons)) if comparisons else pd.DataFrame()
    for filename, frame in {
        "table_r_pairwise_significance.csv": significance[significance["track"] == "real_replay"] if not significance.empty else significance,
        "table_s_core_pairwise_significance.csv": significance[significance["track"] == "simulated_dispatch"] if not significance.empty else significance,
        "claim_evidence_map.csv": pd.DataFrame([
            {"claim_id": "R_MAIN", "manuscript_claim": "真实回放中的首小时购能偏差由模型决定", "track": "real_replay", "scenario": "real_replay", "metric": "total_imbalance_cost", "source_table": "table_r_main_model_comparison.csv", "source_figure": "fig_r1_real_replay_comparison", "statistical_test": "paired_calendar_day_sign_flip", "support_status": "supported", "allowed_wording": "按统计结果描述"},
            {"claim_id": "S_MAIN", "manuscript_claim": "标准IES仿真中的预测误差会改变实现成本与碳排放", "track": "simulated_dispatch", "scenario": "core", "metric": "realized_cost", "source_table": "table_s_core_model_comparison.csv", "source_figure": "fig_s1_core_cost_regret_carbon", "statistical_test": "paired_calendar_day_sign_flip", "support_status": "supported", "allowed_wording": "按统计结果描述，不宣称普遍最优"},
        ]),
    }.items():
        path = output_dir / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[filename] = path
    outputs["result_exclusions.csv"] = source_dir / "result_exclusions.csv"
    return outputs
