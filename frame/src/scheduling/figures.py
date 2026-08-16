"""Publication-oriented Python figures for scheduling evidence (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

COLORS = {"scheme2r": "#2f5597", "hard_share": "#7f8c8d", "dynamic_symmetric": "#55a868", "mmoe_lite": "#c44e52", "ple_lite": "#8172b2"}


def _save(fig: plt.Figure, base: Path) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = [base.with_suffix(ext) for ext in (".svg", ".pdf", ".tiff", ".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], dpi=600, bbox_inches="tight")
    fig.savefig(paths[3], dpi=180, bbox_inches="tight")
    plt.close(fig)
    return paths


def _model_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = frame.groupby("model")[metric].agg(["mean", "std"]).reset_index()
    return grouped.sort_values("model")


def plot_r1(frame: pd.DataFrame, output_base: Path, source_data_path: Path | None = None) -> list[Path]:
    """Real-replay comparison: executed MAE and imbalance cost."""

    metrics = [("grid_mae_executed", "Grid MAE"), ("gas_mae_executed", "Gas MAE"), ("total_imbalance_cost", "Total imbalance cost")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in frame:
            ax.set_visible(False)
            continue
        summary = _model_summary(frame, metric)
        x = np.arange(len(summary))
        ax.bar(x, summary["mean"], yerr=summary["std"].fillna(0.0), color=[COLORS.get(m, "#999999") for m in summary["model"]], capsize=2)
        ax.set_xticks(x, summary["model"], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("normalized index")
    return _save(fig, output_base)


def plot_s1(frame: pd.DataFrame, output_base: Path) -> list[Path]:
    metrics = [("realized_cost", "Realized cost"), ("regret", "Oracle regret"), ("realized_carbon_first_step", "Realized carbon"), ("unserved_electricity", "Unserved electricity")]
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.25), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in frame:
            ax.set_visible(False)
            continue
        summary = _model_summary(frame, metric)
        x = np.arange(len(summary))
        ax.bar(x, summary["mean"], yerr=summary["std"].fillna(0.0), color=[COLORS.get(m, "#999999") for m in summary["model"]], capsize=2)
        ax.set_xticks(x, summary["model"], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel("normalized index")
    return _save(fig, output_base)


def plot_s2(frame: pd.DataFrame, output_base: Path) -> list[Path]:
    metric = "realized_cost"
    summary = frame.groupby(["model", "scenario"])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    for model in summary["model"].unique():
        part = summary[summary["model"] == model]
        ax.plot(part["scenario"], part[metric], marker="o", label=model, color=COLORS.get(model, "#999999"))
    ax.set_ylabel("realized cost (normalized index)")
    ax.set_xlabel("IES scenario")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=2)
    return _save(fig, output_base)


def plot_s3(frame: pd.DataFrame, output_base: Path, date: str | None = None) -> list[Path]:
    data = frame.copy()
    data["timestamp"] = pd.to_datetime(data["origin"])
    # A source table contains five model runs per origin; a typical-day
    # dispatch panel must use one frozen reference model, not the first 24
    # duplicated rows across models.
    if "model" in data.columns and "scheme2r" in set(data["model"].astype(str)):
        data = data[data["model"].astype(str) == "scheme2r"].copy()
    data["date"] = data["timestamp"].dt.strftime("%Y-%m-%d")
    if date is not None:
        dates = [date]
    else:
        daily = data.groupby("date")[["actual_cooling", "actual_heating"]].sum()
        summer = str(daily["actual_cooling"].idxmax())
        winter = str(daily["actual_heating"].idxmax())
        dates = [summer] if summer == winter else [summer, winter]
    fig, axes = plt.subplots(3, len(dates), figsize=(4.1 * len(dates), 5.2), sharex="col", constrained_layout=True, squeeze=False)
    for column, selected_date in enumerate(dates):
        part = data[data["date"] == selected_date].sort_values("timestamp").head(24)
        x = np.arange(len(part))
        if part.empty:
            continue
        panels = (
            (0, "actual_electricity", "electricity demand", "Electricity", (("planned_grid_first_step", "grid"), ("chp_electricity", "CHP"), ("electric_chiller_electricity", "electric chiller"))),
            (1, "actual_heating", "heating demand", "Heat", (("chp_heat", "CHP heat"), ("gas_boiler_heat", "boiler"))),
            (2, "actual_cooling", "cooling demand", "Cooling", (("electric_chiller_cooling", "electric chiller"), ("absorption_chiller_cooling", "absorption chiller"))),
        )
        for row, demand_col, demand_label, ylabel, devices in panels:
            ax = axes[row, column]
            if demand_col in part:
                ax.plot(x, part[demand_col], label=demand_label, color={0: "#2f5597", 1: "#c44e52", 2: "#55a868"}[row])
            for col, label in devices:
                if col in part:
                    ax.plot(x, part[col], label=label)
            ax.set_ylabel(ylabel)
            ax.legend(ncol=1, fontsize=6)
            if row == 0:
                ax.set_title(selected_date)
            if row == 2:
                ax.set_xlabel("forecast origin hour")
    return _save(fig, output_base)


def plot_s4(frame: pd.DataFrame, output_base: Path) -> list[Path]:
    metrics = [("solver_time_seconds", "Solver time"), ("max_balance_residual", "Balance residual"), ("unserved_electricity", "Unserved electricity")]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        if metric not in frame:
            ax.set_visible(False)
            continue
        values = pd.to_numeric(frame[metric], errors="coerce").dropna()
        ax.hist(values, bins=30, color="#4c72b0", alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel("hours")
    return _save(fig, output_base)


def generate_figures(source_dir: Path, figure_dir: Path) -> dict[str, list[Path]]:
    """Generate all four figure families from CSV source data."""

    source_dir = Path(source_dir)
    figure_dir = Path(figure_dir)
    r = pd.read_csv(source_dir / "r_hourly_executed.csv")
    s_core = pd.read_csv(source_dir / "s_hourly_core.csv")
    s_all = pd.read_csv(source_dir / "s_daily_scenarios_by_seed.csv")
    outputs = {
        "fig_r1": plot_r1(r, figure_dir / "fig_r1_real_replay_comparison"),
        "fig_s1": plot_s1(s_core, figure_dir / "fig_s1_core_cost_regret_carbon"),
        "fig_s2": plot_s2(s_all, figure_dir / "fig_s2_scenario_effects"),
        "fig_s3": plot_s3(s_core, figure_dir / "fig_s3_typical_day_dispatch"),
        "fig_s4": plot_s4(s_core, figure_dir / "fig_s4_solver_and_feasibility"),
    }
    for key, frame in (("figure_r1", r), ("figure_s1", s_core), ("figure_s2", s_all), ("figure_s3", s_core), ("figure_s4", s_core)):
        frame.to_csv(source_dir / f"{key}.csv", index=False, encoding="utf-8-sig")
    return outputs
