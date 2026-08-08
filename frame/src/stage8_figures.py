"""Matplotlib figures for the frozen Stage 8 analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def _save(fig: mpl.figure.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_overall(rows: pd.DataFrame, output_dir: Path) -> None:
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    tasks = list(rows["task"].drop_duplicates())
    protocols = list(rows["protocol"].drop_duplicates())
    width = 0.36
    x = np.arange(len(tasks))
    colors = {"full": "#4472A8", "small_sample": "#D98324"}
    for index, protocol in enumerate(protocols):
        subset = rows[rows["protocol"] == protocol].set_index("task").reindex(tasks)
        values = subset["gain_MAE_pct"].to_numpy(dtype=float)
        low = np.maximum(0.0, values - subset["gain_ci_low"].to_numpy(dtype=float))
        high = np.maximum(0.0, subset["gain_ci_high"].to_numpy(dtype=float) - values)
        low = np.nan_to_num(low, nan=0.0)
        high = np.nan_to_num(high, nan=0.0)
        ax.bar(x + (index - (len(protocols) - 1) / 2) * width, values, width,
               label=protocol, color=colors.get(protocol, "#6C757D"),
               yerr=np.vstack([low, high]), capsize=3, error_kw={"elinewidth": 0.8})
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x, tasks)
    ax.set_ylabel("MAE gain (%)")
    ax.set_title("Scheme2R-H4 relative to matched STL-H4")
    ax.legend()
    _save(fig, output_dir / "transfer_gain_overall")


def plot_joint_model_comparison(rows: pd.DataFrame, output_dir: Path) -> None:
    """Main same-track comparison: joint multi-task models only."""

    if rows.empty:
        return
    preferred = ["hard_share", "dynamic_symmetric", "mmoe-lite", "ple-lite", "scheme2r"]
    models = [model for model in preferred if model in set(rows["model"])]
    protocols = [name for name in ("full", "small_sample") if name in set(rows["protocol"])]
    fig, axes = plt.subplots(1, len(protocols), figsize=(7.2, 3.0), squeeze=False)
    colors = {model: "#B8C5D6" for model in models}
    colors["scheme2r"] = "#4472A8"
    labels = {
        "hard_share": "Hard-Share",
        "dynamic_symmetric": "Dynamic\nSymmetric",
        "mmoe-lite": "MMoE-lite",
        "ple-lite": "PLE-lite",
        "scheme2r": "Scheme2R",
    }
    for axis, protocol in zip(axes[0], protocols):
        subset = rows[rows["protocol"] == protocol].set_index("model").reindex(models)
        values = subset["MAE_mean"].to_numpy(dtype=float)
        errors = subset["MAE_std"].to_numpy(dtype=float)
        x = np.arange(len(models))
        axis.bar(
            x, values, yerr=errors, capsize=2.5,
            color=[colors[model] for model in models],
            edgecolor="#FFFFFF", linewidth=0.5,
        )
        axis.set_xticks(x, [labels[model] for model in models], rotation=25, ha="right")
        axis.set_title(protocol.replace("_", " ").title())
        axis.set_ylabel("MAE (five-seed mean ± s.d.)")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.suptitle("Same-track joint multi-task model comparison", y=1.02, fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "joint_model_mae_comparison")


def plot_general_baseline_comparison(rows: pd.DataFrame, output_dir: Path) -> None:
    """Supplementary comparison kept visually separate from the joint leaderboard."""

    if rows.empty:
        return
    protocols = [name for name in ("full", "small_sample") if name in set(rows["protocol"])]
    fig, axes = plt.subplots(1, len(protocols), figsize=(7.2, 2.8), squeeze=False)
    for axis, protocol in zip(axes[0], protocols):
        subset = rows[rows["protocol"] == protocol].sort_values("MAE_mean")
        labels = subset["model"].str.replace("_", " ").tolist()
        values = subset["MAE_mean"].to_numpy(dtype=float)
        errors = subset["MAE_std"].to_numpy(dtype=float)
        x = np.arange(len(labels))
        axis.bar(x, values, yerr=errors, capsize=2.5, color="#A9A9A9")
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.set_title(protocol.replace("_", " ").title())
        axis.set_ylabel("MAE")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.6)
    fig.suptitle("Supplementary general baselines", y=1.02, fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "general_baseline_mae_comparison")


def _heatmap(
    matrix: np.ndarray,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
    title: str,
    colorbar_label: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    finite = matrix[np.isfinite(matrix)]
    limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    limit = max(limit, 1.0)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Target task")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=7)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label(colorbar_label)
    _save(fig, path)


def plot_transfer_horizon(rows: pd.DataFrame, output_dir: Path) -> None:
    if rows.empty:
        return
    for protocol in rows["protocol"].drop_duplicates():
        subset = rows[rows["protocol"] == protocol]
        tasks = list(subset["task"].drop_duplicates())
        matrix = subset.pivot(index="task", columns="horizon_step", values="gain_MAE_pct").reindex(tasks).to_numpy(dtype=float)
        _heatmap(matrix, [f"h+{i}" for i in range(1, matrix.shape[1] + 1)], tasks,
                 f"Task-horizon MAE gain ({protocol})", "MAE gain (%)",
                 output_dir / f"transfer_gain_horizon_{protocol}")


def plot_context(rows: pd.DataFrame, output_dir: Path) -> None:
    if rows.empty or "context_type" not in rows.columns:
        return
    for protocol in rows["protocol"].drop_duplicates():
        subset = rows[(rows["protocol"] == protocol) & (rows["context_type"] == "season")]
        if subset.empty:
            continue
        pivot = subset.groupby(["task", "context_label"], as_index=False)["gain_MAE_pct"].mean()
        matrix = pivot.pivot(index="task", columns="context_label", values="gain_MAE_pct").to_numpy(dtype=float)
        labels = list(pivot["context_label"].drop_duplicates())
        _heatmap(matrix, labels, list(pivot["task"].drop_duplicates()),
                 f"Seasonal MAE gain ({protocol})", "MAE gain (%)",
                 output_dir / f"transfer_gain_season_{protocol}")


def plot_gate_summary(rows: pd.DataFrame, output_dir: Path) -> None:
    if rows.empty:
        return
    for protocol in rows["protocol"].drop_duplicates():
        subset = rows[(rows["protocol"] == protocol) & (rows["context_type"] == "overall")]
        if subset.empty:
            continue
        pivot = subset.groupby(["target_task", "source_task"], as_index=False)["gate_mean"].mean()
        tasks = list(rows["target_task"].drop_duplicates())
        matrix = pivot.pivot(index="target_task", columns="source_task", values="gate_mean").reindex(index=tasks, columns=tasks).to_numpy(dtype=float)
        _heatmap(matrix, tasks, tasks, f"Directed gate intensity ({protocol})", "gate mean",
                 output_dir / f"gate_matrix_{protocol}")


def plot_gate_error_association(rows: pd.DataFrame, output_dir: Path) -> None:
    if rows.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for protocol, group in rows.groupby("protocol"):
        column = "spearman_mean" if "spearman_mean" in group.columns else "spearman_rho"
        numeric = group[column].astype(float).to_numpy()
        values = numeric[np.isfinite(numeric)]
        if values.size:
            ax.scatter(np.arange(len(values)), values, label=protocol, s=18, alpha=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Spearman association")
    ax.set_xlabel("Gate–error analysis cell")
    ax.set_title("Gate intensity and absolute-error improvement")
    ax.set_ylim(-1.05, 1.05)
    ax.legend()
    _save(fig, output_dir / "gate_error_association")


def generate_all_figures(output_dir: Path) -> list[str]:
    figures_dir = output_dir / "figures"
    generated: list[str] = []
    overall = pd.read_csv(output_dir / "transfer_task_overall.csv")
    horizon = pd.read_csv(output_dir / "transfer_task_horizon.csv")
    context_path = output_dir / "transfer_context.csv"
    gate_path = output_dir / "gate_summary.csv"
    assoc_path = output_dir / "gate_error_association.csv"
    joint_path = output_dir / "joint_model_comparison.csv"
    general_path = output_dir / "general_baseline_comparison.csv"
    if joint_path.exists():
        plot_joint_model_comparison(pd.read_csv(joint_path), figures_dir)
    if general_path.exists():
        plot_general_baseline_comparison(pd.read_csv(general_path), figures_dir)
    plot_transfer_overall(overall, figures_dir)
    plot_transfer_horizon(horizon, figures_dir)
    if context_path.exists():
        plot_context(pd.read_csv(context_path), figures_dir)
    if gate_path.exists():
        plot_gate_summary(pd.read_csv(gate_path), figures_dir)
    if assoc_path.exists():
        plot_gate_error_association(pd.read_csv(assoc_path), figures_dir)
    generated.extend(str(path.relative_to(output_dir)) for path in figures_dir.glob("*.svg"))
    generated.extend(str(path.relative_to(output_dir)) for path in figures_dir.glob("*.pdf"))
    generated.extend(str(path.relative_to(output_dir)) for path in figures_dir.glob("*.png"))
    generated.extend(str(path.relative_to(output_dir)) for path in figures_dir.glob("*.tiff"))
    return sorted(generated)
