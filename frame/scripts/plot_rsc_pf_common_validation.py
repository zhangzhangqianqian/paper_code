"""Render a publication-ready common validation comparison for RSC-PF.

The script is a read-only projection of the frozen common-validation table. It
does not retrain models, recompute metrics, or access the sealed test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Mandatory nature-figure editable-text settings.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['pdf.fonttype'] = 42
# Keep the key/value form explicit for automated publication preflight.
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHODS = ("RSC-PF", "iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy")
METHOD_LABELS = {
    "RSC-PF": "RSC-PF",
    "iTransformer-PTO": "iTransformer",
    "DecisionFocused-Online": "DecisionFocused",
    "DigitalTwins-Policy": "DigitalTwins",
}
COLORS = {
    "RSC-PF": "#D55E00",
    "iTransformer-PTO": "#4C78A8",
    "DecisionFocused-Online": "#7A68A6",
    "DigitalTwins-Policy": "#8C8C8C",
}
METRICS = (
    ("forecast_mae", "Forecast MAE", "MAE (source units)", False),
    ("forecast_rmse", "Forecast RMSE", "RMSE (source units)", False),
    ("forecast_wape", "Forecast WAPE", "WAPE (%)", True),
    ("operating_cost", "Operating cost", "Cost (source-index units)", False),
    ("physical_carbon", "Physical carbon", "Carbon index", False),
    ("penalized_objective", "Penalized objective", "Objective (source-index units)", False),
    ("regret_vs_oracle", "Decision regret", "Regret vs Oracle", False),
    ("feasibility_rate", "No-shortage rate", "Windows without shortage (%)", True),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common_table(path: str | Path) -> pd.DataFrame:
    """Load and validate the frozen four-method common-validation table."""

    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(table_path)
    table = pd.read_csv(table_path)
    if table["method_id"].tolist() != list(EXPECTED_METHODS):
        raise ValueError("common-validation methods are missing or out of frozen order")
    if table["seed_count"].astype(int).tolist() != [5] * len(EXPECTED_METHODS):
        raise ValueError("common-validation table does not contain five seeds per method")
    if table["windows_per_seed"].astype(int).tolist() != [8780] * len(EXPECTED_METHODS):
        raise ValueError("common-validation table has an unexpected window count")
    if table["test_set_accessed"].astype(str).str.lower().tolist() != ["false"] * len(EXPECTED_METHODS):
        raise ValueError("common-validation table is not test-set-free")
    for metric, _title, _ylabel, _percent in METRICS:
        for suffix in ("_mean", "_sd"):
            column = f"{metric}{suffix}"
            if column not in table or not np.isfinite(table[column].to_numpy(dtype=float)).all():
                raise ValueError(f"missing/non-finite figure metric: {column}")
    return table


def _panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.04, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def build_common_figure(table: pd.DataFrame):
    """Build the 2x4 quantitative grid; all values come from the source table."""

    plt.rcParams.update({"font.size": 7.2, "axes.linewidth": 0.75, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(2, 4, figsize=(7.09, 4.15), sharex=True)
    x = np.arange(len(EXPECTED_METHODS), dtype=float)
    for index, (metric, title, ylabel, percent) in enumerate(METRICS):
        ax = axes.flat[index]
        y = table[f"{metric}_mean"].to_numpy(dtype=float)
        err = table[f"{metric}_sd"].to_numpy(dtype=float)
        if percent:
            y = y * 100.0
            err = err * 100.0
        ax.axvspan(-0.42, 0.42, color="#FDF1E6", zorder=0)
        for i, method in enumerate(EXPECTED_METHODS):
            ax.errorbar(
                x[i], y[i], yerr=err[i], fmt="o", color=COLORS[method], ecolor=COLORS[method],
                markersize=4.3 if method == "RSC-PF" else 3.8, markeredgecolor="white", markeredgewidth=0.45,
                capsize=2.2, elinewidth=0.9, zorder=3,
            )
        ax.set_title(title, fontsize=8.2, pad=4.0)
        ax.set_ylabel(ylabel, labelpad=2)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.75)
        ax.set_axisbelow(True)
        ax.set_xticks(x, [METHOD_LABELS[m] for m in EXPECTED_METHODS], rotation=28, ha="right")
        for tick in ax.get_xticklabels():
            tick.set_fontsize(6.2)
        ax.tick_params(axis="both", length=2.4, width=0.65, pad=2)
        if percent and metric == "feasibility_rate":
            ax.set_ylim(bottom=0)
        _panel_label(ax, "abcdefgh"[index])
        for tick, method in zip(ax.get_xticklabels(), EXPECTED_METHODS):
            if method == "RSC-PF":
                tick.set_fontweight("bold")
                tick.set_color(COLORS[method])
    fig.suptitle("Common validation comparison of forecasting and dispatch", fontsize=10.5, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.012,
        "Mean ± SD across five seeds; 8,780 validation windows per seed. Lower is better except no-shortage rate. Open-loop four-hour window protocol.",
        ha="center", va="bottom", fontsize=6.2, color="#555555",
    )
    fig.text(
        0.5, 0.032,
        "Exact LP at inference: RSC-PF/iTransformer/DigitalTwins = 0; DecisionFocused = 1 per window.",
        ha="center", va="bottom", fontsize=5.8, color="#555555",
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.96), w_pad=1.15, h_pad=1.1)
    return fig


def render_common_figure(table_path: str | Path, output_root: str | Path) -> dict[str, str]:
    table_path = Path(table_path)
    root = Path(output_root)
    table = load_common_table(table_path)
    for name in ("vector", "raster", "preview", "source_data", "manifests", "qa"):
        (root / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(table_path, root / "source_data" / table_path.name)
    fig = build_common_figure(table)
    stem = "RSC_PF_common_validation_comparison"
    vector_base = root / "vector" / stem
    raster_base = root / "raster" / stem
    preview = root / "preview" / f"{stem}.png"
    fig.savefig(f"{vector_base}.svg", bbox_inches="tight")
    fig.savefig(f"{vector_base}.pdf", bbox_inches="tight")
    fig.savefig(f"{raster_base}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{raster_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(preview, dpi=300, bbox_inches="tight")
    plt.close(fig)
    contract = {
        "core_conclusion": "RSC-PF reduces decision regret and shortage relative to external baselines while trading off pure forecast accuracy.",
        "archetype": "quantitative grid",
        "backend": "python/matplotlib",
        "target_size_mm": {"width": 180, "height": 105},
        "panel_map": {"a": "forecast MAE", "b": "forecast RMSE", "c": "forecast WAPE", "d": "operating cost", "e": "physical carbon", "f": "penalized objective", "g": "decision regret vs Oracle", "h": "no-shortage rate"},
        "statistics": "mean ± SD across five random seeds; no hypothesis-test or p-value claim",
        "source_data": str(root / "source_data" / table_path.name),
        "metric_protocol": "open-loop four-hour validation-window metrics under common validation definitions",
        "reviewer_risks": ["RSC-PF uses zero exact LP calls at inference; DecisionFocused-Online uses one exact LP per window.", "This figure is validation-only and must not be described as sealed-test performance.", "Do not mix these per-window values with closed-loop cumulative v2 totals."],
    }
    contract_path = root / "manifests" / "figure_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    outputs = [root / "vector" / f"{stem}.svg", root / "vector" / f"{stem}.pdf", root / "raster" / f"{stem}.tiff", root / "raster" / f"{stem}.png", preview, root / "source_data" / table_path.name, contract_path]
    manifest = {
        "schema_version": "rsc-pf-common-figure-manifest-v1",
        "figure": stem,
        "status": "complete",
        "backend": "python/matplotlib",
        "source_table": str(table_path),
        "source_table_sha256": _sha256(table_path),
        "test_set_accessed": False,
        "seed_count": 5,
        "methods": list(EXPECTED_METHODS),
        "outputs": [{"path": str(path), "sha256": _sha256(path)} for path in outputs],
    }
    manifest_path = root / "manifests" / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "complete", "figure": str(vector_base), "manifest": str(manifest_path), "preview": str(preview)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation" / "common_validation_summary.csv")
    parser.add_argument("--output-root", type=Path, default=ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation" / "figures" / "rsc_pf_common_comparison_v1")
    args = parser.parse_args(argv)
    print(json.dumps(render_common_figure(args.table, args.output_root), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
