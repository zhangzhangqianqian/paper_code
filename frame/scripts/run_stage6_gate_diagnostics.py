"""阶段 6.4：验证集门控行为诊断与热力图生成。

输入应来自阶段 6.2 或 6.3 的验证结果目录，例如：

    D:\\anaconda\\envs\\pytorch\\python.exe frame\\scripts\\run_stage6_gate_diagnostics.py `
        --input-dir frame\\reports\\stage6_3\\kitakyushu\\small_sample `
        --output-dir frame\\reports\\stage6_4\\kitakyushu\\small_sample

输出 SVG 为主，同时提供 PDF/PNG 预览。所有图表仅使用验证集门控。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gate_diagnostics import (  # noqa: E402
    TASKS,
    compute_gate_diagnostics,
    load_gate_array,
    write_diagnostic_tables,
    write_json,
)
from src.kitakyushu_pipeline import KITAKYUSHU_TASKS  # noqa: E402


TASK_LABELS_BY_DATASET = {
    "heew_total": ("Electricity", "Cooling", "Heating"),
    "kitakyushu_energy_station": ("Electricity", "Cooling", "Heating", "Gas"),
}
GATED_MODELS = (
    "static_gate",
    "dynamic_symmetric",
    "dynamic_directed",
    "scheme2r",
)


def _configure_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _save_figure(fig: plt.Figure, base_path: Path, export_tiff: bool = False) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    if export_tiff:
        fig.savefig(f"{base_path}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def _plot_matrix(
    matrix: np.ndarray,
    title: str,
    base_path: Path,
    cmap: str,
    vmin: float,
    vmax: float,
    colorbar_label: str,
    task_labels: Tuple[str, ...],
    export_tiff: bool = False,
) -> None:
    _configure_publication_style()
    display = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(display, np.nan)
    masked = np.ma.masked_invalid(display)
    fig, ax = plt.subplots(figsize=(3.0, 2.65), constrained_layout=True)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(task_labels)), task_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(task_labels)), task_labels)
    ax.set_xlabel("Source task")
    ax.set_ylabel("Target task")
    ax.set_title(title, pad=8)
    for row in range(len(task_labels)):
        for column in range(len(task_labels)):
            if row == column:
                ax.text(column, row, "—", ha="center", va="center", color="#767676")
            else:
                value = float(matrix[row, column])
                normalized = (value - vmin) / max(vmax - vmin, 1e-9)
                color = "white" if normalized > 0.58 else "#272727"
                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    _save_figure(fig, base_path, export_tiff=export_tiff)


def _plot_run(
    model_name: str,
    candidate_id: str,
    matrices: Dict[str, np.ndarray],
    output_root: Path,
    task_labels: Tuple[str, ...],
    export_tiff: bool = False,
) -> Tuple[str, str]:
    stem = _safe_name(f"{model_name}_{candidate_id}")
    figure_root = output_root / "figures"
    mean_base = figure_root / f"{stem}_gate_mean"
    asymmetry_base = figure_root / f"{stem}_gate_asymmetry"
    _plot_matrix(
        matrices["mean"],
        f"{model_name} / {candidate_id}: mean validation gate",
        mean_base,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="Gate value",
        task_labels=task_labels,
        export_tiff=export_tiff,
    )
    asymmetry_limit = max(float(np.max(np.abs(matrices["asymmetry"]))), 0.05)
    _plot_matrix(
        matrices["asymmetry"],
        f"{model_name} / {candidate_id}: directional asymmetry",
        asymmetry_base,
        cmap="RdBu_r",
        vmin=-asymmetry_limit,
        vmax=asymmetry_limit,
        colorbar_label="G(i,j) − G(j,i)",
        task_labels=task_labels,
        export_tiff=export_tiff,
    )
    return f"figures/{mean_base.name}.svg", f"figures/{asymmetry_base.name}.svg"


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行阶段6.4验证集门控诊断")
    parser.add_argument(
        "--input-dir",
        default="frame/reports/stage6_3/kitakyushu/small_sample",
        help="阶段6.2或6.3验证结果目录",
    )
    parser.add_argument(
        "--output-dir",
        default="frame/reports/stage6_4/kitakyushu/small_sample",
        help="阶段6.4输出目录",
    )
    parser.add_argument(
        "--constant-threshold",
        type=float,
        default=0.01,
        help="判断门控边近似恒定的跨样本标准差阈值",
    )
    parser.add_argument(
        "--asymmetry-threshold",
        type=float,
        default=0.05,
        help="判断有向差异明显的绝对差值阈值",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="允许输入目录缺少某些门控模型；正式实验不建议使用",
    )
    parser.add_argument(
        "--export-tiff",
        action="store_true",
        help="额外导出600 dpi TIFF；默认仅输出SVG、PDF和PNG以控制目录大小",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(TASK_LABELS_BY_DATASET),
        default="kitakyushu_energy_station",
        help="用于确定任务数量和热力图标签的数据协议",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.constant_threshold <= 0 or args.asymmetry_threshold <= 0:
        raise ValueError("两个诊断阈值必须为正数")
    input_root = _resolve_path(args.input_dir)
    output_root = _resolve_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    task_names = (
        TASKS
        if args.dataset == "heew_total"
        else KITAKYUSHU_TASKS
    )
    task_labels = TASK_LABELS_BY_DATASET[args.dataset]

    summary_rows = []
    edge_rows = []
    figure_files = []
    analyzed_runs = []
    missing_runs = []
    for model_name in GATED_MODELS:
        for candidate_id in ("H1", "H2", "H3", "H4"):
            run_dir = input_root / "runs" / model_name / candidate_id
            gate_path = run_dir / "gate_matrix_validation.npz"
            if not gate_path.exists():
                missing_runs.append(f"{model_name}/{candidate_id}")
                continue
            gates = load_gate_array(gate_path, task_names=task_names)
            if gates.ndim == 4:
                step_gates = [
                    (step_index, gates[:, step_index, :, :])
                    for step_index in range(gates.shape[1])
                ]
            else:
                step_gates = [(None, gates)]
            for forecast_step, selected_gates in step_gates:
                diagnostic_candidate = (
                    candidate_id
                    if forecast_step is None
                    else f"{candidate_id}_step{forecast_step + 1}"
                )
                summary, edges, matrices = compute_gate_diagnostics(
                    selected_gates,
                    model_name=model_name,
                    candidate_id=diagnostic_candidate,
                    constant_threshold=args.constant_threshold,
                    asymmetry_threshold=args.asymmetry_threshold,
                    task_names=task_names,
                )
                if forecast_step is not None:
                    summary["forecast_step"] = forecast_step + 1
                    for edge in edges:
                        edge["forecast_step"] = forecast_step + 1
                summary_rows.append(summary)
                edge_rows.extend(edges)
                matrix_path = (
                    output_root
                    / "matrices"
                    / f"{model_name}_{diagnostic_candidate}.npz"
                )
                matrix_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(matrix_path, **matrices)
                mean_figure, asymmetry_figure = _plot_run(
                    model_name,
                    diagnostic_candidate,
                    matrices,
                    output_root,
                    task_labels,
                    export_tiff=args.export_tiff,
                )
                figure_files.extend([mean_figure, asymmetry_figure])
                analyzed_runs.append(
                    {
                        "model": model_name,
                        "candidate_id": diagnostic_candidate,
                        "forecast_step": (
                            None if forecast_step is None else forecast_step + 1
                        ),
                        "sample_count": int(selected_gates.shape[0]),
                        "gate_file": str(gate_path),
                    }
                )

    if missing_runs and not args.allow_missing:
        raise FileNotFoundError(
            "缺少门控验证文件：" + ", ".join(missing_runs) +
            "。请先重新运行阶段6.2/6.3，或明确使用 --allow-missing。"
        )
    if not summary_rows:
        raise ValueError("没有找到可诊断的验证集门控文件")

    write_diagnostic_tables(output_root, summary_rows, edge_rows)
    manifest = {
        "stage": "6.4",
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "dataset": args.dataset,
        "tasks": list(task_names),
        "models": list(GATED_MODELS),
        "analyzed_run_count": len(analyzed_runs),
        "analyzed_runs": analyzed_runs,
        "missing_runs": missing_runs,
        "constant_threshold": float(args.constant_threshold),
        "asymmetry_threshold": float(args.asymmetry_threshold),
        "export_tiff": bool(args.export_tiff),
        "entropy_definition": "normalized Shannon entropy over valid off-diagonal gates",
        "test_set_accessed": False,
        "figure_files": figure_files,
    }
    write_json(manifest, output_root / "stage6_4_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
