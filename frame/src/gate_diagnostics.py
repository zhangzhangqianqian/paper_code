"""阶段 6.4：验证集门控行为统计。

本模块只做数值统计，不读取测试集。绘图由同阶段的 Python 脚本负责。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


TASKS: Tuple[str, ...] = ("electricity", "cooling", "heating")
METRIC_NAMES: Tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "p05",
    "median",
    "p95",
)


def ordered_edges(
    model_name: str,
    task_names: Sequence[str] = TASKS,
) -> Tuple[Tuple[int, int], ...]:
    """返回该模型用于熵/恒定比例的有效门控边。"""

    if model_name == "dynamic_symmetric":
        return unordered_pairs(task_names)
    if model_name in {"static_gate", "dynamic_directed", "scheme2r"}:
        return tuple(
            (target, source)
            for target in range(len(task_names))
            for source in range(len(task_names))
            if target != source
        )
    raise ValueError(f"不支持门控诊断的模型：{model_name}")


def unordered_pairs(
    task_names: Sequence[str] = TASKS,
) -> Tuple[Tuple[int, int], ...]:
    return tuple(
        (target, source)
        for target in range(len(task_names))
        for source in range(target + 1, len(task_names))
    )


def load_gate_array(
    path: str | Path,
    task_names: Sequence[str] = TASKS,
) -> np.ndarray:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"找不到验证集门控文件：{source}")
    with np.load(source, allow_pickle=False) as values:
        if "gates" not in values:
            raise ValueError(f"门控文件缺少 gates 数组：{source}")
        gates = np.asarray(values["gates"], dtype=np.float64)
    task_count = len(task_names)
    if task_count <= 1:
        raise ValueError("task_names至少需要两个任务")
    if gates.ndim == 2 and gates.shape == (task_count, task_count):
        gates = gates[None, :, :]
    if gates.ndim == 4:
        if gates.shape[2:] != (task_count, task_count) or gates.shape[1] <= 0:
            raise ValueError(
                "逐预测步门控数组必须是[N,H,task_count,task_count]："
                f"{gates.shape}"
            )
        diagonal = np.diagonal(gates, axis1=2, axis2=3)
    elif gates.ndim == 3 and gates.shape[1:] == (task_count, task_count):
        diagonal = np.diagonal(gates, axis1=1, axis2=2)
    else:
        raise ValueError(
            f"门控数组必须是[N,{task_count},{task_count}]或"
            f"[N,H,{task_count},{task_count}]：{gates.shape}"
        )
    if gates.shape[0] == 0 or not np.isfinite(gates).all():
        raise ValueError("门控数组不能为空且不能包含 NaN/Inf")
    if not np.allclose(diagonal, 0.0, atol=1e-6):
        raise ValueError("门控对角线必须为 0")
    if np.any(gates < -1e-6) or np.any(gates > 1.0 + 1e-6):
        raise ValueError("门控值必须位于 [0,1]")
    return gates


def _distribution_entropy(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped = np.maximum(values, 0.0)
    totals = clipped.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        clipped,
        totals,
        out=np.zeros_like(clipped),
        where=totals > epsilon,
    )
    entropy = -np.sum(
        np.where(probabilities > epsilon, probabilities * np.log(probabilities), 0.0),
        axis=1,
    )
    normalizer = np.log(values.shape[1])
    return np.divide(
        entropy,
        normalizer,
        out=np.zeros_like(entropy),
        where=normalizer > epsilon,
    )


def _summary(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def compute_gate_diagnostics(
    gates: np.ndarray,
    model_name: str,
    candidate_id: str,
    constant_threshold: float = 0.01,
    asymmetry_threshold: float = 0.05,
    task_names: Sequence[str] = TASKS,
) -> Tuple[Dict[str, object], list[Dict[str, object]], Dict[str, np.ndarray]]:
    """计算一个模型/超参数运行的门控统计。"""

    if constant_threshold <= 0 or asymmetry_threshold <= 0:
        raise ValueError("门控诊断阈值必须为正数")
    task_names = tuple(str(name) for name in task_names)
    if len(task_names) <= 1:
        raise ValueError("task_names至少需要两个任务")
    array = np.asarray(gates, dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (len(task_names), len(task_names)):
        raise ValueError(
            f"gates 必须是 [N,{len(task_names)},{len(task_names)}]：{array.shape}"
        )
    edges = ordered_edges(model_name, task_names)
    edge_values = np.stack([array[:, target, source] for target, source in edges], axis=1)
    edge_std = np.std(edge_values, axis=0)
    entropy = _distribution_entropy(edge_values)
    constant_ratio = float(np.mean(edge_std < constant_threshold))

    mean_matrix = np.mean(array, axis=0)
    std_matrix = np.std(array, axis=0)
    asymmetry_matrix = mean_matrix - mean_matrix.T
    asymmetry_values = []
    edge_rows: list[Dict[str, object]] = []
    for target, source in unordered_pairs(task_names):
        forward = array[:, target, source]
        reverse = array[:, source, target]
        difference = forward - reverse
        abs_difference = np.abs(difference)
        asymmetry_values.append(abs_difference)
        edge_rows.append(
            {
                "model": model_name,
                "candidate_id": candidate_id,
                "forecast_step": None,
                "target_task": task_names[target],
                "source_task": task_names[source],
                "mean_target_from_source": float(np.mean(forward)),
                "mean_source_from_target": float(np.mean(reverse)),
                "signed_difference_mean": float(np.mean(difference)),
                "absolute_difference_mean": float(np.mean(abs_difference)),
                "absolute_difference_std": float(np.std(abs_difference)),
                "absolute_difference_p95": float(np.percentile(abs_difference, 95)),
                "asymmetric_fraction": float(
                    np.mean(abs_difference > asymmetry_threshold)
                ),
            }
        )
    all_asymmetry = np.concatenate(asymmetry_values)
    summary: Dict[str, object] = {
        "model": model_name,
        "candidate_id": candidate_id,
        "forecast_step": None,
        "sample_count": int(array.shape[0]),
        "gate_kind": "static" if model_name == "static_gate" else "dynamic",
        "edge_count_for_entropy": len(edges),
        "gate_mean": float(np.mean(edge_values)),
        "gate_std": float(np.std(edge_values)),
        "entropy_mean": float(np.mean(entropy)),
        "entropy_std": float(np.std(entropy)),
        "entropy_p05": float(np.percentile(entropy, 5)),
        "entropy_median": float(np.median(entropy)),
        "entropy_p95": float(np.percentile(entropy, 95)),
        "constant_edge_ratio": constant_ratio,
        "constant_threshold": float(constant_threshold),
        "asymmetry_mean_abs": float(np.mean(all_asymmetry)),
        "asymmetry_p95_abs": float(np.percentile(all_asymmetry, 95)),
        "asymmetry_fraction": float(np.mean(all_asymmetry > asymmetry_threshold)),
        "asymmetry_threshold": float(asymmetry_threshold),
    }
    matrices = {
        "mean": mean_matrix.astype(np.float32),
        "std": std_matrix.astype(np.float32),
        "asymmetry": asymmetry_matrix.astype(np.float32),
        "entropy": entropy.astype(np.float32),
    }
    return summary, edge_rows, matrices


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"不能写入空诊断表：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostic_tables(
    output_root: str | Path,
    summary_rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
) -> None:
    root = Path(output_root)
    _write_csv(summary_rows, root / "gate_diagnostics_summary.csv")
    _write_csv(edge_rows, root / "gate_asymmetry.csv")


def write_json(value: Mapping[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
