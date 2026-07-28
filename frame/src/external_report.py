"""阶段5.5：外部基线统一结果校验与报告生成。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


MODEL_ORDER = ("dlinear", "mmoe-lite", "softs")
METRICS = ("MAE", "RMSE", "WAPE", "MAPE")
TASK_ORDER = ("electricity", "cooling", "heating")


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON顶层必须是对象：{path}")
    return value


def _write_json(value: Mapping[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def _as_bool(value: object) -> bool:
    return bool(value)


def _run_directory(root: Path, model: str) -> Path:
    return root / model.replace("-", "_")


def load_and_validate_runs(
    root: str | Path,
    protocol: str,
    models: Sequence[str] = MODEL_ORDER,
) -> Dict[str, Dict[str, object]]:
    """读取三个基线结果并校验公平性契约的关键输出。"""

    root_path = Path(root)
    if not models:
        raise ValueError("至少需要一个外部基线结果")
    runs: Dict[str, Dict[str, object]] = {}
    expected_shape_tail = (4, 3)
    expected_sample_counts = None
    expected_window = None
    for model in models:
        run_dir = _run_directory(root_path, model)
        metrics_path = run_dir / "metrics_test.json"
        predictions_path = run_dir / "predictions_test.npz"
        if not metrics_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(
                f"缺少{model}结果，需要{metrics_path.name}和{predictions_path.name}"
            )
        metrics = _read_json(metrics_path)
        if metrics.get("model") != model:
            raise ValueError(
                f"结果目录{run_dir}声明的model为{metrics.get('model')!r}，"
                f"但期望{model!r}"
            )
        if metrics.get("protocol") != protocol:
            raise ValueError(
                f"{model}使用了协议{metrics.get('protocol')!r}，期望{protocol!r}"
            )
        tasks = tuple(metrics.get("tasks", ()))
        if tasks != TASK_ORDER:
            raise ValueError(f"{model}任务顺序错误：{tasks}")
        window = metrics.get("window")
        if not isinstance(window, dict) or (
            int(window.get("lookback", -1)), int(window.get("horizon", -1))
        ) != (24, 4):
            raise ValueError(f"{model}没有遵守24→4窗口：{window}")
        if expected_window is None:
            expected_window = window
        elif window != expected_window:
            raise ValueError("三个基线的窗口设置不一致")

        with np.load(predictions_path, allow_pickle=False) as values:
            prediction_shape = tuple(int(value) for value in values["prediction"].shape)
            target_shape = tuple(int(value) for value in values["target"].shape)
        if prediction_shape != target_shape or prediction_shape[1:] != expected_shape_tail:
            raise ValueError(
                f"{model}预测形状错误：prediction={prediction_shape}, target={target_shape}"
            )
        sample_counts = metrics.get("sample_counts")
        if not isinstance(sample_counts, dict):
            raise ValueError(f"{model}缺少sample_counts")
        if int(sample_counts.get("test", -1)) != prediction_shape[0]:
            raise ValueError(f"{model}测试样本数与预测文件不一致")
        if expected_sample_counts is None:
            expected_sample_counts = sample_counts
        elif sample_counts != expected_sample_counts:
            raise ValueError(
                "三个基线的样本数量不一致；请使用统一编排脚本重新运行，"
                f"当前{model}为{sample_counts}，期望{expected_sample_counts}"
            )

        if _as_bool(metrics.get("future_exogenous_used", True)):
            raise ValueError(f"{model}结果声明使用了未来外生变量")
        runs[model] = {
            "run_dir": str(run_dir),
            "metrics": metrics,
            "prediction_shape": list(prediction_shape),
        }
    return runs


def build_report_rows(
    runs: Mapping[str, Mapping[str, object]],
) -> Dict[str, list[Dict[str, object]]]:
    """将每个模型的JSON指标展开为三类CSV行。"""

    overall_rows: list[Dict[str, object]] = []
    per_task_rows: list[Dict[str, object]] = []
    per_horizon_rows: list[Dict[str, object]] = []
    for model in MODEL_ORDER:
        if model not in runs:
            continue
        metrics = runs[model]["metrics"]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{model}的metrics不是对象")
        config = metrics.get("model_config", {})
        if not isinstance(config, Mapping):
            config = {}
        runtime = metrics.get("runtime_seconds", {})
        if not isinstance(runtime, Mapping):
            runtime = {}
        overall = metrics.get("metrics_original_scale", {}).get(
            "overall_equal_task_mean", {}
        )
        if not isinstance(overall, Mapping):
            raise ValueError(f"{model}缺少总体指标")
        base = {
            "model": model,
            "baseline_name": metrics.get("baseline_name", model),
            "input_mode": metrics.get("input_mode"),
            "exog_used": metrics.get("exog_used"),
            "parameter_count": metrics.get("model_parameter_count"),
            "fit_seconds": runtime.get("fit"),
            "test_evaluation_seconds": runtime.get("test_evaluation"),
            "test_samples_per_second": runtime.get("test_samples_per_second"),
        }
        overall_rows.append(
            {
                **base,
                **{metric: overall.get(metric) for metric in METRICS},
            }
        )

        per_task = metrics.get("metrics_original_scale", {}).get("per_task", {})
        if isinstance(per_task, Mapping):
            for task in TASK_ORDER:
                values = per_task.get(task, {})
                if not isinstance(values, Mapping):
                    continue
                per_task_rows.append(
                    {
                        "model": model,
                        "task": task,
                        **{metric: values.get(metric) for metric in METRICS},
                    }
                )

        per_horizon = metrics.get("metrics_original_scale", {}).get(
            "per_horizon_equal_element_mean", {}
        )
        if isinstance(per_horizon, Mapping):
            for step, values in per_horizon.items():
                if not isinstance(values, Mapping):
                    continue
                per_horizon_rows.append(
                    {
                        "model": model,
                        "horizon_step": step,
                        **{metric: values.get(metric) for metric in METRICS},
                    }
                )

    return {
        "overall": overall_rows,
        "per_task": per_task_rows,
        "per_horizon": per_horizon_rows,
    }


def write_csv(rows: Iterable[Mapping[str, object]], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"不能为零行报告写出CSV：{path}")
    fieldnames = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_unified_report(
    root: str | Path,
    protocol: str,
    runs: Mapping[str, Mapping[str, object]],
) -> Dict[str, object]:
    """写出CSV、Markdown和JSON统一报告，并返回summary对象。"""

    root_path = Path(root)
    rows = build_report_rows(runs)
    write_csv(rows["overall"], root_path / "comparison_overall.csv")
    write_csv(rows["per_task"], root_path / "comparison_per_task.csv")
    write_csv(rows["per_horizon"], root_path / "comparison_per_horizon.csv")

    first_metrics = runs[next(iter(runs))]["metrics"]
    contract = {}
    contract_path = Path(runs[next(iter(runs))]["run_dir"]) / "fairness_contract.json"
    if contract_path.exists():
        contract = _read_json(contract_path)
    overall_rows = rows["overall"]
    markdown_lines = [
        "# 阶段5.5：外部基线统一报告",
        "",
        f"- 协议：`{protocol}`",
        "- 预测协议：24小时历史窗口 → 未来4小时，输出 `[batch, 4, 3]`",
        "- 指标在原始负荷尺度上计算；结果用于基线链路和相对比较，不直接作为正式结论。",
        "",
        "## 总体指标与资源开销",
        "",
        "| 模型 | 输入模式 | 外生变量 | 参数量 | MAE | RMSE | WAPE | MAPE | 训练秒数 | 测试秒数 | 测试样本/秒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        markdown_lines.append(
            "| {model} | {input_mode} | {exog_used} | {parameter_count} | "
            "{MAE:.4f} | {RMSE:.4f} | {WAPE:.4f} | {MAPE:.4f} | "
            "{fit_seconds:.3f} | {test_evaluation_seconds:.3f} | "
            "{test_samples_per_second:.3f} |".format(**row)
        )
    markdown_lines.extend(
        [
            "",
            "## 输入差异与模型关系",
            "",
            "| 模型 | 输入 | 说明 |",
            "|---|---|---|",
            "| DLinear | 历史负荷 | 共享通道线性趋势/季节基线 |",
            "| MMoE-lite | 历史负荷 + 历史气象/日历 | 轻量专家路由基线，不是 Shao 完整模型 |",
            "| SOFTS | 历史负荷 | `minimal_star_adapter`，保留 STAR 核心，不是官方完整复现 |",
            "",
            "## 文件索引",
            "",
            "- `comparison_overall.csv`：总体指标和资源开销；",
            "- `comparison_per_task.csv`：电、冷、热逐任务指标；",
            "- `comparison_per_horizon.csv`：第1—4步指标；",
            "- `summary.json`：协议校验、运行目录和完整结果索引。",
        ]
    )
    (root_path / "fairness_table.md").write_text(
        "\n".join(markdown_lines) + "\n", encoding="utf-8"
    )

    summary: Dict[str, object] = {
        "report_version": "stage5.5",
        "protocol": protocol,
        "tasks": list(TASK_ORDER),
        "window": {"lookback": 24, "horizon": 4},
        "models": list(runs),
        "consistency_checks": {
            "same_protocol": True,
            "same_window": True,
            "same_test_sample_count": True,
            "same_output_tail": True,
            "future_exogenous_used": False,
        },
        "runs": {
            model: {
                "run_dir": value["run_dir"],
                "prediction_shape": value["prediction_shape"],
                "metrics_file": str(Path(value["run_dir"]) / "metrics_test.json"),
            }
            for model, value in runs.items()
        },
        "contract": contract,
        "overall_rows": overall_rows,
    }
    _write_json(summary, root_path / "summary.json")
    return summary
