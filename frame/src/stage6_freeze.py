"""阶段 6.6：只依据验证结果冻结阶段 7 的实验方案。"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .stage6_contract import Stage6SelectionContract, load_stage6_selection_contract


class FreezeValidationError(ValueError):
    """阶段 6.6 输入结果不满足冻结契约。"""


FULL_TRAIN_SAMPLES = 43767
FULL_VALIDATION_SAMPLES = 8781
SMALL_TRAIN_SAMPLES = 1149
SMALL_VALIDATION_SAMPLES = 165


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FreezeValidationError(f"缺少必需文件：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FreezeValidationError(f"无法读取 JSON：{path}") from exc
    if not isinstance(value, dict):
        raise FreezeValidationError(f"JSON 顶层必须是对象：{path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FreezeValidationError(f"缺少必需文件：{path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise FreezeValidationError(f"无法读取 CSV：{path}") from exc


def _assert_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FreezeValidationError(f"{label}目录不存在：{path}")
    if any("smoke" in part.lower() for part in path.parts):
        raise FreezeValidationError(f"禁止使用冒烟目录作为正式冻结输入：{path}")


def _assert_no_test_artifacts(paths: Iterable[Path]) -> None:
    forbidden_names = {"metrics_test.json", "predictions_test.npz"}
    for root in paths:
        for path in root.rglob("*"):
            if path.is_file() and path.name.lower() in forbidden_names:
                raise FreezeValidationError(f"输入目录包含测试集文件：{path}")


def _assert_false(value: object, label: str) -> None:
    if value is not False:
        raise FreezeValidationError(f"{label}必须为 false，实际为 {value!r}")


def _metric_rows(root: Path) -> list[dict[str, Any]]:
    files = sorted(root.rglob("metrics_validation.json"))
    if not files:
        raise FreezeValidationError(f"未发现验证指标文件：{root}")
    rows = []
    for path in files:
        value = _read_json(path)
        value["_path"] = str(path)
        rows.append(value)
    return rows


def _validate_full_stage6_2(
    root: Path,
    contract: Stage6SelectionContract,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _assert_directory(root, "阶段 6.2")
    manifest = _read_json(root / "stage6_2_manifest.json")
    if manifest.get("stage") != "6.2" or manifest.get("protocol") != "full":
        raise FreezeValidationError("阶段 6.2 清单不是正式全年协议")
    if int(manifest.get("candidate_run_count", -1)) != 24:
        raise FreezeValidationError("阶段 6.2 必须包含 24 个正式运行")
    _assert_false(manifest.get("test_set_accessed"), "阶段 6.2 test_set_accessed")
    counts = manifest.get("sample_counts") or {}
    if counts.get("train") != FULL_TRAIN_SAMPLES or counts.get("validation") != FULL_VALIDATION_SAMPLES:
        raise FreezeValidationError(
            "阶段 6.2 样本数不符合 Kitakyushu 全年协议："
            f"{counts!r}，期望 train={FULL_TRAIN_SAMPLES}, "
            f"validation={FULL_VALIDATION_SAMPLES}"
        )
    rows = _metric_rows(root)
    if len(rows) != 24:
        raise FreezeValidationError(f"阶段 6.2 验证指标文件数应为 24，实际为 {len(rows)}")
    for row in rows:
        if row.get("stage") != "6.2" or row.get("protocol") != "full":
            raise FreezeValidationError(f"阶段 6.2 存在非正式运行：{row.get('_path')}")
        _assert_false(row.get("test_set_accessed"), f"{row.get('_path')} test_set_accessed")
        trainer = row.get("trainer_config") or {}
        if trainer.get("max_epochs") != 100 or trainer.get("early_stopping_patience") != 12:
            raise FreezeValidationError(f"阶段 6.2 存在冒烟训练限制：{row.get('_path')}")
        sample_counts = row.get("sample_counts") or {}
        if sample_counts.get("train") != FULL_TRAIN_SAMPLES or sample_counts.get("validation") != FULL_VALIDATION_SAMPLES:
            raise FreezeValidationError(f"阶段 6.2 单次运行样本数异常：{row.get('_path')}")
        if not isinstance(row.get("best_checkpoint") or row.get("best_checkpoints"), dict):
            raise FreezeValidationError(f"阶段 6.2 缺少最佳验证检查点：{row.get('_path')}")
    expected_pairs = {
        (model, str(candidate["candidate_id"]))
        for model in contract.candidate_models
        for candidate in contract.hyperparameter_candidates
    }
    actual_pairs = {(str(row.get("model")), str(row.get("candidate_id"))) for row in rows}
    if actual_pairs != expected_pairs:
        raise FreezeValidationError(
            f"阶段 6.2 候选集合异常：{sorted(actual_pairs)!r}，期望 {sorted(expected_pairs)!r}"
        )
    return manifest, rows


def _validate_small_stage6_3(
    root: Path,
    expected_pairs: set[tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _assert_directory(root, "阶段 6.3")
    manifest = _read_json(root / "stage6_3_manifest.json")
    if manifest.get("stage") != "6.3" or manifest.get("protocol") != "small_sample":
        raise FreezeValidationError("阶段 6.3 清单不是正式小样本协议")
    if int(manifest.get("candidate_run_count", -1)) != 2:
        raise FreezeValidationError("阶段 6.3 必须包含 2 个正式运行")
    _assert_false(manifest.get("test_set_accessed"), "阶段 6.3 test_set_accessed")
    counts = manifest.get("sample_counts") or {}
    if counts.get("train") != SMALL_TRAIN_SAMPLES or counts.get("validation") != SMALL_VALIDATION_SAMPLES:
        raise FreezeValidationError(f"阶段 6.3 样本数异常：{counts!r}")
    actual_models = {
        (str(row.get("model")), str(row.get("candidate_id")))
        for row in _metric_rows(root)
    }
    if actual_models != expected_pairs:
        raise FreezeValidationError(
            f"阶段 6.3 候选集合异常：{sorted(actual_models)!r}，期望 {sorted(expected_pairs)!r}"
        )
    stability = _read_json(root / "validation_stability_summary.json")
    _assert_false(stability.get("test_set_accessed"), "阶段 6.3 稳定性分析 test_set_accessed")
    return manifest, stability


def _validate_stage6_4(root: Path) -> dict[str, Any]:
    _assert_directory(root, "阶段 6.4")
    manifest = _read_json(root / "stage6_4_manifest.json")
    if manifest.get("stage") != "6.4":
        raise FreezeValidationError("阶段 6.4 清单版本错误")
    if int(manifest.get("analyzed_run_count", -1)) != 28:
        raise FreezeValidationError("阶段 6.4 必须分析 28 个门控运行")
    if manifest.get("missing_runs") not in ([], None):
        raise FreezeValidationError(f"阶段 6.4 存在缺失运行：{manifest.get('missing_runs')!r}")
    _assert_false(manifest.get("test_set_accessed"), "阶段 6.4 test_set_accessed")
    if not (root / "gate_diagnostics_summary.csv").exists() or not (root / "gate_asymmetry.csv").exists():
        raise FreezeValidationError("阶段 6.4 缺少门控诊断 CSV")
    return manifest


def _validate_stage6_5(root: Path) -> dict[str, Any]:
    _assert_directory(root, "阶段 6.5")
    manifest = _read_json(root / "stage6_5_manifest.json")
    if manifest.get("stage") != "6.5" or manifest.get("protocol") != "full":
        raise FreezeValidationError("阶段 6.5 清单不是正式全年协议")
    if int(manifest.get("gain_row_count", -1)) != 1920 or int(manifest.get("summary_row_count", -1)) != 240:
        raise FreezeValidationError("阶段 6.5 汇总行数不符合正式全年分析")
    if manifest.get("error_metric_for_significance") != "MAE":
        raise FreezeValidationError("阶段 6.5 冻结要求使用 MAE 进行显著性判断")
    _assert_false(manifest.get("test_set_accessed"), "阶段 6.5 test_set_accessed")
    if not (root / "transfer_gains.csv").exists() or not (root / "negative_transfer_summary.csv").exists():
        raise FreezeValidationError("阶段 6.5 缺少迁移分析 CSV")
    return manifest


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _find_overall_row(rows: Sequence[Mapping[str, Any]], model: str, candidate: str) -> dict[str, Any]:
    for row in rows:
        if str(row.get("model")) == model and str(row.get("candidate_id")) == candidate:
            return dict(row)
    raise FreezeValidationError(f"阶段 6.2 找不到 {model}-{candidate} 的验证结果")


def _candidate_config(contract: Stage6SelectionContract, candidate_id: str) -> dict[str, Any]:
    for candidate in contract.hyperparameter_candidates:
        if str(candidate.get("candidate_id")) == candidate_id:
            return dict(candidate)
    raise FreezeValidationError(f"阶段 6.1 契约中找不到候选配置：{candidate_id}")


def _finite_float(row: Mapping[str, Any], field: str, label: str) -> float:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError) as exc:
        raise FreezeValidationError(f"{label}缺少可解析的{field}") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise FreezeValidationError(f"{label}的{field}不是有限值")
    return value


def _rank_full_candidates(
    rows: Sequence[Mapping[str, Any]],
    contract: Stage6SelectionContract | None = None,
) -> list[dict[str, Any]]:
    """按契约固定的全年验证集规则生成确定性排名。"""

    ranked: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        model = str(row.get("model", ""))
        candidate = str(row.get("candidate_id", ""))
        key = (model, candidate)
        if not model or not candidate or key in seen:
            raise FreezeValidationError(f"阶段 6.2 存在重复或空候选：{key!r}")
        seen.add(key)
        ranked.append(
            {
                **dict(row),
                "_wape": _finite_float(row, "WAPE", f"{model}-{candidate}"),
                "_rmse": _finite_float(row, "RMSE", f"{model}-{candidate}"),
                "_parameter_count": _finite_float(row, "parameter_count", f"{model}-{candidate}"),
                "_fit_seconds": _finite_float(row, "fit_seconds", f"{model}-{candidate}"),
            }
        )
    tolerance = 0.0
    if contract is not None:
        tolerance = float(
            contract.raw.get("selection_rule", {}).get(
                "tie_tolerance_percentage_points", 0.0
            )
        )
    best_wape = min(row["_wape"] for row in ranked)

    def tie_key(row: Mapping[str, Any]) -> tuple[float, float, float, float, str, str]:
        def numeric_or_inf(value: Any) -> float:
            try:
                return float(value) if value is not None else float("inf")
            except (TypeError, ValueError):
                return float("inf")

        return (
            numeric_or_inf(row.get("validation_max_per_task_WAPE")),
            numeric_or_inf(row.get("validation_negative_transfer_rate")),
            row["_parameter_count"],
            row["_fit_seconds"],
            str(row["model"]),
            str(row["candidate_id"]),
        )

    tied = [row for row in ranked if row["_wape"] - best_wape <= tolerance]
    remaining = [row for row in ranked if row not in tied]
    tied.sort(key=tie_key)
    remaining.sort(key=lambda row: (row["_wape"], tie_key(row)))
    ranked = tied + remaining
    for index, row in enumerate(ranked, start=1):
        row["selection_rank"] = index
        for key in ("_wape", "_rmse", "_parameter_count", "_fit_seconds"):
            row.pop(key, None)
    return ranked


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise FreezeValidationError(f"不能写入空冻结检查表：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def freeze_stage6(
    output_root: str | Path,
    stage6_2_root: str | Path,
    stage6_3_root: str | Path,
    stage6_4_root: str | Path,
    stage6_5_root: str | Path,
    contract: Stage6SelectionContract | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """验证阶段 6 结果并生成阶段 7 冻结配置。"""

    contract = contract or load_stage6_selection_contract()
    if contract.raw.get("dataset") != "kitakyushu_energy_station":
        raise FreezeValidationError("阶段 6.6 当前只接受 Kitakyushu 数据协议")
    full_root = Path(stage6_2_root)
    small_root = Path(stage6_3_root)
    gate_root = Path(stage6_4_root)
    transfer_root = Path(stage6_5_root)
    _assert_no_test_artifacts((full_root, small_root, gate_root, transfer_root))
    full_manifest, full_metrics = _validate_full_stage6_2(full_root, contract)
    comparison_rows = _read_csv(full_root / "validation_model_comparison.csv")
    ranked_candidates = _rank_full_candidates(comparison_rows, contract)
    if len(ranked_candidates) < 2:
        raise FreezeValidationError("阶段 6.2 至少需要两个合法候选才能冻结主模型和比较模型")
    primary_row = ranked_candidates[0]
    comparison_row = ranked_candidates[1]
    selected_pairs = {
        (str(primary_row["model"]), str(primary_row["candidate_id"])),
        (str(comparison_row["model"]), str(comparison_row["candidate_id"])),
    }
    small_manifest, small_stability = _validate_small_stage6_3(small_root, selected_pairs)
    gate_manifest = _validate_stage6_4(gate_root)
    transfer_manifest = _validate_stage6_5(transfer_root)
    primary_model = str(primary_row["model"])
    primary_candidate = str(primary_row["candidate_id"])
    comparison_model = str(comparison_row["model"])
    comparison_candidate = str(comparison_row["candidate_id"])

    output = Path(output_root)
    if output.exists() and any(output.iterdir()) and not force:
        raise FreezeValidationError(f"冻结输出目录非空；如需明确覆盖请使用 --force：{output}")
    output.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    config = {
        "freeze_version": "stage6.6",
        "freeze_status": "frozen_for_stage7",
        "frozen_at_utc": now,
        "dataset": contract.raw["dataset"],
        "data_protocol": {
            "years": [2015, 2021],
            "tasks": list(contract.raw["tasks"]),
            "task_order": list(contract.raw["tasks"]),
            "sampling": contract.raw["data_protocol"]["sampling"],
            "lookback": contract.raw["data_protocol"]["lookback"],
            "horizon": contract.raw["data_protocol"]["horizon"],
            "full": contract.raw["data_protocol"]["full"],
            "small_sample": contract.raw["data_protocol"]["small_sample"],
            "standardization": contract.raw["input_output"]["standardization"],
            "future_exogenous_allowed": contract.raw["input_output"]["future_exogenous_allowed"],
        },
        "primary_model": {
            "model": primary_model,
            "candidate_id": primary_candidate,
            "hyperparameters": _candidate_config(contract, primary_candidate),
            "full_validation_metrics": primary_row,
        },
        "comparison_model": {
            "model": comparison_model,
            "candidate_id": comparison_candidate,
            "hyperparameters": _candidate_config(contract, comparison_candidate),
            "full_validation_metrics": comparison_row,
        },
        "training_policy": {
            **dict(contract.raw["training_policy"]),
            "formal_random_seeds": [2026, 2027, 2028, 2029, 2030],
            "resolved_full": dict(contract.training_policy("full")),
            "resolved_small_sample": dict(contract.training_policy("small_sample")),
        },
        "selection_ranking": ranked_candidates,
        "selected_pairs": [
            {"model": primary_model, "candidate_id": primary_candidate, "role": "primary"},
            {"model": comparison_model, "candidate_id": comparison_candidate, "role": "comparison"},
        ],
        "stage7_scope": {
            "protocols": ["full", "small_sample"],
            "ablations": ["A0", "A1", "A2", "A3", "A4"],
            "external_baselines": ["persistence", "seasonal_naive", "dlinear", "mmoe_lite", "softs"],
            "test_usage": "allowed_only_after_stage6_6_freeze",
        },
        "test_set_policy": {
            "status": "sealed_until_stage7",
            "test_metrics_may_be_read": False,
            "test_predictions_may_be_read": False,
        },
        "source_results": {
            "stage6_2": str(full_root),
            "stage6_3": str(small_root),
            "stage6_4": str(gate_root),
            "stage6_5": str(transfer_root),
        },
        "repository_revision": _git_revision(),
    }

    validation_rows = [
        {"source": "stage6.2", "check": "formal_run_count", "value": 24, "status": "pass"},
        {"source": "stage6.2", "check": "train_validation_samples", "value": f"{FULL_TRAIN_SAMPLES}/{FULL_VALIDATION_SAMPLES}", "status": "pass"},
        {"source": "stage6.2", "check": "primary_model", "value": f"{primary_model}-{primary_candidate}", "status": "pass"},
        {"source": "stage6.2", "check": "primary_WAPE", "value": primary_row.get("WAPE"), "status": "pass"},
        {"source": "stage6.2", "check": "comparison_model", "value": f"{comparison_model}-{comparison_candidate}", "status": "pass"},
        {"source": "stage6.3", "check": "formal_run_count", "value": 2, "status": "pass"},
        {"source": "stage6.3", "check": "rank_order_changed", "value": small_stability.get("rank_order_changed"), "status": "recorded"},
        {"source": "stage6.4", "check": "analyzed_run_count", "value": gate_manifest.get("analyzed_run_count"), "status": "pass"},
        {"source": "stage6.5", "check": "gain_row_count", "value": transfer_manifest.get("gain_row_count"), "status": "pass"},
        {"source": "stage6.5", "check": "summary_row_count", "value": transfer_manifest.get("summary_row_count"), "status": "pass"},
        {"source": "all", "check": "test_set_accessed", "value": False, "status": "pass"},
    ]
    record = {
        "freeze_version": "stage6.6",
        "freeze_status": "frozen_for_stage7",
        "frozen_at_utc": now,
        "decision": {
            "primary_model": f"{primary_model}-{primary_candidate}",
            "comparison_model": f"{comparison_model}-{comparison_candidate}",
            "reason": "Candidates were ranked by full-validation overall WAPE, then RMSE, parameter count, fit time, model name, and candidate ID as deterministic tie-breakers.",
            "small_sample_role": "robustness_only; the observed rank reversal does not override full-validation selection.",
        },
        "validation_evidence": {
            "stage6_2_primary": primary_row,
            "stage6_2_comparison": comparison_row,
            "stage6_2_ranking": ranked_candidates,
            "stage6_3_stability": small_stability,
            "stage6_4_manifest": gate_manifest,
            "stage6_5_manifest": transfer_manifest,
        },
        "limitations": [
            "The summer small-sample interval contains near-zero heating demand, so heating WAPE/MAPE are unstable; MAE/RMSE are preferred for that robustness interpretation.",
            "Validation evidence does not establish final test-set generalization until Stage 7.",
        ],
        "test_set_accessed": False,
        "source_results": config["source_results"],
        "repository_revision": config["repository_revision"],
    }
    _write_json(output / "stage6_selected_config.json", config)
    _write_json(output / "stage6_6_freeze_record.json", record)
    _write_csv(output / "stage6_6_validation_summary.csv", validation_rows)
    return {
        "stage": "6.6",
        "status": "frozen_for_stage7",
        "output_dir": str(output),
        "primary_model": f"{primary_model}-{primary_candidate}",
        "comparison_model": f"{comparison_model}-{comparison_candidate}",
        "test_set_accessed": False,
    }
