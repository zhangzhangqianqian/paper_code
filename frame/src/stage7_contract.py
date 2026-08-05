"""阶段 7.0：验证阶段 6.6 冻结配置并生成正式运行契约。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .kitakyushu_pipeline import KITAKYUSHU_SMALL_SAMPLE_SPLIT, KITAKYUSHU_SPLIT, KITAKYUSHU_TASKS


class Stage7ContractError(ValueError):
    """冻结配置不满足阶段 7 正式运行要求。"""


EXPECTED_SEEDS = (2026, 2027, 2028, 2029, 2030)
EXPECTED_ABLATIONS = ("A0", "A1", "A2", "A3", "A4")
EXPECTED_EXTERNAL_BASELINES = (
    "persistence",
    "seasonal_naive",
    "dlinear",
    "mmoe_lite",
    "softs",
)
EXPECTED_INPUT_CONTROLS = ("scheme2r_loads_only",)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Stage7ContractError(f"找不到阶段 6.6 冻结配置：{path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage7ContractError(f"无法读取冻结配置：{path}") from exc
    if not isinstance(value, dict):
        raise Stage7ContractError("冻结配置顶层必须是 JSON 对象")
    return value


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise Stage7ContractError(f"{label}不符合冻结契约：实际={actual!r}，期望={expected!r}")


def _validate_split(actual: object, expected: object, label: str) -> None:
    if not isinstance(actual, Mapping):
        raise Stage7ContractError(f"{label}必须是对象")
    expected_values = {
        "train_start": expected.train_start,
        "train_end": expected.train_end,
        "validation_start": expected.validation_start,
        "validation_end": expected.validation_end,
        "test_start": expected.test_start,
        "test_end": expected.test_end,
    }
    _require_equal(dict(actual), expected_values, label)


def _validate_source_paths(source_results: object) -> None:
    if not isinstance(source_results, Mapping):
        raise Stage7ContractError("source_results必须是对象")
    required = ("stage6_2", "stage6_3", "stage6_4", "stage6_5")
    for key in required:
        value = source_results.get(key)
        if not isinstance(value, str) or not value:
            raise Stage7ContractError(f"缺少冻结结果来源：{key}")
        if "smoke" in value.lower():
            raise Stage7ContractError(f"阶段 7 禁止引用冒烟结果：{value}")


def validate_stage6_freeze_config(config: Mapping[str, Any]) -> None:
    """验证阶段 6.6 冻结配置，不读取数据文件。"""

    _require_equal(config.get("freeze_version"), "stage6.6", "freeze_version")
    _require_equal(config.get("freeze_status"), "frozen_for_stage7", "freeze_status")
    _require_equal(config.get("dataset"), "kitakyushu_energy_station", "dataset")

    protocol = config.get("data_protocol")
    if not isinstance(protocol, Mapping):
        raise Stage7ContractError("data_protocol必须是对象")
    _require_equal(tuple(protocol.get("tasks", ())), tuple(KITAKYUSHU_TASKS), "任务顺序")
    _require_equal(tuple(protocol.get("task_order", ())), tuple(KITAKYUSHU_TASKS), "task_order")
    _require_equal(protocol.get("sampling"), "1h", "sampling")
    _require_equal(protocol.get("lookback"), 24, "lookback")
    _require_equal(protocol.get("horizon"), 4, "horizon")
    _require_equal(protocol.get("standardization"), "train_split_only_zscore", "standardization")
    _require_equal(protocol.get("future_exogenous_allowed"), False, "future_exogenous_allowed")
    _validate_split(protocol.get("full"), KITAKYUSHU_SPLIT, "full split")
    _validate_split(protocol.get("small_sample"), KITAKYUSHU_SMALL_SAMPLE_SPLIT, "small_sample split")

    primary = config.get("primary_model")
    comparison = config.get("comparison_model")
    if not isinstance(primary, Mapping) or not isinstance(comparison, Mapping):
        raise Stage7ContractError("primary_model和comparison_model必须同时存在")
    if not isinstance(primary.get("model"), str) or not primary.get("model"):
        raise Stage7ContractError("primary_model.model必须是非空字符串")
    if not isinstance(comparison.get("model"), str) or not comparison.get("model"):
        raise Stage7ContractError("comparison_model.model必须是非空字符串")
    if (primary.get("model"), primary.get("candidate_id")) == (
        comparison.get("model"), comparison.get("candidate_id")
    ):
        raise Stage7ContractError("primary_model和comparison_model不能是同一候选")
    ablation_reference = config.get("scheme2r_ablation_reference")
    if not isinstance(ablation_reference, Mapping) or ablation_reference.get("model") != "scheme2r":
        raise Stage7ContractError("缺少 Scheme2R 消融参照")
    if not isinstance(ablation_reference.get("hyperparameters"), Mapping):
        raise Stage7ContractError("Scheme2R 消融参照缺少超参数")
    for label, value in (("primary_model.hyperparameters", primary.get("hyperparameters")), ("comparison_model.hyperparameters", comparison.get("hyperparameters"))):
        if not isinstance(value, Mapping) or value.get("candidate_id") is None:
            raise Stage7ContractError(f"{label}缺少有效候选配置")

    training = config.get("training_policy")
    if not isinstance(training, Mapping):
        raise Stage7ContractError("training_policy必须是对象")
    _require_equal(tuple(training.get("formal_random_seeds", ())), EXPECTED_SEEDS, "formal_random_seeds")
    resolved_full = training.get("resolved_full")
    effective_full = resolved_full if isinstance(resolved_full, Mapping) else training
    _require_equal(effective_full.get("device"), "cpu", "training_policy.full.device")
    _require_equal(effective_full.get("loss"), "SmoothL1Loss", "training_policy.full.loss")
    _require_equal(effective_full.get("optimizer"), "AdamW", "training_policy.full.optimizer")
    _require_equal(effective_full.get("max_epochs"), 100, "training_policy.full.max_epochs")
    _require_equal(
        effective_full.get("early_stopping_patience"),
        12,
        "training_policy.full.early_stopping_patience",
    )

    scope = config.get("stage7_scope")
    if not isinstance(scope, Mapping):
        raise Stage7ContractError("stage7_scope必须是对象")
    _require_equal(tuple(scope.get("protocols", ())), ("full", "small_sample"), "stage7_scope.protocols")
    _require_equal(tuple(scope.get("ablations", ())), EXPECTED_ABLATIONS, "stage7_scope.ablations")
    _require_equal(tuple(scope.get("external_baselines", ())), EXPECTED_EXTERNAL_BASELINES, "stage7_scope.external_baselines")
    _require_equal(tuple(scope.get("input_controls", ())), EXPECTED_INPUT_CONTROLS, "stage7_scope.input_controls")
    _require_equal(scope.get("test_usage"), "allowed_only_after_stage6_6_freeze", "stage7_scope.test_usage")

    policy = config.get("test_set_policy")
    if not isinstance(policy, Mapping):
        raise Stage7ContractError("test_set_policy必须是对象")
    _require_equal(policy.get("status"), "sealed_until_stage7", "test_set_policy.status")
    _require_equal(policy.get("test_metrics_may_be_read"), False, "test_set_policy.test_metrics_may_be_read")
    _require_equal(policy.get("test_predictions_may_be_read"), False, "test_set_policy.test_predictions_may_be_read")
    _validate_source_paths(config.get("source_results"))


def build_stage7_contract(config: Mapping[str, Any], freeze_path: str | Path) -> dict[str, Any]:
    """根据已验证的冻结配置生成阶段 7.0 运行契约。"""

    validate_stage6_freeze_config(config)
    protocol = config["data_protocol"]
    training = config["training_policy"]
    resolved_full = training.get("resolved_full")
    resolved_small = training.get("resolved_small_sample")
    if not isinstance(resolved_full, Mapping) or not isinstance(resolved_small, Mapping):
        shared = training.get("shared")
        full = training.get("full")
        small = training.get("small_sample")
        if all(isinstance(value, Mapping) for value in (shared, full, small)):
            resolved_full = {**dict(shared), **dict(full)}
            resolved_small = {**dict(shared), **dict(small)}
        elif all(key in training for key in ("device", "loss", "optimizer", "weight_decay", "gradient_clip_norm")):
            # Legacy freeze files are accepted only as a compatibility path;
            # new Stage 6-R freezes always contain the resolved policies.
            resolved_full = dict(training)
            resolved_small = dict(training)
            resolved_small.update({"batch_size": 32, "max_epochs": 200, "early_stopping_patience": 20})
        else:
            raise Stage7ContractError("冻结配置缺少已解析的全年/小样本训练策略")
    full_policy = dict(resolved_full)
    full_policy["formal_random_seeds"] = list(training["formal_random_seeds"])
    return {
        "contract_version": "stage7.0",
        "contract_status": "ready_for_stage7_smoke",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_source": str(Path(freeze_path)),
        "dataset": config["dataset"],
        "tasks": list(protocol["task_order"]),
        "data_protocol": {
            "sampling": protocol["sampling"],
            "lookback": protocol["lookback"],
            "horizon": protocol["horizon"],
            "full": protocol["full"],
            "small_sample": protocol["small_sample"],
            "standardization": protocol["standardization"],
            "future_exogenous_allowed": protocol["future_exogenous_allowed"],
        },
        "models": {
            "primary": config["primary_model"],
            "comparison": config["comparison_model"],
            "scheme2r_ablation_reference": config["scheme2r_ablation_reference"],
        },
        "training_policy": full_policy,
        "protocol_training_policies": {
            "full": dict(resolved_full),
            "small_sample": dict(resolved_small),
        },
        "stage7_scope": config["stage7_scope"],
        "test_set_policy": {
            "selection_locked": True,
            "test_reading_allowed_in_stage7": True,
            "test_reading_before_stage7": False,
            "test_year": 2021,
        },
        "execution_order": [
            "stage7.0_contract_validation",
            "stage7.1_ablation_interface",
            "stage7.2_smoke",
            "stage7.3_main_and_comparison_formal",
            "stage7.4_ablation_formal",
            "stage7.5_external_baselines_formal",
            "stage7.6_report_and_acceptance",
        ],
    }


def write_stage7_contract(
    freeze_path: str | Path,
    contract_path: str | Path,
    report_path: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """读取冻结配置、生成契约和阶段 7.0 验收报告。"""

    freeze_file = Path(freeze_path)
    contract_file = Path(contract_path)
    report_file = Path(report_path)
    config = _read_json(freeze_file)
    contract = build_stage7_contract(config, freeze_file)
    if contract_file.exists() and not force:
        raise Stage7ContractError(f"阶段 7 契约已存在；如需覆盖请使用 --force：{contract_file}")
    contract_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with contract_file.open("w", encoding="utf-8") as handle:
        json.dump(contract, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    report = {
        "stage": "7.0",
        "status": "ready_for_stage7_smoke",
        "freeze_status": config.get("freeze_status"),
        "contract_file": str(contract_file),
        "freeze_file": str(freeze_file),
        "dataset": config.get("dataset"),
        "task_order": config["data_protocol"]["task_order"],
        "lookback": config["data_protocol"]["lookback"],
        "horizon": config["data_protocol"]["horizon"],
        "formal_random_seeds": config["training_policy"]["formal_random_seeds"],
        "test_set_accessed": False,
        "test_reading_before_stage7": False,
        "checks": [
            {"check": "freeze_status", "status": "pass"},
            {"check": "dataset_and_task_order", "status": "pass"},
            {"check": "full_and_small_sample_protocol", "status": "pass"},
            {"check": "model_and_hyperparameter_freeze", "status": "pass"},
            {"check": "seed_list", "status": "pass"},
            {"check": "test_set_sealed_before_stage7", "status": "pass"},
        ],
    }
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return report
