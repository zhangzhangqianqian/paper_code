"""阶段6.1：验证协议和模型选择规则契约。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from .data_pipeline import (
    FULL_SPLIT,
    HEEW_EXOG_COLUMNS,
    SMALL_SAMPLE_SPLIT,
    TASKS,
)
from .kitakyushu_pipeline import (
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
)
from .models import STAGE6_MODEL_NAMES


DEFAULT_STAGE6_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "stage6_selection_contract.json"
)


def _split_values(split: object) -> dict[str, str]:
    if not isinstance(split, Mapping):
        raise ValueError("时间切分必须是对象")
    return {str(key): str(value) for key, value in split.items()}


def _expected_split(split: object) -> dict[str, str]:
    return {
        "train_start": split.train_start,
        "train_end": split.train_end,
        "validation_start": split.validation_start,
        "validation_end": split.validation_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
    }


def _dataset_protocol(dataset: str) -> tuple[tuple[str, ...], object, object, int]:
    if dataset == "kitakyushu_energy_station":
        return (
            KITAKYUSHU_TASKS,
            KITAKYUSHU_SPLIT,
            KITAKYUSHU_SMALL_SAMPLE_SPLIT,
            len(KITAKYUSHU_EXOG_COLUMNS),
        )
    if dataset == "heew_total":
        return TASKS, FULL_SPLIT, SMALL_SAMPLE_SPLIT, len(HEEW_EXOG_COLUMNS)
    raise ValueError(f"不支持的数据集协议：{dataset!r}")


@dataclass(frozen=True)
class Stage6SelectionContract:
    raw: Mapping[str, Any]

    @property
    def primary_protocol(self) -> str:
        return str(self.raw["validation_policy"]["primary_protocol"])

    @property
    def secondary_protocol(self) -> str:
        return str(self.raw["validation_policy"]["secondary_protocol"])

    @property
    def candidate_models(self) -> Tuple[str, ...]:
        return tuple(str(value) for value in self.raw["candidate_models"])

    @property
    def hyperparameter_candidates(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(self.raw["finite_hyperparameter_candidates"])

    def training_policy(self, protocol: str) -> Mapping[str, Any]:
        """Return the fully resolved policy for ``full`` or ``small_sample``.

        Shared optimizer/device fields are merged with protocol-specific
        batch and early-stopping fields.  Callers must not read the raw JSON
        mapping directly, otherwise the two protocols can silently diverge.
        """

        if protocol not in {"full", "small_sample"}:
            raise ValueError(f"unsupported Stage 6 protocol: {protocol!r}")
        training = self.raw["training_policy"]
        if not isinstance(training, Mapping):
            raise ValueError("training_policy must be an object")
        shared = training.get("shared")
        protocol_policy = training.get(protocol)
        if not isinstance(shared, Mapping) or not isinstance(protocol_policy, Mapping):
            raise ValueError("training_policy must contain shared/full/small_sample objects")
        return {**dict(shared), **dict(protocol_policy)}


def validate_stage6_selection_contract(data: Mapping[str, Any]) -> None:
    required = {
        "contract_version",
        "purpose",
        "dataset",
        "tasks",
        "data_protocol",
        "validation_policy",
        "candidate_models",
        "input_output",
        "training_policy",
        "finite_hyperparameter_candidates",
        "selection_rule",
        "negative_transfer_reference",
        "stage6_outputs",
        "failure_policy",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"阶段6.1契约缺少字段：{missing}")
    if data["contract_version"] != "stage6.1-r1":
        raise ValueError("contract_version必须为stage6.1-r1")
    expected_tasks, full_split, small_sample_split, exog_count = _dataset_protocol(
        str(data["dataset"])
    )
    if list(data["tasks"]) != list(expected_tasks):
        raise ValueError(f"任务顺序必须固定为{expected_tasks}")

    protocol = data["data_protocol"]
    if not isinstance(protocol, Mapping):
        raise ValueError("data_protocol必须是对象")
    if protocol.get("sampling") != "1h":
        raise ValueError("采样间隔必须为1小时")
    if int(protocol.get("lookback", -1)) != 24 or int(protocol.get("horizon", -1)) != 4:
        raise ValueError("阶段6.1必须使用24→4预测协议")
    for name, expected in (("full", full_split), ("small_sample", small_sample_split)):
        if _split_values(protocol.get(name)) != _expected_split(expected):
            raise ValueError(f"{name}时间切分与项目固定协议不一致")

    policy = data["validation_policy"]
    if not isinstance(policy, Mapping):
        raise ValueError("validation_policy必须是对象")
    if policy.get("primary_protocol") != "full":
        raise ValueError("全年协议必须是主要选择协议")
    if policy.get("secondary_protocol") != "small_sample":
        raise ValueError("小样本协议必须是辅助鲁棒性协议")
    if policy.get("secondary_role") != "robustness_only":
        raise ValueError("小样本验证集不得覆盖全年验证集的主要选择作用")
    if policy.get("test_set_status") != "sealed_until_stage7":
        raise ValueError("测试集必须封存到阶段7")
    if policy.get("test_metrics_may_be_read") is not False:
        raise ValueError("阶段6.1禁止读取测试指标")
    if policy.get("test_predictions_may_be_read") is not False:
        raise ValueError("阶段6.1禁止读取测试预测")

    candidates = data["candidate_models"]
    if list(candidates) != list(STAGE6_MODEL_NAMES):
        raise ValueError(f"候选模型必须按项目顺序声明为{STAGE6_MODEL_NAMES}")

    io = data["input_output"]
    if not isinstance(io, Mapping):
        raise ValueError("input_output必须是对象")
    if io.get("input_mode") != "loads_and_exog":
        raise ValueError("阶段6核心模型统一使用loads_and_exog")
    if io.get("loads_shape") != ["batch", 24, len(expected_tasks)]:
        raise ValueError(
            f"loads输入必须是[batch,24,{len(expected_tasks)}]"
        )
    if io.get("exog_shape") != ["batch", 24, exog_count]:
        raise ValueError(f"外生变量输入必须是[batch,24,{exog_count}]")
    if io.get("prediction_shape") != ["batch", 4, len(expected_tasks)]:
        raise ValueError(
            f"预测输出必须是[batch,4,{len(expected_tasks)}]"
        )
    if io.get("future_exogenous_allowed") is not False:
        raise ValueError("阶段6禁止使用未来外生变量")
    if io.get("standardization") != "train_split_only_zscore":
        raise ValueError("标准化必须只在训练切分拟合")

    training = data["training_policy"]
    if not isinstance(training, Mapping):
        raise ValueError("training_policy必须是对象")
    shared = training.get("shared")
    expected_shared = {
        "random_seed": 2026,
        "device": "cpu",
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "early_stopping_monitor": "validation_loss",
        "checkpoint_policy": "restore_best_validation_loss",
        "gradient_clip_norm": 1.0,
    }
    if not isinstance(shared, Mapping):
        raise ValueError("training_policy.shared必须是对象")
    for key, expected in expected_shared.items():
        if shared.get(key) != expected:
            raise ValueError(f"training_policy.shared.{key}必须固定为{expected!r}")
    expected_protocols = {
        "full": {"batch_size": 256, "max_epochs": 100, "early_stopping_patience": 12},
        "small_sample": {"batch_size": 32, "max_epochs": 200, "early_stopping_patience": 20},
    }
    for protocol_name, expected in expected_protocols.items():
        protocol_policy = training.get(protocol_name)
        if not isinstance(protocol_policy, Mapping):
            raise ValueError(f"training_policy.{protocol_name}必须是对象")
        for key, value in expected.items():
            if protocol_policy.get(key) != value:
                raise ValueError(
                    f"training_policy.{protocol_name}.{key}必须固定为{value!r}"
                )

    hyperparameters = data["finite_hyperparameter_candidates"]
    if not isinstance(hyperparameters, list) or len(hyperparameters) != 4:
        raise ValueError("阶段6.1必须预先固定4组有限超参数")
    ids = [candidate.get("candidate_id") for candidate in hyperparameters]
    if ids != ["H1", "H2", "H3", "H4"]:
        raise ValueError("超参数候选编号必须为H1、H2、H3、H4")
    expected_candidate_structure = {
        "H1": {"hidden_dim": 16, "kernel_size": 5, "dilations": [1, 2, 4], "scheme2r_rank": 4},
        "H2": {"hidden_dim": 32, "kernel_size": 5, "dilations": [1, 2, 4], "scheme2r_rank": 8},
        "H3": {"hidden_dim": 32, "kernel_size": 3, "dilations": [1, 2, 4, 8], "scheme2r_rank": 8},
        "H4": {"hidden_dim": 32, "kernel_size": 5, "dilations": [1, 2, 4], "scheme2r_rank": 8},
    }
    signatures = set()
    for candidate in hyperparameters:
        candidate_id = str(candidate.get("candidate_id"))
        expected_structure = expected_candidate_structure[candidate_id]
        for key, expected in expected_structure.items():
            if candidate.get(key) != expected:
                raise ValueError(
                    f"{candidate_id}.{key}必须固定为{expected!r}，实际为{candidate.get(key)!r}"
                )
        if candidate.get("hidden_dim", 0) <= 1:
            raise ValueError("hidden_dim必须大于1")
        if candidate.get("kernel_size", 0) <= 0 or candidate["kernel_size"] % 2 == 0:
            raise ValueError("kernel_size必须为正奇数")
        if not 0.0 <= candidate.get("dropout", -1.0) < 1.0:
            raise ValueError("dropout必须位于[0,1)")
        if candidate.get("learning_rate", 0.0) <= 0:
            raise ValueError("learning_rate必须为正数")
        if candidate.get("scheme2r_kernel_size") != candidate.get("kernel_size"):
            raise ValueError("Scheme2R与通用候选的kernel_size必须一致")
        if candidate.get("scheme2r_dilations") != candidate.get("dilations"):
            raise ValueError("Scheme2R与通用候选的dilations必须一致")
        if candidate.get("scheme2r_rank") <= 0:
            raise ValueError("Scheme2R的低秩维度必须为正整数")
        if candidate.get("scheme2r_gate_hidden_dim") != 16:
            raise ValueError("Scheme2R的门控隐藏维度必须固定为16")
        if candidate.get("scheme2r_step_embedding_dim") != 4:
            raise ValueError("Scheme2R的预测步嵌入维度必须固定为4")
        if candidate.get("prediction_head_hidden_dim") != 16:
            raise ValueError("所有正式候选的预测头隐藏维度必须固定为16")
        signature = json.dumps(
            {
                "hidden_dim": candidate["hidden_dim"],
                "kernel_size": candidate["kernel_size"],
                "dilations": candidate["dilations"],
                "scheme2r_rank": candidate["scheme2r_rank"],
                "dropout": candidate["dropout"],
                "learning_rate": candidate["learning_rate"],
            },
            sort_keys=True,
        )
        if signature in signatures:
            raise ValueError(f"H候选存在重复的有效配置：{candidate_id}")
        signatures.add(signature)

    selection = data["selection_rule"]
    if not isinstance(selection, Mapping):
        raise ValueError("selection_rule必须是对象")
    if selection.get("primary_metric") != "validation_overall_equal_task_mean.WAPE":
        raise ValueError("主要选择指标必须是全年验证集总体WAPE")
    if selection.get("primary_direction") != "minimize":
        raise ValueError("WAPE选择方向必须是minimize")
    if selection.get("primary_dataset") != "full_validation":
        raise ValueError("主要指标必须来自全年验证集")
    if selection.get("small_sample_cannot_override_primary_selection") is not True:
        raise ValueError("小样本验证集不得覆盖主要选择结果")
    if selection.get("test_result_cannot_override_selection") is not True:
        raise ValueError("测试结果不得覆盖选择结果")

    reference = data["negative_transfer_reference"]
    if not isinstance(reference, Mapping):
        raise ValueError("negative_transfer_reference必须是对象")
    if reference.get("reference_model") != "stl_matched" or reference.get("validation_only") is not True:
        raise ValueError("负迁移参照必须是验证集上的STL")

    failure = data["failure_policy"]
    if failure.get("max_recorded_structure_corrections") != 1:
        raise ValueError("阶段6最多允许一轮有记录的结构修正")
    if failure.get("module_stacking_after_failed_gate") is not False:
        raise ValueError("门控失败后不得继续无记录堆叠模块")
    if failure.get("premature_claim_of_negative_transfer_mitigation") is not False:
        raise ValueError("不得提前宣称缓解负迁移")


def load_stage6_selection_contract(
    path: str | Path = DEFAULT_STAGE6_CONTRACT_PATH,
) -> Stage6SelectionContract:
    contract_path = Path(path)
    if not contract_path.exists():
        raise FileNotFoundError(f"找不到阶段6.1契约：{contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("阶段6.1契约顶层必须是对象")
    validate_stage6_selection_contract(data)
    return Stage6SelectionContract(raw=data)
