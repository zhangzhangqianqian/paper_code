"""阶段5.1：外部轻量基线公平性契约及其校验。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .data_pipeline import FULL_SPLIT, SMALL_SAMPLE_SPLIT, TASKS
from .kitakyushu_pipeline import (
    KITAKYUSHU_SMALL_SAMPLE_SPLIT,
    KITAKYUSHU_SPLIT,
    KITAKYUSHU_TASKS,
)


BASELINE_NAMES: Tuple[str, ...] = ("DLinear", "MMoE-lite", "SOFTS")
INPUT_MODES: Tuple[str, ...] = ("loads_only", "loads_and_exog")
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "external_baseline_fairness.json"
)


@dataclass(frozen=True)
class BaselineSpec:
    """一个外部基线在公平性契约中的声明。"""

    name: str
    family: str
    input_mode: str
    exog_support: str
    future_exog_allowed: bool
    exact_reimplementation: bool
    relation_to_shao: str


@dataclass(frozen=True)
class FairnessContract:
    """已通过校验的阶段5.1实验契约。"""

    raw: Mapping[str, Any]

    @property
    def lookback(self) -> int:
        return int(self.raw["data_protocol"]["lookback"])

    @property
    def horizon(self) -> int:
        return int(self.raw["data_protocol"]["horizon"])

    @property
    def task_count(self) -> int:
        return len(self.raw["tasks"])

    @property
    def baseline_specs(self) -> Tuple[BaselineSpec, ...]:
        return tuple(
            BaselineSpec(**spec) for spec in self.raw["baseline_specs"]
        )

    def get_baseline(self, name: str) -> BaselineSpec:
        for spec in self.baseline_specs:
            if spec.name == name:
                return spec
        raise KeyError(f"契约中不存在外部基线 {name!r}")

    def validate_prediction_shape(self, shape: Sequence[int]) -> None:
        """验证模型输出是否符合 ``[batch, horizon, task_count]``。"""

        actual = tuple(int(value) for value in shape)
        expected_tail = (self.horizon, self.task_count)
        if len(actual) != 3 or actual[1:] != expected_tail:
            raise ValueError(
                "预测输出必须是[batch, horizon, task_count]，"
                f"当前得到{actual}，期望第二、三维为{expected_tail}"
            )
        if actual[0] <= 0:
            raise ValueError("预测输出的batch维度必须为正整数")

    def validate_baseline_input(
        self,
        name: str,
        input_mode: str,
        uses_future_exog: bool = False,
    ) -> None:
        """验证单个基线的输入声明，防止静默改变公平性协议。"""

        spec = self.get_baseline(name)
        if input_mode not in INPUT_MODES:
            raise ValueError(
                f"未知输入模式 {input_mode!r}；可选模式为 {INPUT_MODES}"
            )
        if spec.input_mode != input_mode:
            raise ValueError(
                f"基线{name!r}在阶段5.1中声明的输入模式为{spec.input_mode!r}，"
                f"不能直接改为{input_mode!r}"
            )
        if uses_future_exog:
            raise ValueError("公平性契约禁止使用预测时刻之后的真实外生变量")


def _split_mapping(split: object) -> Dict[str, str]:
    if not isinstance(split, Mapping):
        raise ValueError("时间切分必须是映射")
    return {str(key): str(value) for key, value in split.items()}


def _expected_split_values(split: object) -> Dict[str, str]:
    return {
        "train_start": split.train_start,
        "train_end": split.train_end,
        "validation_start": split.validation_start,
        "validation_end": split.validation_end,
        "test_start": split.test_start,
        "test_end": split.test_end,
    }


def _dataset_protocol(dataset: str) -> Tuple[Tuple[str, ...], object, object]:
    if dataset == "kitakyushu_energy_station":
        return KITAKYUSHU_TASKS, KITAKYUSHU_SPLIT, KITAKYUSHU_SMALL_SAMPLE_SPLIT
    if dataset == "heew_total":
        return TASKS, FULL_SPLIT, SMALL_SAMPLE_SPLIT
    raise ValueError(f"不支持的数据集协议：{dataset!r}")


def validate_contract(data: Mapping[str, Any]) -> None:
    """校验契约中的固定实验条件。"""

    required_top = {
        "contract_version",
        "tasks",
        "dataset",
        "data_protocol",
        "input_output",
        "training",
        "evaluation",
        "resource_recording",
        "baseline_specs",
    }
    missing = sorted(required_top - set(data))
    if missing:
        raise ValueError(f"公平性契约缺少字段：{missing}")
    expected_tasks, full_split, small_sample_split = _dataset_protocol(
        str(data["dataset"])
    )
    if list(data["tasks"]) != list(expected_tasks):
        raise ValueError(f"任务顺序必须固定为{expected_tasks}")

    protocol = data["data_protocol"]
    if protocol["sampling"] != "1h":
        raise ValueError("当前公平性契约固定使用1小时采样")
    if int(protocol["lookback"]) != 24 or int(protocol["horizon"]) != 4:
        raise ValueError("当前公平性契约固定使用24→4预测协议")
    splits = protocol["splits"]
    for name, expected in (("full", full_split), ("small_sample", small_sample_split)):
        actual = _split_mapping(splits[name])
        if actual != _expected_split_values(expected):
            raise ValueError(f"{name}时间切分与项目固定协议不一致")
    standardization = protocol["standardization"]
    if standardization["fit_on"] != "train_split_only":
        raise ValueError("标准化参数必须只从训练切分拟合")
    if list(standardization["apply_to"]) != ["train", "validation", "test"]:
        raise ValueError("标准化必须应用于train、validation和test")
    if standardization["fit_targets_and_exog"] is not True:
        raise ValueError("负荷和外生变量必须使用训练切分分别拟合标准化参数")

    input_output = data["input_output"]
    if input_output["loads_shape"] != ["batch", 24, len(expected_tasks)]:
        raise ValueError(
            f"loads输入协议必须为[batch,24,{len(expected_tasks)}]"
        )
    if input_output["exog_shape"] != ["batch", 24, "F"]:
        raise ValueError("exog输入协议必须为[batch,24,F]")
    if input_output["prediction_shape"] != ["batch", 4, len(expected_tasks)]:
        raise ValueError(
            f"预测输出协议必须为[batch,4,{len(expected_tasks)}]"
        )
    if input_output["future_exogenous_allowed"] is not False:
        raise ValueError("不得使用未来真实外生变量")
    if set(input_output["input_modes"]) != set(INPUT_MODES):
        raise ValueError("输入模式必须同时声明loads_only和loads_and_exog")

    training = data["training"]
    if training["loss"] != "SmoothL1Loss":
        raise ValueError("阶段5.1固定使用SmoothL1Loss")
    if training["early_stopping_monitor"] != "validation_loss":
        raise ValueError("Early stopping必须监控validation_loss")
    if training["checkpoint_policy"] != "save_and_restore_best_validation_loss":
        raise ValueError("必须保存并恢复验证集最佳检查点")
    if training["device"] != "cpu":
        raise ValueError("当前阶段固定使用CPU")

    if list(data["evaluation"]["metrics"]) != ["MAE", "RMSE", "WAPE", "MAPE"]:
        raise ValueError("评价指标必须固定为MAE、RMSE、WAPE和MAPE")
    for field in (
        "report_per_task",
        "report_per_horizon",
        "report_overall_equal_task_mean",
    ):
        if data["evaluation"][field] is not True:
            raise ValueError(f"评价契约必须启用{field}")
    if any(data["resource_recording"].get(field) is not True for field in (
        "parameter_count",
        "fit_seconds",
        "test_evaluation_seconds",
        "test_samples_per_second",
    )):
        raise ValueError("必须记录参数量、训练耗时和CPU测试评估耗时")

    specs = data["baseline_specs"]
    if not isinstance(specs, list) or [spec["name"] for spec in specs] != list(BASELINE_NAMES):
        raise ValueError(f"外部基线必须按顺序声明为{BASELINE_NAMES}")
    for spec in specs:
        if spec["input_mode"] not in INPUT_MODES:
            raise ValueError(f"基线{spec['name']}使用了未声明的输入模式")
        if spec["future_exog_allowed"] is not False:
            raise ValueError(f"基线{spec['name']}禁止使用未来外生变量")
        if spec["exact_reimplementation"] is not False:
            raise ValueError("阶段5的外部基线不应被标记为Shao完整复现")
        if not str(spec["relation_to_shao"]).strip():
            raise ValueError(f"基线{spec['name']}缺少与Shao模型的关系说明")


def load_fairness_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> FairnessContract:
    contract_path = Path(path)
    if not contract_path.exists():
        raise FileNotFoundError(f"找不到公平性契约：{contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("公平性契约顶层必须是JSON对象")
    validate_contract(data)
    return FairnessContract(raw=data)
