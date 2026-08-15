"""阶段10.0：预测—调度双轨研究契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import yaml


TASK_ORDER: Tuple[str, ...] = ("electricity", "cooling", "heating", "gas")
TRACK_ORDER: Tuple[str, ...] = ("real_replay", "simulated_dispatch")
DEFAULT_SCHEDULING_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "scheduling_dual_track_contract_v2.yaml"
)


@dataclass(frozen=True)
class SchedulingContract:
    """已经通过硬规则校验的调度研究契约。"""

    raw: Mapping[str, Any]

    @property
    def dataset(self) -> str:
        return str(self.raw["dataset"])

    @property
    def history_hours(self) -> int:
        return int(self.raw["time"]["history_hours"])

    @property
    def horizon_hours(self) -> int:
        return int(self.raw["time"]["horizon_hours"])

    @property
    def train_years(self) -> Tuple[int, ...]:
        return tuple(int(value) for value in self.raw["time"]["train_years"])

    @property
    def validation_year(self) -> int:
        return int(self.raw["time"]["validation_year"])

    @property
    def test_year(self) -> int:
        return int(self.raw["time"]["test_year"])

    @property
    def task_order(self) -> Tuple[str, ...]:
        return tuple(str(value) for value in self.raw["task_order"])

    @property
    def tracks(self) -> Tuple[str, ...]:
        return tuple(str(value) for value in self.raw["tracks"])

    @property
    def gas_role(self) -> str:
        return str(self.raw["gas"]["role"])

    def track_config(self, name: str) -> Mapping[str, Any]:
        if name not in self.tracks:
            raise KeyError(f"契约中不存在调度轨道：{name!r}")
        return self.raw["track_configs"][name]


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}必须是对象")
    return value


def validate_scheduling_contract(data: Mapping[str, Any]) -> None:
    """验证年份、轨道、gas 语义、求解器和未来信息权限。"""

    required = {
        "schema_version",
        "dataset",
        "time",
        "task_order",
        "tracks",
        "track_configs",
        "gas",
        "solver",
        "leakage",
        "outputs",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"调度契约缺少字段：{missing}")
    if data["schema_version"] != "scheduling-dual-track-v2":
        raise ValueError("schema_version必须为scheduling-dual-track-v2")
    if data["dataset"] != "kitakyushu_energy_station":
        raise ValueError("调度契约当前只支持kitakyushu_energy_station")

    time = _require_mapping(data["time"], "time")
    if int(time.get("history_hours", -1)) != 24:
        raise ValueError("历史窗口必须为24小时")
    if int(time.get("horizon_hours", -1)) != 4:
        raise ValueError("调度窗口和预测输出必须为4小时")
    if tuple(int(value) for value in time.get("train_years", ())) != (2015, 2016, 2017, 2018, 2019):
        raise ValueError("训练期必须固定为2015—2019")
    if int(time.get("validation_year", -1)) != 2020:
        raise ValueError("验证年必须固定为2020")
    if int(time.get("test_year", -1)) != 2021:
        raise ValueError("测试年必须固定为2021")

    if tuple(data["task_order"]) != TASK_ORDER:
        raise ValueError(f"任务顺序必须固定为{TASK_ORDER}")
    if tuple(data["tracks"]) != TRACK_ORDER:
        raise ValueError(f"调度轨道必须按顺序声明为{TRACK_ORDER}")

    track_configs = _require_mapping(data["track_configs"], "track_configs")
    output_dirs = []
    for track in TRACK_ORDER:
        cfg = _require_mapping(track_configs.get(track), f"track_configs.{track}")
        output_dir = str(cfg.get("output_subdir", "")).strip()
        if not output_dir:
            raise ValueError(f"{track}必须声明非空output_subdir")
        output_dirs.append(output_dir)
        if cfg.get("ordinary_mode_allow_actual_future") is not False:
            raise ValueError(f"{track}普通模式禁止读取真实未来值")
    if len(set(output_dirs)) != len(output_dirs):
        raise ValueError("real_replay和simulated_dispatch必须使用不同输出目录")
    simulated = track_configs["simulated_dispatch"]
    if simulated.get("gas_demand_balance") is not False:
        raise ValueError("simulated_dispatch不得把gas作为刚性需求平衡")
    if simulated.get("uses_real_equipment") is not False:
        raise ValueError("simulated_dispatch必须明确标记为仿真设备")
    real = track_configs["real_replay"]
    if real.get("uses_real_equipment") is not True:
        raise ValueError("real_replay必须明确使用真实站点运行序列")

    gas = _require_mapping(data["gas"], "gas")
    if gas.get("role") != "station_side_device_consumption":
        raise ValueError("gas必须定义为能源站设备侧聚合燃气消耗")
    if gas.get("main_dispatch_balance") is not False:
        raise ValueError("gas不得作为主调度刚性需求平衡")
    if set(gas.get("allowed_s_uses", ())) != {"diagnostic", "gas_prior_ablation"}:
        raise ValueError("S轨gas只能用于diagnostic和gas_prior_ablation")

    solver = _require_mapping(data["solver"], "solver")
    if solver.get("name") != "scipy.optimize.linprog" or solver.get("method") != "highs":
        raise ValueError("核心求解器必须是scipy.optimize.linprog(method='highs')")
    if solver.get("version_constraint") != ">=1.13,<1.14":
        raise ValueError("SciPy版本约束必须为>=1.13,<1.14")

    leakage = _require_mapping(data["leakage"], "leakage")
    for field in ("ordinary_planning_forbids_actual_future", "test_year_used_for_selection", "test_year_used_for_scaling"):
        if leakage.get(field) is not False and field != "ordinary_planning_forbids_actual_future":
            raise ValueError(f"{field}必须为false")
    if leakage.get("ordinary_planning_forbids_actual_future") is not True:
        raise ValueError("普通计划阶段必须禁止真实未来信息")

    outputs = _require_mapping(data["outputs"], "outputs")
    roots = [str(outputs.get("real_replay_root", "")).strip(), str(outputs.get("simulated_dispatch_root", "")).strip()]
    if any(not root for root in roots) or roots[0] == roots[1]:
        raise ValueError("两条轨道必须声明不同且非空的结果根目录")


def load_scheduling_contract(path: str | Path = DEFAULT_SCHEDULING_CONTRACT_PATH) -> SchedulingContract:
    contract_path = Path(path)
    if not contract_path.exists():
        raise FileNotFoundError(f"找不到调度契约：{contract_path}")
    with contract_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("调度契约顶层必须是对象")
    validate_scheduling_contract(data)
    return SchedulingContract(raw=data)
