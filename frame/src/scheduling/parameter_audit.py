"""阶段10.1：调度参数证据台账与可辨识性审计。"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from .contracts import SchedulingContract


LEDGER_FIELDS: Tuple[str, ...] = (
    "parameter_id",
    "track",
    "component",
    "symbol",
    "value",
    "unit",
    "source_type",
    "source_title",
    "source_url",
    "source_year",
    "derivation",
    "validation_range",
    "data_origin",
)
VALID_TRACKS = {"real_replay", "simulated_dispatch"}
VALID_ORIGINS = {"real", "simulated", "derived"}
REQUIRED_COMPONENT_GROUPS = {
    "grid_chp_substitution": {"grid", "chp"},
    "ec_ac_substitution": {"electric_chiller", "absorption_chiller"},
    "bess_intertemporal_state": {"bess"},
}


@dataclass(frozen=True)
class ParameterRecord:
    parameter_id: str
    track: str
    component: str
    symbol: str
    value: float
    unit: str
    source_type: str
    source_title: str
    source_url: str
    source_year: int
    derivation: str
    validation_range: str
    data_origin: str


@dataclass(frozen=True)
class ParameterLedger:
    records: Tuple[ParameterRecord, ...]

    def for_track(self, track: str) -> Tuple[ParameterRecord, ...]:
        return tuple(record for record in self.records if record.track == track)


@dataclass(frozen=True)
class AuditReport:
    parameter_evidence: str
    decision_space: str
    test_year_used_for_scaling: bool
    issues: Tuple[str, ...]
    required_component_groups: Mapping[str, bool]
    record_count: int

    @property
    def passed(self) -> bool:
        return (
            self.parameter_evidence == "pass"
            and self.decision_space == "pass"
            and not self.test_year_used_for_scaling
            and not self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["required_component_groups"] = dict(self.required_component_groups)
        result["issues"] = list(self.issues)
        result["status"] = "pass" if self.passed else "fail"
        return result


def _nonempty(row: Mapping[str, str], field: str, row_number: int) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"参数台账第{row_number}行缺少{field}")
    return value


def read_parameter_ledger(path: str | Path) -> ParameterLedger:
    ledger_path = Path(path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"找不到参数台账：{ledger_path}")
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = sorted(set(LEDGER_FIELDS) - set(fields))
        if missing:
            raise ValueError(f"参数台账缺少字段：{missing}")
        records = []
        for row_number, row in enumerate(reader, start=2):
            values = {field: _nonempty(row, field, row_number) for field in LEDGER_FIELDS}
            try:
                numeric_value = float(values["value"])
            except ValueError as exc:
                raise ValueError(f"参数台账第{row_number}行value必须为数值") from exc
            if not math.isfinite(numeric_value):
                raise ValueError(f"参数台账第{row_number}行value必须是有限数值")
            if values["track"] not in VALID_TRACKS:
                raise ValueError(f"参数台账第{row_number}行track无效：{values['track']}")
            if values["data_origin"] not in VALID_ORIGINS:
                raise ValueError(f"参数台账第{row_number}行data_origin无效：{values['data_origin']}")
            try:
                source_year = int(values["source_year"])
            except ValueError as exc:
                raise ValueError(f"参数台账第{row_number}行source_year必须为整数") from exc
            records.append(ParameterRecord(value=numeric_value, source_year=source_year, **{
                key: values[key] for key in values if key not in {"value", "source_year"}
            }))
    if not records:
        raise ValueError("参数台账不能为空")
    return ParameterLedger(records=tuple(records))


def audit_identifiability(
    contract: SchedulingContract,
    ledger: ParameterLedger,
    training_stats: Mapping[str, Any],
) -> AuditReport:
    """检查证据完整性、替代路径和跨时段状态是否存在。"""

    issues = []
    if training_stats.get("test_year_used_for_scaling") is not False:
        issues.append("test_year_used_for_scaling必须为false")
    if tuple(training_stats.get("scaling_years", ())) != contract.train_years:
        issues.append("scaling_years必须等于2015—2019")

    records = ledger.records
    parameter_evidence = "pass"
    for record in records:
        if record.track == "simulated_dispatch" and record.data_origin not in {"simulated", "derived"}:
            parameter_evidence = "fail"
            issues.append(f"S轨参数{record.parameter_id}缺少simulated/derived标签")
        if record.track == "real_replay" and record.data_origin != "real":
            parameter_evidence = "fail"
            issues.append(f"R轨参数{record.parameter_id}必须标记为real")

    simulated_components = {
        record.component
        for record in records
        if record.track == "simulated_dispatch"
    }
    group_status = {
        name: required.issubset(simulated_components)
        for name, required in REQUIRED_COMPONENT_GROUPS.items()
    }
    if not all(group_status.values()):
        issues.append("S轨缺少至少一组可辨识的替代路径或跨时段状态")
    decision_space = "pass" if all(group_status.values()) else "fail"

    return AuditReport(
        parameter_evidence=parameter_evidence,
        decision_space=decision_space,
        test_year_used_for_scaling=bool(training_stats.get("test_year_used_for_scaling", False)),
        issues=tuple(issues),
        required_component_groups=group_status,
        record_count=len(records),
    )
