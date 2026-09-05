"""Frozen protocol boundary for the regime-aware formal-v4.3 experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER


SCHEMA_VERSION = "joint-forecast-dispatch-formal-v4.3"
THERMAL_CLASSES = ("off", "cooling", "heating")
_TRANSITIONS = {
    ("gate0", "pilot"),
    ("pilot", "gate1"),
    ("gate1", "gate2"),
}


def _array(payload: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return tuple(value)


@dataclass(frozen=True)
class FormalV43Contract:
    payload: Mapping[str, Any]
    contract_sha256: str
    source_path: Path

    @property
    def train_years(self) -> tuple[int, ...]:
        return tuple(int(value) for value in _array(self.payload, "train_years"))

    @property
    def selection_year(self) -> int:
        return int(self.payload["selection_year"])

    @property
    def evaluation_year(self) -> int:
        return int(self.payload["evaluation_year"])

    @property
    def excluded_years(self) -> tuple[int, ...]:
        return tuple(int(value) for value in _array(self.payload, "excluded_years"))

    @property
    def task_order(self) -> tuple[str, ...]:
        return tuple(str(value) for value in _array(self.payload, "task_order"))

    @property
    def rigid_demands(self) -> tuple[str, ...]:
        return tuple(str(value) for value in _array(self.payload, "rigid_demands"))

    @property
    def exog_order(self) -> tuple[str, ...]:
        return tuple(str(value) for value in _array(self.payload, "exog_order"))

    @property
    def dispatch_order(self) -> tuple[str, ...]:
        return tuple(str(value) for value in _array(self.payload, "dispatch_order"))

    @property
    def status_order(self) -> tuple[str, ...]:
        return tuple(str(value) for value in _array(self.payload, "status_order"))

    @property
    def latent_control_dim(self) -> int:
        return int(self.payload["latent_control_dim"])

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(str(row["name"]) for row in self.payload.get("methods", ()))

    def __getattr__(self, name: str) -> Any:
        try:
            return self.payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def validate(self) -> None:
        if self.payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("formal-v4.3 schema required")
        if self.payload.get("protocol_status") != "frozen":
            raise ValueError("formal-v4.3 contract is not frozen")
        if self.train_years != (2015, 2016, 2017, 2018):
            raise ValueError("formal-v4.3 training years must be 2015-2018")
        if (self.selection_year, self.evaluation_year, self.excluded_years) != (2019, 2020, (2021,)):
            raise ValueError("formal-v4.3 year boundary is invalid")
        if self.task_order != TASK_ORDER or self.exog_order != EXOG_ORDER:
            raise ValueError("task or exogenous order differs from the frozen contract")
        if self.dispatch_order != DISPATCH_ORDER or self.status_order != STATUS_ORDER:
            raise ValueError("dispatch or status order differs from the frozen contract")
        if self.rigid_demands != ("electricity", "cooling", "heating"):
            raise ValueError("gas must remain outside rigid terminal demand")
        if self.payload.get("lookback") != 24 or self.payload.get("horizon") != 4:
            raise ValueError("formal-v4.3 requires a 24-hour history and four-hour horizon")
        if self.latent_control_dim != 15 or int(self.payload.get("dispatch_dim", -1)) != 21:
            raise ValueError("formal-v4.3 control dimensions are invalid")
        if self.payload.get("allow_future_binary_decisions") is not False:
            raise ValueError("future binary decisions are outside formal-v4.3")
        if self.payload.get("gas_semantics") != "station_side_auxiliary_prior":
            raise ValueError("gas must remain a station-side auxiliary prior")
        thermal = self.payload.get("thermal_regime")
        if not isinstance(thermal, Mapping) or tuple(thermal.get("classes", ())) != THERMAL_CLASSES:
            raise ValueError("thermal regime must be off/cooling/heating")
        if bool(thermal.get("hard_calendar_mask")):
            raise ValueError("hard calendar mask is forbidden")
        if bool(thermal.get("hard_training_gate")):
            raise ValueError("hard training gate is forbidden")
        if float(thermal.get("active_epsilon", -1.0)) != 1.0e-9:
            raise ValueError("thermal active epsilon is not frozen")
        if float(thermal.get("probability_temperature", -1.0)) <= 0.0:
            raise ValueError("thermal probability temperature must be positive")
        loss = self.payload.get("forecast_loss")
        if not isinstance(loss, Mapping) or any(float(loss.get(key, -1.0)) < 0.0 for key in (
            "continuous_weight", "regime_weight", "active_magnitude_weight", "point_weight",
            "inactive_leakage_weight", "gas_task_weight", "transition_window_weight",
        )):
            raise ValueError("forecast loss weights are invalid")
        views = self.payload.get("gate1_views")
        if not isinstance(views, Mapping) or not bool(views.get("full_chronology", {}).get("enabled")):
            raise ValueError("full chronology must be a Gate 1 view")
        if not bool(views.get("full_chronology", {}).get("uniform_weight")):
            raise ValueError("full chronology must use uniform weights")
        if not bool(views.get("stress_sample", {}).get("secondary_only")):
            raise ValueError("stress sample must remain secondary")
        if self.payload.get("allow_evaluation_access_before_gate2") is not False:
            raise ValueError("evaluation access must remain fail-closed")

    def gate1_candidate_grid(self) -> tuple[tuple[float, float], ...]:
        grid = self.payload.get("candidate_grid", {})
        lr = tuple(float(value) for value in grid.get("forecaster_lr_multiplier", ()))
        decision = tuple(float(value) for value in grid.get("decision_final", ()))
        if not lr or not decision or any(value <= 0.0 for value in (*lr, *decision)):
            raise ValueError("Gate 1 candidate grid must contain positive values")
        return tuple((left, right) for left in lr for right in decision)


def load_formal_v4_3_contract(path: str | Path) -> FormalV43Contract:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("formal-v4.3 contract must be a JSON object")
    contract = FormalV43Contract(
        payload=MappingProxyType(dict(payload)),
        contract_sha256=hashlib.sha256(raw).hexdigest(),
        source_path=source,
    )
    contract.validate()
    return contract


def assert_gate_transition(contract: FormalV43Contract, completed_gate: str, requested_gate: str) -> None:
    contract.validate()
    if (str(completed_gate), str(requested_gate)) not in _TRANSITIONS:
        raise ValueError(f"invalid formal-v4.3 gate transition: {(completed_gate, requested_gate)}")


__all__ = [
    "FormalV43Contract", "SCHEMA_VERSION", "THERMAL_CLASSES",
    "assert_gate_transition", "load_formal_v4_3_contract",
]
