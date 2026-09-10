"""Frozen scientific boundary for the formal-v4.6 risk-adjusted model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER
METHODS = (
    "RSC-PF", "Decoupled-RSC-PF", "Direct-Policy", "Scheme2R-PTO",
    "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP",
    "Perfect-Information-MPC", "Seasonal-Naive-PTO",
)


SCHEMA_VERSION = "joint-forecast-dispatch-formal-v4.6"
_RISK_ADJUSTMENT = {
    "tasks": ["electricity", "cooling", "heating"],
    "gas_adjustment": 0.0,
    "cap_quantile": 0.90,
    "hidden_width": 64,
    "initial_output_bias": -6.0,
    "regularization_multipliers": [0.5, 1.0, 2.0],
    "risk_size_base_weight": 0.10,
    "off_risk_base_weight": 0.50,
    "j_risk_lr": 0.0005,
}
_PILOT_THRESHOLDS = {
    "maximum_four_task_score_ratio": 1.02,
    "maximum_electricity_wape_ratio": 1.02,
    "maximum_gas_wape_ratio": 1.10,
    "maximum_active_thermal_wape_ratio": 1.05,
    "maximum_normalized_inactive_leakage_ratio": 1.05,
    "maximum_macro_f1_drop": 0.02,
    "minimum_transition_balanced_accuracy_gain": 0.01,
    "maximum_decoupled_decision_gradient": 1.0e-12,
    "maximum_physical_residual": 1.0e-6,
}
_CURRICULUM = {
    "forecast": 1.0,
    "anchor": 0.5,
    "decision_start": 0.05,
    "decision_final": 0.50,
    "imitation_start": 1.0,
    "imitation_final": 0.25,
    "ramp_epochs": 5,
}
_GUARDRAILS = {
    "max_macro_f1_drop": 0.02,
    "max_inactive_leakage_relative_increase": 0.05,
}


def _tuple(payload: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return tuple(value)


@dataclass(frozen=True)
class FormalV46Contract:
    payload: Mapping[str, Any]
    contract_sha256: str
    source_path: Path

    @property
    def train_years(self) -> tuple[int, ...]:
        return tuple(int(value) for value in _tuple(self.payload, "train_years"))

    @property
    def selection_year(self) -> int:
        return int(self.payload["selection_year"])

    @property
    def evaluation_year(self) -> int:
        return int(self.payload["evaluation_year"])

    @property
    def excluded_years(self) -> tuple[int, ...]:
        return tuple(int(value) for value in _tuple(self.payload, "excluded_years"))

    @property
    def forecast_tasks(self) -> tuple[str, ...]:
        return tuple(str(value) for value in _tuple(self.payload, "task_order"))

    @property
    def rigid_demands(self) -> tuple[str, ...]:
        return tuple(str(value) for value in _tuple(self.payload, "rigid_demand_order"))

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(str(row["name"]) for row in self.payload["methods"])

    @property
    def candidate_grid(self) -> tuple[Mapping[str, float], ...]:
        return tuple(MappingProxyType(dict(row)) for row in self.payload["candidate_grid"])

    @property
    def pilot_train_windows(self) -> int:
        return int(self.payload["pilot_budget"]["train_windows"])

    @property
    def control_dim(self) -> int:
        return int(self.payload["latent_control_dim"])

    @property
    def dispatch_dim(self) -> int:
        return len(self.payload["dispatch_order"])

    @property
    def risk_adjustment(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.payload["risk_adjustment"]))

    @property
    def pilot_thresholds(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.payload["pilot_thresholds"]))

    @property
    def joint_curriculum(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.payload["joint_curriculum"]))

    @property
    def forecast_guardrails(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.payload["forecast_guardrails"]))

    def validate(self) -> None:
        if self.payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("formal-v4.6 schema required")
        if self.payload.get("protocol_status") != "frozen":
            raise ValueError("formal-v4.6 contract is not frozen")
        if self.train_years != (2015, 2016, 2017, 2018) or self.selection_year != 2019:
            raise ValueError("formal-v4.6 training and selection years are invalid")
        if self.evaluation_year != 2020 or self.excluded_years != (2021,):
            raise ValueError("formal-v4.6 evaluation boundary is invalid")
        if self.forecast_tasks != tuple(TASK_ORDER):
            raise ValueError("task order differs from the frozen contract")
        if tuple(self.payload.get("exog_order", ())) != tuple(EXOG_ORDER):
            raise ValueError("exogenous order differs from the frozen contract")
        if tuple(self.payload.get("dispatch_order", ())) != tuple(DISPATCH_ORDER):
            raise ValueError("dispatch order differs from the frozen contract")
        if tuple(self.payload.get("status_order", ())) != tuple(STATUS_ORDER):
            raise ValueError("status order differs from the frozen contract")
        if self.rigid_demands != ("electricity", "cooling", "heating"):
            raise ValueError("gas cannot be a rigid terminal demand")
        if self.payload.get("lookback") != 24 or self.payload.get("horizon") != 4:
            raise ValueError("formal-v4.6 requires a 24-hour history and four-hour horizon")
        if self.control_dim != 15 or self.dispatch_dim != 21:
            raise ValueError("formal-v4.6 control dimensions are invalid")
        if self.payload.get("allow_future_binary_decisions") is not False:
            raise ValueError("future binary decisions are outside formal-v4.6")
        if self.payload.get("gas_semantics") != "station_side_auxiliary_prior":
            raise ValueError("gas must remain a station-side auxiliary prior")
        if self.methods != METHODS:
            raise ValueError("formal-v4.6 method matrix is incomplete or reordered")
        regime = self.payload.get("thermal_regime", {})
        if tuple(regime.get("classes", ())) != ("off", "cooling", "heating"):
            raise ValueError("thermal regime classes are invalid")
        if float(regime.get("epsilon", -1.0)) != 1.0e-9 or float(regime.get("laplace_alpha", -1.0)) != 1.0:
            raise ValueError("thermal regime numerical constants are not frozen")
        if regime.get("hard_calendar_mask") is not False or regime.get("hard_scheduling_gate") is not False:
            raise ValueError("hard calendar mask or scheduling gate is forbidden")
        if regime.get("zero_initialize_residuals") is not True:
            raise ValueError("residual heads must be zero initialized")
        if self.pilot_train_windows != 4096:
            raise ValueError("Pilot must use exactly 4096 training windows")
        candidates = tuple((float(row["temperature"]), float(row["inactive_leakage_weight"])) for row in self.candidate_grid)
        if candidates != ((1.0, 0.25), (0.75, 0.25), (1.0, 0.50), (0.75, 0.50)):
            raise ValueError("candidate grid differs from the frozen Pilot contract")
        if float(self.risk_adjustment.get("gas_adjustment", float("nan"))) != 0.0:
            raise ValueError("gas adjustment must be exactly zero")
        if dict(self.risk_adjustment) != _RISK_ADJUSTMENT:
            raise ValueError("risk-adjustment semantics differ from the frozen contract")
        for name, expected in _PILOT_THRESHOLDS.items():
            if float(self.payload["pilot_thresholds"].get(name, float("nan"))) != expected:
                raise ValueError("Pilot thresholds differ from the frozen contract")
        if dict(self.joint_curriculum) != _CURRICULUM:
            raise ValueError("joint curriculum differs from the frozen contract")
        if dict(self.forecast_guardrails) != _GUARDRAILS:
            raise ValueError("forecast guardrails differ from the frozen contract")
        if self.payload.get("allow_evaluation_access_before_gate2") is not False:
            raise ValueError("evaluation access must remain fail closed")

    def __getattr__(self, name: str) -> Any:
        try:
            return self.payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_formal_v4_6_contract(path: str | Path) -> FormalV46Contract:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("formal-v4.6 contract must be a JSON object")
    contract = FormalV46Contract(MappingProxyType(dict(payload)), hashlib.sha256(raw).hexdigest(), source)
    contract.validate()
    return contract


__all__ = ["FormalV46Contract", "SCHEMA_VERSION", "load_formal_v4_6_contract"]
