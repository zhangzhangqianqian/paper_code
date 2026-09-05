"""Frozen protocol boundary for formal-v4.4 development and Pilot."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER


SCHEMA_VERSION = "joint-forecast-dispatch-formal-v4.4"
METHODS = (
    "RSC-PF", "Fair Decoupled", "Direct-Policy", "Scheme2R-PTO",
    "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP",
    "Perfect-Information-MPC", "Seasonal-Naive-PTO",
)


def _tuple(payload: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return tuple(value)


@dataclass(frozen=True)
class FormalV44Contract:
    payload: Mapping[str, Any]
    contract_sha256: str
    source_path: Path

    @property
    def train_years(self) -> tuple[int, ...]:
        return tuple(int(v) for v in _tuple(self.payload, "train_years"))

    @property
    def selection_year(self) -> int:
        return int(self.payload["selection_year"])

    @property
    def evaluation_year(self) -> int:
        return int(self.payload["evaluation_year"])

    @property
    def excluded_years(self) -> tuple[int, ...]:
        return tuple(int(v) for v in _tuple(self.payload, "excluded_years"))

    @property
    def forecast_tasks(self) -> tuple[str, ...]:
        return tuple(str(v) for v in _tuple(self.payload, "task_order"))

    @property
    def rigid_demands(self) -> tuple[str, ...]:
        return tuple(str(v) for v in _tuple(self.payload, "rigid_demand_order"))

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
    def pilot_thresholds(self) -> Mapping[str, float]:
        return MappingProxyType(dict(self.payload["pilot_thresholds"]))

    def validate(self) -> None:
        if self.payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("formal-v4.4 schema required")
        if self.payload.get("protocol_status") != "frozen":
            raise ValueError("formal-v4.4 contract is not frozen")
        if self.train_years != (2015, 2016, 2017, 2018) or self.selection_year != 2019:
            raise ValueError("formal-v4.4 training and selection years are invalid")
        if self.evaluation_year != 2020 or self.excluded_years != (2021,):
            raise ValueError("formal-v4.4 evaluation boundary is invalid")
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
            raise ValueError("formal-v4.4 requires a 24-hour history and four-hour horizon")
        if self.control_dim != 15 or self.dispatch_dim != 21:
            raise ValueError("formal-v4.4 control dimensions are invalid")
        if self.payload.get("allow_future_binary_decisions") is not False:
            raise ValueError("future binary decisions are outside formal-v4.4")
        if self.payload.get("gas_semantics") != "station_side_auxiliary_prior":
            raise ValueError("gas must remain a station-side auxiliary prior")
        if self.methods != METHODS:
            raise ValueError("formal-v4.4 method matrix is incomplete or reordered")
        regime = self.payload.get("thermal_regime", {})
        if tuple(regime.get("classes", ())) != ("off", "cooling", "heating"):
            raise ValueError("thermal regime classes are invalid")
        if float(regime.get("epsilon", -1.0)) != 1.0e-9 or float(regime.get("laplace_alpha", -1.0)) != 1.0:
            raise ValueError("thermal regime numerical constants are not frozen")
        if regime.get("hard_calendar_mask") is not False or regime.get("hard_scheduling_gate") is not False:
            raise ValueError("hard calendar mask or scheduling gate is forbidden")
        if regime.get("zero_initialize_residuals") is not True:
            raise ValueError("residual heads must be zero initialized")
        if int(self.payload["pilot_budget"]["train_windows"]) != 4096:
            raise ValueError("Pilot must use exactly 4096 training windows")
        candidates = tuple((float(row["temperature"]), float(row["inactive_leakage_weight"])) for row in self.candidate_grid)
        if candidates != ((1.0, 0.25), (0.75, 0.25), (1.0, 0.5), (0.75, 0.5)):
            raise ValueError("candidate grid differs from the frozen Pilot contract")
        if tuple(float(v) for v in self.payload["pilot_thresholds"].values())[:6] != (0.30, 0.05, 0.02, 0.02, 0.01, 1.0e-12):
            raise ValueError("Pilot thresholds differ from the frozen contract")
        if float(self.payload["pilot_thresholds"]["maximum_physical_residual"]) != 1.0e-6:
            raise ValueError("physical residual tolerance differs from the frozen contract")
        if self.payload.get("allow_evaluation_access_before_gate2") is not False:
            raise ValueError("evaluation access must remain fail closed")

    def __getattr__(self, name: str) -> Any:
        try:
            return self.payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def load_formal_v4_4_contract(path: str | Path) -> FormalV44Contract:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("formal-v4.4 contract must be a JSON object")
    contract = FormalV44Contract(MappingProxyType(dict(payload)), hashlib.sha256(raw).hexdigest(), source)
    contract.validate()
    return contract


__all__ = ["FormalV44Contract", "METHODS", "SCHEMA_VERSION", "load_formal_v4_4_contract"]
