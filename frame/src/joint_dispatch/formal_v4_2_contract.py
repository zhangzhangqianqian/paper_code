"""Frozen protocol boundary for the RSC-PF formal-v4.2 experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .contract import DISPATCH_ORDER, EXOG_ORDER, STATUS_ORDER, TASK_ORDER


SCHEMA_VERSION = "joint-forecast-dispatch-formal-v4.2"
METHODS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "State-Conditioned-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
    "Perfect-Information-MPC",
    "Seasonal-Naive-PTO",
)
_TRANSITIONS = {
    ("gate0", "pilot"),
    ("pilot", "gate1"),
    ("gate1", "gate2"),
    ("gate2", "gate2_audit"),
    ("gate2_audit", "seed_extension"),
    ("seed_extension", "gate3"),
    ("gate3", "summary"),
}


def _tuple(payload: Mapping[str, Any], key: str) -> tuple[Any, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return tuple(value)


@dataclass(frozen=True)
class FormalV42Contract:
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
    def gate2_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["selection"]["gate2_seeds"])

    @property
    def seed_extension_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["selection"]["seed_extension_seeds"])

    @property
    def gate3_seeds(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["selection"]["gate3_seeds"])

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(str(row["name"]) for row in self.payload["methods"])

    def __getattr__(self, name: str) -> Any:
        try:
            return self.payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def validate(self) -> None:
        if self.payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("formal-v4.2 schema required")
        if self.payload.get("protocol_status") != "frozen":
            raise ValueError("formal-v4.2 contract is not frozen")
        if self.train_years != (2015, 2016, 2017, 2018):
            raise ValueError("formal-v4.2 training years must be 2015-2018")
        if (self.selection_year, self.evaluation_year, self.excluded_years) != (2019, 2020, (2021,)):
            raise ValueError("formal-v4.2 year boundary is invalid")
        if tuple(self.payload.get("task_order", ())) != tuple(TASK_ORDER):
            raise ValueError("task order differs from the frozen contract")
        if tuple(self.payload.get("exog_order", ())) != tuple(EXOG_ORDER):
            raise ValueError("exogenous order differs from the frozen contract")
        if tuple(self.payload.get("dispatch_order", ())) != tuple(DISPATCH_ORDER):
            raise ValueError("dispatch order differs from the frozen contract")
        if tuple(self.payload.get("status_order", ())) != tuple(STATUS_ORDER):
            raise ValueError("status order differs from the frozen contract")
        if self.payload.get("lookback") != 24 or self.payload.get("horizon") != 4:
            raise ValueError("formal-v4.2 requires a 24-hour history and four-hour horizon")
        if self.payload.get("latent_control_dim") != 15 or len(self.dispatch_order) != 21:
            raise ValueError("formal-v4.2 control dimensions are invalid")
        if self.payload.get("allow_future_binary_decisions") is not False:
            raise ValueError("future binary decisions are outside formal-v4.2")
        if self.payload.get("gas_semantics") != "station_side_auxiliary_prior":
            raise ValueError("gas must remain a station-side auxiliary prior")
        if self.methods != METHODS:
            raise ValueError("formal-v4.2 method matrix is incomplete or reordered")
        if self.gate2_seeds != (2026, 2027, 2028):
            raise ValueError("Gate 2 seeds are not frozen")
        if self.seed_extension_seeds != (2029, 2030):
            raise ValueError("seed-extension seeds are not frozen")
        if self.gate3_seeds != (2026, 2027, 2028, 2029, 2030):
            raise ValueError("Gate 3 seeds are not frozen")
        if self.payload["training"].get("stage_order") != ["P", "teacher", "S", "clone", "J"]:
            raise ValueError("training stages are not strictly sequential")
        selection = self.payload.get("selection", {})
        if selection.get("gate1_candidate_parameter") != "stage_j_forecaster_learning_rate_multiplier":
            raise ValueError("Gate 1 candidate parameter is not frozen")
        if tuple(float(value) for value in selection.get("gate1_candidate_values", ())) != (1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
            raise ValueError("Gate 1 candidate values are not frozen")
        pilot = self.payload.get("pilot")
        if not isinstance(pilot, Mapping):
            raise ValueError("formal-v4.2 pilot budget is missing")
        expected_pilot = {
            "seed": 2026, "segment_hours": 96, "segments": 4, "windows": 128,
            "batch_size": 32, "stage_epochs": 3, "rollout_windows": 24,
            "shortage_rate_max": 0.80, "forecaster_learning_rate": 0.001,
            "scheduler_learning_rate": 0.001,
        }
        if set(pilot) != set(expected_pilot):
            raise ValueError("formal-v4.2 pilot fields are not frozen")
        for name, expected in expected_pilot.items():
            actual = pilot[name]
            if isinstance(expected, float):
                if float(actual) != expected:
                    raise ValueError(f"formal-v4.2 pilot {name} is not frozen")
            elif int(actual) != expected:
                raise ValueError(f"formal-v4.2 pilot {name} is not frozen")
        if self.payload.get("allow_evaluation_access_before_gate3") is not False:
            raise ValueError("evaluation access must remain fail-closed")


def load_formal_v4_2_contract(path: str | Path) -> FormalV42Contract:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError("formal-v4.2 contract must be a JSON object")
    contract = FormalV42Contract(
        payload=MappingProxyType(dict(payload)),
        contract_sha256=hashlib.sha256(raw).hexdigest(),
        source_path=source,
    )
    contract.validate()
    return contract


def assert_gate_transition(
    contract: FormalV42Contract,
    completed_gate: str,
    requested_gate: str,
) -> None:
    contract.validate()
    transition = (str(completed_gate), str(requested_gate))
    if transition not in _TRANSITIONS:
        raise ValueError(f"invalid formal-v4.2 gate transition: {transition}")


__all__ = [
    "FormalV42Contract",
    "METHODS",
    "SCHEMA_VERSION",
    "assert_gate_transition",
    "load_formal_v4_2_contract",
]
