"""Validated, hashable heat-pump parameters for the simulated S track."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class HeatPumpParameters:
    """First-round heating-only heat-pump assumptions.

    ``heat_capacity`` is thermal output capacity.  The electric input bound is
    therefore ``heat_capacity / cop``.
    """

    cop: float
    heat_capacity: float
    variable_om_cost: float = 0.0

    def validate(self) -> None:
        if not math.isfinite(float(self.cop)) or float(self.cop) <= 0.0:
            raise ValueError("heat_pump_cop must be finite and positive")
        if not math.isfinite(float(self.heat_capacity)) or float(self.heat_capacity) < 0.0:
            raise ValueError("heat_pump_heat_capacity must be finite and non-negative")
        if not math.isfinite(float(self.variable_om_cost)) or float(self.variable_om_cost) < 0.0:
            raise ValueError("heat_pump_variable_om_cost must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        self.validate()
        return {key: float(value) for key, value in asdict(self).items()}


def _read_mapping(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("heat-pump parameter file must contain a JSON object")
    return payload


def _field(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise ValueError(f"missing heat-pump parameter: {names[0]}")


def load_heat_pump_parameters(path: str | Path) -> HeatPumpParameters:
    payload = _read_mapping(path)
    nested = payload.get("heat_pump", payload)
    if not isinstance(nested, Mapping):
        raise ValueError("heat_pump parameter section must be an object")
    params = HeatPumpParameters(
        cop=float(_field(nested, "heat_pump_cop", "cop")),
        heat_capacity=float(_field(nested, "heat_pump_heat_capacity", "heat_capacity")),
        variable_om_cost=float(nested.get("heat_pump_variable_om_cost", nested.get("variable_om_cost", 0.0))),
    )
    params.validate()
    return params


def heat_pump_parameter_hash(parameters: HeatPumpParameters | Mapping[str, Any]) -> str:
    if isinstance(parameters, HeatPumpParameters):
        canonical = parameters.to_dict()
    else:
        canonical = {
            "cop": float(parameters["cop"]),
            "heat_capacity": float(parameters["heat_capacity"]),
            "variable_om_cost": float(parameters.get("variable_om_cost", 0.0)),
        }
        HeatPumpParameters(**canonical).validate()
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["HeatPumpParameters", "heat_pump_parameter_hash", "load_heat_pump_parameters"]
