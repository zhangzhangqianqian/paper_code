"""Versioned 23-variable dispatch schema for the simulated heat-pump track."""

from __future__ import annotations

from typing import Dict, Tuple

from ..dispatch_schema import VARIABLES as LEGACY_VARIABLES


HEAT_PUMP_VARIABLES: Tuple[str, ...] = tuple(LEGACY_VARIABLES) + ("p_hp", "q_hp")
HEAT_PUMP_INDEX: Dict[str, int] = {
    name: index for index, name in enumerate(HEAT_PUMP_VARIABLES)
}

if len(HEAT_PUMP_VARIABLES) != 23 or len(HEAT_PUMP_INDEX) != 23:
    raise RuntimeError("heat-pump schema must contain 23 unique variables")


__all__ = ["HEAT_PUMP_INDEX", "HEAT_PUMP_VARIABLES"]
