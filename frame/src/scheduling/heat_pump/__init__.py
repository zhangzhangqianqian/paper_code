"""Isolated heat-pump extension for the simulated S-track scheduler.

The package intentionally does not mutate the legacy 21-variable dispatch
schema.  It is a separate boundary for the 23-variable simulation labels.
"""

from .parameters import HeatPumpParameters, heat_pump_parameter_hash, load_heat_pump_parameters
from .schema import HEAT_PUMP_INDEX, HEAT_PUMP_VARIABLES

__all__ = [
    "HEAT_PUMP_INDEX",
    "HEAT_PUMP_VARIABLES",
    "HeatPumpParameters",
    "heat_pump_parameter_hash",
    "load_heat_pump_parameters",
]
