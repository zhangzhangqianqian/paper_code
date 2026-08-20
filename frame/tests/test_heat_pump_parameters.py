from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduling.dispatch_schema import VARIABLES
from src.scheduling.heat_pump.parameters import load_heat_pump_parameters
from src.scheduling.heat_pump.schema import HEAT_PUMP_INDEX, HEAT_PUMP_VARIABLES


def test_heat_pump_schema_is_isolated_from_legacy_schema():
    assert len(VARIABLES) == 21
    assert "p_hp" not in VARIABLES and "q_hp" not in VARIABLES
    assert len(HEAT_PUMP_VARIABLES) == 23
    assert HEAT_PUMP_VARIABLES[-2:] == ("p_hp", "q_hp")
    assert HEAT_PUMP_INDEX["p_hp"] == 21
    assert HEAT_PUMP_INDEX["q_hp"] == 22


@pytest.mark.parametrize(
    ("field", "value"),
    [("heat_pump_cop", 0.0), ("heat_pump_cop", float("nan")), ("heat_pump_heat_capacity", -1.0)],
)
def test_heat_pump_parameters_fail_closed(tmp_path: Path, field: str, value: float):
    path = tmp_path / "params.json"
    payload = {"heat_pump_cop": 3.0, "heat_pump_heat_capacity": 574.1406}
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_heat_pump_parameters(path)


def test_heat_pump_parameters_accept_nested_payload(tmp_path: Path):
    path = tmp_path / "params.json"
    path.write_text(
        json.dumps({"heat_pump": {"cop": 3.0, "heat_capacity": 574.1406}}),
        encoding="utf-8",
    )
    params = load_heat_pump_parameters(path)
    assert params.cop == pytest.approx(3.0)
    assert params.heat_capacity == pytest.approx(574.1406)
