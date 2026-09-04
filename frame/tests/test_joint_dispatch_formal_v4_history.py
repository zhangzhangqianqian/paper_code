from __future__ import annotations

import numpy as np

from src.joint_dispatch.contract import DISPATCH_ORDER, STATUS_ORDER
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries, derive_device_status
from src.joint_dispatch.formal_v4_history import generate_settled_device_trajectory
from src.scheduling.dispatch_lp import DispatchResult


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.2, "unserved_penalty": 100.0,
    "surplus_penalty": 0.1, "chp_ramp_fraction": 0.5,
}


def _base(periods: int = 80, *, start: str = "2018-07-01", gap: bool = False) -> FormalV4BaseSeries:
    timestamps = np.arange(np.datetime64(start), np.datetime64(start) + np.timedelta64(periods, "h"), np.timedelta64(1, "h"))
    if gap:
        timestamps[50:] += np.timedelta64(3, "h")
    t = np.arange(periods, dtype=np.float64)
    tasks = np.column_stack((10.0 + (t % 7.0), 4.0 + (t % 3.0), 5.0 + (t % 5.0), 2.0 + (t % 2.0)))
    exog = np.zeros((periods, 12), dtype=np.float64)
    load = np.concatenate((tasks, exog), axis=1)
    renew = np.column_stack((2.0 + 0.1 * (t % 5.0), 1.0 + 0.1 * (t % 4.0)))
    prices = np.tile(np.array([1.0, 0.6, 0.0]), (periods, 1))
    return FormalV4BaseSeries(load, renew.copy(), renew.copy(), prices, timestamps, "train")


class RecordingSolver:
    def __init__(self) -> None:
        self.inputs = []

    def __call__(self, inputs):
        self.inputs.append(inputs)
        values = {name: np.zeros(4, dtype=np.float64) for name in DISPATCH_ORDER}
        values["soc"][:] = 20.0
        return DispatchResult("optimal", "", 0.0, values, {}, 0.0)


def _receipt() -> dict[str, object]:
    return {"gate0_authorized": True, "capacity_audit": {"selected": {"multiplier": 1.0}}}


def test_trajectory_settles_realized_load_and_uses_only_causal_plan() -> None:
    base = _base()
    solver = RecordingSolver()
    out = generate_settled_device_trajectory(base, PARAMETERS, capacity_receipt=_receipt(), solver=solver)

    assert out.audit.future_label_reads == 0
    assert out.audit.first_model_origin_index == 48
    assert out.audit.solved_hours == len(base.timestamps) - 24 - 3
    assert out.audit.settled_hours == out.audit.solved_hours
    assert out.audit.max_balance_residual <= 1e-6
    assert out.audit.max_conversion_residual <= 1e-6
    assert np.array_equal(out.activity_indicators, derive_device_status(out.settled_dispatch.reshape(1, len(out.settled_dispatch), -1))[0])
    assert np.allclose(out.settled_dispatch[:24], 0.0)
    assert len(solver.inputs) == out.audit.solved_hours
    # At decision index 24, only the first four observed rows are used.
    assert np.array_equal(solver.inputs[0].demand, base.load_and_exog[:4, :3])
    # The first settled action is balanced against the current row (index 24),
    # not copied from the planned four-hour demand.
    assert np.isclose(out.settled_dispatch[24, DISPATCH_ORDER.index("grid")], base.load_and_exog[24, 0])


def test_gap_repeats_warmup_and_resets_state() -> None:
    base = _base(90, gap=True)
    out = generate_settled_device_trajectory(base, PARAMETERS, capacity_receipt=_receipt(), solver=RecordingSolver())
    assert out.audit.reset_indices == (0, 50)
    assert out.audit.first_model_origin_index == 48
    # After the gap, indices 50--73 are warm-up and 74--97 would be the
    # second eligible region; the short fixture ends before a second origin.
    assert not out.settled_mask[50:74].any()
    assert out.settled_mask[74]
    assert not bool(np.isnan(out.initial_soc[74, 0]))
    assert bool(np.isnan(out.initial_soc[50, 0]))


def test_continuous_year_boundary_carries_history() -> None:
    base = _base(80, start="2018-12-30")
    out = generate_settled_device_trajectory(base, PARAMETERS, capacity_receipt=_receipt(), solver=RecordingSolver())
    assert out.audit.reset_indices == (0,)
    assert out.audit.first_model_origin_index == 48
    # There is no artificial reset at the 2018/2019 calendar boundary.
    boundary = int(np.flatnonzero(np.asarray(base.timestamps).astype("datetime64[Y]") == np.datetime64("2019"))[0])
    assert out.settled_mask[boundary]
    assert np.isfinite(out.initial_soc[boundary, 0])
