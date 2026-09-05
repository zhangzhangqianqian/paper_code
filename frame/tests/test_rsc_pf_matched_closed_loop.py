from dataclasses import replace

import numpy as np
import pytest

from src.joint_dispatch.external_v46_data import ExternalV46Split
from src.joint_dispatch.matched_closed_loop import (
    CausalOriginInput,
    PlannedStep,
    initial_state_from_selection,
    planned_horizon_residuals,
    run_matched_closed_loop,
    split_origin,
)
from src.scheduling.dispatch_schema import VARIABLES


def _parameters() -> dict[str, float]:
    return {
        "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0,
        "chp_heat_capacity": 30.0, "gas_boiler_capacity": 40.0,
        "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
        "bess_power_capacity": 10.0, "bess_energy_capacity": 100.0,
        "chp_electric_efficiency": 0.35, "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
        "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9,
        "bess_throughput_cost": 1e-6, "grid_energy_price": 1.0,
        "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25, "carbon_price_default": 0.0,
        "unserved_penalty": 100.0, "chp_ramp_fraction": 0.5,
    }


def _selection(n: int = 4) -> ExternalV46Split:
    return ExternalV46Split(
        load_history=np.zeros((n, 24, 4), dtype=np.float32),
        exog_history=np.zeros((n, 24, 12), dtype=np.float32),
        device_history=np.zeros((n, 24, 17), dtype=np.float32),
        device_status=np.zeros((n, 24, 6), dtype=np.float32),
        scheduler_context=np.zeros((n, 4, 6), dtype=np.float32),
        previous_chp=np.zeros((n, 1), dtype=np.float32),
        forecast_target=np.zeros((n, 4, 4), dtype=np.float32),
        renewable_realized=np.zeros((n, 4, 2), dtype=np.float32),
        target_times=np.arange(np.datetime64("2019-01-01T00"), np.datetime64("2019-01-01T00") + n * np.timedelta64(1, "h"), np.timedelta64(1, "h")),
        split="pilot",
        trajectory_ids=np.full(n, "capacity_bound_causal"),
        source_state_hashes=np.full(n, "source-hash"),
    )


class ZeroProvider:
    method_id = "stub"
    optimizer_role = "none at inference"

    def __init__(self) -> None:
        self.seen_soc: list[float] = []
        self.seen_previous_chp: list[float] = []

    def plan(self, origin: CausalOriginInput) -> PlannedStep:
        self.seen_soc.append(float(origin.scheduler_context[0, 0, 5]))
        self.seen_previous_chp.append(float(origin.previous_chp[0, 0]))
        return PlannedStep(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 2)), np.zeros((4, len(VARIABLES))), 0)


def test_runner_rejects_non_hourly_selection():
    selection = _selection(3)
    bad_times = selection.target_times.copy()
    bad_times[2] += np.timedelta64(2, "h")
    with pytest.raises(ValueError, match="consecutive"):
        run_matched_closed_loop(replace(selection, target_times=bad_times), ZeroProvider(), _parameters(), method_id="stub", seed=2026, warmup_origins=0)


def test_causal_split_has_no_future_label_attributes():
    selection = _selection(2)
    state = initial_state_from_selection(selection)
    causal, labels = split_origin(selection, 0, state)
    assert not hasattr(causal, "forecast_target")
    assert not hasattr(causal, "renewable_realized")
    assert labels.forecast_target.shape == (4, 4)
    assert labels.renewable_realized.shape == (4, 2)


def test_next_origin_receives_previous_settled_soc_and_chp():
    selection = _selection(3)
    provider = ZeroProvider()
    result = run_matched_closed_loop(selection, provider, _parameters(), method_id="stub", seed=2026, warmup_origins=0)
    assert np.isclose(provider.seen_soc[1], result.arrays["final_soc"][0])
    assert np.isclose(provider.seen_previous_chp[1], result.arrays["settled_dispatch"][0, VARIABLES.index("p_chp")])
    assert len(set(result.arrays["state_hashes"].tolist())) == 3


def test_planned_horizon_residuals_inspect_all_four_rows():
    selection = _selection(1)
    state = initial_state_from_selection(selection)
    dispatch = np.zeros((4, len(VARIABLES)))
    dispatch[2, VARIABLES.index("p_chp")] = 20.0
    plan = PlannedStep(np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 2)), dispatch, 0)
    residuals = planned_horizon_residuals(plan, state, _parameters())
    assert residuals.shape == (4, 8)
    assert residuals[2, 4] > 0.0


def test_provider_output_is_independent_of_labels():
    selection = _selection(1)
    state = initial_state_from_selection(selection)
    causal, labels = split_origin(selection, 0, state)
    provider = ZeroProvider()
    first = provider.plan(causal)
    poisoned = replace(labels, forecast_target=np.full((4, 4), 1.0e6), renewable_realized=np.full((4, 2), 1.0e6))
    second = provider.plan(causal)
    assert np.array_equal(first.dispatch, second.dispatch)
    assert not np.array_equal(labels.forecast_target, poisoned.forecast_target)


def test_runner_counts_provider_inference_calls():
    class OneCallProvider(ZeroProvider):
        def plan(self, origin):
            planned = super().plan(origin)
            return replace(planned, inference_lp_calls=1)

    result = run_matched_closed_loop(_selection(4), OneCallProvider(), _parameters(), method_id="pto", seed=2026, warmup_origins=0)
    assert result.inference_lp_calls == 4
