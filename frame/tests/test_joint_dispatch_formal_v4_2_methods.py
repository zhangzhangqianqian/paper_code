from __future__ import annotations

import numpy as np
import pytest
import torch

from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract
from src.joint_dispatch.formal_v4_2_methods import (
    DEPLOYABLE_METHODS,
    build_v42_method,
    registered_method_rows,
)
from src.joint_dispatch.formal_v4_state import FormalV4ClosedLoopState
from src.scheduling.dispatch_schema import VARIABLES


def _state(value: float = 0.5) -> FormalV4ClosedLoopState:
    return FormalV4ClosedLoopState(
        torch.zeros(1, 24, 4), torch.zeros(1, 24, 12), torch.zeros(1, 24, 17), torch.zeros(1, 24, 6),
        torch.full((1, 1), value), torch.zeros(1, 1), np.datetime64("2015-01-01T00:00"), "trajectory-a",
    )


def _planner(window, state):
    plan = np.zeros((4, len(VARIABLES)), dtype=np.float64)
    plan[:, 0] = float(state.initial_soc[0, 0])
    return {"dispatch": plan, "forecast": np.ones((4, 4))}


def _checkpoint():
    return {"model_sha256": "a" * 64, "optimizer_steps": 1}


def test_registered_rows_cover_full_gate2_matrix():
    contract = load_formal_v4_2_contract("configs/joint_forecast_dispatch_formal_v4_2.json")
    rows = registered_method_rows(contract, gate="gate2")
    stochastic = [row for row in rows if row.seed is not None]
    deterministic = [row for row in rows if row.seed is None]
    assert len(stochastic) == 7 * 3
    assert {row.method_id for row in deterministic} == {"Perfect-Information-MPC", "Seasonal-Naive-PTO"}


def test_methods_consume_supplied_rolling_state():
    for method_id in DEPLOYABLE_METHODS:
        method = build_v42_method(method_id, seed=2026, planner=_planner, checkpoint=_checkpoint())
        a = method.plan({}, _state(0.2)); b = method.plan({}, _state(0.8))
        assert a.state_sha256 != b.state_sha256
        assert a.four_hour_plan.shape == (4, 21)


def test_online_optimizer_call_counts_are_truthful():
    assert build_v42_method("RSC-PF", planner=_planner, checkpoint=_checkpoint()).plan({}, _state()).optimizer_calls == 0
    assert build_v42_method("Scheme2R-PTO", planner=_planner, checkpoint=_checkpoint()).plan({}, _state()).optimizer_calls == 1


def test_state_conditioned_pto_loads_exact_stage_p_checkpoint():
    receipt = _checkpoint()
    method = build_v42_method("State-Conditioned-PTO", seed=2026, checkpoint=receipt, planner=_planner)
    assert method.forecaster_sha256 == receipt["model_sha256"]


def test_perfect_information_is_not_deployable():
    method = build_v42_method("Perfect-Information-MPC", planner=_planner)
    assert method.deployable is False


def test_stochastic_method_rejects_missing_checkpoint():
    with pytest.raises(ValueError, match="trained checkpoint"):
        build_v42_method("RSC-PF", seed=2026, planner=_planner)


def test_stochastic_method_rejects_untrained_checkpoint_marker():
    with pytest.raises(ValueError, match="not marked as trained"):
        build_v42_method(
            "Direct-Policy",
            seed=2026,
            planner=_planner,
            checkpoint={"model_sha256": "b" * 64, "optimizer_steps": 0},
        )
