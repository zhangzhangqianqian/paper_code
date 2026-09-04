from __future__ import annotations

import numpy as np
import pytest

from src.joint_dispatch.formal_v4_evaluation import evaluate_formal_v4_arrays


def _arrays():
    prediction = np.ones((2, 4, 4), dtype=float)
    target = np.ones((2, 4, 4), dtype=float)
    dispatch = np.zeros((2, 4, 21), dtype=float)
    dispatch[..., 0] = 3.0
    dispatch[..., 17:20] = 0.0
    demand = np.ones((2, 4, 3), dtype=float)
    return prediction, target, dispatch, demand


def test_forecast_summary_separates_rigid_and_gas():
    prediction, target, dispatch, demand = _arrays()
    result = evaluate_formal_v4_arrays("RSC-PF", prediction, target, dispatch, demand)
    assert result.forecast.rigid_macro_wape == pytest.approx(np.mean(result.forecast.task_wape[:3]))
    assert result.forecast.gas_wape == result.forecast.task_wape[3]
    assert not hasattr(result.forecast, "mean_source_unit_mae")


def test_policy_states_are_not_compared_to_offline_window_oracle():
    prediction, target, dispatch, demand = _arrays()
    result = evaluate_formal_v4_arrays("RSC-PF", prediction, target, dispatch, demand, perfect_information_objective=2.0)
    assert "per_window_regret_vs_oracle" not in result.metric_names
    assert "closed_loop_objective_difference_vs_pi" in result.metric_names


def test_zero_demand_horizon_does_not_poison_task_wape():
    from src.joint_dispatch.formal_v4_evaluation import ForecastMetricTable
    table = ForecastMetricTable(
        mae=np.ones((4, 4)), rmse=np.ones((4, 4)),
        wape=np.asarray([[0.1, np.inf, 0.2, np.nan]] * 4),
    )
    assert np.isfinite(table.task_wape).all()


def test_rollout_preserves_window_axis_and_counts_canonical_optimizer_calls():
    from src.joint_dispatch.formal_v4_evaluation import run_formal_v4_rollout

    class Method:
        method_id = "Seasonal-Naive-PTO"
        online_optimizer_calls_per_window = 1
        deployable = True

        def predict_and_dispatch(self, window, state):
            return {"forecast": np.ones((4, 4)), "target": np.ones((4, 4)), "dispatch": np.pad(np.ones((4, 1)), ((0, 0), (0, 20))), "demand": np.ones((4, 3)), "next_state": {"soc": 0.5, "previous_chp": 0.0}}

    result = run_formal_v4_rollout(Method(), [{"target_time": "2018-01-01T00"}, {"target_time": "2018-01-01T01"}], initial_state={"soc": 0.5})
    assert result.dispatch.optimizer_calls == 2


def test_rollout_rejects_disagreeing_legacy_optimizer_field():
    from src.joint_dispatch.formal_v4_evaluation import run_formal_v4_rollout

    class Method:
        method_id = "bad"
        online_optimizer_calls_per_window = 1
        online_lp_calls_per_window = 0

        def predict_and_dispatch(self, window, state):
            return {"forecast": None, "target": None, "dispatch": np.zeros((4, 21)), "demand": np.ones((4, 3)), "next_state": state}

    with pytest.raises(ValueError, match="disagree"):
        run_formal_v4_rollout(Method(), [{}], initial_state={})
