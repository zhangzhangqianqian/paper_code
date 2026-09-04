from __future__ import annotations

import numpy as np
import torch

from src.joint_dispatch.formal_v4_baselines import (
    DirectPolicyAdapter,
    PerfectInformationMPC,
    Scheme2RPTOMetadata,
    SeasonalNaivePTO,
    StateConditionedPTO,
)


def test_scheme2r_pto_has_no_state_encoder():
    baseline = Scheme2RPTOMetadata()
    assert baseline.forecaster_class == "Scheme2RModel"
    assert baseline.uses_device_history is False
    assert baseline.online_lp_calls_per_window == 1


def test_state_conditioned_pto_reuses_stage_p_exactly():
    baseline = StateConditionedPTO(stage_p_checkpoint_sha256="stage-p-hash")
    assert baseline.forecaster_sha256 == "stage-p-hash"
    assert baseline.online_lp_calls_per_window == 1


def test_pi_mpc_is_reference_only():
    baseline = PerfectInformationMPC()
    assert baseline.uses_realized_future is True
    assert baseline.deployable is False


def test_direct_policy_has_no_forecast_metric_rows():
    baseline = DirectPolicyAdapter(dropout=0.0)
    assert baseline.forecast_metrics_applicable is False
    inputs = {
        "load_history": torch.rand(1, 24, 4), "exog_history": torch.rand(1, 24, 12),
        "device_history": torch.rand(1, 24, 17), "activity_history": torch.zeros(1, 24, 6),
        "scheduler_context": torch.cat((torch.rand(1, 4, 5), torch.full((1, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.zeros(1, 1),
    }
    assert baseline.model(**inputs).forecast_physical is None


def test_seasonal_naive_uses_only_24_hour_causal_lag():
    baseline = SeasonalNaivePTO()
    values = np.arange(40 * 4, dtype=np.float64).reshape(40, 4)
    result = baseline.forecast(values, origin_index=28)
    np.testing.assert_array_equal(result, values[4:8, :3])

