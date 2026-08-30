from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.joint_dispatch.model import JointForecastDispatchModel  # noqa: E402
from src.joint_dispatch.training import build_joint_optimizer, joint_train_step, mix_history_windows, rollin_refresh_epoch  # noqa: E402


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.0,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "unserved_penalty": 100.0, "chp_ramp_fraction": 0.5,
    "pv_capacity": 20.0,
}


def _batch(batch_size=2):
    return {
        "load_history": torch.rand(batch_size, 24, 4),
        "exog_history": torch.rand(batch_size, 24, 12),
        "device_history": torch.rand(batch_size, 24, 21),
        "device_status": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.cat((torch.rand(batch_size, 4, 5), torch.full((batch_size, 4, 1), 0.5)), dim=-1),
        "previous_chp": torch.zeros(batch_size, 1),
        "target_normalized": torch.rand(batch_size, 4, 4),
        "target_physical": torch.rand(batch_size, 4, 4),
        "teacher_dispatch": torch.rand(batch_size, 4, 21),
        "oracle_first_step_objective": torch.ones(batch_size),
    }


def test_one_optimizer_step_updates_forecaster_and_scheduler():
    model = JointForecastDispatchModel.for_test()
    optimizer = build_joint_optimizer(model)
    before_forecaster = [parameter.detach().clone() for parameter in model.forecaster.parameters()]
    before_scheduler = [parameter.detach().clone() for parameter in model.scheduler.parameters()]
    result = joint_train_step(model, _batch(), PARAMETERS, optimizer, epoch=0)
    assert result.gradient_audit.forecaster_nonzero
    assert result.gradient_audit.scheduler_nonzero
    assert any(not torch.equal(old, new) for old, new in zip(before_forecaster, model.forecaster.parameters()))
    assert any(not torch.equal(old, new) for old, new in zip(before_scheduler, model.scheduler.parameters()))


def test_frozen_variant_has_no_optimizer():
    model = JointForecastDispatchModel.for_test()
    optimizer = build_joint_optimizer(model, variant="frozen_pto")
    assert optimizer is None
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_rollin_refresh_and_mixture_are_deterministic():
    from src.joint_dispatch.data import JointWindowSplit
    import numpy as np

    def make(source):
        return JointWindowSplit(
            np.zeros((4, 24, 4), dtype=np.float32), np.zeros((4, 24, 12), dtype=np.float32),
            np.zeros((4, 24, 21), dtype=np.float32), np.zeros((4, 24, 6), dtype=np.float32),
            np.zeros((4, 4, 4), dtype=np.float32), np.zeros((4, 4, 6), dtype=np.float32),
            np.zeros((4, 1), dtype=np.float32), np.zeros((4, 4, 21), dtype=np.float32),
            np.zeros(4, dtype=np.float32), np.datetime64("2020-01-01") + np.arange(4, dtype="timedelta64[h]"), "train", source,
        )

    assert rollin_refresh_epoch(50) == 20
    assert rollin_refresh_epoch(2) == 1
    mixed_a = mix_history_windows(make("causal_lp"), make("joint_policy_rollin"), seed=4)
    mixed_b = mix_history_windows(make("causal_lp"), make("joint_policy_rollin"), seed=4)
    np.testing.assert_array_equal(mixed_a.target_times, mixed_b.target_times)
