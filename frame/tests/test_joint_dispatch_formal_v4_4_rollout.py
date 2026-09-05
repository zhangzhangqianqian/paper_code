from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.joint_dispatch.formal_v4_2_data import fit_train_normalization
from src.joint_dispatch.formal_v4_4_pilot_materializer import materialize_v44_pilot_data
from src.joint_dispatch.formal_v4_4_rollout import evaluate_full_2019_rollout_v44
from src.joint_dispatch.model import JointForecastDispatchModel


class _RolloutModel:
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, **inputs):
        batch = inputs["load_history"].shape[0]
        return SimpleNamespace(
            forecast_physical=torch.ones(batch, 4, 4),
            regime_probabilities=torch.full((batch, 4, 3), 1.0 / 3.0),
            dispatch=torch.zeros(batch, 4, 21),
        )


def test_rollout_uses_full_selection_chronology_and_carries_state(tmp_path, monkeypatch) -> None:
    from tests.test_joint_dispatch_formal_v4_4_teacher import _teacher_fixture
    sources, materialized = _teacher_fixture(tmp_path, monkeypatch)
    normalization = fit_train_normalization(materialized.normalization_source.split)
    parameters = dict(JointForecastDispatchModel._test_parameters())
    parameters.update({
        "unserved_penalty": 100.0, "surplus_penalty": 0.0, "grid_energy_price": 1.0,
        "gas_energy_price": 0.6, "carbon_price_default": 0.0, "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25, "bess_throughput_cost": 1.0e-6,
    })
    result = evaluate_full_2019_rollout_v44(
        model=_RolloutModel(), selection=materialized.selection_full, normalization=normalization,
        parameters=parameters, method_id="test",
    )
    assert result.prediction.shape == (len(materialized.selection_full), 4, 4)
    assert result.planned_dispatch.shape == (len(materialized.selection_full), 4, 21)
    assert result.settled_dispatch.shape == (len(materialized.selection_full), 21)
    assert np.all(np.diff(result.times).astype("timedelta64[h]") > np.timedelta64(0, "h"))
    assert np.isfinite(result.physical_residual).all()
