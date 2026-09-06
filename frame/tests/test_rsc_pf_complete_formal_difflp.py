from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from src.joint_dispatch.complete_formal_difflp import (
    CompleteDiffLPProblemSpec,
    build_difflp_provider,
    one_difflp_training_step,
    train_complete_difflp,
)
from src.joint_dispatch.complete_formal_providers import CompletePlannedStep
from src.joint_dispatch.complete_formal_contract import CompleteFormalContract, MethodSeedKey
from src.joint_dispatch.complete_formal_providers import build_complete_provider_registry
from src.joint_dispatch.formal_v4_diffopt import RIGID_TASKS
from src.joint_dispatch.matched_closed_loop import CausalOriginInput
from src.joint_dispatch.contract import DISPATCH_ORDER
from src.models import Scheme2RModel


class _NativeLayer(nn.Module):
    def forward(self, demand, renewable, prices, initial_soc, previous_chp):
        zero = demand[..., 0] * 0.0
        fields = [zero for _ in range(21)]
        fields[0] = demand[..., 0]
        fields[2] = renewable[..., 0]
        fields[4] = renewable[..., 1]
        fields[18] = demand[..., 1]
        fields[19] = demand[..., 2]
        return torch.stack(fields, dim=-1)


class _ForecastModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(4, 4))

    def forward(self, load_history, exog_history):
        return self.bias.unsqueeze(0).expand(load_history.shape[0], -1, -1)


def _batch():
    return {
        "load_history": torch.ones(1, 24, 4),
        "exog_history": torch.zeros(1, 24, 12),
        "target_normalized": torch.full((1, 4, 4), 0.5),
        "renewable_forecast": torch.zeros(1, 4, 2),
        "prices_and_weights": torch.zeros(1, 4, 3),
        "initial_soc": torch.full((1, 1), 0.5),
        "previous_chp": torch.zeros(1, 1),
    }


def _origin():
    context = np.zeros((1, 4, 6), dtype=np.float32)
    context[:, :, 5] = 0.5
    return CausalOriginInput(
        load_history=np.zeros((1, 24, 4), dtype=np.float32),
        exog_history=np.zeros((1, 24, 12), dtype=np.float32),
        device_history=np.zeros((1, 24, 17), dtype=np.float32),
        activity_history=np.zeros((1, 24, 6), dtype=np.float32),
        scheduler_context=context,
        previous_chp=np.zeros((1, 1), dtype=np.float32),
        renewable_forecast=np.zeros((4, 2), dtype=np.float32),
        origin_time=np.datetime64("2019-01-01T00"),
        trajectory_id="capacity_bound_causal",
    )


def _parameters():
    return {
        "grid_import_capacity": 20.0,
        "chp_electric_capacity": 10.0,
        "chp_heat_capacity": 12.0,
        "gas_boiler_capacity": 20.0,
        "electric_chiller_capacity": 20.0,
        "absorption_chiller_capacity": 20.0,
        "bess_power_capacity": 5.0,
        "bess_energy_capacity": 20.0,
        "chp_electric_efficiency": 0.4,
        "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9,
        "electric_chiller_cop": 3.0,
        "absorption_chiller_cop": 0.8,
        "bess_roundtrip_efficiency": 0.9,
        "bess_throughput_cost": 1.0e-6,
        "unserved_penalty": 100.0,
        "chp_ramp_fraction": 1.0,
    }


def test_difflp_decision_gradient_reaches_forecaster():
    model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0)
    artifact = one_difflp_training_step(model, _NativeLayer(), _batch())
    assert artifact.lp_calls == 1
    assert artifact.forecaster_gradient_norm > 0.0
    assert artifact.decision_forecaster_gradient_norm > 0.0
    assert np.isfinite(artifact.forecaster_gradient_norm)


def test_difflp_provider_reports_one_solve_per_origin():
    provider = build_difflp_provider(
        _parameters(),
        forecaster=_ForecastModel(),
        layer=_NativeLayer(),
    )
    step = provider.plan(_origin())
    assert isinstance(step, CompletePlannedStep)
    assert step.inference_lp_calls == 1
    assert step.forecast.shape == (4, 4)
    assert step.dispatch.shape == (4, 21)


def test_difflp_rigid_balance_uses_only_three_carriers():
    problem_spec = CompleteDiffLPProblemSpec()
    assert problem_spec.rigid_demand_tasks == RIGID_TASKS
    assert problem_spec.rigid_demand_tasks == ("electricity", "cooling", "heating")
    assert problem_spec.output_dimension == len(DISPATCH_ORDER)


def test_difflp_provider_does_not_require_or_accept_gas_as_rigid_balance():
    provider = build_difflp_provider(
        _parameters(),
        forecaster=_ForecastModel(),
        layer=_NativeLayer(),
    )
    step = provider.plan(_origin())
    gas_changed = _origin()
    gas_changed = CausalOriginInput(
        load_history=gas_changed.load_history.copy(),
        exog_history=gas_changed.exog_history.copy(),
        device_history=gas_changed.device_history.copy(),
        activity_history=gas_changed.activity_history.copy(),
        scheduler_context=gas_changed.scheduler_context.copy(),
        previous_chp=gas_changed.previous_chp.copy(),
        renewable_forecast=gas_changed.renewable_forecast.copy(),
        origin_time=gas_changed.origin_time,
        trajectory_id=gas_changed.trajectory_id,
    )
    assert step.forecast.shape == (4, 4)
    assert provider.problem_spec.rigid_demand_tasks == ("electricity", "cooling", "heating")


def test_difflp_is_integrated_into_complete_registry():
    frame_root = Path(__file__).resolve().parents[1]
    contract_path = frame_root / "configs" / "rsc_pf_complete_formal_v1.json"
    registry = build_complete_provider_registry()
    registry.validate(CompleteFormalContract.from_path(contract_path))
    provider = registry.build(
        MethodSeedKey("Differentiable-LP", 2026),
        {
            "contract_path": contract_path,
            "difflp": {
                "parameters": _parameters(),
                "forecaster": _ForecastModel(),
                "layer": _NativeLayer(),
            },
        },
    )
    assert provider.plan(_origin()).inference_lp_calls == 1


def test_complete_difflp_training_writes_lineage_receipt(tmp_path, monkeypatch):
    import src.joint_dispatch.complete_formal_difflp as difflp

    artifact = SimpleNamespace(
        training_receipt={"sample_exposures": 4, "expected_sample_exposures": 4},
        decision_forecaster_gradient_norm=0.25,
        checkpoint_sha256="a" * 64,
    )
    monkeypatch.setattr(difflp, "train_differentiable_lp", lambda *args, **kwargs: artifact)
    train_complete_difflp(
        2026, None, {}, {"parity_residual": 1.0e-8}, _parameters(), tmp_path,
        model=_ForecastModel(), layer=_NativeLayer(),
    )
    receipt = (tmp_path / "COMPLETE_DIFFLP_RECEIPT.json").read_text(encoding="utf-8")
    assert "cvxpylayers_methodology_adaptation" in receipt
    assert "inference_lp_calls_per_origin" in receipt
    assert "a" * 64 in receipt
