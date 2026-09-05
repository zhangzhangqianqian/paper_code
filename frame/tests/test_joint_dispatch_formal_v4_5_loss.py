from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.joint_dispatch.formal_v4_5_contract import load_formal_v4_5_contract
from src.joint_dispatch.formal_v4_5_loss import (
    JointNormalizationV45,
    build_j_optimizer_v45,
    curriculum_weights_v45,
    joint_loss_v45,
)


CONTRACT = load_formal_v4_5_contract("configs/joint_forecast_dispatch_formal_v4_5.json")


def test_curriculum_is_monotone_and_positive() -> None:
    values = [curriculum_weights_v45(epoch, 5) for epoch in range(8)]
    assert values[0]["decision"] == pytest.approx(0.05)
    assert values[4]["decision"] == pytest.approx(0.50)
    assert values[0]["imitation"] == pytest.approx(1.0)
    assert values[4]["imitation"] == pytest.approx(0.25)
    assert all(row["forecast"] == 1.0 and row["anchor"] == 0.5 for row in values)
    assert all(values[i]["decision"] <= values[i + 1]["decision"] for i in range(7))


class _GroupedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.ones(1))
        self.gate = nn.Parameter(torch.ones(1))
        self.magnitude = nn.Parameter(torch.ones(1))
        self.scheduler = nn.Parameter(torch.ones(1))

    def v44_parameter_groups(self):
        return {
            "base": (self.base,), "gate": (self.gate,),
            "magnitude": (self.magnitude,), "scheduler": (self.scheduler,),
        }


def test_optimizer_records_three_joint_learning_rates_and_scheduler_only_decoupled() -> None:
    joint = build_j_optimizer_v45(_GroupedModel(), CONTRACT, mode="joint")
    assert [(group["name"], group["lr"]) for group in joint.param_groups] == [
        ("base", 0.0002), ("head", 0.0005), ("scheduler", 0.0005),
    ]
    decoupled = build_j_optimizer_v45(_GroupedModel(), CONTRACT, mode="decoupled")
    assert [(group["name"], group["lr"]) for group in decoupled.param_groups] == [("scheduler", 0.0005)]


def test_joint_loss_has_anchor_and_nonzero_forecast_gradient() -> None:
    forecast = torch.nn.Parameter(torch.ones(2, 4, 4))
    dispatch = forecast[..., :2].sum(dim=-1, keepdim=True).expand(2, 4, 21)
    output = SimpleNamespace(forecast_normalized=forecast, dispatch=dispatch)
    parent = SimpleNamespace(forecast_normalized=torch.zeros_like(forecast))
    loss = joint_loss_v45(
        output, parent, {"target_normalized": torch.zeros_like(forecast)},
        JointNormalizationV45(1.0, 1.0, 1.0), curriculum_weights_v45(0),
    )
    loss.total.backward()
    assert float(loss.anchor.detach()) > 0.0
    assert forecast.grad is not None and float(forecast.grad.norm()) > 0.0


def test_decision_term_can_be_detached_for_decoupled_branch() -> None:
    forecast = torch.nn.Parameter(torch.ones(1, 4, 4))
    parent = SimpleNamespace(forecast_normalized=torch.zeros_like(forecast))
    output = SimpleNamespace(forecast_normalized=forecast, dispatch=forecast)
    loss = joint_loss_v45(
        output, parent, {"target_normalized": torch.zeros_like(forecast)},
        JointNormalizationV45(1.0, 1.0, 1.0), curriculum_weights_v45(0),
        decision_loss=output.dispatch.detach().square().mean(),
    )
    loss.total.backward()
    assert forecast.grad is not None
