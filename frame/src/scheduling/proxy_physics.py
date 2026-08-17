"""Differentiable dispatch physics, economics, carbon and proxy losses."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .dispatch_lp import VARIABLES
from .proxy_contract import FEATURE_ORDER, LABEL_ORDER


_INDEX = {name: idx for idx, name in enumerate(LABEL_ORDER)}


def _parameter(parameters: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    value = parameters.get(name, default)
    return float(value)


def _check_dispatch(dispatch: Tensor) -> Tensor:
    if dispatch.ndim == 2:
        dispatch = dispatch.unsqueeze(0)
    if dispatch.ndim != 3 or dispatch.shape[-1] != len(LABEL_ORDER):
        raise ValueError("dispatch must have shape [B,H,21] or [H,21]")
    if not torch.isfinite(dispatch).all():
        raise ValueError("dispatch must be finite")
    return dispatch


def _check_inputs(inputs: Tensor, batch: int, horizon: int) -> Tensor:
    if inputs.ndim == 2:
        inputs = inputs.unsqueeze(0)
    if inputs.ndim != 3 or inputs.shape[0] != batch or inputs.shape[1] != horizon or inputs.shape[-1] != len(FEATURE_ORDER):
        raise ValueError("features must have shape [B,H,10]")
    if not torch.isfinite(inputs).all():
        raise ValueError("features must be finite")
    return inputs


def balance_residuals(dispatch: Tensor, features: Tensor) -> Tensor:
    """Return electricity/cooling/heating equality residuals ``[B,H,3]``."""

    dispatch = _check_dispatch(dispatch)
    # The public helper also accepts a bare ``[B,H,3]`` demand tensor for
    # numerical audits; the loss path passes the complete ten-feature input.
    if features.ndim == 3 and features.shape[-1] == 3:
        if features.shape[:2] != dispatch.shape[:2] or not torch.isfinite(features).all():
            raise ValueError("demand must have shape [B,H,3] and be finite")
        demand = features
    else:
        features = _check_inputs(features, dispatch.shape[0], dispatch.shape[1])
        demand = features[..., :3]
    g = lambda name: dispatch[..., _INDEX[name]]
    residual = torch.stack([
        g("grid") + g("pv_use") + g("wt_use") + g("p_chp") + g("p_discharge") + g("slack_e") - g("p_ec") - g("p_charge") - demand[..., 0],
        g("q_ec") + g("q_ac") + g("slack_c") - demand[..., 1],
        g("q_chp") + g("q_gb") + g("slack_h") - g("q_ac_in") - g("q_dump") - demand[..., 2],
    ], dim=-1)
    return residual


def conversion_residuals(dispatch: Tensor, parameters: Mapping[str, Any]) -> Tensor:
    """Return CHP/boiler/electric/absorption conversion residuals ``[B,H,5]``."""

    dispatch = _check_dispatch(dispatch)
    g = lambda name: dispatch[..., _INDEX[name]]
    residual = torch.stack([
        g("p_chp") - _parameter(parameters, "chp_electric_efficiency") * g("g_chp"),
        g("q_chp") - _parameter(parameters, "chp_heat_efficiency") * g("g_chp"),
        g("q_gb") - _parameter(parameters, "gas_boiler_efficiency") * g("g_gb"),
        g("q_ec") - _parameter(parameters, "electric_chiller_cop") * g("p_ec"),
        g("q_ac") - _parameter(parameters, "absorption_chiller_cop") * g("q_ac_in"),
    ], dim=-1)
    return residual


def soc_residuals(dispatch: Tensor, features: Tensor, parameters: Mapping[str, Any]) -> dict[str, Tensor]:
    """Return BESS state recursion and terminal-SOC residuals."""

    dispatch = _check_dispatch(dispatch)
    if features.ndim == 1 and features.shape[0] == dispatch.shape[0]:
        initial_soc = features
    elif features.ndim == 2 and features.shape == (dispatch.shape[0], 1):
        initial_soc = features[:, 0]
    else:
        features = _check_inputs(features, dispatch.shape[0], dispatch.shape[1])
        initial_soc = features[:, 0, 9]
    eta = _parameter(parameters, "bess_roundtrip_efficiency") ** 0.5
    capacity = _parameter(parameters, "bess_energy_capacity")
    soc = dispatch[..., _INDEX["soc"]]
    charge = dispatch[..., _INDEX["p_charge"]]
    discharge = dispatch[..., _INDEX["p_discharge"]]
    # Initial SOC is broadcast over H by the proxy contract; use the first
    # step as the scalar initial state for the recursion.
    initial_energy = initial_soc * capacity
    previous = torch.cat([initial_energy[:, None], soc[:, :-1]], dim=1)
    state = soc - previous - eta * charge + discharge / max(eta, 1e-12)
    terminal = soc[:, -1] - initial_energy
    return {"state": state, "terminal": terminal}


def operating_cost(dispatch: Tensor, features: Tensor, parameters: Mapping[str, Any]) -> Tensor:
    """Compute operating cost excluding carbon price (``[B]``)."""

    dispatch = _check_dispatch(dispatch)
    features = _check_inputs(features, dispatch.shape[0], dispatch.shape[1])
    g = lambda name: dispatch[..., _INDEX[name]]
    grid = g("grid")
    gas = g("g_chp") + g("g_gb")
    throughput = g("p_charge") + g("p_discharge")
    slack = g("slack_e") + g("slack_c") + g("slack_h")
    result = grid * features[..., 6] + gas * features[..., 7]
    result = result + _parameter(parameters, "bess_throughput_cost") * throughput
    result = result + _parameter(parameters, "unserved_penalty") * slack
    return result.sum(dim=1)


def carbon_emissions(dispatch: Tensor, features: Tensor, parameters: Mapping[str, Any] | None = None) -> Tensor:
    """Compute physical grid-plus-gas carbon emissions (``[B]``)."""

    dispatch = _check_dispatch(dispatch)
    features = _check_inputs(features, dispatch.shape[0], dispatch.shape[1])
    parameters = parameters or {}
    g = lambda name: dispatch[..., _INDEX[name]]
    emissions = g("grid") * _parameter(parameters, "grid_emission_factor", 0.5)
    emissions = emissions + (g("g_chp") + g("g_gb")) * _parameter(parameters, "gas_emission_factor", 0.25)
    return emissions.sum(dim=1)


def physics_terms(
    prediction_physical: Tensor,
    features: Tensor,
    parameters: Mapping[str, Any],
    teacher_cost: Tensor | None = None,
    teacher_carbon: Tensor | None = None,
    gas_prior_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    prediction_physical = _check_dispatch(prediction_physical)
    features = _check_inputs(features, prediction_physical.shape[0], prediction_physical.shape[1])
    balance = balance_residuals(prediction_physical, features)
    conversion = conversion_residuals(prediction_physical, parameters)
    soc = soc_residuals(prediction_physical, features, parameters)
    costs = operating_cost(prediction_physical, features, parameters)
    carbon = carbon_emissions(prediction_physical, features, parameters)
    # Residual losses are dimensionless: each physical equation is divided by
    # a safe per-sample demand/frozen-capacity scale.  Raw residual tensors are
    # retained below for physical evaluation and audit reporting.
    demand = features[..., :3]
    electric_capacity = max(
        _parameter(parameters, "grid_import_capacity"),
        _parameter(parameters, "chp_electric_capacity"),
        _parameter(parameters, "bess_power_capacity"),
        _parameter(parameters, "pv_capacity"),
        _parameter(parameters, "wt_capacity"),
        1.0,
    )
    cooling_capacity = max(
        _parameter(parameters, "electric_chiller_capacity"),
        _parameter(parameters, "absorption_chiller_capacity"),
        1.0,
    )
    heating_capacity = max(
        _parameter(parameters, "chp_heat_capacity"),
        _parameter(parameters, "gas_boiler_capacity"),
        1.0,
    )
    balance_scale = torch.maximum(
        torch.abs(demand),
        demand.new_tensor([electric_capacity, cooling_capacity, heating_capacity]),
    ).clamp_min(1.0)
    conversion_scale = prediction_physical.new_tensor([
        max(_parameter(parameters, "chp_electric_capacity"), 1.0),
        max(_parameter(parameters, "chp_heat_capacity"), 1.0),
        max(_parameter(parameters, "gas_boiler_capacity"), 1.0),
        max(_parameter(parameters, "electric_chiller_capacity"), 1.0),
        max(_parameter(parameters, "absorption_chiller_capacity"), 1.0),
    ])
    soc_scale = max(_parameter(parameters, "bess_energy_capacity"), 1.0)
    balance_scaled = balance / balance_scale
    conversion_scaled = conversion / conversion_scale
    soc_state_scaled = soc["state"] / soc_scale
    soc_terminal_scaled = soc["terminal"] / soc_scale
    terms: dict[str, Tensor] = {
        "balance": balance_scaled.pow(2).mean(),
        "conversion": conversion_scaled.pow(2).mean(),
        "soc": 0.5 * soc_state_scaled.pow(2).mean() + 0.5 * soc_terminal_scaled.pow(2).mean(),
        "balance_residual_scaled": balance_scaled,
        "conversion_residual_scaled": conversion_scaled,
        "soc_state_residual_scaled": soc_state_scaled,
        "soc_terminal_residual_scaled": soc_terminal_scaled,
        "balance_scale": balance_scale,
        "conversion_scale": conversion_scale,
        "soc_scale": prediction_physical.new_tensor(soc_scale),
        "predicted_cost": costs,
        "predicted_carbon": carbon,
        "balance_residual": balance,
        "conversion_residual": conversion,
        "soc_state_residual": soc["state"],
        "soc_terminal_residual": soc["terminal"],
    }
    if teacher_cost is not None:
        denom = torch.clamp(torch.abs(teacher_cost), min=1.0)
        terms["cost"] = torch.abs(costs - teacher_cost) / denom
    else:
        terms["cost"] = costs.new_zeros(())
    if teacher_carbon is not None:
        denom = torch.clamp(torch.abs(teacher_carbon), min=1.0)
        terms["carbon"] = torch.abs(carbon - teacher_carbon) / denom
    else:
        terms["carbon"] = carbon.new_zeros(())
    gas_purchase = prediction_physical[..., _INDEX["g_chp"]] + prediction_physical[..., _INDEX["g_gb"]]
    prior = features[..., 3]
    if gas_prior_mask is None:
        gas_prior_mask = torch.ones_like(prior)
    else:
        if gas_prior_mask.ndim == 3 and gas_prior_mask.shape[-1] == 1:
            gas_prior_mask = gas_prior_mask[..., 0]
        if gas_prior_mask.shape != prior.shape:
            raise ValueError("gas_prior_mask must have shape [B,H]")
    error = torch.abs(gas_purchase - prior) / torch.clamp(torch.abs(prior), min=1.0)
    weight = gas_prior_mask.to(error.dtype)
    terms["gas_prior"] = (error * weight).sum() / torch.clamp(weight.sum(), min=1.0)
    terms["cost"] = terms["cost"].mean() if terms["cost"].ndim else terms["cost"]
    terms["carbon"] = terms["carbon"].mean() if terms["carbon"].ndim else terms["carbon"]
    return terms


def physics_aware_loss(
    prediction: Tensor,
    target: Tensor,
    features: Tensor,
    parameters: Mapping[str, Any],
    teacher_cost: Tensor | None = None,
    teacher_carbon: Tensor | None = None,
    gas_prior_mask: Tensor | None = None,
    label_scale: Tensor | Sequence[float] | None = None,
    loss_weights: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return weighted loss and named components.

    ``prediction`` and ``target`` are normalized labels when ``label_scale`` is
    provided; physical residual terms always operate after inverse scaling.
    """

    prediction = _check_dispatch(prediction)
    target = _check_dispatch(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    target = target.to(dtype=prediction.dtype, device=prediction.device)
    if label_scale is None:
        prediction_physical = prediction
        target_for_dispatch = target
    else:
        scale = torch.as_tensor(label_scale, dtype=prediction.dtype, device=prediction.device)
        if scale.ndim != 1 or scale.numel() != len(LABEL_ORDER):
            raise ValueError("label_scale must have shape [21]")
        prediction_physical = prediction * scale
        target_for_dispatch = target
    components = physics_terms(
        prediction_physical,
        features,
        parameters,
        teacher_cost=teacher_cost,
        teacher_carbon=teacher_carbon,
        gas_prior_mask=gas_prior_mask,
    )
    components["dispatch"] = F.smooth_l1_loss(prediction, target_for_dispatch)
    effective_weights = {
        "dispatch": 1.0, "balance": 0.2, "conversion": 0.1, "soc": 0.1,
        "cost": 0.05, "carbon": 0.05, "gas_prior": 0.05,
    }
    if loss_weights is not None:
        effective_weights.update({name: float(value) for name, value in loss_weights.items()})
    if weights is not None:
        # ``weights`` is a concise backwards-compatible spelling; explicit
        # loss_weights wins when both are supplied.
        for name, value in weights.items():
            if loss_weights is None or name not in loss_weights:
                effective_weights[name] = float(value)
    total = prediction.new_zeros(())
    for name in ("dispatch", "balance", "conversion", "soc", "cost", "carbon", "gas_prior"):
        total = total + float(effective_weights.get(name, 0.0)) * components[name]
    components["total"] = total
    return total, components


# Discoverable aliases.
compute_physics_loss = physics_aware_loss
weighted_proxy_loss = physics_aware_loss
proxy_loss = physics_aware_loss
balance_residual = balance_residuals
conversion_residual = conversion_residuals
soc_residual = soc_residuals
compute_operating_cost = operating_cost
compute_carbon_emissions = carbon_emissions


__all__ = [
    "balance_residuals", "conversion_residuals", "soc_residuals", "operating_cost", "carbon_emissions",
    "physics_terms", "physics_aware_loss", "compute_physics_loss", "weighted_proxy_loss",
    "balance_residual", "conversion_residual", "soc_residual", "compute_operating_cost", "compute_carbon_emissions",
    "proxy_loss",
]
