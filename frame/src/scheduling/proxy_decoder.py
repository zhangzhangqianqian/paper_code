"""Feasible-by-construction dispatch decoder for scheduling-proxy v2.

The decoder is deliberately implemented only with PyTorch tensor operations.
It is used by both training and inference, so the float64 cast is kept in the
autograd graph and no optimizer or exact LP is imported here.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor

from .proxy_contract import FEATURE_ORDER, HORIZON, LABEL_ORDER


DECODER_SCHEMA_VERSION = "horizon-reachable-feasible-v2"
DECISION_GROUPS = MappingProxyType(
    {
        "cooling": (0, 4),
        "chp": (4, 8),
        "soc": (8, 11),
        "renewable_pv": (11, 15),
    }
)
CONTROL_DIM = 15
CONTROL_TEMPERATURE = 0.25

_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_ORDER)}
_LABEL_INDEX = {name: index for index, name in enumerate(LABEL_ORDER)}


def _parameter(parameters: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in parameters:
            try:
                value = float(parameters[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"decoder parameter {name} must be finite") from exc
            if not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"decoder parameter {name} must be finite")
            return value
    raise ValueError(f"decoder parameters are missing {names[0]}")


def _parameters(parameters: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(parameters, Mapping):
        raise TypeError("decoder parameters must be a mapping")
    values = {
        "grid": _parameter(parameters, "grid_import_capacity", "Gbar"),
        "chp_p": _parameter(parameters, "chp_electric_capacity", "Pbar_chp"),
        "chp_q": _parameter(parameters, "chp_heat_capacity", "Qbar_chp"),
        "boiler_q": _parameter(parameters, "gas_boiler_capacity", "Qbar_gb"),
        "ec_q": _parameter(parameters, "electric_chiller_capacity", "Qbar_ec"),
        "ac_q": _parameter(parameters, "absorption_chiller_capacity", "Qbar_ac"),
        "bess_p": _parameter(parameters, "bess_power_capacity", "Pbar_b"),
        "bess_e": _parameter(parameters, "bess_energy_capacity", "Ebar_b"),
        "eta_e": _parameter(parameters, "chp_electric_efficiency", "eta_e"),
        "eta_h": _parameter(parameters, "chp_heat_efficiency", "eta_h"),
        "eta_gb": _parameter(parameters, "gas_boiler_efficiency", "eta_gb"),
        "cop_ec": _parameter(parameters, "electric_chiller_cop", "COP_ec"),
        "cop_ac": _parameter(parameters, "absorption_chiller_cop", "COP_ac"),
        "eta_roundtrip": _parameter(parameters, "bess_roundtrip_efficiency", "eta_b_roundtrip", "eta_b"),
        "ramp_fraction": _parameter(parameters, "chp_ramp_fraction", "ramp_fraction"),
    }
    for name, value in values.items():
        if value <= 0.0:
            raise ValueError(f"decoder parameter {name} must be positive")
    # ``eta_b`` is accepted as a compact notation for the per-direction
    # efficiency; the frozen benchmark exposes the round-trip key instead.
    if "bess_roundtrip_efficiency" not in parameters and "eta_b_roundtrip" not in parameters and "eta_b" in parameters:
        values["eta_b"] = _parameter(parameters, "eta_b")
        values["eta_roundtrip"] = values["eta_b"] ** 2
    else:
        values["eta_b"] = values["eta_roundtrip"] ** 0.5
    values["rho"] = values["eta_h"] / values["eta_e"]
    values["ramp"] = values["ramp_fraction"] * values["chp_p"]
    return values


def _check_features(physical_features: Tensor) -> Tensor:
    if not isinstance(physical_features, Tensor):
        raise TypeError("physical_features must be a torch Tensor")
    if physical_features.ndim != 3 or tuple(physical_features.shape[1:]) != (HORIZON, len(FEATURE_ORDER)):
        raise ValueError(f"physical_features must have shape [B,{HORIZON},{len(FEATURE_ORDER)}]")
    features = physical_features.to(dtype=torch.float64)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("physical_features must be finite")
    for name in FEATURE_ORDER[:-1]:
        index = _FEATURE_INDEX[name]
        if bool((features[..., index] < 0.0).any()):
            raise ValueError(f"physical feature {name} must be non-negative")
    soc = features[..., _FEATURE_INDEX["initial_soc"]]
    if bool((soc < 0.0).any()) or bool((soc > 1.0).any()):
        raise ValueError("initial_soc must be in [0,1]")
    if not bool(torch.allclose(soc, soc[:, :1], rtol=0.0, atol=1.0e-12)):
        raise ValueError("initial_soc must be identical across horizon rows")
    return features


def _check_controls(controls: Tensor, batch: int | None = None) -> Tensor:
    if not isinstance(controls, Tensor):
        raise TypeError("controls must be a torch Tensor")
    if controls.ndim != 2 or controls.shape[-1] != CONTROL_DIM:
        raise ValueError(f"controls must have shape [B,{CONTROL_DIM}]")
    if batch is not None and controls.shape[0] != batch:
        raise ValueError("controls and physical_features batch dimensions must match")
    controls = controls.to(dtype=torch.float64)
    if not bool(torch.isfinite(controls).all()):
        raise ValueError("controls must be finite")
    if bool((controls < 0.0).any()) or bool((controls > 1.0).any()):
        raise ValueError("controls must lie in [0,1]")
    return controls


def _minimum(*values: Tensor) -> Tensor:
    result = values[0]
    for value in values[1:]:
        result = torch.minimum(result, value)
    return result


def _maximum(*values: Tensor) -> Tensor:
    result = values[0]
    for value in values[1:]:
        result = torch.maximum(result, value)
    return result


def _fraction(value: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    width = (upper - lower).clamp_min(1.0e-12)
    result = (value - lower) / width
    # A collapsed interval has no meaningful control.  Zero is a deterministic
    # representative and does not alter the decoded physical value.
    result = torch.where((upper - lower).abs() > 1.0e-12, result, torch.zeros_like(result))
    return result.clamp(0.0, 1.0)


def _assemble(values: Mapping[str, Tensor], features: Tensor) -> Tensor:
    dispatch = features.new_zeros((features.shape[0], HORIZON, len(LABEL_ORDER)), dtype=torch.float64)
    for name, value in values.items():
        dispatch[..., _LABEL_INDEX[name]] = value
    return dispatch


def _decode_stages(controls: Tensor, features: Tensor, parameters: Mapping[str, Any]) -> Tensor:
    p = _parameters(parameters)
    demand_e = features[..., _FEATURE_INDEX["electricity"]]
    demand_c = features[..., _FEATURE_INDEX["cooling"]]
    demand_h = features[..., _FEATURE_INDEX["heating"]]
    pv_available = features[..., _FEATURE_INDEX["pv_available"]]
    wt_available = features[..., _FEATURE_INDEX["wt_available"]]
    initial_soc = features[:, 0, _FEATURE_INDEX["initial_soc"]]

    u_cooling = controls[:, 0:4]
    u_chp = controls[:, 4:8]
    u_soc = controls[:, 8:11]
    u_pv = controls[:, 11:15]

    # Cooling allocation.  The boiler-safe absorption bound prevents cooling
    # from consuming heat that cannot be supplied by the boiler.
    qac_safe = torch.minimum(
        features.new_tensor(p["ac_q"]),
        features.new_tensor(p["cop_ac"]) * (features.new_tensor(p["boiler_q"]) - demand_h).clamp_min(0.0),
    )
    cooling_served = torch.minimum(demand_c, features.new_tensor(p["ec_q"]) + qac_safe)
    lower_ec = (cooling_served - qac_safe).clamp_min(0.0)
    upper_ec = torch.minimum(cooling_served, features.new_tensor(p["ec_q"]))
    q_ec = lower_ec + u_cooling * (upper_ec - lower_ec)
    q_ac = cooling_served - q_ec
    p_ec = q_ec / p["cop_ec"]
    q_ac_in = q_ac / p["cop_ac"]
    slack_c = demand_c - cooling_served

    # A forward start-reachable cap followed by a backward viability pass gives
    # a non-empty ramp interval at every step, including time-varying heat and
    # electric demand caps.
    net_before_chp = demand_e + p_ec
    local_chp_cap = _minimum(
        features.new_tensor(p["chp_p"]),
        features.new_tensor(p["chp_q"] / p["rho"]),
        (demand_h + q_ac_in) / p["rho"],
        net_before_chp,
    )
    forward_cap = torch.minimum(
        local_chp_cap,
        features.new_tensor(p["ramp"]) * torch.arange(1, HORIZON + 1, dtype=torch.float64, device=features.device),
    )
    viable = [forward_cap[:, -1]]
    for index in range(HORIZON - 2, -1, -1):
        viable.append(torch.minimum(forward_cap[:, index], viable[-1] + p["ramp"]))
    viable = torch.stack(list(reversed(viable)), dim=1)
    p_chp_values: list[Tensor] = []
    previous_chp = torch.zeros_like(net_before_chp[:, 0])
    for index in range(HORIZON):
        lower_chp = (previous_chp - p["ramp"]).clamp_min(0.0)
        upper_chp = torch.minimum(viable[:, index], previous_chp + p["ramp"])
        if bool((upper_chp < lower_chp - 1.0e-10).any()):
            raise ValueError("decoder constructed an empty CHP ramp interval")
        current_chp = lower_chp + u_chp[:, index] * (upper_chp - lower_chp)
        p_chp_values.append(current_chp)
        previous_chp = current_chp
    p_chp = torch.stack(p_chp_values, dim=1)
    g_chp = p_chp / p["eta_e"]
    q_chp = p_chp * p["rho"]

    # Heat residual and explicit capacity slacks.
    residual_heat = demand_h + q_ac_in
    q_gb = torch.minimum(features.new_tensor(p["boiler_q"]), (residual_heat - q_chp).clamp_min(0.0))
    g_gb = q_gb / p["eta_gb"]
    q_dump = (q_chp + q_gb - residual_heat).clamp_min(0.0)
    slack_h = (residual_heat - q_chp - q_gb).clamp_min(0.0)

    # Battery power intervals are limited by both net electrical demand and
    # the ability to return to the initial energy at the horizon boundary.
    net_after_chp = net_before_chp - p_chp
    charge_max = torch.minimum(
        features.new_tensor(p["bess_p"]),
        (features.new_tensor(p["grid"]) + pv_available + wt_available - net_after_chp).clamp_min(0.0),
    )
    discharge_max = torch.minimum(features.new_tensor(p["bess_p"]), net_after_chp.clamp_min(0.0))
    soc_increase_max = p["eta_b"] * charge_max
    soc_decrease_max = discharge_max / p["eta_b"]
    initial_energy = initial_soc * p["bess_e"]
    states: list[Tensor] = []
    for index in range(HORIZON - 1):
        previous_energy = initial_energy if index == 0 else states[-1]
        future_charge = soc_increase_max[:, index + 1:].sum(dim=1)
        future_discharge = soc_decrease_max[:, index + 1:].sum(dim=1)
        lower_soc = _maximum(
            torch.zeros_like(previous_energy),
            previous_energy - soc_decrease_max[:, index],
            initial_energy - future_charge,
        )
        upper_soc = _minimum(
            features.new_tensor(p["bess_e"]),
            previous_energy + soc_increase_max[:, index],
            initial_energy + future_discharge,
        )
        if bool((upper_soc < lower_soc - 1.0e-10).any()):
            raise ValueError("decoder constructed an empty terminal-reachable SOC interval")
        states.append(lower_soc + u_soc[:, index] * (upper_soc - lower_soc))
    states.append(initial_energy)
    soc = torch.stack(states, dim=1)
    previous_soc = torch.cat([initial_energy[:, None], soc[:, :-1]], dim=1)
    delta_soc = soc - previous_soc
    p_charge = delta_soc.clamp_min(0.0) / p["eta_b"]
    p_discharge = (-delta_soc).clamp_min(0.0) * p["eta_b"]

    # Use as much renewable energy as the residual demand can absorb, then
    # expose only the feasible PV split as a learned degree of freedom.
    net_residual = net_after_chp + p_charge - p_discharge
    renewable_use = torch.minimum(net_residual, pv_available + wt_available)
    lower_pv = (renewable_use - wt_available).clamp_min(0.0)
    upper_pv = torch.minimum(pv_available, renewable_use)
    pv_use = lower_pv + u_pv * (upper_pv - lower_pv)
    wt_use = renewable_use - pv_use
    pv_curt = pv_available - pv_use
    wt_curt = wt_available - wt_use
    grid = torch.minimum(features.new_tensor(p["grid"]), net_residual - renewable_use)
    slack_e = (net_residual - renewable_use - features.new_tensor(p["grid"])).clamp_min(0.0)

    return _assemble(
        {
            "grid": grid,
            "pv_use": pv_use,
            "pv_curt": pv_curt,
            "wt_use": wt_use,
            "wt_curt": wt_curt,
            "g_chp": g_chp,
            "g_gb": g_gb,
            "p_chp": p_chp,
            "q_chp": q_chp,
            "q_gb": q_gb,
            "p_ec": p_ec,
            "q_ec": q_ec,
            "q_ac_in": q_ac_in,
            "q_ac": q_ac,
            "p_charge": p_charge,
            "p_discharge": p_discharge,
            "soc": soc,
            "slack_e": slack_e,
            "slack_c": slack_c,
            "slack_h": slack_h,
            "q_dump": q_dump,
        },
        features,
    )


def decode_feasible_controls(
    controls: Tensor,
    physical_features: Tensor,
    parameters: Mapping[str, Any],
) -> Tensor:
    """Decode normalized controls ``[B,15]`` into physical ``[B,4,21]``."""

    features = _check_features(physical_features)
    checked = _check_controls(controls, batch=int(features.shape[0]))
    return _decode_stages(checked, features, parameters)


def decode_feasible_dispatch(
    logits: Tensor,
    physical_features: Tensor,
    parameters: Mapping[str, Any],
    temperature: float = CONTROL_TEMPERATURE,
) -> Tensor:
    """Apply the fixed-temperature sigmoid and decode a feasible dispatch."""

    try:
        temperature = float(temperature)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    features = _check_features(physical_features)
    if not isinstance(logits, Tensor) or logits.ndim != 2 or logits.shape != (features.shape[0], CONTROL_DIM):
        raise ValueError(f"logits must have shape [B,{CONTROL_DIM}]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    controls = torch.sigmoid(logits.to(dtype=torch.float64) / temperature)
    return _decode_stages(controls, features, parameters)


def recover_teacher_controls(
    dispatch: Tensor,
    physical_features: Tensor,
    parameters: Mapping[str, Any],
) -> Tensor:
    """Recover the 15 decoder controls from a feasible physical dispatch.

    The inverse is intentionally conservative at collapsed intervals: the
    corresponding control is set to zero because every value decodes to the
    same physical endpoint.
    """

    single_dispatch = dispatch.ndim == 2
    if single_dispatch:
        dispatch = dispatch.unsqueeze(0)
    if not isinstance(dispatch, Tensor) or dispatch.ndim != 3 or tuple(dispatch.shape[1:]) != (HORIZON, len(LABEL_ORDER)):
        raise ValueError(f"dispatch must have shape [B,{HORIZON},{len(LABEL_ORDER)}]")
    if not bool(torch.isfinite(dispatch).all()):
        raise ValueError("dispatch must be finite")
    features = _check_features(physical_features)
    if features.shape[0] != dispatch.shape[0]:
        raise ValueError("dispatch and physical_features batch dimensions must match")
    dispatch = dispatch.to(dtype=torch.float64)
    p = _parameters(parameters)
    get_feature = lambda name: features[..., _FEATURE_INDEX[name]]
    get_label = lambda name: dispatch[..., _LABEL_INDEX[name]]
    demand_e = get_feature("electricity")
    demand_c = get_feature("cooling")
    demand_h = get_feature("heating")
    pv_available = get_feature("pv_available")
    wt_available = get_feature("wt_available")
    initial_soc = features[:, 0, _FEATURE_INDEX["initial_soc"]]
    initial_energy = initial_soc * p["bess_e"]

    qac_safe = torch.minimum(
        features.new_tensor(p["ac_q"]),
        features.new_tensor(p["cop_ac"]) * (features.new_tensor(p["boiler_q"]) - demand_h).clamp_min(0.0),
    )
    cooling_served = torch.minimum(demand_c, features.new_tensor(p["ec_q"]) + qac_safe)
    lower_ec = (cooling_served - qac_safe).clamp_min(0.0)
    upper_ec = torch.minimum(cooling_served, features.new_tensor(p["ec_q"]))
    u_cooling = _fraction(get_label("q_ec"), lower_ec, upper_ec)
    q_ec = get_label("q_ec")
    q_ac_in = get_label("q_ac_in")

    net_before_chp = demand_e + q_ec / p["cop_ec"]
    local_chp_cap = _minimum(
        features.new_tensor(p["chp_p"]),
        features.new_tensor(p["chp_q"] / p["rho"]),
        (demand_h + q_ac_in) / p["rho"],
        net_before_chp,
    )
    forward_cap = torch.minimum(
        local_chp_cap,
        features.new_tensor(p["ramp"]) * torch.arange(1, HORIZON + 1, dtype=torch.float64, device=features.device),
    )
    viable = [forward_cap[:, -1]]
    for index in range(HORIZON - 2, -1, -1):
        viable.append(torch.minimum(forward_cap[:, index], viable[-1] + p["ramp"]))
    viable = torch.stack(list(reversed(viable)), dim=1)
    p_chp = get_label("p_chp")
    chp_controls: list[Tensor] = []
    previous_chp = torch.zeros_like(p_chp[:, 0])
    for index in range(HORIZON):
        lower_chp = (previous_chp - p["ramp"]).clamp_min(0.0)
        upper_chp = torch.minimum(viable[:, index], previous_chp + p["ramp"])
        chp_controls.append(_fraction(p_chp[:, index], lower_chp, upper_chp))
        previous_chp = p_chp[:, index]

    net_after_chp = net_before_chp - p_chp
    charge_max = torch.minimum(
        features.new_tensor(p["bess_p"]),
        (features.new_tensor(p["grid"]) + pv_available + wt_available - net_after_chp).clamp_min(0.0),
    )
    discharge_max = torch.minimum(features.new_tensor(p["bess_p"]), net_after_chp.clamp_min(0.0))
    soc_increase_max = p["eta_b"] * charge_max
    soc_decrease_max = discharge_max / p["eta_b"]
    soc = get_label("soc")
    soc_controls: list[Tensor] = []
    for index in range(HORIZON - 1):
        previous_energy = initial_energy if index == 0 else soc[:, index - 1]
        future_charge = soc_increase_max[:, index + 1:].sum(dim=1)
        future_discharge = soc_decrease_max[:, index + 1:].sum(dim=1)
        lower_soc = _maximum(
            torch.zeros_like(previous_energy),
            previous_energy - soc_decrease_max[:, index],
            initial_energy - future_charge,
        )
        upper_soc = _minimum(
            features.new_tensor(p["bess_e"]),
            previous_energy + soc_increase_max[:, index],
            initial_energy + future_discharge,
        )
        soc_controls.append(_fraction(soc[:, index], lower_soc, upper_soc))

    p_charge = get_label("p_charge")
    p_discharge = get_label("p_discharge")
    net_residual = net_after_chp + p_charge - p_discharge
    renewable_use = torch.minimum(net_residual, pv_available + wt_available)
    lower_pv = (renewable_use - wt_available).clamp_min(0.0)
    upper_pv = torch.minimum(pv_available, renewable_use)
    pv_controls = _fraction(get_label("pv_use"), lower_pv, upper_pv)

    controls = torch.cat(
        [
            torch.stack([u_cooling[:, index] for index in range(HORIZON)], dim=1),
            torch.stack(chp_controls, dim=1),
            torch.stack(soc_controls, dim=1),
            pv_controls,
        ],
        dim=1,
    )
    if single_dispatch:
        return controls
    return controls


__all__ = [
    "DECODER_SCHEMA_VERSION",
    "DECISION_GROUPS",
    "CONTROL_DIM",
    "CONTROL_TEMPERATURE",
    "decode_feasible_controls",
    "decode_feasible_dispatch",
    "recover_teacher_controls",
]
