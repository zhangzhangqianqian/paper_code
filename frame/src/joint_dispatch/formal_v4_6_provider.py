"""Frozen RSC-PF provider and checkpoint reconstruction for matched evaluation.

This module intentionally contains no training code.  It reconstructs the
already-frozen formal-v4.6 checkpoint, exposes only the causal origin view to
the model, and returns the nominal forecast, risk-adjusted demand and decoded
dispatch through the common :class:`PlannedStep` contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from .formal_v4_2_data import NormalizationReceiptV42
from .formal_v4_4_pilot_executor import _parameters
from .formal_v4_4_regime import derive_last_observed_regime
from .formal_v4_4_training import sha256_state_dict
from .formal_v4_6_contract import FormalV46Contract, load_formal_v4_6_contract
from .formal_v4_6_model import RiskAdjustedRSCPFModelV46
from .formal_v4_6_risk import RiskCapReceiptV46, load_risk_caps_v46
from .matched_closed_loop import CausalOriginInput, PlannedStep


def _norm(value: np.ndarray, normalization: Any, field: str) -> np.ndarray:
    if normalization is None:
        raise ValueError("RSC-PF provider requires a train-only normalization receipt")
    means = getattr(normalization, "field_mean", None)
    scales = getattr(normalization, "field_scale", None)
    if not isinstance(means, Mapping) or not isinstance(scales, Mapping) or field not in means or field not in scales:
        raise ValueError(f"normalization receipt is missing field {field!r}")
    mean = np.asarray(means[field], dtype=np.float32)
    scale = np.asarray(scales[field], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(f"normalization field {field!r} is invalid")
    return ((np.asarray(value, dtype=np.float32) - mean) / scale).astype(np.float32)


def _origin_inputs(origin: CausalOriginInput, normalization: Any) -> dict[str, torch.Tensor]:
    last = derive_last_observed_regime(np.asarray(origin.load_history, dtype=np.float64))
    return {
        "load_history": torch.as_tensor(_norm(origin.load_history, normalization, "load")),
        "exog_history": torch.as_tensor(_norm(origin.exog_history, normalization, "exog")),
        "device_history": torch.as_tensor(_norm(origin.device_history, normalization, "device")),
        "activity_history": torch.as_tensor(origin.activity_history, dtype=torch.float32),
        # scheduler_context contains renewable forecasts, prices and SOC in
        # their physical units; the model normalizes the combined physical
        # features after the forecast bottleneck.
        "scheduler_context": torch.as_tensor(origin.scheduler_context, dtype=torch.float32),
        "previous_chp": torch.as_tensor(origin.previous_chp, dtype=torch.float32),
        "last_thermal_regime": torch.as_tensor(last, dtype=torch.long),
    }


class RSCPFProvider:
    """Deployable RSC-PF provider with zero inference-time LP calls."""

    optimizer_role = "none at inference"

    def __init__(self, model: RiskAdjustedRSCPFModelV46, normalization: Any | None, parameters: Mapping[str, Any], *, provenance: Mapping[str, Any] | None = None) -> None:
        self.method_id = "RSC-PF"
        self.model = model.eval()
        self.normalization = normalization
        self.parameters = dict(parameters)
        self.provenance = dict(provenance or {})

    def plan(self, origin: CausalOriginInput) -> PlannedStep:
        inputs = _origin_inputs(origin, self.normalization)
        with torch.inference_mode():
            output = self.model(**inputs)
        forecast = output.forecast_nominal_physical.detach().cpu().numpy()[0].astype(np.float64)
        demand = output.scheduler_demand.detach().cpu().numpy()[0].astype(np.float64)
        dispatch = output.dispatch.detach().cpu().numpy()[0].astype(np.float64)
        return PlannedStep(forecast, demand, origin.renewable_forecast.copy(), dispatch, 0)


def _read_capacity(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("capacity receipt must be a JSON object")
    return dict(payload)


def _stage_payload(path: Path) -> tuple[dict[str, Any], Mapping[str, torch.Tensor]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state_dict"), Mapping):
        raise ValueError("J_joint checkpoint is missing a state_dict")
    state_dict = payload["state_dict"]
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()):
        raise ValueError("J_joint state_dict contains invalid entries")
    return dict(payload), state_dict


def load_frozen_rsc_pf_provider(
    run_root: str | Path,
    contract_path: str | Path,
    benchmark_path: str | Path,
    capacity_receipt_path: str | Path,
    *,
    normalization: NormalizationReceiptV42 | Any | None = None,
) -> RSCPFProvider:
    """Reconstruct RSC-PF using only frozen, lineage-checked artifacts."""

    root = Path(run_root)
    contract: FormalV46Contract = load_formal_v4_6_contract(contract_path)
    capacity = _read_capacity(capacity_receipt_path)
    benchmark = yaml.safe_load(Path(benchmark_path).read_text(encoding="utf-8"))
    if not isinstance(benchmark, Mapping):
        raise ValueError("benchmark must be a mapping")
    parameters = _parameters(benchmark, capacity)
    risk_root = root / "pilot" / "risk_caps"
    risk: RiskCapReceiptV46 = load_risk_caps_v46(risk_root)
    if risk.contract_sha256 != contract.contract_sha256:
        raise ValueError("risk-cap contract hash does not match formal-v4.6 contract")
    stage_path = root / "pilot" / "stages" / "J_joint.pt"
    stage, state_dict = _stage_payload(stage_path)
    model = RiskAdjustedRSCPFModelV46(
        transition_probability=torch.full((4, 3, 3), 1.0 / 3.0),
        risk_cap=torch.as_tensor(risk.cap, dtype=torch.float32),
        regime_temperature=float(contract.payload.get("pilot_candidate", {}).get("temperature", 1.0)),
        risk_hidden_width=int(contract.risk_adjustment.get("hidden_width", 64)),
        risk_initial_bias=float(contract.risk_adjustment.get("initial_output_bias", -6.0)),
        decoder_parameters=parameters,
        task_mean=torch.zeros(4), task_scale=torch.ones(4),
        physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10),
        previous_chp_mean=torch.zeros(1), previous_chp_scale=torch.ones(1),
        dropout=0.0,
    )
    model.load_state_dict(state_dict, strict=True)
    state_hash = sha256_state_dict(model)
    expected_state_hash = str(stage.get("final_sha256", stage.get("best_sha256", "")))
    if expected_state_hash and state_hash != expected_state_hash:
        raise ValueError("J_joint state hash does not match checkpoint lineage")
    checkpoint_sha256 = hashlib.sha256(stage_path.read_bytes()).hexdigest()
    provenance = {
        "contract_sha256": contract.contract_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "state_sha256": state_hash,
        "risk_lineage_sha256": risk.lineage_sha256,
        "risk_arrays_sha256": risk.arrays_sha256,
        "scheduler_soc_source": "carried_state",
        "evaluation_year_accessed": False,
    }
    return RSCPFProvider(model, normalization, parameters, provenance=provenance)


__all__ = ["RSCPFProvider", "load_frozen_rsc_pf_provider"]
