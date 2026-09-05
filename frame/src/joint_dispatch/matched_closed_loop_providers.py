"""Causal action providers for the matched closed-loop evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import re

import numpy as np
import torch

from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from ..scheduling.dispatch_schema import VARIABLES
from .external_baseline_data import ExternalBaselineBatch, ExternalNormalization
from .external_baseline_training import _decoder_parameters, load_external_checkpoint
from .external_baselines import build_external_baseline
from .matched_closed_loop import CausalOriginInput, MatchedClosedLoopResult, PlannedStep


def _finite_forecast(value: Any) -> np.ndarray:
    forecast = np.asarray(value, dtype=np.float64)
    if forecast.shape != (4, 4) or not np.isfinite(forecast).all():
        raise ValueError("forecast must have finite shape [4,4]")
    return forecast


def _normalization_value(normalization: Any, name: str) -> np.ndarray:
    value = getattr(normalization, name, None)
    if value is None and name.endswith("_mean"):
        field_mean = getattr(normalization, "field_mean", None)
        if isinstance(field_mean, Mapping):
            value = field_mean.get(name[:-5])
    if value is None and name.endswith("_scale"):
        field_scale = getattr(normalization, "field_scale", None)
        if isinstance(field_scale, Mapping):
            value = field_scale.get(name[:-6])
    if value is None:
        raise ValueError(f"normalization is missing {name}")
    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all() or np.any(array <= 0.0) and "scale" in name:
        raise ValueError(f"normalization {name} is invalid")
    return array


def _dummy_batch(origin: CausalOriginInput, normalization: Any | None, *, normalized: bool) -> ExternalBaselineBatch:
    values = {
        "load_history": origin.load_history.copy(),
        "exog_history": origin.exog_history.copy(),
        "device_history": origin.device_history.copy(),
        "scheduler_context": origin.scheduler_context.copy(),
    }
    if normalized:
        for field, mean_name, scale_name in (
            ("load_history", "load_mean", "load_scale"),
            ("exog_history", "exog_mean", "exog_scale"),
            ("device_history", "device_mean", "device_scale"),
            ("scheduler_context", "scheduler_mean", "scheduler_scale"),
        ):
            if normalization is None:
                raise ValueError("normalized provider requires train-only normalization")
            values[field] = (values[field] - _normalization_value(normalization, mean_name).reshape(1, 1, -1)) / _normalization_value(normalization, scale_name).reshape(1, 1, -1)
    return ExternalBaselineBatch(
        load_history=torch.as_tensor(values["load_history"], dtype=torch.float32),
        exog_history=torch.as_tensor(values["exog_history"], dtype=torch.float32),
        device_history=torch.as_tensor(values["device_history"], dtype=torch.float32),
        device_status=torch.as_tensor(origin.activity_history, dtype=torch.float32),
        scheduler_context=torch.as_tensor(values["scheduler_context"], dtype=torch.float32),
        previous_chp=torch.as_tensor(origin.previous_chp, dtype=torch.float32),
        # These tensors exist solely because the legacy third-party wrapper's
        # constructor requires labelled training batches.  They are created
        # inside this adapter and never populated from a realized origin.
        forecast_target=torch.zeros((1, 4, 4), dtype=torch.float32),
        teacher_dispatch=torch.zeros((1, 4, 21), dtype=torch.float32),
        oracle_first_step_objective=torch.zeros((1,), dtype=torch.float32),
        split="pilot",
    )


def _lp_parameters(origin: CausalOriginInput, parameters: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(parameters)
    context = np.asarray(origin.scheduler_context[0], dtype=np.float64)
    result["grid_energy_price"] = context[:, 2]
    result["gas_energy_price"] = context[:, 3]
    result["carbon_price"] = context[:, 4]
    return result


def solve_pto_from_forecast(
    method_id: str,
    forecast: np.ndarray,
    origin: CausalOriginInput,
    parameters: Mapping[str, Any],
) -> PlannedStep:
    """Solve exactly one causal PTO LP without accepting future labels."""

    if method_id not in {"iTransformer-PTO", "DecisionFocused-Online", "Seasonal-Naive-PTO"}:
        raise ValueError("solve_pto_from_forecast only supports PTO methods")
    prediction = _finite_forecast(forecast)
    lp_parameters = _lp_parameters(origin, parameters)
    context = np.asarray(origin.scheduler_context[0], dtype=np.float64)
    result = solve_dispatch_lp(DispatchInputs(
        demand=np.maximum(prediction[:, :3], 0.0),
        pv_available=np.maximum(origin.renewable_forecast[:, 0], 0.0),
        wt_available=np.maximum(origin.renewable_forecast[:, 1], 0.0),
        parameters=lp_parameters,
        initial_soc=float(context[0, 5]),
        previous_chp=min(float(origin.previous_chp[0, 0]), float(parameters["chp_electric_capacity"])),
    ))
    if not result.success:
        raise RuntimeError(f"{method_id} inference LP failed: {result.message}")
    dispatch = np.column_stack([result.values[name] for name in VARIABLES]).astype(np.float64)
    return PlannedStep(prediction, prediction, origin.renewable_forecast.copy(), dispatch, 1)


class _BaseProvider:
    def __init__(self, method_id: str, model: torch.nn.Module, normalization: Any | None, *, normalized: bool, provenance: Mapping[str, Any] | None = None) -> None:
        self.method_id = method_id
        self.model = model.eval()
        self.normalization = normalization
        self.normalized = bool(normalized)
        self.parameters: dict[str, Any] = {}
        self.provenance = dict(provenance or {})
        self.reproduction_level = "official_source_adaptation" if method_id == "iTransformer-PTO" else "frozen_local_adaptation"

    def _batch(self, origin: CausalOriginInput) -> ExternalBaselineBatch:
        return _dummy_batch(origin, self.normalization, normalized=self.normalized)


class PTOProvider(_BaseProvider):
    optimizer_role = "exact optimizer at inference"

    def plan(self, origin: CausalOriginInput) -> PlannedStep:
        batch = self._batch(origin)
        with torch.inference_mode():
            if self.method_id == "iTransformer-PTO":
                value = self.model(batch.load_history, batch.exog_history)
            else:
                value = self.model(batch).forecast
        prediction = value.detach().cpu().numpy()[0]
        if self.normalization is not None:
            prediction = prediction * _normalization_value(self.normalization, "load_scale") + _normalization_value(self.normalization, "load_mean")
        return solve_pto_from_forecast(self.method_id, prediction, origin, self.parameters)

    def bind_parameters(self, parameters: Mapping[str, Any]) -> "PTOProvider":
        self.parameters = dict(parameters)
        return self


class DigitalTwinsProvider(_BaseProvider):
    optimizer_role = "none at inference"

    def plan(self, origin: CausalOriginInput) -> PlannedStep:
        with torch.inference_mode():
            output = self.model(self._batch(origin))
        forecast = output.forecast_physical.detach().cpu().numpy()[0]
        dispatch = output.dispatch.detach().cpu().numpy()[0]
        return PlannedStep(forecast, forecast, origin.renewable_forecast.copy(), dispatch, 0)


def _safe_method(method_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", method_id).strip("_")


def _load_normalization(config: Mapping[str, Any]) -> Any | None:
    normalization = config.get("normalization")
    if normalization is not None:
        return normalization
    return None


def build_external_provider(method_id: str, seed: int, config: Mapping[str, Any]) -> Any:
    """Build a frozen external provider from a provenance-checked checkpoint."""

    if method_id not in {"iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy"}:
        raise ValueError(f"unknown external method: {method_id}")
    model_options = dict(config.get("model_options", {}))
    if method_id == "DigitalTwins-Policy":
        model_options.setdefault("decoder_parameters", config.get("decoder_parameters", _decoder_parameters()))
    model = build_external_baseline(method_id, model_options)
    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path is None:
        root = Path(config["external_output_root"])
        checkpoint_path = root / "validation" / _safe_method(method_id) / f"seed_{int(seed)}" / "best_checkpoint.pt"
    provenance = load_external_checkpoint(checkpoint_path, expected_method=method_id, expected_seed=int(seed))
    model.load_state_dict(provenance["model_state_dict"], strict=True)
    normalization = _load_normalization(config)
    provider_provenance = {
        "method_id": method_id,
        "seed": int(seed),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_sha256": __import__("hashlib").sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
        "reproduction_level": "official_source_adaptation" if method_id == "iTransformer-PTO" else "frozen_local_adaptation",
    }
    if method_id == "DigitalTwins-Policy":
        return DigitalTwinsProvider(method_id, model, normalization, normalized=False, provenance=provider_provenance)
    parameters = dict(config.get("parameters", _decoder_parameters()))
    return PTOProvider(method_id, model, normalization, normalized=True, provenance=provider_provenance).bind_parameters(parameters)


def load_frozen_rsc_pf_provider(*args: Any, **kwargs: Any) -> Any:
    """Load the RSC-PF provider through the strict v4.6 reconstruction path.

    The concrete checkpoint reconstruction is kept in the formal-v4.6 adapter;
    this wrapper is intentionally explicit so a missing lineage artifact fails
    rather than silently falling back to an unrelated model.
    """

    from .formal_v4_6_provider import load_frozen_rsc_pf_provider as loader
    return loader(*args, **kwargs)


def _legacy_array(saved: Mapping[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in saved:
            return np.asarray(saved[name])
    raise KeyError(f"legacy rollout is missing one of {names}")


def verify_rsc_pf_first_origin(
    result: MatchedClosedLoopResult,
    legacy_rollout: str | Path,
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-6,
) -> dict[str, Any]:
    """Verify the first origin against the historical rollout.

    The first origin has the same materialized SOC in both evaluators.  Later
    origins intentionally differ because the matched runner carries settled
    SOC, whereas the historical evaluator reused the materialized origin SOC.
    Consequently this helper never silently upgrades a first-origin check to
    a full-array equality claim.
    """

    if result.method_id not in {"RSC-PF", "rsc_pf_joint"}:
        raise ValueError("first-origin verification requires an RSC-PF result")
    source = Path(legacy_rollout)
    with np.load(source, allow_pickle=False) as payload:
        saved = {name: np.asarray(payload[name]) for name in payload.files}
    checks: dict[str, bool] = {
        "times": np.array_equal(np.asarray(result.arrays["times"])[0:1].astype("datetime64[ns]"), _legacy_array(saved, "times")[0:1].astype("datetime64[ns]")),
        "forecast": np.allclose(result.arrays["forecast"][0], _legacy_array(saved, "forecast_nominal", "forecast")[0], atol=atol, rtol=rtol),
        "scheduler_demand": np.allclose(result.arrays["scheduler_demand"][0], _legacy_array(saved, "scheduler_demand")[0], atol=atol, rtol=rtol),
        "planned_dispatch": np.allclose(result.arrays["planned_dispatch"][0], _legacy_array(saved, "planned_dispatch")[0], atol=atol, rtol=rtol),
        "settled_dispatch": np.allclose(result.arrays["settled_dispatch"][0], _legacy_array(saved, "settled_dispatch")[0], atol=atol, rtol=rtol),
        "state_hash": str(np.asarray(result.arrays["state_hashes"])[0]) == str(_legacy_array(saved, "state_hashes")[0]),
    }
    if not all(checks.values()):
        raise ValueError(f"RSC-PF first-origin replay mismatch: {checks}")
    return {"checks": checks, "scheduler_soc_source": "carried_state", "full_array_equality_required": False}


def _compare_legacy_arrays(
    result: MatchedClosedLoopResult,
    legacy_rollout: str | Path,
    *,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-6,
) -> dict[str, Any]:
    """Compare the common artifact fields and report maximum differences."""

    with np.load(Path(legacy_rollout), allow_pickle=False) as payload:
        saved = {name: np.asarray(payload[name]) for name in payload.files}
    aliases = {
        "forecast": ("forecast_nominal", "forecast"),
        "scheduler_demand": ("scheduler_demand",),
        "planned_dispatch": ("planned_dispatch",),
        "settled_dispatch": ("settled_dispatch",),
        "state_hashes": ("state_hashes",),
        "times": ("times",),
    }
    checks: dict[str, bool] = {}
    max_abs: dict[str, float] = {}
    for key, names in aliases.items():
        current = np.asarray(result.arrays[key])
        previous = _legacy_array(saved, *names)
        if key == "times":
            current = current.astype("datetime64[ns]").astype(np.int64)
            previous = previous.astype("datetime64[ns]").astype(np.int64)
        if key == "state_hashes":
            checks[key] = np.array_equal(current.astype(str), previous.astype(str))
            max_abs[key] = 0.0 if checks[key] else float("inf")
        else:
            checks[key] = current.shape == previous.shape and bool(np.allclose(current, previous, atol=atol, rtol=rtol))
            max_abs[key] = float(np.max(np.abs(current - previous))) if current.shape == previous.shape else float("inf")
    if not all(checks.values()):
        raise ValueError(f"RSC-PF replay mismatch: {checks}")
    return {"checks": checks, "max_abs": max_abs}


def _compare_legacy_mapping(current: Mapping[str, Any], legacy_rollout: str | Path, *, atol: float, rtol: float) -> dict[str, Any]:
    """Compare a formal-v4.6 legacy result mapping with its saved NPZ."""

    with np.load(Path(legacy_rollout), allow_pickle=False) as payload:
        saved = {name: np.asarray(payload[name]) for name in payload.files}
    aliases = {
        "times": ("times",),
        "forecast_nominal": ("forecast_nominal", "forecast"),
        "scheduler_demand": ("scheduler_demand",),
        "controls": ("controls",),
        "planned_dispatch": ("planned_dispatch",),
        "settled_dispatch": ("settled_dispatch",),
        "state_hashes": ("state_hashes",),
        "initial_soc": ("initial_soc",),
        "previous_chp": ("previous_chp",),
        "shortage": ("shortage",),
        "operating_cost": ("operating_cost",),
        "physical_carbon": ("physical_carbon",),
        "penalized_objective": ("penalized_objective",),
        "realized_demand": ("realized_demand",),
        "realized_renewables": ("realized_renewables",),
        "realized_prices": ("realized_prices",),
    }
    checks: dict[str, bool] = {}; max_abs: dict[str, float] = {}
    for key, names in aliases.items():
        if key not in current:
            continue
        left, right = np.asarray(current[key]), _legacy_array(saved, *names)
        if key in {"times", "state_hashes"}:
            checks[key] = np.array_equal(left.astype(str), right.astype(str))
            max_abs[key] = 0.0 if checks[key] else float("inf")
        else:
            checks[key] = left.shape == right.shape and bool(np.allclose(left, right, atol=atol, rtol=rtol))
            max_abs[key] = float(np.max(np.abs(left - right))) if left.shape == right.shape else float("inf")
    if not all(checks.values()):
        raise ValueError(f"RSC-PF legacy replay mismatch: {checks}")
    return {"checks": checks, "max_abs": max_abs}


def run_and_verify_rsc_pf_legacy_replay(
    provider: Any,
    materialized_root: str | Path,
    legacy_rollout: str | Path,
    artifact_root: str | Path,
    *,
    train_data: str | Path | None = None,
    benchmark: str | Path | None = None,
    capacity_receipt: str | Path | None = None,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-6,
) -> dict[str, Any]:
    """Run the common provider over a cached 2019 view and audit replay.

    This utility is intentionally opt-in: it opens only ``selection_full``
    from a materialized cache and never accepts a sealed test path.
    """

    from .formal_v4_2_data import fit_train_normalization
    from .formal_v4_5_pilot_data import load_v45_pilot_cache
    from .formal_v4_6_rollout import rollout_v46_2019

    if not hasattr(provider, "model") or not hasattr(provider, "parameters"):
        raise ValueError("provider must expose the frozen RSC-PF model and parameters")
    root = Path(materialized_root)
    # Always pass the frozen lineage sources explicitly.  Leaving these as
    # loader defaults would permit an unrelated benchmark to be selected and
    # make a replay mismatch look like a model or numerical failure.
    materialized = load_v45_pilot_cache(
        materialized_root=root,
        train_data=train_data,
        benchmark=benchmark,
        capacity_receipt=capacity_receipt,
    )
    normalization = fit_train_normalization(materialized.normalization_source.split)
    indices = np.arange(len(materialized.selection_full), dtype=np.int64)
    tmp_artifact = Path(artifact_root) / ".legacy_replay_work"
    legacy = rollout_v46_2019(model=provider.model, materialized=materialized, indices=indices, normalization=normalization, parameters=provider.parameters, artifact_root=tmp_artifact, method_id="rsc_pf_joint")
    current = {name: getattr(legacy, name) for name in ("times", "forecast_nominal", "scheduler_demand", "controls", "planned_dispatch", "settled_dispatch", "state_hashes", "initial_soc", "previous_chp", "shortage", "operating_cost", "physical_carbon", "penalized_objective", "realized_demand", "realized_renewables", "realized_prices")}
    checks = _compare_legacy_mapping(current, legacy_rollout, atol=atol, rtol=rtol)
    out = Path(artifact_root) / "provenance"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "rsc_pf_legacy_replay.npz", **current)
    receipt = {
        "schema": "rsc-pf-legacy-replay-v1",
        "method_id": "RSC-PF",
        "rows": int(len(materialized.selection_full)),
        "scheduler_soc_source": "materialized_origin",
        "evaluation_year_accessed": False,
        **checks,
    }
    (out / "RSC_PF_LEGACY_REPLAY.json").write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    return receipt


__all__ = [
    "PTOProvider", "DigitalTwinsProvider", "solve_pto_from_forecast",
    "build_external_provider", "load_frozen_rsc_pf_provider",
    "run_and_verify_rsc_pf_legacy_replay", "verify_rsc_pf_first_origin",
]
