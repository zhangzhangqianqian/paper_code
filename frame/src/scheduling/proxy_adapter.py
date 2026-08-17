"""Scheme2R-to-proxy feature adapter, raw feasibility and exact fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .dispatch_lp import DispatchInputs, solve_dispatch_lp
from .proxy_contract import FEATURE_ORDER, HORIZON, LABEL_ORDER, ProxyContract, contract_sha256, file_sha256
from .proxy_dataset import ProxyNormalizationStats
from .proxy_model import SchedulingProxy
from .proxy_physics import _parameter, balance_residuals, conversion_residuals, soc_residuals
from .synthetic_scenarios import load_benchmark


TASK_ORDER: Tuple[str, ...] = ("electricity", "cooling", "heating", "gas")
RENEWABLE_ORDER: Tuple[str, ...] = ("pv", "wt")


def _is_sha256(value: Any) -> bool:
    text = str(value).lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def validate_proxy_normalization_stats(
    stats: ProxyNormalizationStats,
    contract: ProxyContract | None = None,
    benchmark: Mapping[str, Any] | str | Path | None = None,
) -> None:
    """Validate stats before any adapter construction or feature assembly.

    Adapter inference is an artifact boundary: malformed normalization arrays
    or provenance must fail closed before they can affect model inputs.  A
    benchmark mapping remains supported for the evaluation helper, but an
    external benchmark hash can only be independently checked when a file
    path is supplied.
    """

    if not isinstance(stats, ProxyNormalizationStats):
        raise TypeError("stats must be ProxyNormalizationStats")
    try:
        feature_order = tuple(stats.feature_order)
        label_order = tuple(stats.label_order)
    except TypeError as exc:
        raise ValueError("normalization stats feature/label order metadata is invalid") from exc
    if feature_order != FEATURE_ORDER or label_order != LABEL_ORDER:
        raise ValueError("normalization stats feature/label order mismatch")
    try:
        input_mean = np.asarray(stats.input_mean, dtype=np.float64)
        input_scale = np.asarray(stats.input_scale, dtype=np.float64)
        label_scale = np.asarray(stats.label_scale, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalization stats arrays must be numeric") from exc
    if input_mean.shape != (len(FEATURE_ORDER),) or input_scale.shape != (len(FEATURE_ORDER),):
        raise ValueError("normalization input statistics must have shape [10]")
    if label_scale.shape != (len(LABEL_ORDER),):
        raise ValueError("normalization label scales must have shape [21]")
    if not np.isfinite(input_mean).all() or not np.isfinite(input_scale).all() or not np.isfinite(label_scale).all():
        raise ValueError("normalization stats must be finite")
    if (input_scale <= 0.0).any() or (label_scale <= 0.0).any():
        raise ValueError("normalization stats scales must be strictly positive")
    try:
        epsilon = float(stats.epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalization epsilon must be finite and positive") from exc
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("normalization epsilon must be finite and positive")
    if not _is_sha256(stats.benchmark_sha256) or not _is_sha256(stats.contract_sha256):
        raise ValueError("normalization stats provenance hashes are invalid")
    try:
        train_seed = int(stats.train_seed)
        train_seed_numeric = float(stats.train_seed)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("normalization stats must prove a positive train seed") from exc
    if stats.fit_split != "train" or train_seed <= 0 or not np.isfinite(train_seed_numeric) or train_seed_numeric != float(train_seed):
        raise ValueError("normalization stats must prove a positive train seed")
    if stats.source_type != "pure_simulation" or not str(stats.generator_version):
        raise ValueError("normalization stats source/generator provenance is invalid")
    if contract is not None:
        expected_contract_hash = contract_sha256(contract)
        if stats.contract_sha256.lower() != expected_contract_hash.lower():
            raise ValueError("normalization stats contract SHA-256 mismatch")
        if stats.benchmark_sha256.lower() != str(contract.benchmark_sha256).lower():
            raise ValueError("normalization stats benchmark SHA-256 does not match contract")
        if stats.source_type != contract.source_type or stats.generator_version != contract.generator_version:
            raise ValueError("normalization stats source/generator does not match contract")
        if tuple(contract.feature_order) != FEATURE_ORDER or tuple(contract.label_order) != LABEL_ORDER:
            raise ValueError("proxy contract feature/label order is invalid")
    if benchmark is not None:
        if isinstance(benchmark, (str, Path)):
            actual_benchmark_hash = file_sha256(benchmark)
            if stats.benchmark_sha256.lower() != actual_benchmark_hash.lower():
                raise ValueError("normalization stats benchmark SHA-256 mismatch")
        elif contract is not None:
            raise ValueError("benchmark provenance validation requires a benchmark file path")


@dataclass(frozen=True)
class ProxyAdapterOutput:
    features: np.ndarray  # [N,4,10], raw physical units
    normalized_features: np.ndarray  # [N,4,10]
    gas_prior_mask: np.ndarray  # [N,4]
    clipped_negative_count: int
    task_order: Tuple[str, ...] = TASK_ORDER
    renewable_order: Tuple[str, ...] = RENEWABLE_ORDER

    @property
    def inputs(self) -> np.ndarray:
        return self.features


@dataclass(frozen=True)
class ProxyInferenceResult:
    raw_dispatch: np.ndarray  # [N,4,21]
    safe_dispatch: np.ndarray  # [N,4,21]
    fallback_mask: np.ndarray  # [N]
    raw_feasible_mask: np.ndarray  # [N]
    fallback_reasons: Tuple[str, ...]
    features: np.ndarray
    normalized_features: np.ndarray
    clipped_negative_count: int
    raw_residual: np.ndarray

    @property
    def output(self) -> np.ndarray:
        return self.safe_dispatch


def _benchmark_values(benchmark: Mapping[str, Any] | str | Path | None) -> Mapping[str, float]:
    if benchmark is None:
        return {
            "grid_energy_price": 1.0,
            "gas_energy_price": 0.6,
            "carbon_price_default": 0.0,
            "grid_emission_factor": 0.5,
            "gas_emission_factor": 0.25,
            "bess_throughput_cost": 1.0e-6,
            "unserved_penalty": 100.0,
            "chp_electric_efficiency": 0.35,
            "chp_heat_efficiency": 0.45,
            "gas_boiler_efficiency": 0.9,
            "electric_chiller_cop": 3.5,
            "absorption_chiller_cop": 0.75,
            "bess_roundtrip_efficiency": 0.9,
            "bess_energy_capacity": 1022.4,
        }
    data = load_benchmark(benchmark) if isinstance(benchmark, (str, Path)) else benchmark
    values = data["values"] if isinstance(data, Mapping) and "values" in data else data
    return {str(k): float(v) for k, v in values.items()}


def _prices_array(prices: Any, n: int, values: Mapping[str, float]) -> np.ndarray:
    if prices is None:
        defaults = np.asarray([
            values.get("grid_energy_price", 1.0), values.get("gas_energy_price", 0.6),
            values.get("carbon_price_default", 0.0),
        ], dtype=np.float64)
        return np.broadcast_to(defaults[None, None, :], (n, HORIZON, 3)).copy()
    if isinstance(prices, Mapping):
        try:
            price = np.stack([np.asarray(prices[name], dtype=np.float64) for name in ("grid_price", "gas_price", "carbon_price")], axis=-1)
        except KeyError as exc:
            raise ValueError("price mapping must contain grid_price, gas_price and carbon_price") from exc
    else:
        price = np.asarray(prices, dtype=np.float64)
    if price.ndim == 1 and price.shape == (3,):
        price = np.broadcast_to(price[None, None, :], (n, HORIZON, 3)).copy()
    elif price.ndim == 2 and price.shape == (HORIZON, 3):
        price = np.broadcast_to(price[None, :, :], (n, HORIZON, 3)).copy()
    if price.shape != (n, HORIZON, 3):
        raise ValueError("prices must have shape [3], [4,3] or [N,4,3]")
    if not np.isfinite(price).all() or (price < 0).any():
        raise ValueError("prices must be finite and non-negative")
    # dispatch_lp.py accepts scalar prices.  A time-varying schedule would
    # therefore make proxy features disagree with exact fallback semantics;
    # reject it instead of silently reducing it to a mean price.
    if not np.allclose(price, price[:, :1, :], rtol=0.0, atol=1.0e-12):
        raise ValueError("time-varying price schedules are unsupported: the exact LP uses scalar prices per scenario")
    return price


def _soc_array(initial_soc: Any, n: int) -> np.ndarray:
    soc = np.asarray(initial_soc, dtype=np.float64)
    if soc.ndim == 0:
        soc = np.full(n, float(soc), dtype=np.float64)
    if soc.shape != (n,):
        raise ValueError("initial_soc must be scalar or shape [N]")
    if not np.isfinite(soc).all() or (soc < 0).any() or (soc > 1).any():
        raise ValueError("initial_soc must be finite and in [0,1]")
    return soc


def adapt_scheme2r_inputs(
    load_predictions: np.ndarray,
    renewable_predictions: np.ndarray,
    stats: ProxyNormalizationStats,
    prices: Any = None,
    initial_soc: Any = 0.5,
    benchmark: Mapping[str, Any] | str | Path | None = None,
    use_gas_prior: bool = True,
    gas_prior_mask: np.ndarray | None = None,
    task_order: Sequence[str] = TASK_ORDER,
    renewable_order: Sequence[str] = RENEWABLE_ORDER,
) -> ProxyAdapterOutput:
    """Map frozen Scheme2R forecasts to the exact ten-feature proxy contract."""

    validate_proxy_normalization_stats(stats, benchmark=benchmark)
    if tuple(task_order) != TASK_ORDER:
        raise ValueError(f"task order must be exactly {TASK_ORDER}")
    if tuple(renewable_order) != RENEWABLE_ORDER:
        raise ValueError(f"renewable order must be exactly {RENEWABLE_ORDER}")
    loads = np.asarray(load_predictions, dtype=np.float64)
    if isinstance(renewable_predictions, Mapping):
        if "predictions" not in renewable_predictions:
            raise ValueError("renewable mapping must contain predictions")
        renewable_predictions = renewable_predictions["predictions"]
    renewables = np.asarray(renewable_predictions, dtype=np.float64)
    if loads.ndim != 3 or loads.shape[1:] != (HORIZON, 4):
        raise ValueError("load_predictions must have shape [N,4,4] in electricity/cooling/heating/gas order")
    if renewables.ndim != 3 or renewables.shape[1:] != (HORIZON, 2):
        raise ValueError("renewable_predictions must have shape [N,4,2] in PV/WT order")
    if loads.shape[0] != renewables.shape[0]:
        raise ValueError("load and renewable sample counts must match")
    if not np.isfinite(loads).all() or not np.isfinite(renewables).all():
        raise ValueError("forecast inputs must be finite")
    n = loads.shape[0]
    negative_count = int((loads < 0).sum() + (renewables < 0).sum())
    loads = np.maximum(loads, 0.0)
    renewables = np.maximum(renewables, 0.0)
    values = _benchmark_values(benchmark)
    price_array = _prices_array(prices, n, values)
    soc = _soc_array(initial_soc, n)
    gas = loads[:, :, 3] if use_gas_prior else np.zeros((n, HORIZON), dtype=np.float64)
    if gas_prior_mask is None:
        mask = np.ones((n, HORIZON), dtype=np.float64) if use_gas_prior else np.zeros((n, HORIZON), dtype=np.float64)
    else:
        mask = np.asarray(gas_prior_mask, dtype=np.float64)
        if mask.shape == (n, HORIZON, 1):
            mask = mask[..., 0]
        if mask.shape != (n, HORIZON) or not np.isin(mask, [0.0, 1.0]).all():
            raise ValueError("gas_prior_mask must have shape [N,4] and be binary")
        if not use_gas_prior:
            mask = np.zeros_like(mask)
            gas = np.zeros_like(gas)
    # The mask is part of the auxiliary station-side channel semantics.  Apply
    # it before assembly so masked gas can never leak into proxy features.
    gas = gas * mask
    features = np.concatenate([
        loads[:, :, :3], gas[..., None], renewables,
        price_array,
        np.broadcast_to(soc[:, None, None], (n, HORIZON, 1)),
    ], axis=-1)
    if features.shape != (n, HORIZON, len(FEATURE_ORDER)):
        raise AssertionError("proxy feature assembly shape mismatch")
    normalized = stats.transform_inputs(features)
    return ProxyAdapterOutput(
        features=features,
        normalized_features=normalized,
        gas_prior_mask=mask,
        clipped_negative_count=negative_count,
    )


def evaluate_raw_feasibility(
    dispatch: np.ndarray,
    features: np.ndarray,
    parameters: Mapping[str, Any],
    tolerance: float = 1.0e-3,
) -> dict[str, np.ndarray | float]:
    """Evaluate raw proxy outputs before any safety fallback."""

    y = np.asarray(dispatch, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if y.ndim != 3 or y.shape[1:] != (HORIZON, len(LABEL_ORDER)):
        raise ValueError("dispatch must have shape [N,4,21]")
    if x.shape != (y.shape[0], HORIZON, len(FEATURE_ORDER)):
        raise ValueError("features must have shape [N,4,10]")
    finite = np.isfinite(y).all(axis=(1, 2))
    # Use torch helpers so feasibility uses exactly the same equations as loss.
    yt = torch.as_tensor(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float64)
    xt = torch.as_tensor(x, dtype=torch.float64)
    balance = balance_residuals(yt, xt).detach().numpy()
    conversion = conversion_residuals(yt, parameters).detach().numpy()
    soc = soc_residuals(yt, xt, parameters)
    soc_state = soc["state"].detach().numpy()
    soc_terminal = soc["terminal"].detach().numpy()
    # The frozen LP has finite upper bounds for dispatch variables and two
    # renewable split equalities.  Keep these diagnostics separate so raw
    # proxy feasibility is not confused with fallback feasibility.
    idx = {name: position for position, name in enumerate(LABEL_ORDER)}
    pv_split = np.abs(y[:, :, idx["pv_use"]] + y[:, :, idx["pv_curt"]] - x[:, :, 4])
    wt_split = np.abs(y[:, :, idx["wt_use"]] + y[:, :, idx["wt_curt"]] - x[:, :, 5])
    renewable_residual = np.maximum(np.max(pv_split, axis=1), np.max(wt_split, axis=1))
    upper_bounds = {
        "grid": _parameter(parameters, "grid_import_capacity"),
        "g_chp": _parameter(parameters, "chp_electric_capacity") / max(_parameter(parameters, "chp_electric_efficiency"), 1e-12),
        "g_gb": _parameter(parameters, "gas_boiler_capacity") / max(_parameter(parameters, "gas_boiler_efficiency"), 1e-12),
        "p_chp": _parameter(parameters, "chp_electric_capacity"),
        "q_chp": _parameter(parameters, "chp_heat_capacity"),
        "q_gb": _parameter(parameters, "gas_boiler_capacity"),
        "p_ec": _parameter(parameters, "electric_chiller_capacity") / max(_parameter(parameters, "electric_chiller_cop"), 1e-12),
        "q_ec": _parameter(parameters, "electric_chiller_capacity"),
        "q_ac_in": _parameter(parameters, "absorption_chiller_capacity") / max(_parameter(parameters, "absorption_chiller_cop"), 1e-12),
        "q_ac": _parameter(parameters, "absorption_chiller_capacity"),
        "p_charge": _parameter(parameters, "bess_power_capacity"),
        "p_discharge": _parameter(parameters, "bess_power_capacity"),
        "soc": _parameter(parameters, "bess_energy_capacity"),
    }
    safe_y = np.nan_to_num(y, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
    bound_residual = np.zeros(y.shape[0], dtype=np.float64)
    for name, upper in upper_bounds.items():
        bound_residual = np.maximum(bound_residual, np.max(safe_y[:, :, idx[name]] - float(upper), axis=1))
    bound_residual = np.maximum(bound_residual, 0.0)
    ramp = _parameter(parameters, "chp_ramp_fraction") * _parameter(parameters, "chp_electric_capacity")
    p_chp = safe_y[:, :, idx["p_chp"]]
    p_change = np.concatenate([p_chp[:, :1], np.diff(p_chp, axis=1)], axis=1)
    ramp_residual = np.maximum(np.max(np.abs(p_change) - ramp, axis=1), 0.0)
    residual = np.maximum.reduce([
        np.max(np.abs(balance), axis=(1, 2)),
        np.max(np.abs(conversion), axis=(1, 2)),
        np.max(np.abs(soc_state), axis=1),
        np.abs(soc_terminal),
        renewable_residual,
        bound_residual,
        ramp_residual,
    ])
    nonnegative = np.all(np.nan_to_num(y, nan=-np.inf, posinf=np.inf, neginf=-np.inf) >= -float(tolerance), axis=(1, 2))
    feasible = finite & nonnegative & (residual <= float(tolerance))
    return {
        "feasible_mask": feasible,
        "residual": residual,
        "balance_residual": np.max(np.abs(balance), axis=(1, 2)),
        "conversion_residual": np.max(np.abs(conversion), axis=(1, 2)),
        "soc_residual": np.maximum(np.max(np.abs(soc_state), axis=1), np.abs(soc_terminal)),
        "renewable_residual": renewable_residual,
        "renewable_split_residual": renewable_residual,
        "bound_residual": bound_residual,
        "capacity_residual": bound_residual,
        "ramp_residual": ramp_residual,
        "ramp_constraint_residual": ramp_residual,
        "feasible_rate": np.asarray(float(np.mean(feasible))),
    }


class Scheme2RProxyAdapter:
    """Stateful adapter holding train-only normalization and frozen prices."""

    def __init__(
        self,
        stats: ProxyNormalizationStats,
        benchmark: Mapping[str, Any] | str | Path | None = None,
        contract: ProxyContract | None = None,
    ):
        self.stats = stats
        self.benchmark = benchmark
        self.contract = contract
        validate_proxy_normalization_stats(stats, contract=contract, benchmark=benchmark)
        self.parameters = _benchmark_values(benchmark)

    def prepare(self, *args: Any, **kwargs: Any) -> ProxyAdapterOutput:
        kwargs.setdefault("benchmark", self.benchmark)
        return adapt_scheme2r_inputs(*args, stats=self.stats, **kwargs)

    def predict(self, model: SchedulingProxy, prepared: ProxyAdapterOutput) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            normalized = model(torch.from_numpy(prepared.normalized_features.astype(np.float32))).cpu().numpy()
        physical = self.stats.inverse_dispatch(normalized)
        return np.asarray(physical, dtype=np.float64)

    def infer(
        self,
        model: SchedulingProxy,
        load_predictions: np.ndarray,
        renewable_predictions: np.ndarray,
        prices: Any = None,
        initial_soc: Any = 0.5,
        use_gas_prior: bool = True,
        allow_exact_fallback: bool | None = None,
        feasibility_tolerance: float | None = None,
    ) -> ProxyInferenceResult:
        prepared = self.prepare(
            load_predictions, renewable_predictions, prices=prices, initial_soc=initial_soc,
            use_gas_prior=use_gas_prior,
        )
        raw = self.predict(model, prepared)
        tolerance = float(feasibility_tolerance if feasibility_tolerance is not None else self.contract.safety["feasibility_tolerance"] if self.contract is not None else 1e-3)
        report = evaluate_raw_feasibility(raw, prepared.features, self.parameters, tolerance=tolerance)
        raw_mask = np.asarray(report["feasible_mask"], dtype=bool)
        allow = bool(allow_exact_fallback if allow_exact_fallback is not None else self.contract.safety["allow_exact_fallback"] if self.contract is not None else True)
        safe = raw.copy()
        fallback = np.zeros(raw.shape[0], dtype=bool)
        reasons: list[str] = ["none"] * raw.shape[0]
        if allow:
            for index in np.flatnonzero(~raw_mask):
                try:
                    params = dict(self.parameters)
                    params["grid_energy_price"] = float(np.mean(prepared.features[index, :, 6]))
                    params["gas_energy_price"] = float(np.mean(prepared.features[index, :, 7]))
                    params["carbon_price"] = float(np.mean(prepared.features[index, :, 8]))
                    result = solve_dispatch_lp(DispatchInputs(
                        demand=prepared.features[index, :, :3],
                        pv_available=prepared.features[index, :, 4],
                        wt_available=prepared.features[index, :, 5],
                        parameters=params,
                        initial_soc=float(prepared.features[index, 0, 9]),
                    ))
                    if result.success:
                        safe[index] = np.stack([result.values[name] for name in LABEL_ORDER], axis=-1)
                        fallback[index] = True
                        reasons[index] = "raw_infeasible_exact_lp_fallback"
                    else:
                        reasons[index] = "raw_infeasible_lp_failed"
                except Exception as exc:  # fallback must never hide raw metrics
                    reasons[index] = "raw_infeasible_fallback_error:" + type(exc).__name__
        return ProxyInferenceResult(
            raw_dispatch=raw,
            safe_dispatch=safe,
            fallback_mask=fallback,
            raw_feasible_mask=raw_mask,
            fallback_reasons=tuple(reasons),
            features=prepared.features,
            normalized_features=prepared.normalized_features,
            clipped_negative_count=prepared.clipped_negative_count,
            raw_residual=np.asarray(report["residual"], dtype=np.float64),
        )


# Alias names used by the plan and tests.
ProxyAdapter = Scheme2RProxyAdapter
build_proxy_inputs = adapt_scheme2r_inputs
adapt_forecasts = adapt_scheme2r_inputs
adapt_scheme2r_predictions = adapt_scheme2r_inputs
safe_proxy_inference = Scheme2RProxyAdapter.infer


__all__ = [
    "TASK_ORDER", "RENEWABLE_ORDER", "ProxyAdapterOutput", "ProxyInferenceResult",
    "validate_proxy_normalization_stats", "adapt_scheme2r_inputs", "build_proxy_inputs", "adapt_forecasts", "evaluate_raw_feasibility",
    "adapt_scheme2r_predictions",
    "Scheme2RProxyAdapter", "ProxyAdapter", "safe_proxy_inference",
]
