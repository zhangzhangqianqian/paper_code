"""Evaluation metrics with explicit raw-proxy versus exact-fallback reporting."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dispatch_lp import DispatchInputs, solve_dispatch_lp
from .proxy_adapter import ProxyInferenceResult, Scheme2RProxyAdapter, evaluate_raw_feasibility
from .proxy_contract import FEATURE_ORDER, LABEL_ORDER
from .proxy_dataset import LabeledProxySplit, ProxyNormalizationStats, scenario_id_digest
from .proxy_model import SchedulingProxy
from .proxy_physics import carbon_emissions, operating_cost


def _array_metrics(prediction: np.ndarray, target: np.ndarray, stats: ProxyNormalizationStats | None = None) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 3 or pred.shape[-1] != len(LABEL_ORDER):
        raise ValueError("prediction and target must both have shape [N,4,21]")
    diff = pred - truth
    result = {
        "dispatch_mae": float(np.mean(np.abs(diff))),
        "dispatch_rmse": float(np.sqrt(np.mean(diff ** 2))),
    }
    if stats is not None:
        scale = stats.label_scale.reshape(1, 1, -1)
        normalized = diff / scale
        result.update({
            "normalized_dispatch_mae": float(np.mean(np.abs(normalized))),
            "normalized_dispatch_rmse": float(np.sqrt(np.mean(normalized ** 2))),
        })
    return result


def dispatch_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    features: np.ndarray,
    teacher_cost: np.ndarray,
    teacher_carbon: np.ndarray,
    parameters: Mapping[str, Any],
    stats: ProxyNormalizationStats | None = None,
    feasibility_tolerance: float = 1e-3,
    teacher_objective: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute imitation, economic, carbon and raw feasibility metrics."""

    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    metrics = _array_metrics(pred, target, stats)
    pred_cost = operating_cost(torch.from_numpy(pred), torch.from_numpy(x), parameters).detach().numpy()
    pred_carbon = carbon_emissions(torch.from_numpy(pred), torch.from_numpy(x), parameters).detach().numpy()
    teacher_cost = np.asarray(teacher_cost, dtype=np.float64)
    teacher_carbon = np.asarray(teacher_carbon, dtype=np.float64)
    cost_gap = pred_cost - teacher_cost
    cost_denom = np.maximum(np.abs(teacher_cost), 1.0)
    carbon_gap = pred_carbon - teacher_carbon
    carbon_denom = np.maximum(np.abs(teacher_carbon), 1.0)
    metrics.update({
        "predicted_cost_mean": float(np.mean(pred_cost)),
        "teacher_cost_mean": float(np.mean(teacher_cost)),
        "cost_gap_mean": float(np.mean(cost_gap)),
        "cost_regret_mean": float(np.mean(np.maximum(cost_gap, 0.0))),
        "cost_regret_relative": float(np.mean(np.maximum(cost_gap, 0.0) / cost_denom)),
        "predicted_carbon_mean": float(np.mean(pred_carbon)),
        "teacher_carbon_mean": float(np.mean(teacher_carbon)),
        "carbon_absolute_error": float(np.mean(np.abs(carbon_gap))),
        "carbon_relative_error": float(np.mean(np.abs(carbon_gap) / carbon_denom)),
    })
    # The frozen LP objective is operating cost plus carbon-price times
    # physical grid/gas emissions.  Price features are scalar-per-scenario in
    # the teacher, though the proxy contract stores their H-step broadcast.
    carbon_price = np.mean(x[:, :, 8], axis=1)
    predicted_objective = pred_cost + carbon_price * pred_carbon
    metrics["predicted_objective_mean"] = float(np.mean(predicted_objective))
    if teacher_objective is not None:
        teacher_objective = np.asarray(teacher_objective, dtype=np.float64)
        if teacher_objective.shape != (pred.shape[0],) or not np.isfinite(teacher_objective).all():
            raise ValueError("teacher_objective must have shape [N] and be finite")
        objective_gap = predicted_objective - teacher_objective
        objective_denom = np.maximum(np.abs(teacher_objective), 1.0)
        metrics.update({
            "teacher_objective_mean": float(np.mean(teacher_objective)),
            "objective_gap_mean": float(np.mean(objective_gap)),
            "objective_gap_absolute_mean": float(np.mean(np.abs(objective_gap))),
            "objective_gap_relative_absolute_mean": float(np.mean(np.abs(objective_gap) / objective_denom)),
            "signed_objective_gap_mean": float(np.mean(objective_gap)),
            "absolute_objective_gap_mean": float(np.mean(np.abs(objective_gap))),
            "relative_absolute_objective_gap_mean": float(np.mean(np.abs(objective_gap) / objective_denom)),
            "objective_gap_signed_mean": float(np.mean(objective_gap)),
            "objective_gap_abs_mean": float(np.mean(np.abs(objective_gap))),
            "objective_gap_relative_abs_mean": float(np.mean(np.abs(objective_gap) / objective_denom)),
        })
    feasible = evaluate_raw_feasibility(pred, x, parameters, feasibility_tolerance)
    metrics.update({
        "balance_residual_max": float(np.max(np.asarray(feasible["balance_residual"]))),
        "conversion_residual_max": float(np.max(np.asarray(feasible["conversion_residual"]))),
        "soc_residual_max": float(np.max(np.asarray(feasible["soc_residual"]))),
        "renewable_residual_max": float(np.max(np.asarray(feasible["renewable_residual"]))),
        "bound_residual_max": float(np.max(np.asarray(feasible["bound_residual"]))),
        "ramp_residual_max": float(np.max(np.asarray(feasible["ramp_residual"]))),
        "raw_feasible_rate": float(np.mean(np.asarray(feasible["feasible_mask"]))),
        "raw_residual_mean": float(np.mean(np.asarray(feasible["residual"]))),
    })
    return metrics


def _latency_summary(values_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(list(values_ms), dtype=np.float64)
    if values.size == 0:
        return {"median_ms": 0.0, "p95_ms": 0.0}
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def measure_proxy_latency(
    model: SchedulingProxy,
    normalized_features: np.ndarray,
    repeats: int = 5,
    warmup: int = 2,
) -> dict[str, Any]:
    x = torch.from_numpy(np.asarray(normalized_features, dtype=np.float32))
    if x.ndim != 3:
        raise ValueError("normalized_features must have shape [N,4,10]")
    repeats = max(1, int(repeats))
    warmup = max(0, int(warmup))
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        for index in range(min(warmup, x.shape[0])):
            model(x[index:index + 1])
        batch_values: list[float] = []
        scenario_values: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            batch_values.append((time.perf_counter() - start) * 1000.0)
            for index in range(x.shape[0]):
                start = time.perf_counter()
                model(x[index:index + 1])
                scenario_values.append((time.perf_counter() - start) * 1000.0)
    batch_summary = _latency_summary(batch_values)
    scenario_summary = _latency_summary(scenario_values)
    return {
        "units": "ms",
        "batch_size": int(x.shape[0]),
        "repeats": repeats,
        "warmup_runs": warmup,
        "proxy_batch_median_ms": batch_summary["median_ms"],
        "proxy_batch_p95_ms": batch_summary["p95_ms"],
        "proxy_per_scenario_median_ms": scenario_summary["median_ms"],
        "proxy_per_scenario_p95_ms": scenario_summary["p95_ms"],
        "median_ms": batch_summary["median_ms"],
        "p95_ms": batch_summary["p95_ms"],
    }


def measure_exact_lp_latency(
    features: np.ndarray,
    parameters: Mapping[str, Any],
    repeats: int = 1,
    warmup: int = 1,
) -> dict[str, Any]:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (4, 10):
        raise ValueError("features must have shape [N,4,10]")
    repeats = max(1, int(repeats))
    warmup = max(0, int(warmup))
    def solve_one(index: int) -> float:
        start = time.perf_counter()
        solve_dispatch_lp(DispatchInputs(
            demand=x[index, :, :3], pv_available=x[index, :, 4], wt_available=x[index, :, 5],
            parameters=dict(parameters, grid_energy_price=float(np.mean(x[index, :, 6])), gas_energy_price=float(np.mean(x[index, :, 7])), carbon_price=float(np.mean(x[index, :, 8]))),
            initial_soc=float(x[index, 0, 9]),
        ))
        return (time.perf_counter() - start) * 1000.0
    for _ in range(warmup):
        for index in range(x.shape[0]):
            solve_one(index)
    values: list[float] = []
    for _ in range(repeats):
        for index in range(x.shape[0]):
            values.append(solve_one(index))
    summary = _latency_summary(values)
    return {
        "units": "ms",
        "mode": "exact_lp_per_scenario",
        "batch_size": 1,
        "scenario_count": int(x.shape[0]),
        "repeats": repeats,
        "warmup_runs": warmup,
        "exact_lp_per_scenario_median_ms": summary["median_ms"],
        "exact_lp_per_scenario_p95_ms": summary["p95_ms"],
        "median_ms": summary["median_ms"],
        "p95_ms": summary["p95_ms"],
    }


def evaluate_inference_result(
    result: ProxyInferenceResult,
    target: np.ndarray,
    teacher_cost: np.ndarray,
    teacher_carbon: np.ndarray,
    parameters: Mapping[str, Any],
    stats: ProxyNormalizationStats | None = None,
    feasibility_tolerance: float = 1e-3,
    teacher_objective: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return separate raw and fallback-assisted metric blocks."""

    raw = dispatch_metrics(
        result.raw_dispatch, target, result.features, teacher_cost, teacher_carbon, parameters,
        stats=stats, feasibility_tolerance=feasibility_tolerance, teacher_objective=teacher_objective,
    )
    safe = dispatch_metrics(
        result.safe_dispatch, target, result.features, teacher_cost, teacher_carbon, parameters,
        stats=stats, feasibility_tolerance=feasibility_tolerance, teacher_objective=teacher_objective,
    )
    fallback = np.asarray(result.fallback_mask, dtype=bool)
    return {
        "raw_proxy": raw,
        "safe_with_exact_fallback": safe,
        "fallback_rate": float(np.mean(fallback)),
        "fallback_count": int(fallback.sum()),
        "safe_success_rate": float(np.mean(np.asarray(safe["raw_feasible_rate"]) if isinstance(safe["raw_feasible_rate"], np.ndarray) else np.asarray([safe["raw_feasible_rate"]]))),
        "clipped_negative_count": int(result.clipped_negative_count),
        "fallback_reasons": dict((reason, int(result.fallback_reasons.count(reason))) for reason in set(result.fallback_reasons)),
    }


def evaluate_gas_prior_ablation(
    model: SchedulingProxy,
    adapter: Scheme2RProxyAdapter,
    split: LabeledProxySplit,
    use_gas_prior: bool,
    allow_exact_fallback: bool = False,
    feasibility_tolerance: float = 1e-3,
) -> tuple[ProxyInferenceResult, np.ndarray]:
    result = adapter.infer(
        model,
        load_predictions=np.concatenate([split.demand, split.inputs[:, :, 3:4]], axis=-1),
        renewable_predictions=np.stack([split.pv_available, split.wt_available], axis=-1),
        prices=split.prices,
        initial_soc=split.initial_soc,
        use_gas_prior=use_gas_prior,
        allow_exact_fallback=allow_exact_fallback,
        feasibility_tolerance=feasibility_tolerance,
    )
    return result, result.raw_dispatch


def evaluate_model_on_split(
    model: SchedulingProxy,
    split: LabeledProxySplit,
    stats: ProxyNormalizationStats,
    parameters: Mapping[str, Any],
    adapter: Scheme2RProxyAdapter | None = None,
    allow_exact_fallback: bool = True,
    feasibility_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Evaluate a labelled split and both gas-prior modes on identical samples."""

    if adapter is None:
        adapter = Scheme2RProxyAdapter(stats, benchmark={"values": parameters})
    loads = np.concatenate([split.demand, split.inputs[:, :, 3:4]], axis=-1)
    renewable = np.stack([split.pv_available, split.wt_available], axis=-1)
    result = adapter.infer(
        model, loads, renewable, prices=split.prices, initial_soc=split.initial_soc,
        use_gas_prior=True, allow_exact_fallback=allow_exact_fallback,
        feasibility_tolerance=feasibility_tolerance,
    )
    metrics = evaluate_inference_result(
        result, split.dispatch, split.teacher_cost, split.teacher_carbon, parameters, stats, feasibility_tolerance,
        teacher_objective=split.teacher_objective,
    )
    # No-prior ablation uses exactly the same scenarios; its raw and fallback
    # blocks remain separate as required by the contract.
    no_prior = adapter.infer(
        model, loads, renewable, prices=split.prices, initial_soc=split.initial_soc,
        use_gas_prior=False, allow_exact_fallback=allow_exact_fallback,
        feasibility_tolerance=feasibility_tolerance,
    )
    metrics["gas_prior_ablation"] = evaluate_inference_result(
        no_prior, split.dispatch, split.teacher_cost, split.teacher_carbon, parameters, stats, feasibility_tolerance,
        teacher_objective=split.teacher_objective,
    )
    metrics["proxy_latency"] = measure_proxy_latency(model, result.normalized_features)
    metrics["exact_lp_latency"] = measure_exact_lp_latency(result.features, parameters)
    metrics["provenance"] = {
        "split": split.split,
        "seed": int(split.seed),
        "source_type": split.source_type,
        "generator_version": split.generator_version,
        "benchmark_sha256": split.benchmark_sha256,
        "contract_sha256": split.contract_sha256,
        "feature_order": list(FEATURE_ORDER),
        "label_order": list(LABEL_ORDER),
        "selection_split": "validation",
        "test_split_used_for_selection": False,
    }
    return metrics


def save_metrics(metrics: Mapping[str, Any], path: str | Path) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(metrics), indent=2, sort_keys=True), encoding="utf-8")


def load_prediction_artifact(
    path: str | Path,
    *,
    expected_split: str,
    expected_count: int | None = None,
    expected_seed: int | None = None,
    expected_scenario_ids: np.ndarray | None = None,
    expected_scenario_id_digest: str | None = None,
    expected_benchmark_sha256: str | None = None,
    expected_contract_sha256: str | None = None,
    expected_source_type: str = "pure_simulation",
    expected_generator_version: str = "synthetic-scheduling-v1",
) -> dict[str, np.ndarray]:
    """Fail-closed prediction artifact loader for evaluation/reload paths."""

    if expected_split not in {"train", "validation", "test"}:
        raise ValueError("prediction artifact requires an explicit expected split")
    if expected_seed is None:
        raise ValueError("prediction artifact requires an expected seed")
    if expected_scenario_ids is None and expected_scenario_id_digest is None:
        raise ValueError("prediction artifact requires expected scenario IDs or digest")
    required = {
        "raw_prediction", "safe_prediction", "target", "fallback_mask", "raw_feasible_mask",
        "features", "scenario_ids", "teacher_objective", "benchmark_sha256", "contract_sha256",
        "source_type", "generator_version", "split", "seed", "sample_count", "scenario_id_digest",
        "feature_order", "label_order",
    }
    with np.load(path, allow_pickle=False) as payload:
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"prediction artifact is missing required fields: {missing}")
        split = str(np.asarray(payload["split"]).item())
        if split != expected_split:
            raise ValueError("prediction artifact split mismatch")
        seed_value = np.asarray(payload["seed"])
        sample_count_value = np.asarray(payload["sample_count"])
        if seed_value.ndim != 0 or not np.issubdtype(seed_value.dtype, np.integer):
            raise ValueError("prediction artifact seed must be a scalar integer")
        if sample_count_value.ndim != 0 or not np.issubdtype(sample_count_value.dtype, np.integer):
            raise ValueError("prediction artifact sample_count must be a scalar integer")
        stored_seed = int(seed_value.item())
        stored_count = int(sample_count_value.item())
        if stored_seed <= 0 or stored_count <= 0:
            raise ValueError("prediction artifact seed/sample_count must be positive")
        if expected_seed is not None and stored_seed != int(expected_seed):
            raise ValueError("prediction artifact seed mismatch")
        source_type = str(np.asarray(payload["source_type"]).item())
        generator_version = str(np.asarray(payload["generator_version"]).item())
        if source_type != expected_source_type or generator_version != expected_generator_version:
            raise ValueError("prediction artifact source/generator mismatch")
        benchmark_hash = str(np.asarray(payload["benchmark_sha256"]).item()).lower()
        contract_hash = str(np.asarray(payload["contract_sha256"]).item()).lower()
        if len(benchmark_hash) != 64 or len(contract_hash) != 64 or any(char not in "0123456789abcdef" for char in benchmark_hash + contract_hash):
            raise ValueError("prediction artifact hashes are invalid")
        if expected_benchmark_sha256 is not None and benchmark_hash != str(expected_benchmark_sha256).lower():
            raise ValueError("prediction artifact benchmark SHA-256 mismatch")
        if expected_contract_sha256 is not None and contract_hash != str(expected_contract_sha256).lower():
            raise ValueError("prediction artifact contract SHA-256 mismatch")
        feature_order = tuple(str(value) for value in np.asarray(payload["feature_order"]).tolist())
        label_order = tuple(str(value) for value in np.asarray(payload["label_order"]).tolist())
        if feature_order != FEATURE_ORDER or label_order != LABEL_ORDER:
            raise ValueError("prediction artifact feature/label order mismatch")
        digest_value = np.asarray(payload["scenario_id_digest"])
        if digest_value.ndim != 0:
            raise ValueError("prediction artifact scenario_id_digest must be scalar")
        stored_digest = str(digest_value.item()).lower()
        if len(stored_digest) != 64 or any(char not in "0123456789abcdef" for char in stored_digest):
            raise ValueError("prediction artifact scenario_id_digest is invalid")
        arrays = {name: np.asarray(payload[name]) for name in required if name not in {"benchmark_sha256", "contract_sha256", "source_type", "generator_version", "split", "feature_order", "label_order"}}
    raw = np.asarray(arrays["raw_prediction"], dtype=np.float64)
    safe = np.asarray(arrays["safe_prediction"], dtype=np.float64)
    target = np.asarray(arrays["target"], dtype=np.float64)
    features = np.asarray(arrays["features"], dtype=np.float64)
    count = int(raw.shape[0]) if raw.ndim else -1
    if raw.shape != (count, 4, len(LABEL_ORDER)) or safe.shape != raw.shape or target.shape != raw.shape:
        raise ValueError("prediction arrays must have shape [N,4,21]")
    if features.shape != (count, 4, len(FEATURE_ORDER)):
        raise ValueError("prediction features must have shape [N,4,10]")
    if stored_count != count:
        raise ValueError("prediction artifact sample_count does not match arrays")
    if np.asarray(arrays["fallback_mask"]).shape != (count,) or np.asarray(arrays["raw_feasible_mask"]).shape != (count,) or np.asarray(arrays["scenario_ids"]).shape != (count,) or np.asarray(arrays["teacher_objective"]).shape != (count,):
        raise ValueError("prediction audit arrays must have shape [N]")
    if expected_count is not None and count != int(expected_count):
        raise ValueError("prediction artifact sample count mismatch")
    if not np.isfinite(raw).all() or not np.isfinite(safe).all() or not np.isfinite(target).all() or not np.isfinite(features).all():
        raise ValueError("prediction artifact contains non-finite arrays")
    raw_ids = np.asarray(arrays["scenario_ids"])
    if not np.issubdtype(raw_ids.dtype, np.integer):
        raise ValueError("prediction scenario IDs must be stored as integer values")
    ids = np.asarray(raw_ids, dtype=np.int64)
    actual_digest = scenario_id_digest(ids)
    if actual_digest.lower() != stored_digest:
        raise ValueError("prediction scenario_id_digest mismatch")
    if expected_scenario_id_digest is not None and stored_digest != str(expected_scenario_id_digest).lower():
        raise ValueError("prediction scenario IDs/digest do not match expected split")
    if expected_scenario_ids is not None and not np.array_equal(ids, np.asarray(expected_scenario_ids, dtype=np.int64)):
        raise ValueError("prediction scenario IDs do not match expected split")
    if np.unique(ids).size != count:
        raise ValueError("prediction scenario IDs must be unique")
    arrays["scenario_ids"] = ids
    for mask_name in ("fallback_mask", "raw_feasible_mask"):
        mask = np.asarray(arrays[mask_name])
        if mask.dtype == np.bool_:
            arrays[mask_name] = mask.astype(bool, copy=False)
        elif np.issubdtype(mask.dtype, np.number) and np.isfinite(mask).all() and np.isin(mask, [0, 1]).all():
            arrays[mask_name] = mask.astype(bool)
        else:
            raise ValueError(f"prediction {mask_name} must be boolean or strictly binary")
    return arrays


# Aliases.
evaluate_predictions = dispatch_metrics
evaluate_proxy = evaluate_model_on_split
evaluate_proxy_predictions = dispatch_metrics


__all__ = [
    "dispatch_metrics", "measure_proxy_latency", "measure_exact_lp_latency", "evaluate_inference_result",
    "evaluate_gas_prior_ablation", "evaluate_model_on_split", "evaluate_predictions", "evaluate_proxy",
    "evaluate_proxy_predictions", "save_metrics",
    "load_prediction_artifact",
]
