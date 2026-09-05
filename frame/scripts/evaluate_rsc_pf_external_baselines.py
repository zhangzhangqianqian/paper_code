"""Common validation metrics and optimizer-role accounting for external baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.data import benchmark_lp_generation  # noqa: E402
from src.joint_dispatch.evaluation import evaluate_forecast  # noqa: E402
from src.joint_dispatch.external_baseline_data import ExternalBaselineBatch, fit_external_normalization  # noqa: E402
from src.joint_dispatch.external_baseline_training import (  # noqa: E402
    METHODS,
    SEEDS,
    _batches,
    _build,
    _config_paths,
    _decoder_parameters,
    _forbidden_test_path,
    _load_data,
    _split_manifest,
    _METHOD_SAFE,
    load_external_checkpoint,
)
from src.joint_dispatch.pto import PTOForecasts, solve_pto_windows  # noqa: E402
from src.joint_dispatch.contract import DISPATCH_ORDER  # noqa: E402
from src.scheduling.dispatch_lp import DispatchInputs  # noqa: E402
from src.joint_dispatch.formal_v4_recourse import settle_first_step_v4  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _lp_parameters() -> dict[str, float]:
    values = _decoder_parameters()
    values.update({
        "bess_throughput_cost": 1.0e-6,
        "grid_energy_price": 1.0,
        "gas_energy_price": 0.6,
        "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25,
        "carbon_price_default": 0.0,
        "unserved_penalty": 100.0,
    })
    return values


def _predict(method_id: str, model: torch.nn.Module, split, normalization, batch_size: int) -> tuple[np.ndarray, np.ndarray | None]:
    forecasts: list[np.ndarray] = []
    dispatches: list[np.ndarray] = []
    normalized = method_id != "DigitalTwins-Policy"
    model.eval()
    with torch.no_grad():
        for batch in _batches(split, normalization=normalization, batch_size=batch_size, seed=0, shuffle=False, normalized=normalized):
            if method_id == "iTransformer-PTO":
                value = model(batch.load_history, batch.exog_history).cpu().numpy()
                value = value * normalization.load_scale.reshape(1, 1, -1) + normalization.load_mean.reshape(1, 1, -1)
            elif method_id == "DecisionFocused-Online":
                value = model(batch).forecast.detach().cpu().numpy()
                value = value * normalization.load_scale.reshape(1, 1, -1) + normalization.load_mean.reshape(1, 1, -1)
            else:
                output = model(batch)
                value = output.forecast_physical.detach().cpu().numpy()
                dispatches.append(output.dispatch.detach().cpu().numpy())
            forecasts.append(np.asarray(value, dtype=np.float32))
    prediction = np.concatenate(forecasts, axis=0)
    dispatch = np.concatenate(dispatches, axis=0) if dispatches else None
    if not np.isfinite(prediction).all() or (dispatch is not None and not np.isfinite(dispatch).all()):
        raise RuntimeError(f"{method_id} produced non-finite evaluation outputs")
    return prediction, dispatch


def _dispatch_metrics(dispatch: np.ndarray, split, *, lp_calls: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if dispatch.shape != (len(split), 4, len(DISPATCH_ORDER)):
        raise ValueError("dispatch output does not match [N,4,21]")
    context = np.asarray(split.scheduler_context, dtype=np.float64)
    demand = np.asarray(split.forecast_target[..., :3], dtype=np.float64)
    values = np.asarray(dispatch, dtype=np.float64)
    index = {name: DISPATCH_ORDER.index(name) for name in DISPATCH_ORDER}
    served = np.stack((
        values[..., index["grid"]] + values[..., index["pv_use"]] + values[..., index["wt_use"]] + values[..., index["p_chp"]] + values[..., index["p_discharge"]] - values[..., index["p_ec"]] - values[..., index["p_charge"]],
        values[..., index["q_ec"]] + values[..., index["q_ac"]],
        values[..., index["q_chp"]] + values[..., index["q_gb"]] - values[..., index["q_ac_in"]] - values[..., index["q_dump"]],
    ), axis=-1)
    shortage = np.maximum(demand - served, 0.0)
    grid = values[..., index["grid"]]
    gas = values[..., index["g_chp"]] + values[..., index["g_gb"]]
    operating = grid * context[..., 2] + gas * context[..., 3]
    carbon = grid * 0.5 + gas * 0.25
    objective = operating + context[..., 4] * carbon + 100.0 * shortage.sum(axis=-1)
    first_objective = objective[:, 0]
    # Use the same canonical one-step physical settlement as RSC-PF for the
    # comparable metrics. The raw four-hour values above remain as open-loop
    # diagnostics, while the first executed action is projected against the
    # realized first-hour demand/renewables and the recorded initial state.
    settled = settle_first_step_v4(
        torch.as_tensor(values[:, 0, :], dtype=torch.float64),
        torch.as_tensor(np.asarray(split.forecast_target[:, 0, :3], dtype=np.float64)),
        torch.as_tensor(np.asarray(split.renewable_realized[:, 0, :], dtype=np.float64)),
        _lp_parameters(),
        initial_soc=torch.as_tensor(np.asarray(context[:, 0, 5:6], dtype=np.float64)),
        previous_chp=torch.as_tensor(np.asarray(split.previous_chp, dtype=np.float64)),
        grid_price=torch.as_tensor(np.asarray(context[:, 0, 2], dtype=np.float64)),
        gas_price=torch.as_tensor(np.asarray(context[:, 0, 3], dtype=np.float64)),
        carbon_price=torch.as_tensor(np.asarray(context[:, 0, 4], dtype=np.float64)),
    )
    settled_operating = settled.operating_cost.detach().cpu().numpy()
    settled_carbon = settled.physical_carbon.detach().cpu().numpy()
    settled_objective = settled.penalized_objective.detach().cpu().numpy()
    settled_shortage = settled.shortage.detach().cpu().numpy()
    settled_feasible = np.max(settled_shortage, axis=-1) <= 1.0e-8
    oracle = np.asarray(split.oracle_first_step_objective, dtype=np.float64)
    per_window = {
        "operating_cost": operating.sum(axis=1),
        "physical_carbon": carbon.sum(axis=1),
        "penalized_objective": objective.sum(axis=1),
        "operating_cost_first_step": settled_operating,
        "physical_carbon_first_step": settled_carbon,
        "penalized_objective_first_step": settled_objective,
        "first_step_objective": settled_objective,
        "regret_vs_oracle": settled_objective - oracle,
        "shortage": shortage.sum(axis=(1, 2)),
        "shortage_first_step": settled_shortage.sum(axis=-1),
        "feasible": (np.max(shortage, axis=(1, 2)) <= 1.0e-8),
        "feasible_first_step": settled_feasible,
    }
    summary = {
        "operating_cost_mean": float(per_window["operating_cost"].mean()),
        "physical_carbon_mean": float(per_window["physical_carbon"].mean()),
        "penalized_objective_mean": float(per_window["penalized_objective"].mean()),
        "first_step_objective_mean": float(per_window["first_step_objective"].mean()),
        "regret_vs_oracle_mean": float(per_window["regret_vs_oracle"].mean()),
        "shortage_mean": float(per_window["shortage"].mean()),
        "feasibility_rate": float(per_window["feasible"].mean()),
        # These are the comparable rolling-execution statistics: only the
        # first action of each four-hour prediction window is settled.
        "operating_cost_first_step_mean": float(per_window["operating_cost_first_step"].mean()),
        "physical_carbon_first_step_mean": float(per_window["physical_carbon_first_step"].mean()),
        "penalized_objective_first_step_mean": float(per_window["penalized_objective_first_step"].mean()),
        "shortage_first_step_mean": float(per_window["shortage_first_step"].mean()),
        "feasibility_rate_first_step": float(per_window["feasible_first_step"].mean()),
        "exact_lp_calls": int(lp_calls),
        "windows": int(len(split)),
    }
    return summary, per_window


def evaluate_external_validation(method_id: str, seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one method/seed on the shared validation split only."""

    if method_id not in METHODS:
        raise ValueError(f"unknown external method: {method_id}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    paths = _config_paths(config)
    train, validation, normalization = _load_data(paths)
    checkpoint = paths["output_root"] / "validation" / _METHOD_SAFE[method_id] / f"seed_{int(seed)}" / "best_checkpoint.pt"
    receipt_path = checkpoint.parent / "training_receipt.json"
    if _forbidden_test_path(checkpoint) or not checkpoint.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"formal validation checkpoint/receipt is missing: {checkpoint}")
    provenance = load_external_checkpoint(checkpoint, expected_method=method_id, expected_seed=seed)
    model = _build(method_id)
    model.load_state_dict(provenance["model_state_dict"])
    started = time.perf_counter()
    prediction, direct_dispatch = _predict(method_id, model, validation, normalization, int(config.get("evaluation_batch_size", 256)))
    forecast_table = evaluate_forecast(prediction, validation.forecast_target)
    if method_id in {"iTransformer-PTO", "DecisionFocused-Online"}:
        cache = solve_pto_windows(
            PTOForecasts(method_id, prediction, validation.forecast_target), validation, _lp_parameters()
        )
        dispatch = cache.dispatch
        lp_calls = cache.offline_exact_lp_calls
        lp_success_rate = float(cache.success.mean())
    else:
        dispatch = np.asarray(direct_dispatch, dtype=np.float32)
        lp_calls = 0
        lp_success_rate = 1.0
    metrics, per_window = _dispatch_metrics(dispatch, validation, lp_calls=lp_calls)
    elapsed = float(time.perf_counter() - started)
    metrics.update({
        "forecast_mae_mean": float(np.nanmean(forecast_table.mae)),
        "forecast_rmse_mean": float(np.nanmean(forecast_table.rmse)),
        "forecast_wape_mean": float(np.nanmean(forecast_table.wape)),
        "forecast_mae_by_task_horizon": forecast_table.mae.tolist(),
        "forecast_rmse_by_task_horizon": forecast_table.rmse.tolist(),
        "forecast_wape_by_task_horizon": forecast_table.wape.tolist(),
        "lp_success_rate": lp_success_rate,
        "evaluation_seconds": elapsed,
        "evaluation_windows_per_second": float(len(validation) / max(elapsed, 1.0e-12)),
    })
    evaluation_dir = checkpoint.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evaluation_dir / "validation_metrics.npz", **per_window)
    receipt = {
        "schema_version": "rsc-pf-external-evaluation-receipt-v1",
        "method_id": method_id,
        "seed": int(seed),
        "stage": "validation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics": metrics,
        "optimizer_role": "exact optimizer at inference" if method_id == "DecisionFocused-Online" else "none at inference" if method_id == "DigitalTwins-Policy" else "none at inference",
        "exact_lp_calls": int(lp_calls),
        "test_set_accessed": False,
        "split_manifest": _split_manifest(paths),
    }
    (evaluation_dir / "evaluation_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def write_external_validation_manifest(output_root: Path) -> Path:
    """Aggregate complete per-seed receipts without opening the test split."""

    root = Path(output_root)
    entries: list[dict[str, Any]] = []
    for method in METHODS:
        safe = _METHOD_SAFE[method]
        for seed in SEEDS:
            receipt_path = root / "validation" / safe / f"seed_{seed}" / "evaluation" / "evaluation_receipt.json"
            if not receipt_path.is_file():
                raise FileNotFoundError(f"missing evaluation receipt: {receipt_path}")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("test_set_accessed") is not False:
                raise ValueError("evaluation receipt is not test-set-free")
            entries.append(receipt)
    manifest = {
        "schema_version": "rsc-pf-external-validation-manifest-v1",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "entries": entries,
        "test_set_accessed": False,
        "metric_definitions": {"forecast": ["MAE", "RMSE", "WAPE"], "dispatch": ["operating_cost", "physical_carbon", "regret_vs_oracle", "shortage", "feasibility_rate"]},
        "optimizer_roles": {"iTransformer-PTO": "none at inference", "DecisionFocused-Online": "exact optimizer at inference", "DigitalTwins-Policy": "none at inference"},
    }
    destination = root / "external_validation_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def benchmark_external_lp_resources(output_root: Path, split, *, required_solves: int = 500) -> dict[str, Any]:
    """Run the fixed 500-case LP resource gate on validation inputs only."""

    if split.split != "validation":
        raise ValueError("LP resource benchmark is validation-only")
    if len(split) < int(required_solves):
        raise ValueError("validation split is smaller than the LP resource benchmark")
    cases = []
    for index in range(int(required_solves)):
        context = split.scheduler_context[index]
        parameters = _lp_parameters()
        parameters["grid_energy_price"] = context[:, 2]
        parameters["gas_energy_price"] = context[:, 3]
        parameters["carbon_price"] = context[:, 4]
        cases.append(DispatchInputs(
            demand=np.asarray(split.forecast_target[index, :, :3], dtype=np.float64),
            pv_available=np.asarray(context[:, 0], dtype=np.float64),
            wt_available=np.asarray(context[:, 1], dtype=np.float64),
            parameters=parameters,
            initial_soc=float(context[0, 5]),
        ))
    receipt = benchmark_lp_generation(
        cases,
        projected_total_solves=2 * len(SEEDS) * len(split),
        required_solves=int(required_solves),
        max_projected_hours=24.0,
    )
    payload = asdict(receipt) if hasattr(receipt, "__dataclass_fields__") else dict(receipt)
    payload["test_set_accessed"] = False
    payload["validation_windows"] = int(len(split))
    destination = Path(output_root) / "lp_resource_gate.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen external baselines on validation only")
    parser.add_argument("--method", choices=METHODS, required=False)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--config", type=Path, default=FRAME_ROOT / "configs" / "rsc_pf_external_baseline_implementation_v1.json")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    config["evaluation_batch_size"] = int(args.evaluation_batch_size)
    methods = METHODS if args.all_methods else ((args.method,) if args.method else METHODS)
    seeds = SEEDS if args.all_seeds else ((args.seed,) if args.seed else SEEDS)
    receipts = [evaluate_external_validation(method, int(seed), config) for method in methods for seed in seeds]
    if args.all_methods and args.all_seeds:
        paths = _config_paths(config)
        manifest = write_external_validation_manifest(paths["output_root"])
        _train_split, validation_split, _normalization = _load_data(paths)
        resource_gate = benchmark_external_lp_resources(paths["output_root"], validation_split)
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_payload["lp_resource_gate"] = resource_gate
        manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        manifest = None
    print(json.dumps({"evaluated": len(receipts), "manifest": str(manifest) if manifest else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
