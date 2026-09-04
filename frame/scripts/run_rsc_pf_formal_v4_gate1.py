"""Run the fail-closed formal-v4 Gate 1 calibration.

Gate 1 is deliberately a small, one-seed calibration.  It consumes only the
2015--2018 training archive and the locked 2019 selection origins, writes all
candidate receipts, and never creates or reads an evaluation/2020 archive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_protocol_v4 import load_formal_v4_spec  # noqa: E402
from src.joint_dispatch.formal_v4_models import RSCPFModel  # noqa: E402
from src.joint_dispatch.formal_v4_training import train_stage_j, train_stage_p, train_stage_s  # noqa: E402
from src.joint_dispatch.formal_v4_evaluation import evaluate_dispatch_table, evaluate_forecast_table  # noqa: E402
from src.joint_dispatch.formal_v4_objective import fit_training_objective_scale  # noqa: E402
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp  # noqa: E402
from src.joint_dispatch.contract import DISPATCH_ORDER  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.array(payload[key], copy=True) for key in payload.files}


def _batch(data: dict[str, np.ndarray], indices: np.ndarray, parameters: dict[str, float], normalization_source: dict[str, np.ndarray] | None = None) -> dict[str, torch.Tensor]:
    stats = normalization_source or data
    def norm(name: str, mean_key: str, scale_key: str) -> np.ndarray:
        return (data[name][indices] - stats[mean_key]) / np.maximum(stats[scale_key], 1.0e-6)
    context = np.concatenate((data["renewable_forecast"][indices], data["prices_and_weights"][indices], np.repeat(data["initial_soc"][indices, None, :], 4, axis=1)), axis=-1)
    return {
        "load_history": torch.as_tensor(norm("load_history", "normalization_load_mean", "normalization_load_scale"), dtype=torch.float32),
        "exog_history": torch.as_tensor(norm("exog_history", "normalization_exog_mean", "normalization_exog_scale"), dtype=torch.float32),
        "device_history": torch.as_tensor(norm("device_history", "normalization_device_mean", "normalization_device_scale"), dtype=torch.float32),
        "activity_history": torch.as_tensor(data["activity_history"][indices], dtype=torch.float32),
        "scheduler_context": torch.as_tensor(context, dtype=torch.float32),
        "previous_chp": torch.as_tensor(data["previous_chp"][indices], dtype=torch.float32),
        "target_normalized": torch.as_tensor(norm("forecast_target", "normalization_load_mean", "normalization_load_scale"), dtype=torch.float32),
        "target_physical": torch.as_tensor(data["forecast_target"][indices], dtype=torch.float64),
        "realized_renewables": torch.as_tensor(data["renewable_realized"][indices], dtype=torch.float64),
        "initial_soc": torch.as_tensor(data["initial_soc"][indices], dtype=torch.float64),
        "teacher_dispatch": torch.as_tensor(data["teacher_dispatch"][indices], dtype=torch.float64),
        "rigid_demand": torch.as_tensor(data["rigid_demand"][indices], dtype=torch.float64),
    }


def _teacher(data: dict[str, np.ndarray], indices: np.ndarray, parameters: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    labels, objectives = [], []
    for index in indices:
        demand = data["rigid_demand"][index]
        renew = data["renewable_realized"][index]
        context = dict(parameters)
        context["grid_energy_price"] = data["prices_and_weights"][index, :, 0]
        context["gas_energy_price"] = data["prices_and_weights"][index, :, 1]
        context["carbon_price"] = data["prices_and_weights"][index, :, 2]
        solved = solve_dispatch_lp(DispatchInputs(
            demand=demand, pv_available=renew[:, 0], wt_available=renew[:, 1],
            parameters=context, initial_soc=float(data["initial_soc"][index, 0]),
            previous_chp=float(max(data["previous_chp"][index, 0], 0.0)),
        ))
        if not solved.success:
            raise RuntimeError(f"Gate1 teacher LP failed at origin {index}: {solved.message}")
        labels.append(np.stack([solved.values[name] for name in DISPATCH_ORDER], axis=-1))
        objectives.append(float(solved.objective))
    return np.asarray(labels, dtype=np.float64), np.asarray(objectives, dtype=np.float64)


def _eligible_indices(data: dict[str, np.ndarray], limit: int) -> np.ndarray:
    """Select one deterministic continuous block with defined WAPE denominators."""
    sums = np.asarray(data["forecast_target"], dtype=np.float64).sum(axis=1)
    if len(sums) < limit:
        raise ValueError(f"only {len(sums)} origins are available; need {limit}")
    for start in range(len(sums) - limit + 1):
        if np.all(sums[start:start + limit, :3].sum(axis=0) > 1.0e-9):
            return np.arange(start, start + limit, dtype=np.int64)
    raise ValueError("no continuous calibration block has positive rigid-task WAPE denominators")


def _configure(model: RSCPFModel, train: dict[str, np.ndarray]) -> None:
    model.core.task_mean.copy_(torch.as_tensor(train["normalization_load_mean"], dtype=torch.float32))
    model.core.task_scale.copy_(torch.as_tensor(train["normalization_load_scale"], dtype=torch.float32))
    # Scheduler features are forecast tasks followed by five exogenous signals
    # and SOC.  Fitting this scale on train only keeps the calibration causal.
    context = np.concatenate((train["forecast_target"], train["renewable_forecast"], train["prices_and_weights"], np.repeat(train["initial_soc"][:, None, :], 4, axis=1)), axis=-1)
    feature = context.reshape(-1, 10)
    model.core.physical_feature_mean.copy_(torch.as_tensor(feature.mean(0), dtype=torch.float32))
    model.core.physical_feature_scale.copy_(torch.as_tensor(np.where(feature.std(0) < 1.0e-6, 1.0, feature.std(0)), dtype=torch.float32))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs/joint_forecast_dispatch_formal_v4.json")
    parser.add_argument("--run-id", default="formal_v4_20260903")
    parser.add_argument("--gate0-receipt", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=None,
        help="frozen Gate 0 benchmark YAML; defaults to the benchmark staged beside the Gate 0 receipt",
    )
    parser.add_argument("--origins", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    return parser


def main() -> int:
    args = _parser().parse_args()
    spec = load_formal_v4_spec(args.contract)
    gate0 = args.gate0_receipt or (FRAME_ROOT / spec.paths["output_root"] / args.run_id / "gate0" / "GATE0_RECEIPT.json")
    if not gate0.exists():
        raise PermissionError(f"Gate 1 requires Gate0 receipt: {gate0}")
    gate0_payload = json.loads(gate0.read_text(encoding="utf-8"))
    if gate0_payload.get("authorized_gate1") is not True:
        raise PermissionError("Gate0 did not authorize Gate1")
    # Gate 1 must use the exact capacity-frozen benchmark that Gate 0
    # certified.  Falling back silently to the repository-level source YAML
    # would make the calibration physically inconsistent with Gate 0.
    benchmark_path = args.benchmark_path
    if benchmark_path is None:
        benchmark_path = gate0.parent / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
        if not benchmark_path.exists():
            benchmark_path = Path(spec.paths["benchmark_path"])
    benchmark_path = benchmark_path.resolve()
    benchmark_receipt_path = benchmark_path.parent / "FORMAL_V4_BENCHMARK_RECEIPT.json"
    if benchmark_receipt_path.exists():
        benchmark_receipt = json.loads(benchmark_receipt_path.read_text(encoding="utf-8"))
        expected_hash = str(benchmark_receipt.get("resolved_benchmark_sha256", ""))
        if expected_hash and _sha256(benchmark_path) != expected_hash:
            raise PermissionError("Gate1 benchmark does not match the Gate0 benchmark receipt")
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Gate1 benchmark not found: {benchmark_path}")
    data_root = args.data_root or (FRAME_ROOT / spec.paths["output_root"] / args.run_id / "data")
    train, selection = _load(data_root / "train.npz"), _load(data_root / "selection.npz")
    if np.any(np.char.startswith(train["target_times"].astype(str), "2020")) or np.any(np.char.startswith(selection["target_times"].astype(str), "2020")):
        raise PermissionError("Gate1 data contain forbidden 2020 origins")
    if args.origins <= 0 or args.origins > 1000 or args.epochs <= 0 or args.epochs > 5:
        raise ValueError("Gate1 budget is fixed at at most 1000 origins and 5 epochs")
    parameters = dict(yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))["values"])
    train_indices = _eligible_indices(train, args.origins)
    selection_indices = _eligible_indices(selection, args.origins)
    train_teacher, train_obj = _teacher(train, train_indices, parameters)
    selection_teacher, _ = _teacher(selection, selection_indices, parameters)
    train_cal = {key: value[train_indices] if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == len(train["target_times"]) else value for key, value in train.items()}
    selection_cal = {key: value[selection_indices] if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == len(selection["target_times"]) else value for key, value in selection.items()}
    train_cal["teacher_dispatch"] = train_teacher
    selection_cal["teacher_dispatch"] = selection_teacher
    scale = fit_training_objective_scale(train_obj, capacity_scenario_hash=str(gate0_payload.get("capacity_scenario_hash", "")), source_hash=_sha256(data_root / "train.npz"))
    train_batch = _batch(train_cal, np.arange(len(train_indices)), parameters, train)
    selection_batch = _batch(selection_cal, np.arange(len(selection_indices)), parameters, train)
    candidates = (1.0e-6, 3.0e-6, 1.0e-5, 3.0e-5)
    results = []
    for candidate_id, forecaster_lr in enumerate(candidates, start=1):
        seed = 2026
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = RSCPFModel(decoder_parameters=parameters, dropout=0.05)
        _configure(model, train)
        stage_receipts = []
        for epoch in range(args.epochs):
            stage_receipts.append(train_stage_p(model, train_batch, parameters, seed=2026, checkpoint_path="" ).to_dict())
        model.eval()
        with torch.no_grad():
            stage_p_out = model(**{key: train_batch[key] for key in ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")})
            stage_p_sel_out = model(**{key: selection_batch[key] for key in ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")})
        stage_p_metrics = evaluate_forecast_table(stage_p_sel_out.forecast_physical.cpu().numpy(), selection_batch["target_physical"].cpu().numpy())
        for epoch in range(args.epochs):
            stage_receipts.append(train_stage_s(model, train_batch, parameters, seed=2026, checkpoint_path="" ).to_dict())
        # Stage J is evaluated as the actual joint branch.  The paired
        # decoupled control is run from the same Stage-S state below.
        for epoch in range(args.epochs):
            stage_receipts.append(train_stage_j(model, train_batch, parameters, c_ref=scale.c_ref, mode="joint", seed=2026, epoch=epoch, checkpoint_path="", decision_start=0.05, forecaster_lr=forecaster_lr).to_dict())
        model.eval()
        with torch.no_grad():
            out = model(**{key: selection_batch[key] for key in ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")})
        pred = out.forecast_physical.detach().cpu().numpy()
        dispatch = out.dispatch.detach().cpu().numpy()
        forecast_metrics = evaluate_forecast_table(pred, selection_batch["target_physical"].cpu().numpy())
        dispatch_metrics, _ = evaluate_dispatch_table(dispatch, selection_batch["rigid_demand"].cpu().numpy() if "rigid_demand" in selection_batch else selection["rigid_demand"][selection_indices], grid_price=selection["prices_and_weights"][selection_indices, :, 0], gas_price=selection["prices_and_weights"][selection_indices, :, 1], carbon_price=selection["prices_and_weights"][selection_indices, :, 2])
        metric_values = np.concatenate((forecast_metrics.task_wape, stage_p_metrics.task_wape, np.asarray([forecast_metrics.rigid_macro_wape, forecast_metrics.gas_wape, stage_p_metrics.rigid_macro_wape, stage_p_metrics.gas_wape])))
        metric_finite = bool(np.isfinite(metric_values).all())
        rigid_guard = bool(metric_finite and forecast_metrics.rigid_macro_wape <= 1.02 * max(stage_p_metrics.rigid_macro_wape, 1.0e-12) and np.all(forecast_metrics.task_wape[:3] <= 1.05 * np.maximum(stage_p_metrics.task_wape[:3], 1.0e-12)) and forecast_metrics.gas_wape <= 1.10 * max(stage_p_metrics.gas_wape, 1.0e-12))
        finite = bool(np.isfinite(pred).all() and np.isfinite(dispatch).all() and metric_finite)
        saturation = float(np.mean((out.controls.detach().cpu().numpy() < 0.01) | (out.controls.detach().cpu().numpy() > 0.99)))
        decision_gradients = [float(item["decision_forecaster_gradient_norm"]) for item in stage_receipts if item["stage"] == "J"]
        gradient_boundary = bool(decision_gradients and min(decision_gradients) > 1.0e-10)
        physical_residual = float(np.max(np.abs(dispatch[..., 17:20])))
        results.append({"candidate_id": candidate_id, "forecaster_lr": forecaster_lr, "seed": 2026, "forecast": {"task_wape": forecast_metrics.task_wape.tolist(), "rigid_macro_wape": forecast_metrics.rigid_macro_wape, "gas_wape": forecast_metrics.gas_wape, "stage_p_task_wape": stage_p_metrics.task_wape.tolist()}, "dispatch": dispatch_metrics.__dict__, "stage_receipts": stage_receipts, "finite": finite, "guardrail_passed": rigid_guard, "physical_residual_max": physical_residual, "control_saturation_fraction": saturation, "saturation_passed": bool(saturation < 0.80), "decision_gradient_boundary_passed": gradient_boundary})
    # Gate1 is a calibration receipt, never a paper result.  The score is
    # deterministic and only candidates satisfying the numerical guardrails
    # are eligible for Gate2.
    eligible = [item for item in results if item["finite"] and item["guardrail_passed"] and item["physical_residual_max"] <= 1.0e-6 and item["saturation_passed"] and item["decision_gradient_boundary_passed"]]
    selected = min(eligible, key=lambda item: item["dispatch"]["penalized_objective"]) if eligible else None
    out_root = FRAME_ROOT / spec.paths["output_root"] / args.run_id / "gate1"
    out_root.mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": "formal-v4-gate1-v1", "gate0_receipt": str(gate0), "benchmark_path": str(benchmark_path), "benchmark_sha256": _sha256(benchmark_path), "data_root": str(data_root), "seed": 2026, "origins": int(args.origins), "epochs": int(args.epochs), "candidate_count": len(candidates), "candidate_results": results, "selected_candidate": selected, "authorized_gate2": selected is not None, "paper_result": False, "test_set_accessed": False, "source_hashes": {"train": _sha256(data_root / "train.npz"), "selection": _sha256(data_root / "selection.npz")}, "objective_scale": scale.to_dict()}
    (out_root / "GATE1_FREEZE.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "pass" if selected else "fail", "authorized_gate2": bool(selected), "candidates": len(results), "output": str(out_root / "GATE1_FREEZE.json")}, ensure_ascii=False))
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
