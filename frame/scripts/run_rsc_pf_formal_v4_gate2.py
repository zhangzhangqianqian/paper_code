"""Run the frozen three-seed formal-v4 Gate 2 selection check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

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


def _teacher(data: dict[str, np.ndarray], parameters: dict[str, float], cache: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        with np.load(cache, allow_pickle=False) as payload:
            return np.array(payload["dispatch"]), np.array(payload["objective"])
    labels, objectives = [], []
    for index in range(len(data["target_times"])):
        context = dict(parameters)
        context["grid_energy_price"] = data["prices_and_weights"][index, :, 0]
        context["gas_energy_price"] = data["prices_and_weights"][index, :, 1]
        context["carbon_price"] = data["prices_and_weights"][index, :, 2]
        result = solve_dispatch_lp(DispatchInputs(
            demand=data["rigid_demand"][index], pv_available=data["renewable_realized"][index, :, 0],
            wt_available=data["renewable_realized"][index, :, 1], parameters=context,
            initial_soc=float(data["initial_soc"][index, 0]), previous_chp=float(max(data["previous_chp"][index, 0], 0.0)),
        ))
        if not result.success:
            raise RuntimeError(f"Gate2 teacher LP failed at origin {index}: {result.message}")
        labels.append(np.stack([result.values[name] for name in DISPATCH_ORDER], axis=-1)); objectives.append(float(result.objective))
    dispatch, objective = np.asarray(labels, dtype=np.float64), np.asarray(objectives, dtype=np.float64)
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, dispatch=dispatch, objective=objective)
    return dispatch, objective


def _batch(data: dict[str, np.ndarray], indices: np.ndarray, stats: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    norm = lambda name, mean, scale: (data[name][indices] - stats[mean]) / np.maximum(stats[scale], 1.0e-6)
    context = np.concatenate((data["renewable_forecast"][indices], data["prices_and_weights"][indices], np.repeat(data["initial_soc"][indices, None, :], 4, axis=1)), axis=-1)
    return {"load_history": torch.as_tensor(norm("load_history", "normalization_load_mean", "normalization_load_scale"), dtype=torch.float32), "exog_history": torch.as_tensor(norm("exog_history", "normalization_exog_mean", "normalization_exog_scale"), dtype=torch.float32), "device_history": torch.as_tensor(norm("device_history", "normalization_device_mean", "normalization_device_scale"), dtype=torch.float32), "activity_history": torch.as_tensor(data["activity_history"][indices], dtype=torch.float32), "scheduler_context": torch.as_tensor(context, dtype=torch.float32), "previous_chp": torch.as_tensor(data["previous_chp"][indices], dtype=torch.float64), "target_normalized": torch.as_tensor(norm("forecast_target", "normalization_load_mean", "normalization_load_scale"), dtype=torch.float32), "target_physical": torch.as_tensor(data["forecast_target"][indices], dtype=torch.float64), "realized_renewables": torch.as_tensor(data["renewable_realized"][indices], dtype=torch.float64), "initial_soc": torch.as_tensor(data["initial_soc"][indices], dtype=torch.float64), "teacher_dispatch": torch.as_tensor(data["teacher_dispatch"][indices], dtype=torch.float64), "rigid_demand": torch.as_tensor(data["rigid_demand"][indices], dtype=torch.float64)}


def _configure(model: RSCPFModel, train: dict[str, np.ndarray], parameters: dict[str, float]) -> None:
    model.core.task_mean.copy_(torch.as_tensor(train["normalization_load_mean"], dtype=torch.float32)); model.core.task_scale.copy_(torch.as_tensor(train["normalization_load_scale"], dtype=torch.float32))
    feature = np.concatenate((train["forecast_target"], train["renewable_forecast"], train["prices_and_weights"], np.repeat(train["initial_soc"][:, None, :], 4, axis=1)), axis=-1).reshape(-1, 10)
    model.core.physical_feature_mean.copy_(torch.as_tensor(feature.mean(0), dtype=torch.float32)); model.core.physical_feature_scale.copy_(torch.as_tensor(np.where(feature.std(0) < 1.0e-6, 1.0, feature.std(0)), dtype=torch.float32))


def _train(model: RSCPFModel, data: dict[str, np.ndarray], indices: np.ndarray, stats: dict[str, np.ndarray], parameters: dict[str, float], teacher: np.ndarray, c_ref: float, epochs: int, mode: str, lr: float) -> None:
    data = dict(data); data["teacher_dispatch"] = teacher
    batch_size = 512
    for epoch in range(epochs):
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]; batch = _batch(data, part, stats)
            train_stage_p(model, batch, parameters, seed=2026)
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]; batch = _batch(data, part, stats)
            train_stage_s(model, batch, parameters, seed=2026)
        for start in range(0, len(indices), batch_size):
            part = indices[start:start + batch_size]; batch = _batch(data, part, stats)
            train_stage_j(model, batch, parameters, c_ref=c_ref, mode=mode, seed=2026, epoch=epoch, forecaster_lr=lr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-id", default="formal_v4_1_gate2"); parser.add_argument("--gate1", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--benchmark-path", type=Path, default=None); parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    gate1 = json.loads(args.gate1.read_text(encoding="utf-8"));
    if gate1.get("authorized_gate2") is not True: raise PermissionError("Gate1 did not authorize Gate2")
    if gate1.get("paper_result") is not False: raise PermissionError("Gate1 receipt is not calibration-only")
    train, selection = _load(args.data_root / "train.npz"), _load(args.data_root / "selection.npz")
    if any("2020" in str(value) for value in selection["target_times"]): raise PermissionError("Gate2 selection archive contains 2020")
    benchmark_path = args.benchmark_path
    if benchmark_path is None:
        benchmark_path = Path(str(gate1.get("benchmark_path", "")))
    if not benchmark_path or not benchmark_path.is_file():
        raise FileNotFoundError(f"Gate2 frozen benchmark not found: {benchmark_path}")
    benchmark_path = benchmark_path.resolve()
    expected_benchmark_hash = str(gate1.get("benchmark_sha256", ""))
    if expected_benchmark_hash and _sha256(benchmark_path) != expected_benchmark_hash:
        raise PermissionError("Gate2 benchmark does not match the Gate1 calibration receipt")
    parameters = dict(yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))["values"])
    train_idx, selection_idx = np.arange(len(train["target_times"])), np.arange(len(selection["target_times"]))
    train_teacher, train_obj = _teacher(train, parameters, args.data_root / "gate2_train_teacher.npz"); selection_teacher, _ = _teacher(selection, parameters, args.data_root / "gate2_selection_teacher.npz")
    train["teacher_dispatch"], selection["teacher_dispatch"] = train_teacher, selection_teacher
    stats = train; scale = fit_training_objective_scale(train_obj, source_hash=_sha256(args.data_root / "train.npz"))
    results = []
    for seed in (2026, 2027, 2028):
        for method in ("RSC-PF", "Decoupled-RSC-PF"):
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = RSCPFModel(decoder_parameters=parameters, dropout=0.05); _configure(model, train, parameters)
            _train(model, train, train_idx, stats, parameters, train_teacher, scale.c_ref, args.epochs, "joint" if method == "RSC-PF" else "decoupled", float(gate1["selected_candidate"]["forecaster_lr"]))
            model.eval(); preds, dispatches = [], []
            with torch.no_grad():
                for start in range(0, len(selection_idx), 512):
                    batch = _batch(selection, selection_idx[start:start + 512], stats); out = model(**{key: batch[key] for key in ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")}); preds.append(out.forecast_physical.cpu().numpy()); dispatches.append(out.dispatch.cpu().numpy())
            prediction, dispatch = np.concatenate(preds), np.concatenate(dispatches); fm = evaluate_forecast_table(prediction, selection["forecast_target"]); dm, _ = evaluate_dispatch_table(dispatch, selection["rigid_demand"], grid_price=selection["prices_and_weights"][..., 0], gas_price=selection["prices_and_weights"][..., 1], carbon_price=selection["prices_and_weights"][..., 2])
            results.append({"seed": seed, "method": method, "forecast": {"task_wape": fm.task_wape.tolist(), "rigid_macro_wape": fm.rigid_macro_wape, "gas_wape": fm.gas_wape}, "dispatch": dm.__dict__, "test_set_accessed": False})
    rsc = [x for x in results if x["method"] == "RSC-PF"]; dec = [x for x in results if x["method"] == "Decoupled-RSC-PF"]
    mean_diff = float(np.mean([a["dispatch"]["penalized_objective"] - b["dispatch"]["penalized_objective"] for a, b in zip(rsc, dec)])); favors = int(sum(a["dispatch"]["penalized_objective"] < b["dispatch"]["penalized_objective"] for a, b in zip(rsc, dec)))
    authorized = bool(len(results) == 6 and mean_diff < 0.0 and favors >= 2 and all(np.isfinite(x["forecast"]["rigid_macro_wape"]) for x in results))
    root = FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_1" / args.run_id / "gate2"; root.mkdir(parents=True, exist_ok=False)
    payload = {"schema_version": "formal-v4.1-gate2-v1", "gate1_receipt": str(args.gate1), "benchmark_path": str(benchmark_path), "benchmark_sha256": _sha256(benchmark_path), "data_root": str(args.data_root), "seeds": [2026, 2027, 2028], "epochs": args.epochs, "results": results, "mean_rsc_minus_decoupled_objective": mean_diff, "rsc_favor_count": favors, "authorized_gate3": authorized, "paper_result": False, "test_set_accessed": False}
    (root / "GATE2_DECISION.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"); print(json.dumps({"status": "pass" if authorized else "fail", "authorized_gate3": authorized, "output": str(root / "GATE2_DECISION.json")}, ensure_ascii=False)); return 0 if authorized else 2


if __name__ == "__main__": raise SystemExit(main())
