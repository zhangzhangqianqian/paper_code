"""Run the fail-closed Differentiable-LP evidence gate for formal-v4.

This is a small, training-free check.  It uses only fixed training-period
windows, the frozen benchmark parameters, and the isolated NumPy/PyTorch/
CVXPYlayers environment.  It never reads 2019 selection outputs or 2020.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.common_evaluation import evaluate_four_hour_plan
from src.joint_dispatch.formal_v4_diffopt import (
    DifferentiableIESLayer,
    DifferentiableLPGateReceipt,
    DifferentiableLPForecasterAdapter,
)
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from src.joint_dispatch.formal_v4_gate0_evidence import write_immutable_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameters(benchmark_path: Path) -> dict[str, float]:
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    values = dict(benchmark["values"])
    return {str(key): float(value) for key, value in values.items() if isinstance(value, (int, float))}


def _resolve_inside(run_root: Path, path: Path, name: str) -> Path:
    root = run_root.resolve()
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} must be inside the supplied run root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _fixed_training_windows(archive_path: Path, parameters: Mapping[str, float], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load deterministic train-only windows from a materialized NPZ archive."""

    with np.load(archive_path, allow_pickle=False) as payload:
        split = np.asarray(payload["split"]).astype(str)
        if split.ndim == 0:
            split = np.repeat(split.reshape(1), payload["rigid_demand"].shape[0])
        if split.shape[0] < count or not np.all(split[:count] == "train"):
            raise ValueError("DiffLP gate requires at least window-count train-only rows")
        demand = np.asarray(payload["rigid_demand"][:count], dtype=np.float64)
        renew = np.asarray(payload["renewable_forecast"][:count], dtype=np.float64)
        prices3 = np.asarray(payload["prices_and_weights"][:count], dtype=np.float64)
        initial_soc = np.asarray(payload["initial_soc"][:count], dtype=np.float64).reshape(count)
        previous_chp = np.asarray(payload["previous_chp"][:count], dtype=np.float64).reshape(count)
        target_times = np.asarray(payload["target_times"][:count])
    if demand.shape != (count, 4, 3) or renew.shape != (count, 4, 2) or prices3.shape != (count, 4, 3):
        raise ValueError("materialized train archive has an unexpected DiffLP shape")
    carbon_weight = prices3[..., 2]
    prices = np.stack((prices3[..., 0], prices3[..., 1], carbon_weight * float(parameters.get("grid_emission_factor", 0.0)), carbon_weight * float(parameters.get("gas_emission_factor", 0.0))), axis=-1)
    if not np.isfinite(demand).all() or not np.isfinite(renew).all() or not np.isfinite(prices).all():
        raise ValueError("materialized train archive contains non-finite values")
    return demand, renew, prices, initial_soc, previous_chp, np.arange(count, dtype=np.int64)


def _residual(ev: Any) -> float:
    return max(
        float(np.max(np.abs(ev.balance_residual))),
        float(np.max(np.abs(ev.conversion_residual))),
        float(np.max(np.abs(ev.soc_residual))),
        float(np.max(np.abs(ev.renewable_residual))),
        float(ev.capacity_violation), float(ev.ramp_violation),
        float(ev.simultaneous_charge_discharge),
    )


def _gradient_probe(layer: DifferentiableIESLayer, parameters: Mapping[str, float], renew: np.ndarray, prices: np.ndarray, soc: float, chp: float) -> float:
    class TinyForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3, dtype=torch.float64)

        def forward(self, history: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.linear(history)) + 1.0

    forecaster = TinyForecaster()
    adapter = DifferentiableLPForecasterAdapter(forecaster, layer)
    history = torch.ones((1, 4, 3), dtype=torch.float64)
    output = adapter(
        history,
        torch.as_tensor(renew[None], dtype=torch.float64),
        torch.as_tensor(prices[None], dtype=torch.float64),
        torch.as_tensor([soc], dtype=torch.float64),
        torch.as_tensor([chp], dtype=torch.float64),
    )
    gradient = torch.autograd.grad(output.square().mean(), tuple(forecaster.parameters()), allow_unused=True)
    return float(sum(float(g.abs().sum()) for g in gradient if g is not None))


def _require_isolated_environment(lock_path: Path) -> Mapping[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = Path(str(lock.get("python_executable", ""))).resolve()
    actual = Path(sys.executable).resolve()
    if not expected or actual != expected:
        raise RuntimeError(f"DiffLP gate must run in isolated environment {expected}; current interpreter is {actual}")
    if lock.get("native_layer_probe", {}).get("eligible_for_gate0") is not True:
        raise RuntimeError("isolated environment has no eligible native cvxpylayers probe")
    return lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--window-count", type=int, default=100)
    parser.add_argument("--lock-path", type=Path, default=FRAME_ROOT / "requirements" / "formal_v4_diffopt.lock")
    args = parser.parse_args(argv)
    if args.window_count < 100:
        raise ValueError("window-count must be at least 100")
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    benchmark_path = _resolve_inside(run_root, args.benchmark_path, "benchmark-path")
    train_archive = _resolve_inside(run_root, args.train_archive, "train-archive")
    output_path = (args.output_receipt if args.output_receipt.is_absolute() else run_root / args.output_receipt).resolve()
    try:
        output_path.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("output-receipt must be inside the supplied run root") from exc
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite DiffLP gate receipt: {output_path}")
    lock = _require_isolated_environment(args.lock_path.resolve())
    parameters = _parameters(benchmark_path)
    demand, renew, prices, initial_soc, previous_chp, origins = _fixed_training_windows(train_archive, parameters, args.window_count)
    source_lock = args.lock_path.resolve()
    native_probe_passed = bool(lock.get("native_layer_probe", {}).get("eligible_for_gate0") is True)
    started = time.perf_counter()
    dpp_passed = True
    finite_solves = 0
    objective_gaps: list[float] = []
    residuals: list[float] = []
    reason = "pass"
    try:
        layer = DifferentiableIESLayer(parameters, tie_break_throughput_cost=1.0e-5)
        for i in range(args.window_count):
            output = layer(
                torch.as_tensor(demand[i], dtype=torch.float64),
                torch.as_tensor(renew[i], dtype=torch.float64),
                torch.as_tensor(prices[i], dtype=torch.float64),
                torch.as_tensor(initial_soc[i], dtype=torch.float64),
                torch.as_tensor(previous_chp[i], dtype=torch.float64),
            )[0].detach().numpy()
            reference = solve_dispatch_lp(DispatchInputs(
                demand=demand[i], pv_available=renew[i, :, 0], wt_available=renew[i, :, 1],
                parameters=parameters, initial_soc=float(initial_soc[i]), previous_chp=float(previous_chp[i]),
            ))
            evaluation = evaluate_four_hour_plan(
                dispatch=output, demand=demand[i], renewables=renew[i], initial_soc=float(initial_soc[i]),
                previous_chp=float(previous_chp[i]), parameters=parameters, tolerance=1.0e-6,
            )
            if not reference.success or not np.isfinite(output).all() or not np.isfinite(evaluation.penalized_objective):
                raise ValueError(f"non-finite or failed solve at fixed origin {int(origins[i])}")
            finite_solves += 1
            objective_gaps.append(abs(evaluation.penalized_objective - reference.objective) / max(abs(reference.objective), 1.0))
            residuals.append(_residual(evaluation))
        gradient_norm = _gradient_probe(layer, parameters, renew[0], prices[0], float(initial_soc[0]), float(previous_chp[0]))
    except Exception as exc:
        dpp_passed = False
        gradient_norm = 0.0
        reason = f"exception: {exc}"
    elapsed = time.perf_counter() - started
    parity_passed = bool(objective_gaps) and max(objective_gaps) <= 1.0e-4
    physical_residual_passed = bool(residuals) and max(residuals) <= 1.0e-6
    gradient_passed = np.isfinite(gradient_norm) and gradient_norm > 0.0
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_margin = float(memory.available / max(memory.total, 1))
    except Exception:
        memory_margin = 0.0
    # The complete formal run is conservatively projected as 10,000 windows
    # with a 2x timing margin; no evaluation data is read for this estimate.
    projected_hours = (elapsed / max(args.window_count, 1)) * 10000.0 * 2.0 / 3600.0
    eligible = all((
        dpp_passed, finite_solves >= args.window_count, args.window_count >= 100,
        parity_passed, physical_residual_passed, gradient_passed,
        native_probe_passed, memory_margin >= 0.20, projected_hours <= 24.0,
    ))
    if not eligible and reason == "pass":
        reason = "mandatory Differentiable-LP evidence did not pass"
    receipt = DifferentiableLPGateReceipt(
        eligible=eligible, dpp_passed=dpp_passed, finite_solves=finite_solves,
        parity_passed=parity_passed, physical_residual_passed=physical_residual_passed,
        gradient_passed=gradient_passed, native_probe_passed=native_probe_passed,
        memory_margin_fraction=memory_margin, projected_p95_hours=projected_hours,
        source_environment=str(lock.get("python_executable", "")), reason=reason,
    )
    receipt.validate()
    payload = receipt.to_dict()
    payload.update({
        "schema_version": "formal-v4.1-differentiable-lp-gate-v1",
        "backend": {"solve_method": "SCS", "eps": 1.0e-10, "max_iters": 100000, "dtype": "float64", "tie_break_throughput_cost": 1.0e-5},
        "windows": args.window_count, "training_years": [2015, 2016, 2017, 2018],
        "fixed_origin_indices_sha256": hashlib.sha256(origins.tobytes()).hexdigest(),
        "max_relative_objective_gap": float(max(objective_gaps, default=float("inf"))),
        "p95_relative_objective_gap": float(np.quantile(objective_gaps, 0.95)) if objective_gaps else float("inf"),
        "max_physical_residual": float(max(residuals, default=float("inf"))),
        "p95_physical_residual": float(np.quantile(residuals, 0.95)) if residuals else float("inf"),
        "gradient_norm": gradient_norm, "elapsed_seconds": elapsed,
        "lock_sha256": _sha256(source_lock), "benchmark_path": benchmark_path.relative_to(run_root).as_posix(),
        "train_archive_path": train_archive.relative_to(run_root).as_posix(),
        "benchmark_sha256": _sha256(benchmark_path), "train_archive_sha256": _sha256(train_archive),
        "test_set_accessed": False,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_immutable_json(output_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
