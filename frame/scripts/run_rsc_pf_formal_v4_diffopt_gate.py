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
from src.joint_dispatch.formal_protocol_v4 import load_formal_v4_spec
from src.joint_dispatch.formal_v4_diffopt import (
    DifferentiableIESLayer,
    DifferentiableLPGateReceipt,
    DifferentiableLPForecasterAdapter,
)
from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from src.scheduling.renewables import pv_available, wt_available


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parameters(spec: Any) -> dict[str, float]:
    benchmark = yaml.safe_load(Path(spec.paths["benchmark_path"]).read_text(encoding="utf-8"))
    values = dict(benchmark["values"])
    with Path(spec.paths["parameter_ledger_path"]).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("parameter_id") and row.get("value") not in (None, ""):
                # The benchmark stores resolved capacities; ledger capacity
                # entries are derivation ratios and must not overwrite them.
                values.setdefault(str(row["parameter_id"]), float(row["value"]))
    return {str(key): float(value) for key, value in values.items() if isinstance(value, (int, float))}


def _fixed_training_windows(spec: Any, parameters: Mapping[str, float], count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw, _ = read_kitakyushu_canonical(Path(r"D:\Paper\Kitakyushu dataset"), years=list(spec.train_years))
    frame, _ = clean_kitakyushu_dataframe(raw)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    loads = frame[["electricity", "cooling", "heating"]].to_numpy(dtype=np.float64)
    profile = {
        "pv_rated_capacity": float(parameters.get("pv_capacity", 1.0)),
        "pv_reference_irradiance": float(parameters.get("pv_reference_irradiance", 1000.0)),
        "pv_conversion_efficiency": float(parameters.get("pv_conversion_efficiency", 0.2)),
        "pv_reference_temperature": float(parameters.get("pv_reference_temperature", 25.0)),
        "pv_temperature_coefficient": float(parameters.get("pv_temperature_coefficient", 0.0)),
        "wt_rated_capacity": float(parameters.get("wt_capacity", 1.0)),
        "wt_cut_in_speed": float(parameters.get("wt_cut_in_speed", 3.0)),
        "wt_rated_speed": float(parameters.get("wt_rated_speed", 12.0)),
        "wt_cut_out_speed": float(parameters.get("wt_cut_out_speed", 25.0)),
    }
    renewable = np.column_stack((pv_available(frame, profile), wt_available(frame, profile)))
    timestamps = np.asarray(frame["timestamp"], dtype="datetime64[ns]")
    origins: list[int] = []
    for origin in range(24, len(frame) - 3):
        if np.all(np.diff(timestamps[origin - 24:origin + 4]) == np.timedelta64(1, "h")):
            origins.append(origin)
            if len(origins) >= count:
                break
    if len(origins) < count:
        raise ValueError(f"only {len(origins)} fixed training windows available; need {count}")
    ix = np.asarray(origins, dtype=int)
    demand = np.stack([loads[i:i + 4] for i in ix])
    renew = np.stack([renewable[i:i + 4] for i in ix])
    grid = float(parameters.get("grid_energy_price", 1.0))
    gas = float(parameters.get("gas_energy_price", 0.6))
    carbon = float(parameters.get("carbon_price_default", 0.0))
    grid_ef = float(parameters.get("grid_emission_factor", 0.0))
    gas_ef = float(parameters.get("gas_emission_factor", 0.0))
    prices = np.tile(np.asarray([grid, gas, carbon * grid_ef, carbon * gas_ef], dtype=np.float64), (count, 4, 1))
    initial_soc = np.full(count, 0.5, dtype=np.float64)
    previous_chp = np.zeros(count, dtype=np.float64)
    return demand, renew, prices, initial_soc, previous_chp, ix


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4.json")
    parser.add_argument("--output", type=Path, default=FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4" / "protocol" / "DIFFERENTIABLE_LP_GATE.json")
    parser.add_argument("--windows", type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite DiffLP gate receipt: {args.output}")
    spec = load_formal_v4_spec(args.contract)
    parameters = _parameters(spec)
    demand, renew, prices, initial_soc, previous_chp, origins = _fixed_training_windows(spec, parameters, args.windows)
    source_lock = FRAME_ROOT / "requirements" / "formal_v4_diffopt.lock"
    lock = json.loads(source_lock.read_text(encoding="utf-8"))
    native_probe_passed = bool(lock.get("native_layer_probe", {}).get("eligible_for_gate0") is True)
    started = time.perf_counter()
    dpp_passed = True
    finite_solves = 0
    objective_gaps: list[float] = []
    residuals: list[float] = []
    reason = "pass"
    try:
        layer = DifferentiableIESLayer(parameters, tie_break_throughput_cost=1.0e-5)
        for i in range(args.windows):
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
    projected_hours = (elapsed / max(args.windows, 1)) * 10000.0 * 2.0 / 3600.0
    eligible = all((
        dpp_passed, finite_solves >= args.windows, args.windows >= 100,
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
        "schema_version": "formal-v4-diffopt-gate-v1",
        "backend": {"solve_method": "SCS", "eps": 1.0e-10, "max_iters": 100000, "dtype": "float64", "tie_break_throughput_cost": 1.0e-5},
        "windows": args.windows, "training_years": list(spec.train_years),
        "fixed_origin_indices_sha256": hashlib.sha256(origins.tobytes()).hexdigest(),
        "max_relative_objective_gap": float(max(objective_gaps, default=float("inf"))),
        "p95_relative_objective_gap": float(np.quantile(objective_gaps, 0.95)) if objective_gaps else float("inf"),
        "max_physical_residual": float(max(residuals, default=float("inf"))),
        "p95_physical_residual": float(np.quantile(residuals, 0.95)) if residuals else float("inf"),
        "gradient_norm": gradient_norm, "elapsed_seconds": elapsed,
        "lock_sha256": _sha256(source_lock), "contract_sha256": _sha256(args.contract),
        "test_set_accessed": False,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
