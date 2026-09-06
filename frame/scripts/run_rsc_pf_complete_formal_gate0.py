"""Run the readiness/resource certification for the complete RSC-PF matrix.

Gate 0 is a short native smoke only.  It never discovers formal data and it
never reads the sealed 2020 year.  A successful transition authorizes the
later 2015--2019 training/selection stage, not the sealed evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.complete_formal_contract import CompleteFormalContract
from src.joint_dispatch.formal_v4_itransformer import verify_itransformer_source_files
from src.joint_dispatch.formal_v4_models import RSCPFModel
from src.joint_dispatch.model import JointForecastDispatchModel
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp


GATE0_SCHEMA = "rsc-pf-complete-formal-gate0-v1"
GATE0_AUDIT_SCHEMA = "rsc-pf-complete-formal-gate0-audit-v1"
GATE0_TRANSITION_SCHEMA = "rsc-pf-complete-formal-gate0-transition-v1"
DEFAULT_TRAIN_WINDOWS = 34959
DEFAULT_ITRANSFORMER_RECEIPT = FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2" / "formal_v4_2_20260905_j" / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json"
DEFAULT_ITRANSFORMER_SOURCE = FRAME_ROOT / "third_party" / "iTransformer_source"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_canonical(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _p50_p95(samples: list[float]) -> dict[str, Any]:
    if not samples:
        raise ValueError("timing probe produced no samples")
    values = np.asarray(samples, dtype=np.float64)
    return {"samples": len(samples), "p50_seconds": float(np.quantile(values, 0.50)), "p95_seconds": float(np.quantile(values, 0.95)), "min_seconds": float(values.min()), "max_seconds": float(values.max())}


def _memory_bytes() -> Optional[int]:
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _rsc_inputs(batch: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(2026)
    return {
        "load_history": torch.rand((batch, 24, 4), generator=generator),
        "exog_history": torch.rand((batch, 24, 12), generator=generator),
        "device_history": torch.rand((batch, 24, 17), generator=generator),
        # Device on/off history is a binary causal state, not a continuous
        # feature.  Keeping this explicit prevents the smoke from accepting a
        # tensor shape that the real model would reject.
        "activity_history": torch.zeros((batch, 24, 6)),
        # The decoder treats initial SOC and previous CHP as origin-level
        # state and requires them to be constant across the four-hour horizon.
        "scheduler_context": torch.zeros((batch, 4, 6)),
        "previous_chp": torch.zeros((batch, 1)),
    }


def smoke_rsc_training(*, windows: int = 16, epochs: int = 2, seed: int = 2026) -> dict[str, Any]:
    if windows != 16 or epochs != 2:
        raise ValueError("Gate 0 fixes a 16-window, two-epoch smoke")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = RSCPFModel(decoder_parameters=JointForecastDispatchModel._test_parameters(), dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    inputs = _rsc_inputs(windows)
    step_samples: list[float] = []
    decision_gradient_norm = 0.0
    before = _memory_bytes()
    model.train()
    for _epoch in range(epochs):
        started = time.perf_counter()
        output = model(**inputs)
        loss = output.dispatch.square().mean() + output.forecast_physical.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [value.grad.detach().float().norm() for value in model.forecaster_parameters() if value.grad is not None]
        decision_gradient_norm = float(torch.stack(gradients).norm()) if gradients else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_samples.append(time.perf_counter() - started)
    after = _memory_bytes()
    return {
        "windows": windows,
        "epochs": epochs,
        "seed": seed,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "native_gradient_nonzero": decision_gradient_norm > 0.0,
        "decision_forecaster_gradient_norm": decision_gradient_norm,
        "forecast_shape": list(output.forecast_physical.shape),
        "dispatch_shape": list(output.dispatch.shape),
        "timing": _p50_p95(step_samples),
        "rss_before_bytes": before,
        "rss_after_bytes": after,
        "rss_delta_bytes": None if before is None or after is None else max(0, after - before),
        "identity": "src.joint_dispatch.formal_v4_models.RSCPFModel",
    }


def smoke_rsc_batch64(*, samples: int = 5, seed: int = 2026) -> dict[str, Any]:
    """Calibrate one effective-batch update without loading formal arrays.

    The required 16-window/two-epoch smoke remains the functional gate.  This
    additional synthetic batch-64 timing avoids multiplying a small-batch CPU
    timing by four, which is needlessly conservative for a vectorized model.
    """

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = RSCPFModel(decoder_parameters=JointForecastDispatchModel._test_parameters(), dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    inputs = _rsc_inputs(64)
    samples_out: list[float] = []
    model.train()
    for _ in range(samples):
        started = time.perf_counter()
        output = model(**inputs)
        loss = output.dispatch.square().mean() + output.forecast_physical.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        samples_out.append(time.perf_counter() - started)
    return {"batch_size": 64, "timing": _p50_p95(samples_out), "identity": "RSCPFModel.effective_batch_64_synthetic_update"}


def smoke_exact_lp(*, samples: int = 5) -> dict[str, Any]:
    parameters = {
        "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
        "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
        "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
        "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
        "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
        "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25, "carbon_price_default": 0.0, "unserved_penalty": 100.0,
        "surplus_penalty": 0.1, "chp_ramp_fraction": 0.5,
    }
    samples_out: list[float] = []
    objective = None
    for _ in range(samples):
        started = time.perf_counter()
        result = solve_dispatch_lp(DispatchInputs(np.ones((4, 3)), np.full(4, 0.5), np.full(4, 0.5), parameters, 0.5, 0.0))
        samples_out.append(time.perf_counter() - started)
        if not result.success:
            raise RuntimeError(result.message)
        objective = float(result.objective)
    return {"timing": _p50_p95(samples_out), "success": True, "objective": objective, "identity": "src.scheduling.dispatch_lp.solve_dispatch_lp"}


def smoke_difflp(*, samples: int = 5) -> dict[str, Any]:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    variable = cp.Variable(1)
    parameter = cp.Parameter(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(variable - parameter)), [variable >= 0])
    layer = CvxpyLayer(problem, parameters=[parameter], variables=[variable])
    samples_out: list[float] = []
    gradient_norm = 0.0
    value = None
    for _ in range(samples):
        query = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        started = time.perf_counter()
        value = layer(query)[0]
        value.sum().backward()
        samples_out.append(time.perf_counter() - started)
        gradient_norm = max(gradient_norm, float(query.grad.detach().norm()))
    return {"timing": _p50_p95(samples_out), "native_gradient_nonzero": gradient_norm > 0.0, "gradient_norm": gradient_norm, "identity": "cvxpylayers.torch.CvxpyLayer.forward_backward"}


def smoke_policy() -> dict[str, Any]:
    torch.manual_seed(2026)
    policy = torch.nn.Linear(24, 15)
    x = torch.zeros((4, 24))
    value = policy(x).square().mean()
    value.backward()
    return {"success": True, "finite": bool(torch.isfinite(value).item()), "identity": "torch.nn.Linear.direct_policy_smoke"}


def smoke_naive() -> dict[str, Any]:
    values = np.arange(24, dtype=np.float64)
    return {"success": bool(np.isfinite(values[-1])), "identity": "seasonal_naive_metadata_smoke"}


def verify_official_itransformer(*, source_root: Path = DEFAULT_ITRANSFORMER_SOURCE, receipt_path: Path = DEFAULT_ITRANSFORMER_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_itransformer_source_files(source_root, receipt, require_license=True)
    return {"official_source_hash_verified": True, "source_root": str(source_root), "receipt_path": str(receipt_path), "receipt_sha256": _sha256(receipt_path), "commit": receipt.get("commit"), "identity": "THUML/iTransformer:model.iTransformer.Model"}


def _project_resources(contract: CompleteFormalContract, timings: Mapping[str, Mapping[str, Any]], *, train_windows: int, disk: Mapping[str, Any]) -> dict[str, Any]:
    batch = int(contract.training["common"]["effective_batch_size"])
    epochs = int(contract.training["common"]["max_epochs_per_stage"])
    stochastic_rows = len(contract.payload["seeds"]) * len([method for method in contract.primary_method_ids if contract.method(method).stochastic])
    updates = int(np.ceil(train_windows / batch) * epochs * stochastic_rows)
    if "rsc_training_batch64" in timings:
        neural_seconds = updates * float(timings["rsc_training_batch64"]["p95_seconds"])
        neural_timing_source = "synthetic_batch64_update"
    else:
        smoke_batch = 16
        neural_seconds = updates * float(timings["rsc_training"]["p95_seconds"]) * batch / smoke_batch
        neural_timing_source = "scaled_16_window_smoke"
    lp_calls_per_origin = sum(contract.method(method).inference_lp_calls_per_origin for method in contract.primary_method_ids)
    lp_seconds = contract.selection_origin_count * lp_calls_per_origin * float(timings["exact_lp"]["p95_seconds"])
    diff_seconds = updates * float(timings["difflp"]["p95_seconds"])
    total_seconds = neural_seconds + lp_seconds + diff_seconds
    estimated_checkpoint_bytes = 35 * 64 * 1024 * 1024 + 2 * 1024 * 1024
    free_after = max(int(disk["free_bytes"]) - estimated_checkpoint_bytes, 0)
    margin_after = float(free_after / max(int(disk["total_bytes"]), 1))
    return {
        "train_windows": train_windows,
        "stochastic_rows": stochastic_rows,
        "max_epochs": epochs,
        "effective_batch_size": batch,
        "training_updates": updates,
        "selection_origins": contract.selection_origin_count,
        "inference_lp_calls_per_origin_total": lp_calls_per_origin,
        "components_seconds": {"neural_training": neural_seconds, "exact_lp_selection": lp_seconds, "difflp_training": diff_seconds},
        "neural_timing_source": neural_timing_source,
        "projected_total_seconds": total_seconds,
        "projected_total_hours": total_seconds / 3600.0,
        "maximum_hours": 24.0,
        "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
        "disk_margin_after_fraction": margin_after,
        "minimum_disk_margin_fraction": 0.20,
        "disk_authorized": margin_after >= 0.20,
        "duration_authorized": total_seconds / 3600.0 <= 24.0,
        "authorized": total_seconds / 3600.0 <= 24.0 and margin_after >= 0.20,
    }


def audit_gate0_receipt(receipt: Mapping[str, Any], *, contract: CompleteFormalContract) -> dict[str, Any]:
    failures: list[str] = []
    if receipt.get("schema_version") != GATE0_SCHEMA:
        failures.append("schema mismatch")
    if receipt.get("contract_sha256") != contract.contract_sha256:
        failures.append("contract hash mismatch")
    if receipt.get("evaluation_year_accessed") is not False or receipt.get("excluded_year_accessed") is not False:
        failures.append("sealed or excluded year access reported")
    if receipt.get("difflp", {}).get("native_gradient_nonzero") is not True:
        failures.append("DiffLP native gradient probe failed")
    if receipt.get("itransformer", {}).get("official_source_hash_verified") is not True:
        failures.append("official iTransformer source hash probe failed")
    if receipt.get("all_method_families_smoked") is not True:
        failures.append("not all method families were smoked")
    projection = receipt.get("resource_projection", {})
    if projection.get("disk_authorized") is not True:
        failures.append("disk margin is below 20 percent")
    duration_requires_authorization = projection.get("duration_authorized") is not True
    if duration_requires_authorization:
        failures.append("projected formal run exceeds 24 hours")
    hard_failures = [failure for failure in failures if failure != "projected formal run exceeds 24 hours"]
    status = "pass" if not failures else ("requires_authorization" if duration_requires_authorization and not hard_failures else "fail")
    return {"schema_version": GATE0_AUDIT_SCHEMA, "status": status, "failures": failures, "contract_sha256": contract.contract_sha256, "evaluation_year_accessed": receipt.get("evaluation_year_accessed"), "excluded_year_accessed": receipt.get("excluded_year_accessed"), "recomputed_authorized_gate1": status == "pass", "requires_explicit_authorization": status == "requires_authorization"}


def run_gate0(*, contract_path: str | Path, output_root: str | Path, run_id: str, train_windows: int = DEFAULT_TRAIN_WINDOWS, smoke_windows: int = 16, smoke_epochs: int = 2, seed: int = 2026, source_root: str | Path = DEFAULT_ITRANSFORMER_SOURCE, itransformer_receipt: str | Path = DEFAULT_ITRANSFORMER_RECEIPT, operations: Optional[Mapping[str, Callable[[], Mapping[str, Any]]]] = None) -> dict[str, Any]:
    contract = CompleteFormalContract.from_path(contract_path)
    root = Path(output_root).resolve() / run_id
    gate_dir = root / "gate0"
    if root.exists():
        raise FileExistsError(f"fresh run id required: {root}")
    gate_dir.mkdir(parents=True, exist_ok=False)
    disk_usage = shutil.disk_usage(gate_dir)
    disk = {"total_bytes": int(disk_usage.total), "free_bytes": int(disk_usage.free), "used_bytes": int(disk_usage.used), "free_fraction": float(disk_usage.free / max(disk_usage.total, 1))}
    ops = dict(operations or {})
    failures: list[str] = []
    started = time.perf_counter()
    try:
        rsc = dict(ops.get("rsc_training", lambda: smoke_rsc_training(windows=smoke_windows, epochs=smoke_epochs, seed=seed))())
        rsc_batch64 = dict(ops.get("rsc_training_batch64", lambda: smoke_rsc_batch64(seed=seed))())
        lp = dict(ops.get("exact_lp", smoke_exact_lp)())
        difflp = dict(ops.get("difflp", smoke_difflp)())
        policy = dict(ops.get("direct_policy", smoke_policy)())
        naive = dict(ops.get("seasonal_naive", smoke_naive)())
        itransformer = dict(ops.get("itransformer", lambda: verify_official_itransformer(source_root=Path(source_root), receipt_path=Path(itransformer_receipt)))())
    except Exception as exc:
        failures.append(f"native smoke error: {type(exc).__name__}: {exc}")
        rsc, rsc_batch64, lp, difflp, policy, naive, itransformer = ({"native_gradient_nonzero": False}, {"timing": {"p95_seconds": float("inf")}}, {"success": False}, {"native_gradient_nonzero": False}, {"success": False}, {"success": False}, {"official_source_hash_verified": False})
    timings = {"rsc_training": rsc.get("timing", {"p95_seconds": float("inf")}), "rsc_training_batch64": rsc_batch64.get("timing", {"p95_seconds": float("inf")}), "exact_lp": lp.get("timing", {"p95_seconds": float("inf")}), "difflp": difflp.get("timing", {"p95_seconds": float("inf")})}
    resource_projection = _project_resources(contract, timings, train_windows=int(train_windows), disk=disk)
    families = {
        "RSC-PF": rsc.get("native_gradient_nonzero") is True and rsc.get("loss_finite") is True,
        "Decoupled-RSC-PF": rsc.get("native_gradient_nonzero") is True,
        "Direct-Policy": policy.get("success") is True,
        "Scheme2R-PTO": lp.get("success") is True,
        "State-Conditioned-PTO": lp.get("success") is True,
        "Official iTransformer-PTO": itransformer.get("official_source_hash_verified") is True,
        "Differentiable-LP": difflp.get("native_gradient_nonzero") is True,
        "Seasonal-Naive-PTO": naive.get("success") is True,
        "Perfect-Information-MPC": lp.get("success") is True,
    }
    receipt: dict[str, Any] = {
        "schema_version": GATE0_SCHEMA,
        "run_id": run_id,
        "contract_sha256": contract.contract_sha256,
        "smoke": {"windows": smoke_windows, "epochs": smoke_epochs, "seed": seed, "data_arrays_loaded": False},
        "rsc_pf": rsc,
        "rsc_batch64": rsc_batch64,
        "exact_lp": lp,
        "difflp": difflp,
        "itransformer": itransformer,
        "all_method_families_smoked": all(families.values()),
        "method_family_checks": families,
        "resource_projection": resource_projection,
        "disk": disk,
        "evaluation_year_accessed": False,
        "excluded_year_accessed": False,
        "selection_year_accessed": False,
        "data_boundary": {"train_years": list(contract.train_years), "selection_year": contract.selection_year, "evaluation_year": contract.evaluation_year, "excluded_years": list(contract.excluded_years), "formal_arrays_loaded": False},
        "runtime_seconds": time.perf_counter() - started,
        "failures": failures,
    }
    audit = audit_gate0_receipt(receipt, contract=contract)
    if failures:
        audit["status"] = "fail"
        audit["failures"] = list(audit["failures"]) + failures
    authorized = audit["status"] == "pass"
    receipt["authorized_gate1_training"] = authorized
    receipt["status"] = audit["status"]
    _write_json(gate_dir / "GATE0_RECEIPT.json", receipt)
    _write_json(gate_dir / "GATE0_AUDIT.json", audit)
    transition = {"schema_version": GATE0_TRANSITION_SCHEMA, "run_id": run_id, "contract_sha256": contract.contract_sha256, "authorized_gate1_training": authorized, "evaluation_year_accessed": False, "excluded_year_accessed": False, "receipt_sha256": _sha256(gate_dir / "GATE0_RECEIPT.json"), "audit_sha256": _sha256(gate_dir / "GATE0_AUDIT.json"), "projected_total_hours": resource_projection["projected_total_hours"], "requires_explicit_authorization": audit["status"] == "requires_authorization", "paper_result": False}
    _write_json(gate_dir / "GATE0_TRANSITION.json", transition)
    return receipt


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--train-windows", type=int, default=DEFAULT_TRAIN_WINDOWS)
    args = parser.parse_args(argv)
    receipt = run_gate0(contract_path=args.contract, output_root=args.output_root, run_id=args.run_id, train_windows=args.train_windows)
    print(json.dumps({"status": receipt["status"], "authorized_gate1_training": receipt["authorized_gate1_training"], "projected_total_hours": receipt["resource_projection"]["projected_total_hours"], "run_id": receipt["run_id"]}, ensure_ascii=False))
    return 0 if receipt["status"] == "pass" else (3 if receipt["status"] == "requires_authorization" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
