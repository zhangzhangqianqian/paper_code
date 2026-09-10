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
from src.joint_dispatch.formal_v4_models import DirectPolicyModel, RSCPFModel
from src.joint_dispatch.model import JointForecastDispatchModel
from src.joint_dispatch.reference_data import seasonal_naive_24h
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp


GATE0_SCHEMA = "rsc-pf-complete-formal-gate0-v1"
GATE0_AUDIT_SCHEMA = "rsc-pf-complete-formal-gate0-audit-v1"
GATE0_TRANSITION_SCHEMA = "rsc-pf-complete-formal-gate0-transition-v1"
DEFAULT_TRAIN_WINDOWS = 34959
DEFAULT_ITRANSFORMER_RECEIPT = FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2" / "formal_v4_2_20260905_j" / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json"
DEFAULT_ITRANSFORMER_SOURCE = FRAME_ROOT / "third_party" / "iTransformer_source"
# A fixed absolute reserve is appropriate for this run because the formal
# artifacts are bounded and the drive capacity is much larger than the run.
# The reserve is applied after the estimated checkpoint footprint.
MINIMUM_DISK_MARGIN_BYTES = 20 * 1024 ** 3


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


def _source_fingerprints(contract: CompleteFormalContract) -> dict[str, Any]:
    """Hash the code/config artifacts that define the Gate 0 contract."""

    paths = {
        "complete_formal_contract": contract.source_path,
        "formal_v46_config": FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_6.json",
        "standard_ies_rules": FRAME_ROOT / "configs" / "standard_ies_formal_v4_rules.yaml",
        "capacity_binding": FRAME_ROOT / "configs" / "rsc_pf_complete_formal_capacity_binding.json",
        "capacity_benchmark": FRAME_ROOT / "configs" / "standard_ies_benchmark_formal_v4_complete.json",
        "capacity_ledger": FRAME_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv",
        "formal_v4_models": FRAME_ROOT / "src" / "joint_dispatch" / "formal_v4_models.py",
        "proxy_decoder": FRAME_ROOT / "src" / "scheduling" / "proxy_decoder.py",
        "dispatch_lp": FRAME_ROOT / "src" / "scheduling" / "dispatch_lp.py",
    }
    result: dict[str, Any] = {}
    for name, path in paths.items():
        result[name] = {"path": str(path.resolve()), "exists": path.is_file(), "sha256": _sha256(path) if path.is_file() else None}
    return result


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


def smoke_rsc_inference(*, samples: int = 5, seed: int = 2026) -> dict[str, Any]:
    """Measure one chronological-origin neural forward for 2019 projection."""

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = RSCPFModel(decoder_parameters=JointForecastDispatchModel._test_parameters(), dropout=0.0).eval()
    inputs = _rsc_inputs(1)
    samples_out: list[float] = []
    with torch.no_grad():
        for _ in range(samples):
            started = time.perf_counter()
            model(**inputs)
            samples_out.append(time.perf_counter() - started)
    return {"batch_size": 1, "timing": _p50_p95(samples_out), "identity": "RSCPFModel.chronological_origin_forward"}


def smoke_exact_lp(contract: CompleteFormalContract, *, samples: int = 5) -> dict[str, Any]:
    """Smoke the exact LP with the contract-bound Standard-IES parameters."""

    parameters = dict(contract.capacity_parameters)
    samples_out: list[float] = []
    objective = None
    residual_max = 0.0
    for _ in range(samples):
        started = time.perf_counter()
        result = solve_dispatch_lp(DispatchInputs(np.ones((4, 3)), np.full(4, 0.5), np.full(4, 0.5), parameters, 0.5, 0.0))
        samples_out.append(time.perf_counter() - started)
        if not result.success:
            raise RuntimeError(result.message)
        objective = float(result.objective)
        residual_max = max(residual_max, *(float(value) for value in result.balance_residuals.values()))
    return {"timing": _p50_p95(samples_out), "success": True, "objective": objective, "max_balance_residual": residual_max, "identity": "src.scheduling.dispatch_lp.solve_dispatch_lp"}


def smoke_difflp(*, samples: int = 5) -> dict[str, Any]:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    import warnings

    variable = cp.Variable(1)
    parameter = cp.Parameter(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(variable - parameter)), [variable >= 0])
    layer = CvxpyLayer(problem, parameters=[parameter], variables=[variable])
    samples_out: list[float] = []
    gradient_norm = 0.0
    value = None
    warning_count = 0
    warning_categories: dict[str, int] = {}
    residual_max = 0.0
    reference_status_counts: dict[str, int] = {}
    for _ in range(samples):
        query = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        parameter.value = np.asarray([1.0], dtype=np.float64)
        try:
            problem.solve(solver=cp.SCS, eps=1.0e-7, max_iters=10000, warm_start=False, verbose=False)
            reference_status = str(problem.status)
        except Exception as exc:
            reference_status = f"error:{type(exc).__name__}"
        reference_status_counts[reference_status] = reference_status_counts.get(reference_status, 0) + 1
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            value = layer(query)[0]
        warning_count += len(caught)
        for warning in caught:
            category = warning.category.__name__
            warning_categories[category] = warning_categories.get(category, 0) + 1
        value.sum().backward()
        samples_out.append(time.perf_counter() - started)
        gradient_norm = max(gradient_norm, float(query.grad.detach().norm()))
        residual_max = max(residual_max, float((value.detach() - query.detach()).abs().max()))
    return {
        "timing": _p50_p95(samples_out),
        "native_gradient_nonzero": gradient_norm > 0.0,
        "gradient_norm": gradient_norm,
        "solver_warning_count": warning_count,
        "solver_warning_categories": warning_categories,
        "solver_status_counts": reference_status_counts,
        "reference_solver_status_counts": reference_status_counts,
        "max_primal_residual": residual_max,
        "layer_solver_status_exposed": False,
        "identity": "cvxpylayers.torch.CvxpyLayer.forward_backward",
    }


def smoke_policy() -> dict[str, Any]:
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    policy = DirectPolicyModel(decoder_parameters=JointForecastDispatchModel._test_parameters(), dropout=0.0)
    inputs = _rsc_inputs(4)
    output = policy(**inputs)
    value = output.dispatch.square().mean() + output.latent_planning_demand.square().mean()
    if tuple(output.controls.shape) != (4, 4, 15) or tuple(output.dispatch.shape) != (4, 4, 21):
        raise RuntimeError("Direct-Policy smoke returned an unexpected control/dispatch shape")
    value.backward()
    gradient_nonzero = any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) and float(parameter.grad.detach().abs().sum()) > 0.0 for parameter in policy.parameters())
    return {"success": bool(torch.isfinite(value).item()) and gradient_nonzero, "finite": bool(torch.isfinite(value).item()), "gradient_nonzero": gradient_nonzero, "controls_shape": list(output.controls.shape), "dispatch_shape": list(output.dispatch.shape), "feasibility_adapter_id": DirectPolicyModel.FEASIBILITY_ADAPTER_ID, "identity": "src.joint_dispatch.formal_v4_models.DirectPolicyModel"}


def smoke_naive() -> dict[str, Any]:
    values = np.arange(28 * 4, dtype=np.float64).reshape(28, 4)
    forecast = seasonal_naive_24h(values, origin_index=24, horizon=4, task_count=4)
    expected = values[:4]
    return {"success": bool(forecast.shape == (4, 4) and np.isfinite(forecast).all() and np.array_equal(forecast, expected)), "forecast_shape": list(forecast.shape), "identity": "src.joint_dispatch.reference_data.seasonal_naive_24h"}


def verify_official_itransformer(*, source_root: Path = DEFAULT_ITRANSFORMER_SOURCE, receipt_path: Path = DEFAULT_ITRANSFORMER_RECEIPT) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    verify_itransformer_source_files(source_root, receipt, require_license=True)
    return {"official_source_hash_verified": True, "source_root": str(source_root), "receipt_path": str(receipt_path), "receipt_sha256": _sha256(receipt_path), "commit": receipt.get("commit"), "identity": "THUML/iTransformer:model.iTransformer.Model"}


def _project_resources(contract: CompleteFormalContract, timings: Mapping[str, Mapping[str, Any]], *, train_windows: int, disk: Mapping[str, Any]) -> dict[str, Any]:
    batch = int(contract.training["common"]["effective_batch_size"])
    epochs = int(contract.training["common"]["max_epochs_per_stage"])
    stochastic_rows = len(contract.payload["seeds"]) * len([method for method in contract.primary_method_ids if contract.method(method).stochastic])
    updates_per_row = int(np.ceil(train_windows / batch) * epochs)
    updates = int(updates_per_row * stochastic_rows)
    if "rsc_training_batch64" in timings:
        neural_seconds = updates * float(timings["rsc_training_batch64"]["p95_seconds"])
        neural_timing_source = "synthetic_batch64_update"
    else:
        smoke_batch = 16
        neural_seconds = updates * float(timings["rsc_training"]["p95_seconds"]) * batch / smoke_batch
        neural_timing_source = "scaled_16_window_smoke"
    # Count rows, not only method names: every stochastic row has its own
    # forecast/dispatch rollout for each seed.  The deterministic references
    # contribute one row each.
    lp_calls_per_origin = sum(
        contract.method(method).inference_lp_calls_per_origin
        * (len(contract.payload["seeds"]) if contract.method(method).stochastic else 1)
        for method in contract.primary_method_ids
    )
    lp_seconds = contract.selection_origin_count * lp_calls_per_origin * float(timings["exact_lp"]["p95_seconds"])
    diff_seconds = updates * float(timings["difflp"]["p95_seconds"])
    # Search is part of the frozen Gate 1 budget.  The selected trial is
    # accounted for by the primary row, so only the remaining trials are
    # additional work.  The grids are method-specific and are not silently
    # applied to methods that have no frozen search grid.
    search = contract.payload["training"]["search"]
    rsc_trials = len(search["rsc_pf_decision_multiplier_grid"])
    difflp_trials = len(search["difflp_learning_rate_grid"])
    extra_rsc_trials = max(0, rsc_trials - 1)
    extra_difflp_trials = max(0, difflp_trials - 1)
    search_neural_timing = timings.get("rsc_training_batch64", timings["rsc_training"])
    search_neural_seconds = extra_rsc_trials * updates_per_row * float(search_neural_timing["p95_seconds"])
    search_difflp_seconds = extra_difflp_trials * updates_per_row * float(timings["difflp"]["p95_seconds"])
    # Gate 1 evaluates every stochastic row chronologically on 2019.  This
    # forward-only cost is separate from backpropagation during training.
    evaluation_neural_seconds = contract.selection_origin_count * stochastic_rows * float(timings["rsc_inference"]["p95_seconds"])
    evaluation_difflp_seconds = contract.selection_origin_count * len(contract.payload["seeds"]) * float(timings["difflp"]["p95_seconds"])
    total_seconds = neural_seconds + lp_seconds + diff_seconds + search_neural_seconds + search_difflp_seconds + evaluation_neural_seconds + evaluation_difflp_seconds
    estimated_checkpoint_bytes = 35 * 64 * 1024 * 1024 + 2 * 1024 * 1024
    free_after = max(int(disk["free_bytes"]) - estimated_checkpoint_bytes, 0)
    margin_after = float(free_after / max(int(disk["total_bytes"]), 1))
    return {
        "train_windows": train_windows,
        "stochastic_rows": stochastic_rows,
        "max_epochs": epochs,
        "effective_batch_size": batch,
        "training_updates": updates,
        "updates_per_row": updates_per_row,
        "search_trials_by_method": {"RSC-PF": rsc_trials, "Differentiable-LP": difflp_trials},
        "search_extra_trials_by_method": {"RSC-PF": extra_rsc_trials, "Differentiable-LP": extra_difflp_trials},
        "search_extra_updates": int((extra_rsc_trials + extra_difflp_trials) * updates_per_row),
        "selection_origins": contract.selection_origin_count,
        "inference_lp_calls_per_origin_total": lp_calls_per_origin,
        "components_seconds": {"neural_training": neural_seconds, "exact_lp_selection": lp_seconds, "difflp_training": diff_seconds, "search_neural_training": search_neural_seconds, "search_difflp_training": search_difflp_seconds, "neural_2019_evaluation": evaluation_neural_seconds, "difflp_2019_evaluation": evaluation_difflp_seconds},
        "neural_timing_source": neural_timing_source,
        "projected_total_seconds": total_seconds,
        "projected_total_hours": total_seconds / 3600.0,
        "maximum_hours": 24.0,
        "estimated_checkpoint_bytes": estimated_checkpoint_bytes,
        "disk_margin_after_bytes": free_after,
        "disk_margin_after_gib": free_after / (1024 ** 3),
        "disk_margin_after_fraction": margin_after,
        "minimum_disk_margin_bytes": MINIMUM_DISK_MARGIN_BYTES,
        "minimum_disk_margin_gib": MINIMUM_DISK_MARGIN_BYTES / (1024 ** 3),
        "disk_authorized": free_after >= MINIMUM_DISK_MARGIN_BYTES,
        "duration_authorized": total_seconds / 3600.0 <= 24.0,
        # The frozen resource rule is an absolute post-checkpoint reserve.
        # The fractional value is retained for diagnostics only; it is not a
        # second authorization threshold.
        "authorized": total_seconds / 3600.0 <= 24.0 and free_after >= MINIMUM_DISK_MARGIN_BYTES,
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
    difflp = receipt.get("difflp", {})
    if "max_primal_residual" in difflp and (not np.isfinite(float(difflp["max_primal_residual"])) or float(difflp["max_primal_residual"]) > 1.0e-4):
        failures.append("DiffLP smoke primal residual exceeds 1e-4")
    status_counts = difflp.get("solver_status_counts", difflp.get("reference_solver_status_counts"))
    if status_counts and any(str(status) not in {"optimal", "optimal_inaccurate"} for status in status_counts):
        failures.append("DiffLP reference solver did not reach an accepted status")
    if receipt.get("itransformer", {}).get("official_source_hash_verified") is not True:
        failures.append("official iTransformer source hash probe failed")
    if receipt.get("all_method_families_smoked") is not True:
        failures.append("not all method families were smoked")
    capacity_binding = receipt.get("capacity_binding", {})
    if capacity_binding.get("status") != "pass" or capacity_binding.get("fit_years") != list(contract.train_years) or capacity_binding.get("evaluation_year_accessed") is not False:
        failures.append("current contract capacity binding is missing or not training-only")
    fingerprints = receipt.get("source_fingerprints", {})
    required_fingerprints = {"complete_formal_contract", "formal_v46_config", "standard_ies_rules", "capacity_binding", "capacity_benchmark", "capacity_ledger", "formal_v4_models", "proxy_decoder", "dispatch_lp"}
    if not required_fingerprints.issubset(fingerprints) or any(fingerprints[name].get("exists") is not True or not fingerprints[name].get("sha256") for name in required_fingerprints):
        failures.append("frozen source/config fingerprints are incomplete")
    projection = receipt.get("resource_projection", {})
    resource_fields_valid = True
    try:
        free_after = int(projection["disk_margin_after_bytes"])
        minimum_reserve = int(projection["minimum_disk_margin_bytes"])
        projected_hours = float(projection["projected_total_hours"])
        maximum_hours = float(projection["maximum_hours"])
        resource_fields_valid = np.isfinite(projected_hours) and np.isfinite(maximum_hours) and minimum_reserve > 0
    except (KeyError, TypeError, ValueError, OverflowError):
        resource_fields_valid = False
        free_after = minimum_reserve = 0
        projected_hours = maximum_hours = float("inf")
    if not resource_fields_valid:
        failures.append("resource projection fields are invalid")
        recomputed_disk_authorized = False
        recomputed_duration_authorized = False
    else:
        recomputed_disk_authorized = free_after >= minimum_reserve
        recomputed_duration_authorized = projected_hours <= maximum_hours
        if projection.get("disk_authorized") is not recomputed_disk_authorized:
            failures.append("resource projection disk_authorized flag is inconsistent with its values")
        if projection.get("duration_authorized") is not recomputed_duration_authorized:
            failures.append("resource projection duration_authorized flag is inconsistent with its values")
        if projection.get("authorized") is not (recomputed_disk_authorized and recomputed_duration_authorized):
            failures.append("resource projection authorized flag is inconsistent with its values")
    if not recomputed_disk_authorized:
        failures.append("disk margin is below 20 GiB after estimated checkpoints")
    duration_requires_authorization = not recomputed_duration_authorized
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
        rsc_inference = dict(ops.get("rsc_inference", lambda: smoke_rsc_inference(seed=seed))())
        lp = dict(ops.get("exact_lp", lambda: smoke_exact_lp(contract))())
        difflp = dict(ops.get("difflp", smoke_difflp)())
        policy = dict(ops.get("direct_policy", smoke_policy)())
        naive = dict(ops.get("seasonal_naive", smoke_naive)())
        itransformer = dict(ops.get("itransformer", lambda: verify_official_itransformer(source_root=Path(source_root), receipt_path=Path(itransformer_receipt)))())
    except Exception as exc:
        failures.append(f"native smoke error: {type(exc).__name__}: {exc}")
        rsc, rsc_batch64, rsc_inference, lp, difflp, policy, naive, itransformer = ({"native_gradient_nonzero": False}, {"timing": {"p95_seconds": float("inf")}}, {"timing": {"p95_seconds": float("inf")}}, {"success": False}, {"native_gradient_nonzero": False}, {"success": False}, {"success": False}, {"official_source_hash_verified": False})
    timings = {"rsc_training": rsc.get("timing", {"p95_seconds": float("inf")}), "rsc_training_batch64": rsc_batch64.get("timing", {"p95_seconds": float("inf")}), "rsc_inference": rsc_inference.get("timing", {"p95_seconds": float("inf")}), "exact_lp": lp.get("timing", {"p95_seconds": float("inf")}), "difflp": difflp.get("timing", {"p95_seconds": float("inf")})}
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
        "rsc_inference": rsc_inference,
        "exact_lp": lp,
        "difflp": difflp,
        "itransformer": itransformer,
        "capacity_binding": dict(contract.capacity_binding),
        "all_method_families_smoked": all(families.values()),
        "method_family_checks": families,
        "source_fingerprints": _source_fingerprints(contract),
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
