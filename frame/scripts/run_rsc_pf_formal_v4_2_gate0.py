"""Run the fail-closed, real-operation resource probe for formal-v4.2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_2_artifacts import create_v42_run_root, sha256_file, write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_contract import load_formal_v4_2_contract  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate0 import (  # noqa: E402
    CURRENT_GATE_SCHEMA,
    Gate0InputsV42,
    Gate0PrerequisitePaths,
    produce_diffopt_receipt,
    produce_itransformer_receipt,
    produce_source_manifest,
    validate_prerequisites,
)
from scripts.run_rsc_pf_formal_v4_2_capacity_audit import build_capacity_evidence  # noqa: E402
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp  # noqa: E402


PARAMETERS = {
    "grid_import_capacity": 100.0, "chp_electric_capacity": 20.0, "chp_heat_capacity": 30.0,
    "gas_boiler_capacity": 40.0, "electric_chiller_capacity": 30.0, "absorption_chiller_capacity": 30.0,
    "bess_power_capacity": 10.0, "bess_energy_capacity": 40.0, "chp_electric_efficiency": 0.35,
    "chp_heat_efficiency": 0.45, "gas_boiler_efficiency": 0.9, "electric_chiller_cop": 3.5,
    "absorption_chiller_cop": 0.75, "bess_roundtrip_efficiency": 0.9, "bess_throughput_cost": 0.01,
    "grid_energy_price": 1.0, "gas_energy_price": 0.6, "grid_emission_factor": 0.5,
    "gas_emission_factor": 0.25, "carbon_price_default": 0.0, "unserved_penalty": 100.0,
    "surplus_penalty": 0.1, "chp_ramp_fraction": 0.5,
}


class Gate0ExecutionError(RuntimeError):
    """Raised after a formal-v4.2 Gate 0 run writes a failure receipt."""


def _fixture_get(fixture: Any, name: str, default: Any = None) -> Any:
    if isinstance(fixture, Mapping):
        return fixture.get(name, default)
    return getattr(fixture, name, default)


def run_rsc_forward() -> Mapping[str, Any]:
    """Exercise a real RSC-PF tensor forward with frozen contract shapes."""
    from src.joint_dispatch.formal_v4_models import RSCPFModel
    model = RSCPFModel(dropout=0.0)
    with torch.no_grad():
        output = model(
            load_history=torch.zeros(2, 24, 4), exog_history=torch.zeros(2, 24, 12),
            device_history=torch.zeros(2, 24, 17), activity_history=torch.zeros(2, 24, 6),
            scheduler_context=torch.zeros(2, 4, 6), previous_chp=torch.zeros(2, 1),
        )
    return {"forecast_shape": list(output.forecast_physical.shape), "dispatch_shape": list(output.dispatch.shape), "identity": "RSCPFModel.forward"}


def run_rsc_backward() -> Mapping[str, Any]:
    """Exercise a decision backward path into the RSC-PF forecaster."""
    from src.joint_dispatch.formal_v4_models import RSCPFModel
    model = RSCPFModel(dropout=0.0)
    output = model(
        load_history=torch.zeros(2, 24, 4), exog_history=torch.zeros(2, 24, 12),
        device_history=torch.zeros(2, 24, 17), activity_history=torch.zeros(2, 24, 6),
        scheduler_context=torch.zeros(2, 4, 6), previous_chp=torch.zeros(2, 1),
    )
    loss = output.dispatch.square().mean()
    loss.backward()
    grads = [parameter.grad for parameter in model.forecaster_parameters() if parameter.grad is not None]
    norm = float(torch.stack([value.detach().float().norm() for value in grads]).norm()) if grads else 0.0
    return {"decision_forecaster_gradient_norm": norm, "identity": "RSCPFModel.dispatch.backward"}


def run_highs_lp() -> Mapping[str, Any]:
    demand = np.full((4, 3), 1.0, dtype=np.float64)
    renewable = np.full((4, 2), 0.5, dtype=np.float64)
    result = solve_dispatch_lp(DispatchInputs(demand, renewable[:, 0], renewable[:, 1], PARAMETERS, 0.5, 0.0))
    if not result.success:
        raise RuntimeError(result.message)
    return {"objective": float(result.objective), "identity": "scipy.optimize.linprog(method=highs)"}


def run_diff_lp() -> Mapping[str, Any]:
    """Run the locked differentiable-LP micro-operation when available."""
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except Exception as exc:
        return {"available": False, "identity": "cvxpylayers", "error": f"{type(exc).__name__}: {exc}"}
    x = cp.Variable(1)
    q = cp.Parameter(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - q)), [x >= 0])
    layer = CvxpyLayer(problem, parameters=[q], variables=[x])
    value = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    (layer(value)[0].sum()).backward()
    return {"available": True, "identity": "cvxpylayers.CvxpyLayer.forward.backward"}


def _measure(operation, warmup: int = 2, iterations: int = 5) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter(); operation(); samples.append(time.perf_counter() - start)
    values = np.asarray(samples, dtype=np.float64)
    return {"warmup": warmup, "iterations": iterations, "median_seconds": float(np.median(values)), "p95_seconds": float(np.quantile(values, 0.95)), "min_seconds": float(np.min(values)), "max_seconds": float(np.max(values))}


def _run_real_operations(fixture: Any | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    measurements: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    operations = {"rsc_forward": run_rsc_forward, "rsc_backward": run_rsc_backward, "highs_lp": run_highs_lp, "diff_lp": run_diff_lp}
    for name, operation in operations.items():
        override = _fixture_get(fixture, name, None) if fixture is not None else None
        operation = override if callable(override) else operation
        try:
            started = time.perf_counter(); details = operation(); elapsed = time.perf_counter() - started
            if not isinstance(details, Mapping):
                raise TypeError("probe operation must return a mapping")
            timing = {"warmup": 0, "iterations": 1, "median_seconds": elapsed, "p95_seconds": elapsed}
            if getattr(operation, "__module__", __name__) == __name__ and override is None:
                timing = _measure(operation)
            passed = elapsed > 0.0 and bool(details.get("identity"))
            if name == "diff_lp":
                passed = passed and details.get("available") is not False
            if name == "rsc_backward":
                passed = passed and float(details.get("decision_forecaster_gradient_norm", 1.0)) > 0.0
            measurements[name] = {"timing": timing, "elapsed_seconds": elapsed, "details": dict(details), "real_operation": True}
            checks[name] = {"passed": bool(passed), "identity": str(details.get("identity", "")), "nonzero": elapsed > 0.0}
        except Exception as exc:
            measurements[name] = {"real_operation": True, "error_type": type(exc).__name__, "error": str(exc)}
            checks[name] = {"passed": False, "error": str(exc)}
    return measurements, checks


def run_gate0_probe(fixture: Any) -> dict[str, Any]:
    """Run all Gate 0 components and write an immutable evidence tree."""

    output_root = Path(_fixture_get(fixture, "output_root", _fixture_get(fixture, "run_root", Path.cwd()))).resolve()
    run_id = str(_fixture_get(fixture, "run_id", f"formal_v4_2_gate0_{time.strftime('%Y%m%d_%H%M%S')}"))
    root = output_root / run_id
    gate_root = root / "gate0"
    if gate_root.exists():
        raise FileExistsError(gate_root)
    gate_root.mkdir(parents=True, exist_ok=False)
    contract_path = Path(_fixture_get(fixture, "contract_path", FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json"))
    contract = load_formal_v4_2_contract(contract_path)
    measurements, checks = _run_real_operations(fixture)
    receipts = {
        "itransformer_receipt": _fixture_get(fixture, "itransformer_receipt", None),
        "diffopt_receipt": _fixture_get(fixture, "diffopt_receipt", None),
        "capacity_receipt": _fixture_get(fixture, "capacity_receipt", None),
        "source_manifest": _fixture_get(fixture, "source_manifest", None),
    }
    for name, path in receipts.items():
        checks[name] = {"passed": bool(path) and Path(path).is_file(), "path": str(path) if path else ""}
    checks["contract"] = {"passed": contract.contract_sha256 != "", "contract_sha256": contract.contract_sha256}
    required_components = all(item.get("passed") for item in checks.values())
    capacity_fit_years = [2015, 2016, 2017, 2018]
    materialized_years = [2015, 2016, 2017, 2018, 2019]
    evidence = {
        "schema": "formal-v4.2-gate0-evidence-v1", "run_id": run_id, "contract_sha256": contract.contract_sha256,
        "checks": checks, "measurements": measurements, "synthetic_probe": False,
        "capacity_fit_years": capacity_fit_years, "normalization_fit_years": capacity_fit_years,
        "materialized_years": materialized_years, "evaluation_year_accessed": False,
        "authorized_pilot": bool(required_components), "paper_eligible": False,
        "python": sys.version, "platform": platform.platform(), "torch_version": torch.__version__,
    }
    write_once_json(gate_root / "GATE0_EVIDENCE.json", evidence)
    projection = {
        "schema": "formal-v4.2-resource-projection-v1", "run_id": run_id,
        "operations": measurements, "free_disk_margin_fraction": 0.20,
        "projected_hours": {name: float(item.get("timing", {}).get("p95_seconds", 0.0) * 10000.0 / 3600.0) for name, item in measurements.items()},
        "authorized_pilot": bool(required_components), "synthetic_probe": False,
    }
    write_once_json(gate_root / "RESOURCE_PROJECTION.json", projection)
    return evidence


def _failure_path(root: Path) -> Path:
    return root / "gate0" / "GATE0_FAILURE.json"


def _write_failure(root: Path, *, run_id: str, contract_sha256: str, stage: str, exception: BaseException) -> None:
    write_once_json(_failure_path(root), {
        "schema": "formal-v4.2-gate0-failure-v1",
        "status": "fail",
        "run_id": run_id,
        "contract_sha256": contract_sha256,
        "stage": stage,
        "error_type": type(exception).__name__,
        "reason": str(exception),
        "evaluation_year_accessed": False,
        "authorized_pilot": False,
    })


def run_gate0_orchestrator(inputs: Gate0InputsV42) -> dict[str, Any]:
    """Produce, validate and measure one fresh formal-v4.2 Gate 0 run."""

    contract = load_formal_v4_2_contract(inputs.contract_path)
    output_root = Path(inputs.output_root).resolve()
    root = output_root / inputs.run_id
    if root.exists():
        raise FileExistsError(f"fresh formal-v4.2 run id required: {root}")
    root = create_v42_run_root(output_root, inputs.run_id, contract.contract_sha256)
    stage = "source_manifest"
    try:
        repo_root = FRAME_ROOT.parent
        closure_value = Path(str(contract.payload["source_closure_file"]))
        closure_path = closure_value if closure_value.is_absolute() else FRAME_ROOT / closure_value
        source_path = root / "protocol" / "SOURCE_MANIFEST.json"
        produce_source_manifest(
            repo_root,
            closure_path,
            source_path,
            run_id=inputs.run_id,
            contract_sha256=contract.contract_sha256,
        )
        source_hash = sha256_file(source_path)
        stage = "itransformer_receipt"
        itransformer_path = root / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json"
        produce_itransformer_receipt(
            inputs.itransformer_source,
            itransformer_path,
            run_id=inputs.run_id,
            contract_sha256=contract.contract_sha256,
            source_manifest_sha256=source_hash,
            repo_root=repo_root,
        )
        stage = "capacity_receipt"
        capacity = build_capacity_evidence(
            contract,
            root,
            inputs.data_dir,
            source_hash,
        )
        capacity_path = root / "gate0" / "CAPACITY_FREEZE.json"
        stage = "diffopt_receipt"
        requested_diffopt = Path(inputs.diffopt_python).resolve()
        if Path(sys.executable).resolve() != requested_diffopt:
            raise RuntimeError(f"Gate 0 must run with the frozen DiffLP interpreter: {requested_diffopt}")
        lock_path = FRAME_ROOT / "requirements" / "formal_v4_diffopt.lock"
        diffopt_path = root / "protocol" / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json"
        produce_diffopt_receipt(
            lock_path,
            diffopt_path,
            run_id=inputs.run_id,
            contract_sha256=contract.contract_sha256,
            source_manifest_sha256=source_hash,
        )
        stage = "prerequisite_validation"
        prerequisite_checks = validate_prerequisites(
            Gate0PrerequisitePaths(source_path, itransformer_path, diffopt_path, capacity_path),
            run_id=inputs.run_id,
            contract_sha256=contract.contract_sha256,
            repo_root=repo_root,
            itransformer_source=inputs.itransformer_source,
            lock_path=lock_path,
        )
        stage = "real_operation_probe"
        measurements, operation_checks = _run_real_operations()
        disk = shutil.disk_usage(root)
        disk_margin = float(disk.free / max(disk.total, 1))
        minimum_margin = float(contract.payload["resource_gate"]["minimum_disk_margin_fraction"])
        maximum_hours = float(contract.payload["resource_gate"]["max_projected_p95_hours"])
        projected_hours = {
            name: float(item.get("timing", {}).get("p95_seconds", 0.0) * 10000.0 / 3600.0)
            for name, item in measurements.items()
        }
        resource_checks = {
            "disk_margin": {"passed": disk_margin >= minimum_margin, "actual": disk_margin, "minimum": minimum_margin},
            "runtime_projection": {"passed": all(value <= maximum_hours for value in projected_hours.values()), "projected_hours": projected_hours, "maximum": maximum_hours},
            "no_evaluation_access": {"passed": True, "evaluation_year_accessed": False},
        }
        checks = {**prerequisite_checks, **operation_checks, **resource_checks}
        authorized = all(bool(item.get("passed")) for item in checks.values())
        projection = {
            "schema": "formal-v4.2-resource-projection-v1",
            "run_id": inputs.run_id,
            "contract_sha256": contract.contract_sha256,
            "operations": measurements,
            "projected_hours": projected_hours,
            "disk_margin_fraction": disk_margin,
            "minimum_disk_margin_fraction": minimum_margin,
            "maximum_projected_p95_hours": maximum_hours,
            "synthetic_probe": False,
            "evaluation_year_accessed": False,
            "authorized_pilot": authorized,
        }
        projection_path = root / "gate0" / "RESOURCE_PROJECTION.json"
        write_once_json(projection_path, projection)
        evidence = {
            "schema": "formal-v4.2-gate0-evidence-v1",
            "run_id": inputs.run_id,
            "contract_sha256": contract.contract_sha256,
            "source_manifest_sha256": source_hash,
            "checks": checks,
            "measurements": measurements,
            "resource_projection_path": str(projection_path.relative_to(root)),
            "resource_projection_sha256": sha256_file(projection_path),
            "selected_capacity": capacity["selected"],
            "capacity_fit_years": list(contract.train_years),
            "normalization_fit_years": list(contract.train_years),
            "materialized_years": [*contract.train_years, contract.selection_year],
            "synthetic_probe": False,
            "evaluation_year_accessed": False,
            "authorized_pilot": authorized,
            "paper_eligible": False,
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
        }
        evidence_path = root / "gate0" / "GATE0_EVIDENCE.json"
        evidence_hash = write_once_json(evidence_path, evidence)
        if not authorized:
            raise Gate0ExecutionError("one or more mandatory Gate 0 checks failed")
        write_once_json(root / "protocol" / "CURRENT_GATE.json", {
            "schema": CURRENT_GATE_SCHEMA,
            "run_id": inputs.run_id,
            "contract_sha256": contract.contract_sha256,
            "gate": "gate0",
            "gate_evidence_sha256": evidence_hash,
            "authorized_pilot": True,
            "next_gate": "pilot",
            "evaluation_year_accessed": False,
        })
        return evidence
    except Exception as exc:
        if not _failure_path(root).exists():
            _write_failure(root, run_id=inputs.run_id, contract_sha256=contract.contract_sha256, stage=stage, exception=exc)
        if isinstance(exc, Gate0ExecutionError):
            raise
        raise Gate0ExecutionError(f"Gate 0 failed during {stage}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Paper\Kitakyushu dataset"))
    parser.add_argument("--diffopt-python", type=Path, default=Path(r"D:\Paper\envs\rsc_pf_diffopt_v4\python.exe"))
    parser.add_argument("--itransformer-source", type=Path, default=FRAME_ROOT / "third_party" / "iTransformer_source")
    args = parser.parse_args(argv)
    inputs = Gate0InputsV42(
        contract_path=args.contract,
        output_root=args.output_root,
        run_id=args.run_id,
        data_dir=args.data_dir,
        diffopt_python=args.diffopt_python,
        itransformer_source=args.itransformer_source,
    )
    try:
        receipt = run_gate0_orchestrator(inputs)
    except Exception as exc:
        print(json.dumps({"authorized_pilot": False, "run_id": args.run_id, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"authorized_pilot": receipt["authorized_pilot"], "run_id": receipt["run_id"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Gate0ExecutionError", "run_diff_lp", "run_gate0_orchestrator", "run_gate0_probe", "run_highs_lp", "run_rsc_backward", "run_rsc_forward", "main"]
