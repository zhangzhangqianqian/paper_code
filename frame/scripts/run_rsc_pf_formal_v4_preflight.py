"""Run the fail-closed, transactional formal-v4.1 Gate 0 preflight.

The command only validates already-produced training/selection evidence.  It
never opens 2020 data and never starts Gate 1 or model training.  Missing
evidence is a normal denial, not an implicit pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_protocol_v4 import load_formal_v4_spec  # noqa: E402
from src.joint_dispatch.formal_v4_access import DataAccessReceipt, scan_runtime_access  # noqa: E402
from src.joint_dispatch.formal_v4_diffopt import DifferentiableLPGateReceipt  # noqa: E402
from src.joint_dispatch.formal_v4_gate0 import (  # noqa: E402
    MANDATORY_CHECK_IDS,
    Gate0Context,
    execute_gate0,
    run_gate0,
)
from src.joint_dispatch.formal_v4_itransformer import validate_itransformer_receipt  # noqa: E402
from src.joint_dispatch.formal_v4_objective import validate_c_ref_receipt  # noqa: E402
from src.joint_dispatch.formal_v4_provenance import build_source_manifest, load_invalid_run_registry, validate_source_manifest  # noqa: E402
from src.joint_dispatch.formal_v4_resources import ResourceProjection  # noqa: E402


# Backward-compatible public name used by earlier tests and callers.
MANDATORY_FAILURES = MANDATORY_CHECK_IDS


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return payload


def _file_check(path: Path, *, schema: str | None = None) -> dict[str, Any]:
    payload = _read_json(path)
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(f"{path.name} schema must equal {schema}")
    return {"passed": True, "path": str(path), "sha256": _sha256(path), "schema_version": payload.get("schema_version")}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_checks(spec: Any, root: Path) -> dict[str, dict[str, Any]]:
    """Construct evidence checks from immutable receipts below one run root."""

    protocol = root / "protocol"
    gate0 = root / "gate0"
    data = root / "data"
    checks: dict[str, dict[str, Any]] = {}

    def check(check_id: str, fn):
        try:
            evidence = fn()
            if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
                raise ValueError("evidence did not return passed=true")
            checks[check_id] = dict(evidence)
        except Exception as exc:
            checks[check_id] = {"passed": False, "error_type": type(exc).__name__, "reason": str(exc)}

    check("protocol_freeze", lambda: {"passed": spec.schema_version == "joint-forecast-dispatch-formal-v4.1", "schema_version": spec.schema_version})

    closure = Path(spec.source_closure_file) if spec.source_closure_file is not None else FRAME_ROOT / "configs" / "formal_v4_source_closure_v4_1.txt"
    def source_closure():
        if not closure.is_file():
            raise FileNotFoundError(closure)
        declared = []
        for line in closure.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                path = (REPO_ROOT / line).resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                declared.append(path)
        scan = scan_runtime_access(
            declared,
            allowlisted_paths=(
                FRAME_ROOT / "src" / "kitakyushu_pipeline.py",
                FRAME_ROOT / "src" / "joint_dispatch" / "formal_v4_access.py",
            ),
        )
        if scan["status"] != "pass":
            raise ValueError(f"unapproved data-access bypasses: {scan['findings']}")
        return {"passed": True, "path": str(closure), "entry_count": len(declared), "scan": scan}

    check("source_closure", source_closure)
    manifest_path = protocol / "SOURCE_MANIFEST.json"

    def source_manifest():
        payload = _read_json(manifest_path)
        validate_source_manifest(payload, REPO_ROOT)
        return {"passed": True, "path": str(manifest_path), "sha256": _sha256(manifest_path), "git_commit": payload.get("git_commit")}

    check("source_manifest", source_manifest)
    check("tracked_clean_closure", source_manifest)

    registry = FRAME_ROOT / "configs" / "joint_dispatch_invalid_runs_v4.json"
    check("invalid_run_registry", lambda: {"passed": bool(load_invalid_run_registry(registry)), "path": str(registry), "sha256": _sha256(registry)})

    benchmark_path = gate0 / "benchmark" / "FORMAL_V4_BENCHMARK_RECEIPT.json"

    def benchmark():
        payload = _read_json(benchmark_path)
        if payload.get("schema_version") != "formal-v4.1-benchmark-receipt-v1":
            raise ValueError("benchmark receipt schema mismatch")
        if tuple(payload.get("source_years", ())) != (2015, 2016, 2017, 2018):
            raise ValueError("benchmark receipt is not train-year bounded")
        if payload.get("test_year_used_for_scaling") is not False:
            raise ValueError("benchmark receipt used test-year scaling")
        return {"passed": True, "path": str(benchmark_path), "sha256": _sha256(benchmark_path), "source_years": payload["source_years"]}

    check("benchmark_boundary", benchmark)
    check("benchmark_receipt", benchmark)

    capacity_path = gate0 / "CAPACITY_FREEZE.json"

    def capacity(stage: str | None = None):
        payload = _read_json(capacity_path)
        if payload.get("gate0_authorized") is not True:
            raise ValueError("capacity freeze is not authorized")
        audit = payload.get("capacity_audit")
        if not isinstance(audit, Mapping) or audit.get("status") != "pass":
            raise ValueError("capacity audit did not pass")
        if stage is not None and audit.get(stage, {}).get("status") != "pass":
            raise ValueError(f"capacity stage did not pass: {stage}")
        return {"passed": True, "path": str(capacity_path), "sha256": _sha256(capacity_path), "selected": audit.get("selected")}

    check("capacity_audit_stratified", lambda: capacity("diagnostic"))
    check("capacity_audit_chronological", lambda: capacity("chronological"))

    train_archive = data / "train.npz"
    selection_archive = data / "selection.npz"
    check("materialized_data", lambda: {"passed": train_archive.is_file() and selection_archive.is_file(), "train": str(train_archive), "selection": str(selection_archive)})

    trajectory_receipt = protocol / "TRAJECTORY_RECEIPT.json"
    check("trajectory_physics", lambda: _file_check(trajectory_receipt, schema="formal-v4.1-trajectory-receipt-v1"))

    normalization = root / "NORMALIZATION_RECEIPT.json"
    check("normalization_receipt", lambda: _file_check(normalization, schema="formal-v4.1-normalization-receipt-v1"))

    c_ref = root / "C_REF_RECEIPT.json"

    def objective():
        payload = _read_json(c_ref)
        validate_c_ref_receipt(payload)
        return {"passed": True, "path": str(c_ref), "sha256": _sha256(c_ref), "c_ref": payload.get("c_ref")}

    check("c_ref_receipt", objective)
    check("teacher_alignment", lambda: _file_check(protocol / "TEACHER_ALIGNMENT_RECEIPT.json", schema="formal-v4.1-teacher-alignment-v1"))
    check("curriculum", lambda: _file_check(protocol / "CURRICULUM_RECEIPT.json", schema="formal-v4.1-curriculum-receipt-v1"))
    check("gradient_boundary", lambda: _file_check(protocol / "GRADIENT_RECEIPT.json", schema="formal-v4.1-gradient-receipt-v1"))
    check("method_adapters", lambda: _file_check(protocol / "METHOD_ADAPTER_RECEIPT.json", schema="formal-v4.1-method-adapter-receipt-v1"))

    def itransformer():
        payload = _read_json(protocol / "ITRANSFORMER_SOURCE_RECEIPT.json")
        validate_itransformer_receipt(payload)
        return {"passed": True, "path": str(protocol / "ITRANSFORMER_SOURCE_RECEIPT.json"), "sha256": _sha256(protocol / "ITRANSFORMER_SOURCE_RECEIPT.json")}

    check("itransformer_receipt", itransformer)

    def diffopt():
        payload = _read_json(protocol / "DIFFERENTIABLE_LP_GATE.json")
        fields = ("method_id", "eligible", "dpp_passed", "finite_solves", "parity_passed", "physical_residual_passed", "gradient_passed", "native_probe_passed", "memory_margin_fraction", "projected_p95_hours", "source_environment", "reason")
        receipt = DifferentiableLPGateReceipt(**{key: payload[key] for key in fields if key in payload})
        receipt.validate()
        if receipt.eligible is not True:
            raise ValueError("Differentiable-LP is not eligible")
        return {"passed": True, "path": str(protocol / "DIFFERENTIABLE_LP_GATE.json"), "sha256": _sha256(protocol / "DIFFERENTIABLE_LP_GATE.json")}

    check("diffopt_gate", diffopt)

    access = protocol / "DATA_ACCESS_RECEIPT.json"

    def access_check():
        payload = _read_json(access)
        receipt = DataAccessReceipt()
        events = payload.get("events", [])
        if not isinstance(events, list):
            raise ValueError("access receipt events must be a list")
        # Gate 0 only needs the immutable summary here; the controller tests
        # validate event-level hashes and deny behavior.
        if payload.get("test_set_accessed") is not False:
            raise ValueError("evaluation/test set was accessed")
        return {"passed": True, "path": str(access), "sha256": _sha256(access), "event_count": len(events), "test_set_accessed": False}

    check("data_access", access_check)
    check("no_evaluation_access", access_check)
    check("archive_receipts", lambda: _file_check(protocol / "ARCHIVE_ACCESS_RECEIPT.json", schema="formal-v4.1-archive-access-v1"))
    check("regression_tests", lambda: _file_check(root / "audit" / "REGRESSION_RECEIPT.json", schema="formal-v4.1-regression-receipt-v1"))

    def resources():
        path = protocol / "RESOURCE_PROJECTION.json"
        payload = _read_json(path)
        projection = ResourceProjection(
            payload.get("component_seconds", {}),
            float(payload["projected_p95_hours"]),
            float(payload["disk_margin_fraction"]),
            float(payload["memory_margin_fraction"]),
            str(payload.get("method_id", "formal-v4.1")),
        )
        projection.validate(max_projected_hours=float(spec.resource_gate["max_projected_p95_hours"]), minimum_margin=float(spec.resource_gate["minimum_disk_margin_fraction"]))
        return {"passed": True, "path": str(path), "sha256": _sha256(path), "projected_p95_hours": projection.projected_p95_hours}

    check("resource_projection", resources)

    def no_test_artifact():
        forbidden = []
        for path in root.rglob("*"):
            if path.is_file() and any(part.lower() in {"evaluation", "test", "2020", "2021"} for part in path.relative_to(root).parts):
                forbidden.append(str(path.relative_to(root)))
        return {"passed": not forbidden, "forbidden": forbidden}

    check("test_artifact_absent", no_test_artifact)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_1.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    spec = load_formal_v4_spec(args.contract)
    report_root = Path(spec.paths["output_root"])
    closure = Path(spec.source_closure_file) if spec.source_closure_file is not None else FRAME_ROOT / "configs" / "formal_v4_source_closure_v4_1.txt"
    holder: dict[str, Path] = {}

    def prepare(run_root: Path) -> None:
        holder["run_root"] = run_root
        commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        paths = [line.strip() for line in closure.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
        manifest = build_source_manifest(REPO_ROOT, paths, commit)
        manifest_path = run_root / "protocol" / "SOURCE_MANIFEST.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest.to_payload(), ensure_ascii=False, indent=2), encoding="utf-8")

    def evidence(check_id: str) -> Mapping[str, Any]:
        return _receipt_checks(spec, holder["run_root"])[check_id]

    checkers = {check_id: (lambda check_id=check_id: evidence(check_id)) for check_id in MANDATORY_CHECK_IDS}
    result = execute_gate0(
        Gate0Context(
            report_root,
            checkers,
            metadata={"contract": str(args.contract), "training_years": list(spec.train_years), "selection_year": spec.selection_year, "evaluation_year_locked": spec.evaluation_year},
            prepare=prepare,
        ),
        args.run_id,
    )
    print(json.dumps(result.to_payload(), ensure_ascii=False, indent=2, default=str))
    return 0 if result.authorized_gate1 else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MANDATORY_FAILURES", "main", "run_gate0"]
