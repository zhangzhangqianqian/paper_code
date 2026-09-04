"""Independently audit a formal-v4.1 Gate 0 receipt.

This module intentionally does not import or call the Gate 0 decision
function.  It reopens the receipt, recomputes hashes and checks the
authorization invariants from an independent implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


AUDITOR_SCHEMA = "formal-v4.1-independent-gate0-audit-v1"
MANDATORY_IDS: tuple[str, ...] = (
    "protocol_freeze", "source_manifest", "source_closure", "tracked_clean_closure",
    "invalid_run_registry", "benchmark_boundary", "benchmark_receipt",
    "capacity_audit_stratified", "capacity_audit_chronological", "materialized_data",
    "trajectory_physics", "normalization_receipt", "c_ref_receipt", "teacher_alignment",
    "curriculum", "gradient_boundary", "method_adapters", "itransformer_receipt",
    "diffopt_gate", "data_access", "archive_receipts", "regression_tests",
    "resource_projection", "test_artifact_absent", "no_evaluation_access",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _audit_payload(run_root: Path) -> dict[str, Any]:
    gate_root = run_root / "gate0" if (run_root / "gate0").is_dir() else run_root
    run_root = gate_root.parent
    receipt_path = gate_root / "GATE0_RECEIPT.json"
    diagnostic_path = gate_root / "GATE0_DIAGNOSTIC.json"
    receipt = _json(receipt_path)
    errors: list[str] = []
    if receipt.get("schema_version") != "formal-v4.1-gate0-receipt-v1":
        errors.append("receipt schema mismatch")
    ids = tuple(receipt.get("mandatory_check_ids", ()))
    if ids != MANDATORY_IDS:
        errors.append("mandatory check registry mismatch")
    checks = receipt.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(MANDATORY_IDS):
        errors.append("receipt checks are incomplete or contain unknown IDs")
    else:
        for check_id in MANDATORY_IDS:
            item = checks[check_id]
            if not isinstance(item, Mapping) or not isinstance(item.get("passed"), bool):
                errors.append(f"check {check_id} does not contain a strict boolean")
    if not diagnostic_path.is_file():
        errors.append("diagnostic receipt is missing")
    else:
        diagnostic = _json(diagnostic_path)
        expected = receipt.get("diagnostic_sha256")
        if expected != _sha256(diagnostic_path):
            errors.append("diagnostic receipt hash mismatch")
        if diagnostic.get("schema_version") != "formal-v4.1-gate0-receipt-v1":
            errors.append("diagnostic receipt schema mismatch")

    authorized = receipt.get("authorized_gate1") is True
    auth_path = gate_root / "GATE0_AUTHORIZATION.json"
    if authorized:
        if not auth_path.is_file():
            errors.append("authorization marker is missing")
        else:
            auth = _json(auth_path)
            if auth.get("schema_version") != "formal-v4.1-gate0-authorization-v1" or auth.get("authorized_gate1") is not True:
                errors.append("authorization marker is invalid")
            if auth.get("receipt_sha256") != _sha256(receipt_path):
                errors.append("authorization marker does not bind receipt")
    elif auth_path.exists():
        errors.append("failed Gate 0 contains an authorization marker")

    # Evaluation/test data must not be materialized below the run root.  This
    # check is deliberately independent of the access-controller implementation.
    forbidden: list[str] = []
    for path in run_root.rglob("*"):
        if path.is_file() and any(part.lower() in {"evaluation", "2020", "2021"} for part in path.relative_to(run_root).parts):
            forbidden.append(str(path.relative_to(run_root)))
    if forbidden:
        errors.append("evaluation artifact is present below run root")

    access_check = checks.get("data_access", {}) if isinstance(checks, Mapping) else {}
    access_evidence = access_check.get("evidence", {}) if isinstance(access_check, Mapping) else {}
    if access_evidence.get("test_set_accessed") is True:
        errors.append("access receipt reports test-set access")
    return {
        "schema_version": AUDITOR_SCHEMA,
        "run_root": str(run_root),
        "receipt_sha256": _sha256(receipt_path),
        "authorized_gate1": authorized,
        "verified": not errors,
        "errors": errors,
        "forbidden_artifacts": forbidden,
    }


def audit_gate0_run(run_root: str | Path, *, write_receipt: bool = True) -> dict[str, Any]:
    supplied = Path(run_root).resolve()
    root = supplied.parent if supplied.name.lower() == "gate0" else supplied
    payload = _audit_payload(root)
    if write_receipt:
        destination = root / "INDEPENDENT_GATE0_AUDIT.json"
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite independent audit receipt: {destination}")
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_gate0_run(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AUDITOR_SCHEMA", "audit_gate0_run", "main"]
