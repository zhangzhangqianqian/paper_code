"""Transactional, fail-closed Gate 0 orchestration for formal-v4.1.

Gate 0 is intentionally an authorization boundary rather than an experiment
runner.  It evaluates a complete, closed set of evidence-producing checks,
keeps diagnostics when a check fails, and writes the authorization marker only
after the final receipt has been committed to a fresh run directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable, Mapping


GATE0_SCHEMA = "formal-v4.1-gate0-receipt-v1"
GATE0_AUTHORIZATION_SCHEMA = "formal-v4.1-gate0-authorization-v1"

# This registry is deliberately exhaustive.  Adding a new scientific
# prerequisite requires changing the registry and its tests, rather than
# silently adding an optional boolean to a receipt.
MANDATORY_CHECK_IDS: tuple[str, ...] = (
    "protocol_freeze",
    "source_manifest",
    "source_closure",
    "tracked_clean_closure",
    "invalid_run_registry",
    "benchmark_boundary",
    "benchmark_receipt",
    "capacity_audit_stratified",
    "capacity_audit_chronological",
    "materialized_data",
    "trajectory_physics",
    "normalization_receipt",
    "c_ref_receipt",
    "teacher_alignment",
    "curriculum",
    "gradient_boundary",
    "method_adapters",
    "itransformer_receipt",
    "diffopt_gate",
    "data_access",
    "archive_receipts",
    "regression_tests",
    "resource_projection",
    "test_artifact_absent",
    "no_evaluation_access",
)


def _safe_run_id(run_id: str) -> str:
    value = str(run_id)
    if not value or value in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("run_id must be a simple non-empty directory name")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Gate 0 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(encoded)


@dataclass(frozen=True)
class Gate0CheckResult:
    check_id: str
    passed: bool
    evidence: Mapping[str, Any]

    def to_payload(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "passed": bool(self.passed), "evidence": dict(self.evidence)}


@dataclass(frozen=True)
class Gate0Result:
    run_id: str
    run_root: Path
    authorized_gate1: bool
    checks: Mapping[str, Gate0CheckResult]
    receipt_path: Path | None = None
    authorization_path: Path | None = None

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(check_id for check_id, check in self.checks.items() if not check.passed)

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": GATE0_SCHEMA,
            "run_id": self.run_id,
            "authorized_gate1": bool(self.authorized_gate1),
            "mandatory_check_ids": list(MANDATORY_CHECK_IDS),
            "checks": {key: value.to_payload() for key, value in self.checks.items()},
            "mandatory_failures": list(self.failures),
        }


@dataclass(frozen=True)
class Gate0Context:
    """Inputs to the orchestrator.

    ``checkers`` are small evidence-producing functions.  A checker may return
    a mapping containing a strict boolean ``passed`` field, or a bare boolean.
    Exceptions are converted into a failed check and retained in diagnostics.
    """

    output_root: Path
    checkers: Mapping[str, Callable[[], Mapping[str, Any] | bool]]
    metadata: Mapping[str, Any] = ()
    expected_check_ids: tuple[str, ...] = MANDATORY_CHECK_IDS
    prepare: Callable[[Path], Any] | None = None


def _coerce_check(check_id: str, value: Mapping[str, Any] | bool) -> Gate0CheckResult:
    if isinstance(value, bool):
        return Gate0CheckResult(check_id, value, {})
    if not isinstance(value, Mapping) or "passed" not in value or not isinstance(value["passed"], bool):
        raise TypeError("checker must return a bool or a mapping with boolean 'passed'")
    evidence = {str(key): item for key, item in value.items() if key != "passed"}
    return Gate0CheckResult(check_id, bool(value["passed"]), evidence)


def evaluate_gate0(checkers: Mapping[str, Callable[[], Mapping[str, Any] | bool]], *, expected_check_ids: tuple[str, ...] = MANDATORY_CHECK_IDS) -> dict[str, Gate0CheckResult]:
    """Evaluate all checks without writing files (used by unit tests)."""

    expected = tuple(expected_check_ids)
    if len(expected) != len(set(expected)) or any(not item for item in expected):
        raise ValueError("Gate 0 check registry must contain unique non-empty IDs")
    actual = set(checkers)
    missing = set(expected) - actual
    unknown = actual - set(expected)
    if missing or unknown:
        raise ValueError(f"Gate 0 check registry mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    results: dict[str, Gate0CheckResult] = {}
    for check_id in expected:
        try:
            results[check_id] = _coerce_check(check_id, checkers[check_id]())
        except Exception as exc:  # diagnostics are part of the fail-closed contract
            results[check_id] = Gate0CheckResult(check_id, False, {"error_type": type(exc).__name__, "error": str(exc)})
    return results


def execute_gate0(context: Gate0Context, run_id: str) -> Gate0Result:
    """Run Gate 0 transactionally below one fresh ``run_id/gate0`` tree."""

    safe_id = _safe_run_id(run_id)
    output_root = Path(context.output_root).resolve()
    gate_root = output_root / safe_id / "gate0"
    if gate_root.exists():
        raise FileExistsError(f"refusing to overwrite Gate 0 output: {gate_root}")
    gate_root.mkdir(parents=True)
    marker = gate_root / "IN_PROGRESS.json"
    marker.write_text(json.dumps({"schema_version": "formal-v4.1-gate0-in-progress-v1", "run_id": safe_id}, indent=2), encoding="utf-8")

    try:
        if context.prepare is not None:
            context.prepare(gate_root.parent)
        checks = evaluate_gate0(context.checkers, expected_check_ids=context.expected_check_ids)
        authorized = all(check.passed for check in checks.values())
        result = Gate0Result(safe_id, gate_root, authorized, checks)
        diagnostics = result.to_payload()
        diagnostics["metadata"] = dict(context.metadata) if isinstance(context.metadata, Mapping) else {}
        diagnostics["in_progress_marker_sha256"] = _sha256_bytes(marker.read_bytes())
        _write_immutable(gate_root / "GATE0_DIAGNOSTIC.json", diagnostics)

        receipt_payload = result.to_payload()
        receipt_payload.update({
            "metadata": dict(context.metadata) if isinstance(context.metadata, Mapping) else {},
            "diagnostic_sha256": _sha256_bytes((gate_root / "GATE0_DIAGNOSTIC.json").read_bytes()),
        })
        receipt_path = gate_root / "GATE0_RECEIPT.json"
        _write_immutable(receipt_path, receipt_payload)
        authorization_path: Path | None = None
        if authorized:
            authorization_path = gate_root / "GATE0_AUTHORIZATION.json"
            auth_payload = {
                "schema_version": GATE0_AUTHORIZATION_SCHEMA,
                "run_id": safe_id,
                "authorized_gate1": True,
                "receipt_sha256": _sha256_bytes(receipt_path.read_bytes()),
            }
            _write_immutable(authorization_path, auth_payload)
        marker.unlink(missing_ok=True)
        return Gate0Result(safe_id, gate_root, authorized, checks, receipt_path, authorization_path)
    except Exception:
        # Keep the marker and all diagnostics written so far.  The absence of
        # GATE0_AUTHORIZATION.json is the fail-closed signal.
        raise


def run_gate0(fixture: Any) -> Any:
    """Compatibility helper for lightweight tests and callers.

    The old implementation accepted a set of injected failure IDs.  Preserve
    that test seam while routing through the same exhaustive registry and
    strict evaluator used by the transactional runner.
    """

    failures = set(getattr(fixture, "failures", set()))
    checkers = {
        check_id: (lambda check_id=check_id: {"passed": check_id not in failures})
        for check_id in MANDATORY_CHECK_IDS
    }
    checks = evaluate_gate0(checkers)
    return SimpleNamespace(
        authorized_gate1=all(check.passed for check in checks.values()),
        failures=tuple(check_id for check_id, check in checks.items() if not check.passed),
        checks=checks,
    )


__all__ = [
    "GATE0_AUTHORIZATION_SCHEMA", "GATE0_SCHEMA", "Gate0CheckResult", "Gate0Context", "Gate0Result",
    "MANDATORY_CHECK_IDS", "evaluate_gate0", "execute_gate0", "run_gate0",
]
