"""Fail-closed execution and independent audit primitives for formal RSC-PF runs.

The runner in this module is intentionally small and data-agnostic.  It can
exercise the gate protocol with a synthetic row matrix, but it never discovers
or reads real 2019/2020 arrays by itself.  A production runner must pass an
explicit, lineage-stamped row mapping to these functions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .complete_formal_contract import CompleteFormalContract, MethodSeedKey


@dataclass(frozen=True)
class GateDecision:
    stage: str
    authorized_next_gate: bool
    expected_rows: tuple[MethodSeedKey, ...]
    missing_rows: tuple[str, ...]
    failures: tuple[str, ...]
    receipt_sha256: str

    @property
    def authorized_gate2(self) -> bool:
        """Compatibility alias used by gate-transition tests."""

        return self.authorized_next_gate


@dataclass(frozen=True)
class AuditedRow:
    key: MethodSeedKey
    inference_lp_calls: int
    origin_count: int
    finite: bool
    chronological: bool
    physical_feasible: bool
    shortage_total: float
    evaluation_year_accessed: bool
    excluded_year_accessed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class AuditReceipt:
    stage: str
    status: str
    execution_receipt_sha256: str
    recomputed_row_count: int
    failures: tuple[str, ...]
    evaluation_year_accessed: bool
    excluded_year_accessed: bool
    rows: Mapping[str, AuditedRow]


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, MethodSeedKey):
        return {"method_id": value.method_id, "seed": value.seed}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256(value: Any) -> str:
    payload = json.dumps(_canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _key(value: Any) -> MethodSeedKey:
    if isinstance(value, MethodSeedKey):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        return MethodSeedKey(str(value[0]), None if value[1] is None else int(value[1]))
    if isinstance(value, str) and "/" in value:
        method, seed = value.rsplit("/", 1)
        return MethodSeedKey(method, None if seed == "deterministic" else int(seed))
    if isinstance(value, Mapping) and "method_id" in value:
        seed = value.get("seed")
        return MethodSeedKey(str(value["method_id"]), None if seed is None else int(seed))
    raise TypeError(f"unsupported formal row key: {value!r}")


def _label(key: MethodSeedKey) -> str:
    return f"{key.method_id}/{key.seed if key.seed is not None else 'deterministic'}"


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _normalize_rows(rows: Mapping[Any, Any]) -> dict[MethodSeedKey, Any]:
    if not isinstance(rows, Mapping):
        raise TypeError("formal rows must be a mapping")
    return {_key(key): value for key, value in rows.items()}


def synthetic_rows(contract: CompleteFormalContract, stage: str = "gate1", origin_count: Optional[int] = None) -> dict[tuple[str, Optional[int]], dict[str, Any]]:
    """Return metadata-only rows for gate tests; no data arrays are loaded."""

    if stage not in {"gate1", "gate2"}:
        raise ValueError("stage must be gate1 or gate2")
    count = int(origin_count if origin_count is not None else (contract.selection_origin_count if stage == "gate1" else contract.evaluation_origin_count))
    rows: dict[tuple[str, Optional[int]], dict[str, Any]] = {}
    for key in contract.expected_rows(stage):
        spec = contract.method(key.method_id)
        rows[(key.method_id, key.seed)] = {
            "method_id": key.method_id,
            "seed": key.seed,
            "stage": stage,
            "synthetic": True,
            "paper_result": False,
            "complete": True,
            "origin_count": count,
            "inference_lp_calls": spec.inference_lp_calls_per_origin * count,
            "finite": True,
            "chronological": True,
            "physical_feasible": True,
            "shortage_total": 0.0,
            "evaluation_year_accessed": False,
            "excluded_year_accessed": False,
            "test_set_accessed": False,
        }
    return rows


def decide_gate(contract: CompleteFormalContract, rows: Mapping[Any, Any], *, stage: str) -> GateDecision:
    expected = contract.expected_rows(stage)
    normalized = _normalize_rows(rows)
    failures: list[str] = []
    missing: list[str] = []
    expected_set = set(expected)
    for key in expected:
        row = normalized.get(key)
        if row is None:
            missing.append(_label(key))
            continue
        if _row_value(row, "complete", True) is not True:
            failures.append(f"{_label(key)} is incomplete")
        if _row_value(row, "stage", stage) != stage:
            failures.append(f"{_label(key)} has stage {_row_value(row, 'stage')!r}")
        if _row_value(row, "evaluation_year_accessed", False) is not False:
            failures.append(f"{_label(key)} accessed the evaluation year")
        if _row_value(row, "excluded_year_accessed", False) is not False:
            failures.append(f"{_label(key)} accessed an excluded year")
        if _row_value(row, "test_set_accessed", False) is not False:
            failures.append(f"{_label(key)} accessed the test set")
    for key in sorted(set(normalized) - expected_set, key=_label):
        failures.append(f"unexpected row {_label(key)}")
    receipt = _sha256({"stage": stage, "expected": expected, "rows": normalized, "missing": missing, "failures": failures})
    return GateDecision(stage, not missing and not failures, expected, tuple(missing), tuple(failures), receipt)


def decide_gate1(contract: CompleteFormalContract, rows: Mapping[Any, Any]) -> GateDecision:
    return decide_gate(contract, rows, stage="gate1")


def decide_gate2(contract: CompleteFormalContract, rows: Mapping[Any, Any]) -> GateDecision:
    return decide_gate(contract, rows, stage="gate2")


def _audit_one(contract: CompleteFormalContract, key: MethodSeedKey, row: Any, origin_count: int, stage: str) -> AuditedRow:
    spec = contract.method(key.method_id)
    expected_calls = spec.inference_lp_calls_per_origin * origin_count
    declared_calls = int(_row_value(row, "inference_lp_calls", expected_calls))
    declared_origins = int(_row_value(row, "origin_count", origin_count))
    finite = bool(_row_value(row, "finite", True))
    chronological = bool(_row_value(row, "chronological", True))
    feasible = bool(_row_value(row, "physical_feasible", True))
    shortage = float(_row_value(row, "shortage_total", 0.0))
    eval_access = bool(_row_value(row, "evaluation_year_accessed", False))
    excluded_access = bool(_row_value(row, "excluded_year_accessed", False))
    failures: list[str] = []
    if declared_origins != origin_count:
        failures.append(f"origin count {declared_origins} != {origin_count}")
    if declared_calls != expected_calls:
        failures.append(f"inference LP calls {declared_calls} != {expected_calls}")
    if not finite:
        failures.append("non-finite metric or rollout value")
    if not chronological:
        failures.append("chronology/state-carry audit failed")
    if not feasible:
        failures.append("physical feasibility audit failed")
    if shortage < 0 or not (shortage < float("inf")):
        failures.append("invalid shortage value")
    if eval_access:
        failures.append("evaluation year was accessed")
    if excluded_access:
        failures.append("excluded year was accessed")
    return AuditedRow(key, declared_calls, declared_origins, finite, chronological, feasible, shortage, eval_access, excluded_access, tuple(failures))


def audit_rows(contract: CompleteFormalContract, rows: Mapping[Any, Any], *, origin_count: int, stage: str = "gate1", execution_receipt_sha256: Optional[str] = None) -> AuditReceipt:
    """Recompute row accounting independently of the training runner."""

    if stage not in {"gate1", "gate2"}:
        raise ValueError("stage must be gate1 or gate2")
    normalized = _normalize_rows(rows)
    expected = contract.expected_rows(stage)
    audited: dict[str, AuditedRow] = {}
    failures: list[str] = []
    for key in expected:
        row = normalized.get(key)
        if row is None:
            failures.append(f"missing row {_label(key)}")
            continue
        item = _audit_one(contract, key, row, int(origin_count), stage)
        audited[_label(key)] = item
        failures.extend(f"{_label(key)}: {failure}" for failure in item.failures)
    for key in sorted(set(normalized) - set(expected), key=_label):
        failures.append(f"unexpected row {_label(key)}")
    evaluation_accessed = any(row.evaluation_year_accessed for row in audited.values())
    excluded_accessed = any(row.excluded_year_accessed for row in audited.values())
    status = "pass" if not failures and len(audited) == len(expected) else "fail"
    execution_hash = execution_receipt_sha256 or _sha256(normalized)
    return AuditReceipt(stage, status, execution_hash, len(audited), tuple(failures), evaluation_accessed, excluded_accessed, MappingProxyType(dict(audited)))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_canonical(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_row_artifacts(stage_dir: Path, key: MethodSeedKey, row: Any) -> None:
    name = key.method_id.replace(" ", "_").replace("/", "_")
    seed = str(key.seed) if key.seed is not None else "deterministic"
    row_dir = stage_dir / "rows" / name / seed
    row_dir.mkdir(parents=True, exist_ok=True)
    row_payload = dict(row) if isinstance(row, Mapping) else {"value": repr(row)}
    training_receipt = {"method_id": key.method_id, "seed": key.seed, "stage": row_payload.get("stage"), "synthetic": bool(row_payload.get("synthetic", False)), "row": row_payload}
    receipt_path = row_dir / "TRAINING_RECEIPT.json"
    if receipt_path.exists():
        try:
            previous = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PermissionError(f"cannot reuse unreadable row receipt: {receipt_path}") from exc
        if _canonical(previous.get("row")) != _canonical(row_payload):
            raise PermissionError(f"row lineage/hash mismatch; refusing reuse: {receipt_path}")
    else:
        _write_json(receipt_path, training_receipt)
    checkpoint = row_dir / "checkpoint.pt"
    if not checkpoint.exists():
        if bool(row_payload.get("synthetic", False)):
            checkpoint.write_bytes(b"synthetic-checkpoint-placeholder\n")
        else:
            raise PermissionError(f"real row is missing its checkpoint: {checkpoint}")
    evaluation_receipt = {"method_id": key.method_id, "seed": key.seed, "stage": row_payload.get("stage"), "metrics_status": "metadata-only" if row_payload.get("synthetic", False) else "provided"}
    evaluation_path = row_dir / "EVALUATION_RECEIPT.json"
    if evaluation_path.exists():
        try:
            previous_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PermissionError(f"cannot reuse unreadable evaluation receipt: {evaluation_path}") from exc
        if _canonical(previous_evaluation) != _canonical(evaluation_receipt):
            raise PermissionError(f"evaluation lineage mismatch; refusing reuse: {evaluation_path}")
    else:
        _write_json(evaluation_path, evaluation_receipt)


def _run_gate(contract: CompleteFormalContract, output_dir: str | Path, rows: Mapping[Any, Any], *, stage: str, run_id: str) -> tuple[GateDecision, AuditReceipt, Path]:
    decision = decide_gate(contract, rows, stage=stage)
    if not decision.authorized_next_gate:
        raise PermissionError(f"{stage} gate rejected: {decision.missing_rows or decision.failures}")
    normalized = _normalize_rows(rows)
    origin_count = contract.selection_origin_count if stage == "gate1" else contract.evaluation_origin_count
    audit = audit_rows(contract, normalized, origin_count=origin_count, stage=stage, execution_receipt_sha256=decision.receipt_sha256)
    if audit.status != "pass":
        raise PermissionError(f"{stage} independent audit rejected: {audit.failures}")
    stage_dir = Path(output_dir).resolve() / run_id / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    for key, row in normalized.items():
        _write_row_artifacts(stage_dir, key, row)
    _write_json(stage_dir / "EXECUTION_RECEIPT.json", {"stage": stage, "run_id": run_id, "synthetic": all(bool(_row_value(row, "synthetic", False)) for row in normalized.values()), "contract_sha256": contract.contract_sha256, "decision": decision, "rows": normalized})
    _write_json(stage_dir / "AUDIT_RECEIPT.json", {"stage": audit.stage, "status": audit.status, "execution_receipt_sha256": audit.execution_receipt_sha256, "recomputed_row_count": audit.recomputed_row_count, "failures": audit.failures, "evaluation_year_accessed": audit.evaluation_year_accessed, "excluded_year_accessed": audit.excluded_year_accessed})
    return decision, audit, stage_dir


def run_gate1(contract: CompleteFormalContract, output_dir: str | Path, rows: Optional[Mapping[Any, Any]] = None, *, run_id: str = "complete_formal_gate1_synthetic") -> tuple[GateDecision, AuditReceipt, Path]:
    """Run only the metadata/synthetic Gate 1 protocol."""

    payload = rows if rows is not None else synthetic_rows(contract, "gate1")
    decision, audit, stage_dir = _run_gate(contract, output_dir, payload, stage="gate1", run_id=run_id)
    normalized = _normalize_rows(payload)
    synthetic = any(_row_value(row, "synthetic", True) is not False for row in normalized.values())
    paper_result = all(_row_value(row, "paper_result", False) is True for row in normalized.values())
    authorized_gate2 = bool(audit.status == "pass" and not synthetic and paper_result)
    transition = {"schema_version": "rsc-pf-complete-formal-gate1-transition-v1", "contract_sha256": contract.contract_sha256, "authorized_gate2": authorized_gate2, "evaluation_year_accessed": False, "test_set_accessed": False, "execution_receipt_sha256": decision.receipt_sha256, "audit_status": audit.status, "synthetic": synthetic, "paper_result": paper_result}
    _write_json(stage_dir / "GATE1_TRANSITION.json", transition)
    guarded_decision = GateDecision(decision.stage, authorized_gate2, decision.expected_rows, decision.missing_rows, decision.failures, decision.receipt_sha256)
    return guarded_decision, audit, stage_dir


def run_gate2(contract: CompleteFormalContract, output_dir: str | Path, gate1_transition_path: str | Path, rows: Optional[Mapping[Any, Any]] = None, *, run_id: str = "complete_formal_gate2_synthetic") -> tuple[GateDecision, AuditReceipt, Path]:
    """Run Gate 2 only after a real, audited Gate 1 transition is accepted."""

    contract.authorize_evaluation(gate1_transition_path)
    if rows is None:
        raise PermissionError("Gate 2 requires explicit non-synthetic evaluated rows")
    normalized = _normalize_rows(rows)
    if any(_row_value(row, "synthetic", True) is not False for row in normalized.values()):
        raise PermissionError("Gate 2 refuses synthetic rows")
    if any(_row_value(row, "paper_result", False) is not True for row in normalized.values()):
        raise PermissionError("Gate 2 requires paper-result rows")
    payload = rows
    return _run_gate(contract, output_dir, payload, stage="gate2", run_id=run_id)


def audit_complete_formal(contract: CompleteFormalContract, rows: Mapping[Any, Any], *, stage: str, origin_count: int, execution_receipt_sha256: Optional[str] = None) -> AuditReceipt:
    return audit_rows(contract, rows, origin_count=origin_count, stage=stage, execution_receipt_sha256=execution_receipt_sha256)


__all__ = [
    "AuditReceipt",
    "AuditedRow",
    "GateDecision",
    "audit_complete_formal",
    "audit_rows",
    "decide_gate1",
    "decide_gate2",
    "run_gate1",
    "run_gate2",
    "synthetic_rows",
]
