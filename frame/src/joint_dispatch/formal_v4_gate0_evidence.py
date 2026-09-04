"""Content-aware validators for formal-v4.1 Gate 0 evidence receipts.

Gate 0 receipts are scientific evidence, not merely JSON markers.  This
module deliberately keeps validation independent from the evidence builders:
the preflight can reopen a receipt, verify its hashes and invariants, and
never execute the code that produced it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


GATE0_RECEIPT_SCHEMAS: dict[str, str] = {
    "trajectory": "formal-v4.1-trajectory-receipt-v1",
    "teacher_alignment": "formal-v4.1-teacher-alignment-v1",
    "curriculum": "formal-v4.1-curriculum-receipt-v1",
    "gradient": "formal-v4.1-gradient-receipt-v1",
    "method_adapter": "formal-v4.1-method-adapter-receipt-v1",
    "data_access": "formal-v4.1-data-access-v1",
    "archive_access": "formal-v4.1-archive-access-v1",
    "resource_projection": "formal-v4.1-resource-projection-v1",
}

FORMAL_V4_METHOD_IDS: tuple[str, ...] = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "State-Conditioned-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
    "Perfect-Information-MPC",
    "Seasonal-Naive-PTO",
)

STEP_WEIGHTS: tuple[float, ...] = (0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)
FORECAST_TASK_WEIGHTS: tuple[float, ...] = (1.0, 1.0, 1.0, 0.25)
_MAX_PHYSICAL_RESIDUAL = 1.0e-6
_MIN_RESOURCE_MARGIN = 0.20
_MAX_PROJECTED_HOURS = 24.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(payload: Mapping[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in payload]
    if missing:
        _fail(f"receipt is missing required fields: {', '.join(missing)}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a JSON object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(f"{name} must be a JSON array")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a non-empty string")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        _fail(f"{name} must be a strict boolean")
    return value


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        _fail(f"{name} must be finite")
    return result


def _sha256_string(value: Any, name: str) -> str:
    result = _string(value, name).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        _fail(f"{name} must be a lowercase SHA-256 digest")
    return result


def _commit_string(value: Any, name: str) -> str:
    result = _string(value, name).lower()
    if len(result) not in {40, 64} or any(char not in "0123456789abcdef" for char in result):
        _fail(f"{name} must be a hexadecimal Git commit digest")
    return result


def _resolve_inside(path: str, run_root: Path, name: str) -> Path:
    root = run_root.resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{name} points outside the Gate 0 run root") from exc
    return resolved


def _verify_artifact_hash(
    payload: Mapping[str, Any],
    *,
    path_field: str,
    hash_field: str,
    run_root: Path,
) -> None:
    path = _resolve_inside(_string(payload.get(path_field), path_field), run_root, path_field)
    expected = _sha256_string(payload.get(hash_field), hash_field)
    if not path.is_file():
        _fail(f"{path_field} does not identify a file: {path}")
    actual = _sha256(path)
    if actual != expected:
        _fail(f"{hash_field} does not match {path_field}")


def _validate_schema(payload: Mapping[str, Any], receipt_name: str) -> None:
    expected = GATE0_RECEIPT_SCHEMAS.get(receipt_name)
    if expected is None:
        _fail(f"unknown Gate 0 receipt name: {receipt_name}")
    if payload.get("schema_version") != expected:
        _fail(f"{receipt_name} schema must equal {expected}")


def _validate_common_probe(payload: Mapping[str, Any], name: str) -> None:
    _require(payload, "schema_version", "protocol_id", "source_commit", "probe_only", "test_set_accessed")
    _string(payload["protocol_id"], f"{name}.protocol_id")
    _commit_string(payload["source_commit"], f"{name}.source_commit")
    if _bool(payload["probe_only"], f"{name}.probe_only") is not True:
        _fail(f"{name} must be marked probe_only")
    if _bool(payload["test_set_accessed"], f"{name}.test_set_accessed") is not False:
        _fail(f"{name} reports evaluation/test-set access")


def _validate_trajectory(payload: Mapping[str, Any], run_root: Path) -> None:
    _validate_schema(payload, "trajectory")
    _require(
        payload,
        "protocol_id", "source_commit", "capacity_receipt_path", "capacity_receipt_sha256",
        "benchmark_path", "benchmark_sha256", "train_archive_path", "train_archive_sha256",
        "selection_archive_path", "selection_archive_sha256", "trajectory_id",
        "trajectory_path", "trajectory_file_sha256", "trajectory_sha256", "rule_version", "trajectory_audit",
    )
    for path_field, hash_field in (
        ("capacity_receipt_path", "capacity_receipt_sha256"),
        ("benchmark_path", "benchmark_sha256"),
        ("train_archive_path", "train_archive_sha256"),
        ("selection_archive_path", "selection_archive_sha256"),
        ("trajectory_path", "trajectory_file_sha256"),
    ):
        _verify_artifact_hash(payload, path_field=path_field, hash_field=hash_field, run_root=run_root)
    _string(payload["protocol_id"], "trajectory.protocol_id")
    _commit_string(payload["source_commit"], "trajectory.source_commit")
    _string(payload["trajectory_id"], "trajectory.trajectory_id")
    _sha256_string(payload["trajectory_sha256"], "trajectory.trajectory_sha256")
    if payload["rule_version"] != "formal-v4.1-causal-realized-settlement-v1":
        _fail("trajectory settlement rule is not the formal-v4.1 causal rule")
    audit = _mapping(payload["trajectory_audit"], "trajectory.trajectory_audit")
    required_audit = (
        "solved_hours", "settled_hours", "warmup_hours", "future_label_reads",
        "max_balance_residual", "max_conversion_residual", "max_soc_recursion_residual",
        "max_chp_ramp_violation", "max_renewable_availability_violation",
        "max_soc_bound_violation",
    )
    _require(audit, *required_audit)
    for key in ("solved_hours", "settled_hours"):
        if _finite(audit[key], f"trajectory_audit.{key}") <= 0:
            _fail(f"trajectory_audit.{key} must be positive")
    for key in ("warmup_hours", "future_label_reads"):
        if _finite(audit[key], f"trajectory_audit.{key}") < 0:
            _fail(f"trajectory_audit.{key} must be non-negative")
    if _finite(audit["future_label_reads"], "trajectory_audit.future_label_reads") != 0.0:
        _fail("trajectory contains future-label reads")
    for key in required_audit[4:]:
        if _finite(audit[key], f"trajectory_audit.{key}") > _MAX_PHYSICAL_RESIDUAL:
            _fail(f"trajectory physical audit exceeds {_MAX_PHYSICAL_RESIDUAL}: {key}")


def _validate_teacher_alignment(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, "teacher_alignment")
    _validate_common_probe(payload, "teacher_alignment")
    _require(
        payload, "production_overlay_deferred_until_stage_p", "stage_p_checkpoint_sha256",
        "window_count", "timestamps_aligned", "state_aligned", "lp_feasible",
        "forecast_shape", "dispatch_shape", "train_archive_sha256", "capacity_receipt_sha256",
        "normalization_sha256", "solver_sha256", "implementation_sha256", "production_overlay_written",
    )
    if _bool(payload["production_overlay_deferred_until_stage_p"], "teacher_alignment.production_overlay_deferred_until_stage_p") is not True:
        _fail("teacher alignment must defer production overlay until Stage P")
    if payload["stage_p_checkpoint_sha256"] is not None:
        _fail("Gate 0 teacher alignment must not invent a Stage P checkpoint hash")
    for key in ("train_archive_sha256", "capacity_receipt_sha256", "normalization_sha256", "solver_sha256", "implementation_sha256"):
        _sha256_string(payload[key], f"teacher_alignment.{key}")
    if int(payload["window_count"]) != 100:
        _fail("teacher alignment probe must use exactly 100 windows")
    for key in ("timestamps_aligned", "state_aligned", "lp_feasible"):
        if _bool(payload[key], f"teacher_alignment.{key}") is not True:
            _fail(f"teacher alignment field failed: {key}")
    if list(payload["forecast_shape"]) != [4, 4] or list(payload["dispatch_shape"]) != [4, 21]:
        _fail("teacher alignment probe shape mismatch")
    if _bool(payload["production_overlay_written"], "teacher_alignment.production_overlay_written") is not False:
        _fail("Gate 0 teacher alignment must not write a production overlay")


def _validate_curriculum(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, "curriculum")
    _validate_common_probe(payload, "curriculum")
    _require(payload, "step_weights", "forecast_task_weights", "gas_forecast_weight", "rigid_task_weights", "epochs")
    if tuple(float(value) for value in payload["step_weights"]) != STEP_WEIGHTS:
        _fail("curriculum step weights are not frozen")
    if tuple(float(value) for value in payload["forecast_task_weights"]) != FORECAST_TASK_WEIGHTS:
        _fail("curriculum forecast task weights are not frozen")
    if _finite(payload["gas_forecast_weight"], "curriculum.gas_forecast_weight") != 0.25:
        _fail("curriculum gas forecast weight must be 0.25")
    if tuple(float(value) for value in payload["rigid_task_weights"]) != (1.0, 1.0, 1.0):
        _fail("curriculum rigid-task weights are not all 1.0")
    epochs = _list(payload["epochs"], "curriculum.epochs")
    if not epochs:
        _fail("curriculum must record at least one epoch")
    for index, item in enumerate(epochs):
        row = _mapping(item, f"curriculum.epochs[{index}]")
        _require(row, "epoch", "forecast", "imitation", "decision")
        for key in ("forecast", "imitation", "decision"):
            if _finite(row[key], f"curriculum.epochs[{index}].{key}") < 0.0:
                _fail(f"curriculum weight is negative: {key}")
    first = _mapping(epochs[0], "curriculum.epochs[0]")
    if _finite(first["decision"], "curriculum.epochs[0].decision") <= 0.0:
        _fail("decision loss is inactive at the first Stage J epoch")


def validate_c_ref_gate0_receipt(payload: Mapping[str, Any], *, run_root: str | Path) -> None:
    """Validate the stronger Gate 0 C-ref contract, including file lineage."""

    _require(
        payload,
        "schema_version", "source_split", "train_archive_sha256", "capacity_receipt_sha256",
        "capacity_scenario_hash", "objective_implementation_sha256", "step_weights",
        "sample_count", "c_ref", "c_ref_sha256",
    )
    if payload.get("schema_version") != "formal-v4.1-c-ref-receipt-v1":
        _fail("C_ref receipt schema mismatch")
    if payload.get("source_split") != "train":
        _fail("C_ref receipt must be fitted on train")
    if int(payload["sample_count"]) != 34959:
        _fail("C_ref receipt must contain exactly 34959 train windows")
    if tuple(float(value) for value in payload["step_weights"]) != STEP_WEIGHTS:
        _fail("C_ref receipt step weights are not frozen")
    if _finite(payload["c_ref"], "C_ref.c_ref") <= 0.0:
        _fail("C_ref must be positive")
    for key in ("train_archive_sha256", "capacity_receipt_sha256", "capacity_scenario_hash", "objective_implementation_sha256", "c_ref_sha256"):
        _sha256_string(payload[key], f"C_ref.{key}")
    root = Path(run_root).resolve()
    train_path = root / "data" / "train.npz"
    capacity_path = root / "gate0" / "CAPACITY_FREEZE.json"
    if not train_path.is_file() or _sha256(train_path) != str(payload["train_archive_sha256"]):
        _fail("C_ref train archive hash does not match the run root")
    if not capacity_path.is_file() or _sha256(capacity_path) != str(payload["capacity_receipt_sha256"]):
        _fail("C_ref capacity receipt hash does not match the run root")
    # Reuse the canonical self-hash implementation from the objective module.
    from .formal_v4_objective import validate_c_ref_receipt

    validate_c_ref_receipt(payload, train_archive_sha256=str(payload["train_archive_sha256"]), capacity_receipt_sha256=str(payload["capacity_receipt_sha256"]))


def _validate_gradient(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, "gradient")
    _validate_common_probe(payload, "gradient")
    _require(payload, "initial_model_hashes", "joint", "decoupled")
    hashes = _mapping(payload["initial_model_hashes"], "gradient.initial_model_hashes")
    _require(hashes, "joint", "decoupled")
    if _sha256_string(hashes["joint"], "gradient.initial_model_hashes.joint") != _sha256_string(hashes["decoupled"], "gradient.initial_model_hashes.decoupled"):
        _fail("gradient probe models do not share identical initialization")
    for mode in ("joint", "decoupled"):
        row = _mapping(payload[mode], f"gradient.{mode}")
        _require(row, "forecaster_decision_gradient_norm", "scheduler_decision_gradient_norm")
        forecaster = _finite(row["forecaster_decision_gradient_norm"], f"gradient.{mode}.forecaster_decision_gradient_norm")
        scheduler = _finite(row["scheduler_decision_gradient_norm"], f"gradient.{mode}.scheduler_decision_gradient_norm")
        if scheduler <= 0.0:
            _fail(f"{mode} scheduler decision gradient is zero")
        if mode == "joint" and forecaster <= 0.0:
            _fail("joint decision loss does not reach the forecaster")
        if mode == "decoupled" and forecaster > 1.0e-12:
            _fail("decoupled decision loss reaches the forecaster")
    if _bool(payload.get("checkpoint_created", False), "gradient.checkpoint_created") is not False:
        _fail("gradient probe created a checkpoint")


def _validate_method_adapter(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, "method_adapter")
    _validate_common_probe(payload, "method_adapter")
    _require(payload, "benchmark_sha256", "train_archive_sha256")
    _sha256_string(payload["benchmark_sha256"], "method_adapter.benchmark_sha256")
    _sha256_string(payload["train_archive_sha256"], "method_adapter.train_archive_sha256")
    methods = _list(payload.get("methods"), "method_adapter.methods")
    rows = {_string(_mapping(item, "method_adapter.methods[]").get("method_id"), "method_adapter.method_id"): _mapping(item, "method_adapter.methods[]") for item in methods}
    if set(rows) != set(FORMAL_V4_METHOD_IDS) or len(rows) != len(FORMAL_V4_METHOD_IDS):
        _fail("method adapter registry does not contain exactly the nine formal-v4 methods")
    for method_id, row in rows.items():
        _require(row, "role", "deployable", "produces_forecast", "forecast_metrics_applicable", "online_optimizer_calls_per_window", "expected_dispatch_shape", "probe_status")
        _string(row["role"], f"method_adapter.{method_id}.role")
        probe_status = _string(row["probe_status"], f"method_adapter.{method_id}.probe_status")
        if probe_status not in {"passed", "contract_only"}:
            _fail(f"invalid probe status for {method_id}")
        if method_id in {"Official iTransformer-PTO", "Differentiable-LP", "Perfect-Information-MPC"}:
            if probe_status != "contract_only":
                _fail(f"{method_id} must remain contract-only in Gate 0")
        elif probe_status != "passed":
            _fail(f"{method_id} executable adapter probe did not pass")
        if list(row["expected_dispatch_shape"]) != [4, 21]:
            _fail(f"dispatch shape mismatch for {method_id}")
        calls = int(row["online_optimizer_calls_per_window"])
        if calls < 0:
            _fail(f"negative optimizer call count for {method_id}")
        if method_id in {"RSC-PF", "Decoupled-RSC-PF", "Direct-Policy"} and calls != 0:
            _fail(f"{method_id} must not call an online optimizer")
        if method_id in {"Scheme2R-PTO", "State-Conditioned-PTO", "Official iTransformer-PTO", "Differentiable-LP", "Perfect-Information-MPC", "Seasonal-Naive-PTO"} and calls != 1:
            _fail(f"{method_id} must call one online optimizer")
        if method_id == "Perfect-Information-MPC" and _bool(row["deployable"], f"method_adapter.{method_id}.deployable") is not False:
            _fail("Perfect-Information-MPC must be marked nondeployable")
        if method_id == "Direct-Policy" and _bool(row["forecast_metrics_applicable"], f"method_adapter.{method_id}.forecast_metrics_applicable") is not False:
            _fail("Direct-Policy forecast metrics must be marked not applicable")
        if _bool(row["produces_forecast"], f"method_adapter.{method_id}.produces_forecast"):
            if list(row.get("expected_forecast_shape", ())) != [4, 4]:
                _fail(f"forecast shape mismatch for {method_id}")


def _validate_data_access(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, "data_access")
    _require(payload, "events", "test_set_accessed", "blocked_count")
    if _bool(payload["test_set_accessed"], "data_access.test_set_accessed") is not False:
        _fail("data-access receipt reports test-set access")
    events = _list(payload["events"], "data_access.events")
    allowed_splits: set[str] = set()
    denied_evaluation = False
    for index, item in enumerate(events):
        row = _mapping(item, f"data_access.events[{index}]")
        _require(row, "split", "years", "decision", "path", "path_sha256", "caller")
        split = _string(row["split"], f"data_access.events[{index}].split")
        years = tuple(sorted({int(value) for value in _list(row["years"], f"data_access.events[{index}].years")}))
        decision = _string(row["decision"], f"data_access.events[{index}].decision")
        _string(row["path"], f"data_access.events[{index}].path")
        _sha256_string(row["path_sha256"], f"data_access.events[{index}].path_sha256") if row["path_sha256"] else None
        _string(row["caller"], f"data_access.events[{index}].caller")
        if 2020 in years or 2021 in years:
            if decision == "allow":
                _fail("data-access receipt contains an allowed 2020/2021 read")
            if split == "evaluation" and decision == "deny":
                denied_evaluation = True
        if decision == "allow" and split in {"train", "selection"}:
            allowed_splits.add(split)
    if not {"train", "selection"}.issubset(allowed_splits):
        _fail("data-access receipt does not contain separate train and selection reads")
    if not denied_evaluation:
        _fail("data-access receipt lacks an explicit denied evaluation probe")


def _validate_archive_access(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, "archive_access")
    _require(payload, "events", "test_set_accessed")
    if _bool(payload["test_set_accessed"], "archive_access.test_set_accessed") is not False:
        _fail("archive receipt reports test-set access")
    events = _list(payload["events"], "archive_access.events")
    splits: set[str] = set()
    for index, item in enumerate(events):
        row = _mapping(item, f"archive_access.events[{index}]")
        _require(row, "split", "years", "container_path", "container_sha256", "member_name", "member_sha256")
        split = _string(row["split"], f"archive_access.events[{index}].split")
        years = {int(value) for value in _list(row["years"], f"archive_access.events[{index}].years")}
        if years.intersection({2020, 2021}):
            _fail("archive receipt contains a forbidden evaluation year")
        if split not in {"train", "selection"}:
            _fail(f"archive receipt contains unknown split: {split}")
        _string(row["container_path"], f"archive_access.events[{index}].container_path")
        _sha256_string(row["container_sha256"], f"archive_access.events[{index}].container_sha256")
        _string(row["member_name"], f"archive_access.events[{index}].member_name")
        _sha256_string(row["member_sha256"], f"archive_access.events[{index}].member_sha256")
        splits.add(split)
    if not {"train", "selection"}.issubset(splits):
        _fail("archive receipt does not contain train and selection members")


def _validate_resource_projection(payload: Mapping[str, Any]) -> None:
    _validate_schema(payload, "resource_projection")
    _require(payload, "rows", "worst_case_method_id", "worst_case_projected_hours", "free_memory_fraction", "free_disk_fraction")
    rows = _list(payload["rows"], "resource_projection.rows")
    if not rows:
        _fail("resource projection must contain method rows")
    maximum = 0.0
    method_ids: set[str] = set()
    for index, item in enumerate(rows):
        row = _mapping(item, f"resource_projection.rows[{index}]")
        _require(row, "method_id", "sample_count", "p50_seconds", "p95_seconds", "projected_hours", "peak_memory_bytes")
        method_id = _string(row["method_id"], f"resource_projection.rows[{index}].method_id")
        method_ids.add(method_id)
        if int(row["sample_count"]) != 500:
            _fail(f"resource projection row {method_id} must contain 500 samples")
        p95 = _finite(row["projected_hours"], f"resource_projection.{method_id}.projected_hours")
        if p95 < 0.0:
            _fail(f"resource projection is negative for {method_id}")
        maximum = max(maximum, p95)
        _finite(row["p50_seconds"], f"resource_projection.{method_id}.p50_seconds")
        _finite(row["p95_seconds"], f"resource_projection.{method_id}.p95_seconds")
        if int(row["peak_memory_bytes"]) < 0:
            _fail(f"resource projection peak memory is negative for {method_id}")
    worst_method = _string(payload["worst_case_method_id"], "resource_projection.worst_case_method_id")
    if worst_method not in method_ids:
        _fail("resource projection worst-case method is not present in rows")
    worst_case = _finite(payload["worst_case_projected_hours"], "resource_projection.worst_case_projected_hours")
    if worst_case > _MAX_PROJECTED_HOURS:
        _fail("resource projection exceeds 24 hours")
    if abs(worst_case - maximum) > 1.0e-9:
        _fail("resource projection worst-case duration is not the maximum row")
    if maximum > _MAX_PROJECTED_HOURS:
        _fail("resource projection exceeds 24 hours")
    if _finite(payload["free_memory_fraction"], "resource_projection.free_memory_fraction") < _MIN_RESOURCE_MARGIN:
        _fail("resource projection memory margin is below 20%")
    if _finite(payload["free_disk_fraction"], "resource_projection.free_disk_fraction") < _MIN_RESOURCE_MARGIN:
        _fail("resource projection disk margin is below 20%")


def validate_gate0_receipt(
    receipt_name: str,
    payload: Mapping[str, Any],
    *,
    run_root: str | Path,
) -> None:
    """Validate one typed formal-v4.1 receipt, raising on any violation."""

    if not isinstance(payload, Mapping):
        raise ValueError("Gate 0 receipt must be a JSON object")
    root = Path(run_root).resolve()
    validators = {
        "trajectory": lambda: _validate_trajectory(payload, root),
        "teacher_alignment": lambda: _validate_teacher_alignment(payload),
        "curriculum": lambda: _validate_curriculum(payload),
        "gradient": lambda: _validate_gradient(payload),
        "method_adapter": lambda: _validate_method_adapter(payload),
        "data_access": lambda: _validate_data_access(payload),
        "archive_access": lambda: _validate_archive_access(payload),
        "resource_projection": lambda: _validate_resource_projection(payload),
    }
    try:
        validators[receipt_name]()
    except KeyError as exc:
        raise ValueError(f"unknown Gate 0 receipt name: {receipt_name}") from exc


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write one canonical JSON receipt and refuse to overwrite an artifact."""

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite Gate 0 evidence: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary receipt already exists: {temporary}")
    temporary.write_bytes(encoded)
    temporary.replace(destination)


def load_gate0_receipt(path: str | Path, receipt_name: str, *, run_root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a receipt before returning its decoded mapping."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Gate 0 receipt must be a JSON object: {source}")
    validate_gate0_receipt(receipt_name, payload, run_root=run_root or source.parent.parent)
    return dict(payload)


__all__ = [
    "FORMAL_V4_METHOD_IDS",
    "FORECAST_TASK_WEIGHTS",
    "GATE0_RECEIPT_SCHEMAS",
    "STEP_WEIGHTS",
    "load_gate0_receipt",
    "validate_c_ref_gate0_receipt",
    "validate_gate0_receipt",
    "write_immutable_json",
]
