from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.joint_dispatch.formal_v4_gate0_evidence import (
    FORMAL_V4_METHOD_IDS,
    validate_gate0_receipt,
    write_immutable_json,
)


_COMMIT = "a" * 40
_HASH = "b" * 64


def _write(root: Path, relative: str, content: bytes = b"evidence") -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _trajectory(root: Path) -> dict:
    hashes = {
        "capacity": _write(root, "gate0/CAPACITY_FREEZE.json", b"capacity"),
        "benchmark": _write(root, "gate0/benchmark/FORMAL_V4_BENCHMARK_RECEIPT.json", b"benchmark"),
        "train": _write(root, "data/train.npz", b"train"),
        "selection": _write(root, "data/selection.npz", b"selection"),
    }
    return {
        "schema_version": "formal-v4.1-trajectory-receipt-v1",
        "protocol_id": "formal-v4.1-trajectory-causal-audit-v1",
        "source_commit": _COMMIT,
        "capacity_receipt_path": "gate0/CAPACITY_FREEZE.json",
        "capacity_receipt_sha256": hashes["capacity"],
        "benchmark_path": "gate0/benchmark/FORMAL_V4_BENCHMARK_RECEIPT.json",
        "benchmark_sha256": hashes["benchmark"],
        "train_archive_path": "data/train.npz",
        "train_archive_sha256": hashes["train"],
        "selection_archive_path": "data/selection.npz",
        "selection_archive_sha256": hashes["selection"],
        "trajectory_id": "trajectory-001",
        "trajectory_sha256": _HASH,
        "rule_version": "formal-v4.1-causal-realized-settlement-v1",
        "trajectory_audit": {
            "solved_hours": 10,
            "settled_hours": 10,
            "warmup_hours": 24,
            "future_label_reads": 0,
            "max_balance_residual": 1.0e-8,
            "max_conversion_residual": 1.0e-8,
            "max_soc_recursion_residual": 1.0e-8,
            "max_chp_ramp_violation": 0.0,
            "max_renewable_availability_violation": 0.0,
            "max_soc_bound_violation": 0.0,
        },
    }


def _probe_common(schema: str, protocol_id: str) -> dict:
    return {
        "schema_version": schema,
        "protocol_id": protocol_id,
        "source_commit": _COMMIT,
        "probe_only": True,
        "test_set_accessed": False,
    }


def _teacher() -> dict:
    payload = _probe_common("formal-v4.1-teacher-alignment-v1", "formal-v4.1-teacher-probe-v1")
    payload.update({
        "production_overlay_deferred_until_stage_p": True,
        "stage_p_checkpoint_sha256": None,
        "window_count": 100,
        "timestamps_aligned": True,
        "state_aligned": True,
        "lp_feasible": True,
        "forecast_shape": [4, 4],
        "dispatch_shape": [4, 21],
    })
    return payload


def _curriculum() -> dict:
    payload = _probe_common("formal-v4.1-curriculum-receipt-v1", "formal-v4.1-curriculum-probe-v1")
    payload.update({
        "step_weights": [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        "forecast_task_weights": [1.0, 1.0, 1.0, 0.25],
        "gas_forecast_weight": 0.25,
        "rigid_task_weights": [1.0, 1.0, 1.0],
        "epochs": [
            {"epoch": 0, "forecast": 1.0, "imitation": 1.0, "decision": 0.05},
            {"epoch": 10, "forecast": 1.0, "imitation": 0.5, "decision": 0.5},
            {"epoch": 20, "forecast": 1.0, "imitation": 0.0, "decision": 1.0},
        ],
    })
    return payload


def _gradient() -> dict:
    payload = _probe_common("formal-v4.1-gradient-receipt-v1", "formal-v4.1-gradient-probe-v1")
    payload.update({
        "initial_model_hashes": {"joint": _HASH, "decoupled": _HASH},
        "joint": {"forecaster_decision_gradient_norm": 0.5, "scheduler_decision_gradient_norm": 0.8},
        "decoupled": {"forecaster_decision_gradient_norm": 0.0, "scheduler_decision_gradient_norm": 0.8},
        "checkpoint_created": False,
    })
    return payload


def _methods() -> dict:
    payload = _probe_common("formal-v4.1-method-adapter-receipt-v1", "formal-v4.1-method-adapter-probe-v1")
    zero_optimizer = {"RSC-PF", "Decoupled-RSC-PF", "Direct-Policy"}
    rows = []
    for method_id in FORMAL_V4_METHOD_IDS:
        rows.append({
            "method_id": method_id,
            "role": "reference" if method_id == "Perfect-Information-MPC" else "model",
            "deployable": method_id != "Perfect-Information-MPC",
            "produces_forecast": method_id != "Direct-Policy",
            "forecast_metrics_applicable": method_id != "Direct-Policy",
            "online_optimizer_calls_per_window": 0 if method_id in zero_optimizer else 1,
            "expected_forecast_shape": None if method_id == "Direct-Policy" else [4, 4],
            "expected_dispatch_shape": [4, 21],
        })
    payload["methods"] = rows
    return payload


def _data_access() -> dict:
    return {
        "schema_version": "formal-v4.1-data-access-v1",
        "events": [
            {"split": "train", "years": [2015, 2016, 2017, 2018], "decision": "allow", "path": "source/train.zip", "path_sha256": _HASH, "caller": "canonical-loader"},
            {"split": "selection", "years": [2019], "decision": "allow", "path": "source/selection.zip", "path_sha256": _HASH, "caller": "canonical-loader"},
            {"split": "evaluation", "years": [2020], "decision": "deny", "path": "source/evaluation.zip", "path_sha256": "", "caller": "gate0-denial-probe"},
        ],
        "test_set_accessed": False,
        "blocked_count": 1,
    }


def _archive_access() -> dict:
    return {
        "schema_version": "formal-v4.1-archive-access-v1",
        "events": [
            {"split": "train", "years": [2018], "container_path": "source/train.zip", "container_sha256": _HASH, "member_name": "2018.csv", "member_sha256": _HASH},
            {"split": "selection", "years": [2019], "container_path": "source/selection.zip", "container_sha256": _HASH, "member_name": "2019.csv", "member_sha256": _HASH},
        ],
        "test_set_accessed": False,
    }


def _resources() -> dict:
    return {
        "schema_version": "formal-v4.1-resource-projection-v1",
        "rows": [
            {"method_id": "RSC-PF", "sample_count": 500, "p50_seconds": 0.1, "p95_seconds": 0.2, "projected_hours": 1.0, "peak_memory_bytes": 100},
            {"method_id": "Differentiable-LP", "sample_count": 500, "p50_seconds": 0.2, "p95_seconds": 0.3, "projected_hours": 2.0, "peak_memory_bytes": 200},
        ],
        "worst_case_method_id": "Differentiable-LP",
        "worst_case_projected_hours": 2.0,
        "free_memory_fraction": 0.5,
        "free_disk_fraction": 0.5,
    }


def test_valid_typed_receipts_are_accepted(tmp_path: Path) -> None:
    validate_gate0_receipt("trajectory", _trajectory(tmp_path), run_root=tmp_path)
    validate_gate0_receipt("teacher_alignment", _teacher(), run_root=tmp_path)
    validate_gate0_receipt("curriculum", _curriculum(), run_root=tmp_path)
    validate_gate0_receipt("gradient", _gradient(), run_root=tmp_path)
    validate_gate0_receipt("method_adapter", _methods(), run_root=tmp_path)
    validate_gate0_receipt("data_access", _data_access(), run_root=tmp_path)
    validate_gate0_receipt("archive_access", _archive_access(), run_root=tmp_path)
    validate_gate0_receipt("resource_projection", _resources(), run_root=tmp_path)


@pytest.mark.parametrize(
    ("name", "mutator", "message"),
    [
        ("trajectory", lambda payload: payload["trajectory_audit"].update(future_label_reads=1), "future-label"),
        ("teacher_alignment", lambda payload: payload.update(stage_p_checkpoint_sha256=_HASH), "checkpoint"),
        ("curriculum", lambda payload: payload["epochs"][0].update(decision=0.0), "decision"),
        ("gradient", lambda payload: payload["joint"].update(forecaster_decision_gradient_norm=0.0), "forecaster"),
        ("data_access", lambda payload: payload["events"].append({"split": "evaluation", "years": [2020], "decision": "allow", "path": "x", "path_sha256": _HASH, "caller": "bad"}), "allowed"),
        ("archive_access", lambda payload: payload["events"].append({"split": "evaluation", "years": [2020], "container_path": "x", "container_sha256": _HASH, "member_name": "2020.csv", "member_sha256": _HASH}), "forbidden"),
        ("resource_projection", lambda payload: payload.update(worst_case_projected_hours=25.0), "24 hours"),
    ],
)
def test_materially_invalid_receipt_is_rejected(tmp_path: Path, name: str, mutator, message: str) -> None:
    payload = {
        "trajectory": _trajectory,
        "teacher_alignment": lambda root: _teacher(),
        "curriculum": lambda root: _curriculum(),
        "gradient": lambda root: _gradient(),
        "data_access": lambda root: _data_access(),
        "archive_access": lambda root: _archive_access(),
        "resource_projection": lambda root: _resources(),
    }[name](tmp_path)
    mutator(payload)
    with pytest.raises(ValueError, match=message):
        validate_gate0_receipt(name, payload, run_root=tmp_path)


def test_method_registry_rejects_missing_method(tmp_path: Path) -> None:
    payload = _methods()
    payload["methods"] = payload["methods"][:-1]
    with pytest.raises(ValueError, match="exactly the nine"):
        validate_gate0_receipt("method_adapter", payload, run_root=tmp_path)


def test_artifact_hash_and_path_are_bound_to_run_root(tmp_path: Path) -> None:
    payload = _trajectory(tmp_path)
    payload["benchmark_path"] = "..\\outside.json"
    with pytest.raises(ValueError, match="outside"):
        validate_gate0_receipt("trajectory", payload, run_root=tmp_path)


def test_write_immutable_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "protocol" / "receipt.json"
    write_immutable_json(path, {"schema_version": "x"})
    with pytest.raises(FileExistsError):
        write_immutable_json(path, {"schema_version": "y"})
