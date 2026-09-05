"""Run the bounded formal-v4.5 training-only diagnostic; never starts a Pilot."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_pilot_executor import _parameters  # noqa: E402
from src.joint_dispatch.formal_v4_4_artifacts import write_json_once  # noqa: E402
from src.joint_dispatch.formal_v4_5_artifacts import audit_v45_selection, write_v45_stage_receipt  # noqa: E402
from src.joint_dispatch.formal_v4_5_contract import load_formal_v4_5_contract  # noqa: E402
from src.joint_dispatch.formal_v4_5_pilot_data import (  # noqa: E402
    MaterializedTrainingV45, load_v45_training_cache, materialize_v45_training_only,
)
from src.joint_dispatch.formal_v4_5_pilot_executor import execute_training_stages_v45  # noqa: E402
from src.joint_dispatch.formal_v4_5_training import curriculum_weights_v45  # noqa: E402


def _fixture_diagnostic(config_path: str | Path, output_root: str | Path, max_batches: int | None) -> Mapping[str, Any]:
    contract = load_formal_v4_5_contract(config_path)
    weights = [curriculum_weights_v45(epoch, int(contract.joint_curriculum["ramp_epochs"])) for epoch in range(6)]
    receipt = {
        "schema": "formal-v4.5-diagnostic-v1",
        "mode": "fixture",
        "accessed_years": list(contract.train_years),
        "selection_year_accessed": False,
        "evaluation_year_accessed": False,
        "pilot_authorized": False,
        "checks": {
            "curriculum_finite": bool(all(np.isfinite(list(row.values())).all() for row in weights)),
            "curriculum_monotone": bool(all(weights[i]["decision"] <= weights[i + 1]["decision"] for i in range(5))),
        },
        "max_batches": None if max_batches is None else int(max_batches),
    }
    Path(output_root).mkdir(parents=True, exist_ok=True)
    write_json_once(Path(output_root) / "DIAGNOSTIC_RECEIPT.json", receipt)
    return receipt


def _truncate_collection(collection: Any, limit: int) -> Any:
    array_fields = {
        "load_history", "exog_history", "renewable_history", "device_history",
        "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
        "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
        "target_times", "trajectory_ids", "state_hashes",
    }
    updates = {
        field.name: getattr(collection.split, field.name)[:limit]
        for field in fields(collection.split) if field.name in array_fields
    }
    return collection.__class__(replace(collection.split, **updates), collection.role)


def run_formal_v45_diagnostic(
    *,
    config: str | Path,
    output_root: str | Path,
    max_batches: int | None = None,
    train_data: str | Path | None = None,
    benchmark: str | Path | None = None,
    capacity_receipt: str | Path | None = None,
    split_path: str | Path | None = None,
    materialized_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Run fixtures when sources are omitted, otherwise run training-only data."""

    if materialized_root is not None:
        contract = load_formal_v4_5_contract(config)
        materialized = load_v45_training_cache(
            materialized_root=materialized_root, train_data=train_data,
            benchmark=benchmark, capacity_receipt=capacity_receipt,
        )
        if max_batches is not None:
            limit = max(1, int(max_batches)) * int(contract.payload["pilot_budget"]["batch_size"])
            materialized = MaterializedTrainingV45(
                train=_truncate_collection(materialized.train, limit),
                early_stop=_truncate_collection(materialized.early_stop, limit),
                normalization_source=materialized.normalization_source,
                lineage=materialized.lineage,
            )
        return _run_materialized_diagnostic(
            contract=contract, materialized=materialized, output_root=output_root,
            benchmark=benchmark, capacity_receipt=capacity_receipt, max_batches=max_batches,
        )
    if any(value is None for value in (train_data, benchmark, capacity_receipt, split_path)):
        if any(value is not None for value in (train_data, benchmark, capacity_receipt, split_path)):
            raise ValueError("real diagnostic requires train_data, benchmark, capacity_receipt, and split_path together")
        return _fixture_diagnostic(config, output_root, max_batches)
    contract = load_formal_v4_5_contract(config)
    batch_size = int(contract.payload["pilot_budget"]["batch_size"])
    limit = None if max_batches is None else max(1, int(max_batches)) * batch_size
    with np.load(split_path, allow_pickle=False) as arrays:
        split = {
            "train": np.asarray(arrays["train"], dtype=np.int64),
            "early_stop": np.asarray(arrays["early_stop"], dtype=np.int64),
        }
    if limit is not None:
        split = {name: values[:limit] for name, values in split.items()}
    materialized = materialize_v45_training_only(
        train_data=train_data, benchmark=benchmark, capacity_receipt=capacity_receipt,
        split=split, artifact_root=output_root,
    )
    benchmark_payload = yaml.safe_load(Path(benchmark).read_text(encoding="utf-8"))
    capacity_payload = json.loads(Path(capacity_receipt).read_text(encoding="utf-8"))
    parameters = _parameters(benchmark_payload, capacity_payload)
    bundle = execute_training_stages_v45(
        materialized=materialized, contract=contract, artifact_root=output_root,
        benchmark=benchmark, capacity_receipt=capacity_receipt,
        seed=int(contract.payload["pilot_budget"]["seed"]), parameters=parameters,
        epoch_cap=2 if max_batches is not None else None,
    )
    output_root = Path(output_root)
    stage_root = output_root / "stages"
    stage_root.mkdir(parents=True, exist_ok=True)
    group_rates = [
        {"name": "base", "lr": float(contract.payload["pilot_budget"]["j_forecaster_lr"])},
        {"name": "head", "lr": float(contract.payload["pilot_budget"]["j_head_lr"])},
        {"name": "scheduler", "lr": float(contract.payload["pilot_budget"]["j_scheduler_lr"])},
    ]
    decoupled_rates = [{"name": "scheduler", "lr": float(contract.payload["pilot_budget"]["j_scheduler_lr"])}]
    joint_root = stage_root / "J_joint"; decoupled_root = stage_root / "J_decoupled"
    write_v45_stage_receipt(joint_root / "STAGE_SELECTION.json", bundle.j.joint, bundle.j.joint.validation_history, group_rates, accessed_years=list(materialized.lineage["years"]))
    write_v45_stage_receipt(decoupled_root / "STAGE_SELECTION.json", bundle.j.decoupled, bundle.j.decoupled.validation_history, decoupled_rates, accessed_years=list(materialized.lineage["years"]))
    joint_audit = audit_v45_selection(joint_root, contract)
    decoupled_audit = audit_v45_selection(decoupled_root, contract)
    joint_grad = dict(bundle.j.joint.gradient_norms)
    decoupled_grad = dict(bundle.j.decoupled.gradient_norms)
    checks = {
        "year_firewall": list(materialized.lineage["years"]) == list(contract.train_years) and not materialized.lineage["selection_year_accessed"] and not materialized.lineage["evaluation_year_accessed"],
        "joint_decision_to_base": joint_grad.get("decision_to_base", 0.0) > 0.0,
        "joint_decision_to_gate": joint_grad.get("decision_to_gate", 0.0) > 0.0,
        "joint_decision_to_magnitude": joint_grad.get("decision_to_magnitude", 0.0) > 0.0,
        "decoupled_forecast_boundary": all(decoupled_grad.get(name, 0.0) <= 1.0e-12 for name in ("decision_to_base", "decision_to_gate", "decision_to_magnitude")),
        "joint_selection_audited": joint_audit["authorized_gate1"] is False,
        "decoupled_selection_audited": decoupled_audit["authorized_gate1"] is False,
    }
    receipt = {
        "schema": "formal-v4.5-diagnostic-v1",
        "mode": "training_only",
        "accessed_years": list(materialized.lineage["years"]),
        "selection_year_accessed": bool(materialized.lineage["selection_year_accessed"]),
        "evaluation_year_accessed": bool(materialized.lineage["evaluation_year_accessed"]),
        "pilot_authorized": False,
        "checks": checks,
        "gradient_norms": {"joint": joint_grad, "decoupled": decoupled_grad},
        "stage_audits": {"joint": joint_audit, "decoupled": decoupled_audit},
        "lineage": dict(materialized.lineage),
    }
    write_json_once(output_root / "DIAGNOSTIC_RECEIPT.json", receipt)
    return receipt


def _run_materialized_diagnostic(
    *, contract: Any, materialized: MaterializedTrainingV45, output_root: str | Path,
    benchmark: str | Path | None, capacity_receipt: str | Path | None,
    max_batches: int | None,
) -> Mapping[str, Any]:
    if benchmark is None or capacity_receipt is None:
        raise ValueError("real diagnostic requires benchmark and capacity_receipt")
    benchmark_payload = yaml.safe_load(Path(benchmark).read_text(encoding="utf-8"))
    capacity_payload = json.loads(Path(capacity_receipt).read_text(encoding="utf-8"))
    parameters = _parameters(benchmark_payload, capacity_payload)
    bundle = execute_training_stages_v45(
        materialized=materialized, contract=contract, artifact_root=output_root,
        benchmark=benchmark, capacity_receipt=capacity_receipt,
        seed=int(contract.payload["pilot_budget"]["seed"]), parameters=parameters,
        epoch_cap=2 if max_batches is not None else None,
    )
    output_root = Path(output_root)
    stage_root = output_root / "stages"
    stage_root.mkdir(parents=True, exist_ok=True)
    group_rates = [
        {"name": "base", "lr": float(contract.payload["pilot_budget"]["j_forecaster_lr"])},
        {"name": "head", "lr": float(contract.payload["pilot_budget"]["j_head_lr"])},
        {"name": "scheduler", "lr": float(contract.payload["pilot_budget"]["j_scheduler_lr"])},
    ]
    decoupled_rates = [{"name": "scheduler", "lr": float(contract.payload["pilot_budget"]["j_scheduler_lr"])}]
    joint_root = stage_root / "J_joint"; decoupled_root = stage_root / "J_decoupled"
    write_v45_stage_receipt(joint_root / "STAGE_SELECTION.json", bundle.j.joint, bundle.j.joint.validation_history, group_rates, accessed_years=list(materialized.lineage["years"]))
    write_v45_stage_receipt(decoupled_root / "STAGE_SELECTION.json", bundle.j.decoupled, bundle.j.decoupled.validation_history, decoupled_rates, accessed_years=list(materialized.lineage["years"]))
    joint_audit = audit_v45_selection(joint_root, contract)
    decoupled_audit = audit_v45_selection(decoupled_root, contract)
    gradient_norms = {
        "joint": dict(bundle.j.joint.gradient_norms),
        "decoupled": dict(bundle.j.decoupled.gradient_norms),
    }
    joint_history = bundle.j.joint.validation_history
    checks = {
        "year_firewall": list(materialized.lineage["years"]) == list(contract.train_years) and not materialized.lineage["selection_year_accessed"] and not materialized.lineage["evaluation_year_accessed"],
        "joint_decision_to_base": gradient_norms["joint"].get("decision_to_base", 0.0) > 0.0,
        "joint_decision_to_gate": gradient_norms["joint"].get("decision_to_gate", 0.0) > 0.0,
        "joint_decision_to_magnitude": gradient_norms["joint"].get("decision_to_magnitude", 0.0) > 0.0,
        "decoupled_forecast_boundary": all(gradient_norms["decoupled"].get(name, 0.0) <= 1.0e-12 for name in ("decision_to_base", "decision_to_gate", "decision_to_magnitude")),
        "joint_guardrails_finite": bool(joint_history) and all(
            bool(float(row.get("eligible", 0.0))) and np.isfinite(float(row["forecast"]))
            and np.isfinite(float(row["anchor"])) and float(row["forecast"]) <= 1.02
            and float(row["anchor"]) <= 0.25 for row in joint_history
        ),
        "joint_selection_audited": joint_audit["authorized_gate1"] is False,
        "decoupled_selection_audited": decoupled_audit["authorized_gate1"] is False,
    }
    receipt = {
        "schema": "formal-v4.5-diagnostic-v1", "mode": "training_cache",
        "accessed_years": list(materialized.lineage["years"]),
        "selection_year_accessed": bool(materialized.lineage["selection_year_accessed"]),
        "evaluation_year_accessed": bool(materialized.lineage["evaluation_year_accessed"]),
        "pilot_authorized": False, "max_batches": max_batches,
        "checks": checks, "gradient_norms": gradient_norms,
        "stage_audits": {"joint": joint_audit, "decoupled": decoupled_audit},
        "lineage": dict(materialized.lineage),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_once(output_root / "DIAGNOSTIC_RECEIPT.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-data", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--capacity-receipt", type=Path)
    parser.add_argument("--split-path", type=Path)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    try:
        receipt = run_formal_v45_diagnostic(
            config=args.config, output_root=args.output_root,
            max_batches=args.max_batches, train_data=args.train_data,
            benchmark=args.benchmark, capacity_receipt=args.capacity_receipt,
            split_path=args.split_path,
            materialized_root=args.materialized_root,
        )
    except Exception as exc:
        print(json.dumps({"pilot_authorized": False, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"pilot_authorized": receipt["pilot_authorized"], "checks": receipt["checks"]}, ensure_ascii=False))
    return 0 if all(bool(value) for value in receipt["checks"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
