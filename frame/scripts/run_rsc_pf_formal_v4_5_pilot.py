"""Run the formal-v4.5 Pilot on the frozen train/2019 cache; never opens 2020."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_artifacts import canonical_sha256, sha256_file, write_json_once  # noqa: E402
from src.joint_dispatch.formal_v4_4_pilot_gate import authorize_pilot_v44  # noqa: E402
from src.joint_dispatch.formal_v4_5_contract import load_formal_v4_5_contract  # noqa: E402
from src.joint_dispatch.formal_v4_5_pilot_data import load_v45_pilot_cache  # noqa: E402
from src.joint_dispatch.formal_v4_5_pilot_executor import execute_real_pilot_v45  # noqa: E402


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


def _select_collection(collection: Any, indices: np.ndarray) -> Any:
    array_fields = {
        "load_history", "exog_history", "renewable_history", "device_history",
        "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
        "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
        "target_times", "trajectory_ids", "state_hashes",
    }
    updates = {
        field.name: getattr(collection.split, field.name)[indices]
        for field in fields(collection.split) if field.name in array_fields
    }
    return collection.__class__(replace(collection.split, **updates), collection.role)


def run_formal_v45_pilot(
    *,
    config: str | Path,
    materialized_root: str | Path,
    train_data: str | Path,
    benchmark: str | Path,
    capacity_receipt: str | Path,
    output_root: str | Path,
    run_id: str,
    epoch_cap: int | None = None,
    max_batches: int | None = None,
    max_selection: int | None = None,
) -> dict[str, Any]:
    # Keep the Windows CPU runtime deterministic and avoid oversubscribing the
    # BLAS/torch worker pools during long rolling evaluation.
    torch.set_num_threads(1)
    contract = load_formal_v4_5_contract(config)
    materialized = load_v45_pilot_cache(
        materialized_root=materialized_root, train_data=train_data,
        benchmark=benchmark, capacity_receipt=capacity_receipt,
    )
    if max_batches is not None:
        limit = max(1, int(max_batches)) * int(contract.payload["pilot_budget"]["batch_size"])
        materialized = type(materialized)(
            train=_truncate_collection(materialized.train, limit),
            early_stop=_truncate_collection(materialized.early_stop, limit),
            selection_full=materialized.selection_full,
            selection_stress=materialized.selection_stress,
            normalization_source=materialized.normalization_source,
            lineage=materialized.lineage,
        )
    if max_selection is not None:
        selection_limit = max(1, int(max_selection))
        selection_indices = np.linspace(0, len(materialized.selection_full) - 1, min(selection_limit, len(materialized.selection_full)), dtype=np.int64)
        stress_indices = np.linspace(0, len(materialized.selection_stress) - 1, min(selection_limit, len(materialized.selection_stress)), dtype=np.int64)
        materialized = type(materialized)(
            train=materialized.train, early_stop=materialized.early_stop,
            selection_full=_select_collection(materialized.selection_full, selection_indices),
            selection_stress=_select_collection(materialized.selection_stress, stress_indices),
            normalization_source=materialized.normalization_source,
            lineage=materialized.lineage,
        )
    root = Path(output_root).resolve() / str(run_id)
    if root.exists():
        protected = (root / "pilot" / "PILOT_RECEIPT.json", root / "pilot" / "PILOT_AUDIT.json", root / "PILOT_TRANSITION.json")
        if any(path.exists() for path in protected):
            raise FileExistsError(root)
        # A failed run may have left only deterministic teacher caches. They
        # are safe to reuse; any other partial artifact requires a new run ID.
        partial = [path for path in root.rglob("*") if path.is_file() and "teacher" not in path.parts]
        if partial:
            raise FileExistsError(root)
    pilot_root = root / "pilot"
    pilot_root.mkdir(parents=True, exist_ok=True)
    result = execute_real_pilot_v45(
        materialized=materialized, contract=contract, artifact_root=pilot_root,
        benchmark=benchmark, capacity_receipt=capacity_receipt,
        seed=int(contract.payload["pilot_budget"]["seed"]), epoch_cap=epoch_cap,
    )
    joint_metrics = result["metrics"]["rsc_pf_joint"]
    receipt: dict[str, Any] = {
        "schema": "formal-v4.5-pilot-receipt-v1",
        "run_id": str(run_id),
        "contract_sha256": contract.contract_sha256,
        "lineage": {
            "source_manifest_sha256": sha256_file(Path(materialized_root) / "data" / "MATERIALIZED_LINEAGE.json"),
            "train_data_sha256": sha256_file(train_data),
            "selection_data_sha256": sha256_file(Path(materialized_root) / "data" / "selection_full.npz"),
            "selection_cache_lineage_sha256": str(materialized.lineage["source_materialized_lineage_sha256"]),
            "contract_sha256": contract.contract_sha256,
        },
        "accessed_years": [2015, 2016, 2017, 2018, 2019],
        "evaluation_year_accessed": False,
        "finite_values": [
            float(joint_metrics["four_task_score"]),
            float(result["joint"]["penalized_objective"]),
            float(result["decoupled"]["penalized_objective"]),
        ],
        "comparisons": result["comparisons"],
        "regime": {
            "transition_balanced_accuracy_gain": float(joint_metrics["transition"]["balanced_accuracy_gain"]),
            "macro_f1": float(joint_metrics["macro_f1"]),
            "prior_macro_f1": float(joint_metrics["prior_macro_f1"]),
        },
        "metrics": result["metrics"],
        "joint": result["joint"], "decoupled": result["decoupled"],
        "physics": result["physics"], "rows": result["rows"],
        "selection_lineage": dict(materialized.lineage),
    }
    decision = authorize_pilot_v44(receipt, contract)  # shared frozen criteria
    receipt["authorized_gate1"] = bool(decision.authorized_gate1)
    receipt["failures"] = list(decision.failures)
    receipt["criteria"] = dict(decision.criteria)
    write_json_once(pilot_root / "PILOT_RECEIPT.json", receipt)
    audit = {
        "schema": "formal-v4.5-pilot-audit-v1",
        "receipt_sha256": canonical_sha256(receipt),
        "metrics_sha256": canonical_sha256(result["metrics"]),
        "authorized_gate1": bool(decision.authorized_gate1),
        "failures": list(decision.failures),
        "evaluation_year_accessed": False,
    }
    audit_hash = write_json_once(pilot_root / "PILOT_AUDIT.json", audit)
    transition = {
        "schema": "formal-v4.5-pilot-transition-v1",
        "run_id": str(run_id),
        "contract_sha256": contract.contract_sha256,
        "authorized_gate1": bool(decision.authorized_gate1),
        "evaluation_year_accessed": False,
        "audit_sha256": audit_hash,
    }
    write_json_once(root / "PILOT_TRANSITION.json", transition)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--capacity-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--epoch-cap", type=int)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--max-selection", type=int)
    args = parser.parse_args(argv)
    try:
        receipt = run_formal_v45_pilot(
            config=args.config, materialized_root=args.materialized_root,
            train_data=args.train_data, benchmark=args.benchmark,
            capacity_receipt=args.capacity_receipt, output_root=args.output_root,
            run_id=args.run_id, epoch_cap=args.epoch_cap, max_batches=args.max_batches,
            max_selection=args.max_selection,
        )
    except Exception as exc:
        print(json.dumps({"authorized_gate1": False, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"authorized_gate1": receipt["authorized_gate1"], "failures": receipt["failures"]}, ensure_ascii=False))
    return 0 if receipt["authorized_gate1"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
