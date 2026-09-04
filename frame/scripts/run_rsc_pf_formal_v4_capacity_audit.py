"""Build and certify the training-only formal-v4.1 cooling capacity.

The runner creates a complete run-root benchmark/base bundle before solving the
two-stage capacity audit.  It never reads 2020 data and never starts model
training.  A capacity freeze is written only when one candidate passes both
the stratified and chronological checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from scripts.build_rsc_pf_formal_v4_data import _build_base, _save_base  # noqa: E402
from scripts.build_rsc_pf_formal_v4_benchmark import build_from_data  # noqa: E402
from src.joint_dispatch.formal_protocol_v4 import SCHEMA_VERSION_V41, load_formal_v4_spec  # noqa: E402
from src.joint_dispatch.formal_v4_capacity import (  # noqa: E402
    CapacityAuditReceipt,
    CapacityOriginManifest,
    run_capacity_audit,
    select_capacity_origins,
)
from src.joint_dispatch.formal_v4_artifacts import sha256_file  # noqa: E402
from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical  # noqa: E402


FREEZE_SCHEMA = "formal-v4-capacity-freeze-v1"
AUDIT_SCHEMA = "formal-v4.1-capacity-audit-result-v1"


def _ledger(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "parameter_id" not in rows.fieldnames or "value" not in rows.fieldnames:
            raise ValueError("parameter ledger must contain parameter_id and value columns")
        return {str(row["parameter_id"]): float(row["value"]) for row in rows if row.get("value") not in (None, "")}


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal-v4.1 capacity artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _load_base(path: Path):
    from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries

    with np.load(path, allow_pickle=False) as payload:
        return FormalV4BaseSeries(
            payload["load_and_exog"], payload["renewable_forecast"],
            payload["renewable_realized"], payload["prices_and_weights"],
            payload["timestamps"], str(np.asarray(payload["split"]).item()),
        )


def _load_origin_manifest(path: Path) -> CapacityOriginManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = {key: np.asarray(value, dtype=np.float64) for key, value in payload["demand_summaries"].items()}
    return CapacityOriginManifest(
        origin_indices=np.asarray(payload["origin_indices"], dtype=np.int64),
        origin_timestamps=np.asarray(payload["origin_timestamps"], dtype="datetime64[ns]"),
        strata=np.asarray(payload["strata"], dtype=str),
        demand_summaries=summary,
        source_base_sha256=str(payload["source_base_sha256"]),
        selection_config={str(key): int(value) for key, value in payload["selection_config"].items()},
        schema_version=str(payload.get("schema_version", "formal-v4.1-capacity-origin-manifest-v1")),
        manifest_sha256=str(payload.get("manifest_sha256", "")),
    )


def build_capacity_freeze_payload(
    audit: CapacityAuditReceipt,
    parameters: Mapping[str, Any],
    origin_manifest: CapacityOriginManifest,
    *,
    run_id: str,
    benchmark_receipt_sha256: str,
) -> dict[str, Any]:
    """Convert the solver receipt to the freeze schema consumed by Gate 0."""

    if audit.status != "pass" or audit.selected is None:
        raise ValueError("cannot freeze a capacity audit without a passing selection")
    selected = dict(audit.selected)
    diagnostic_row = dict(selected.get("diagnostic_row", {}))
    chronological_row = {key: value for key, value in selected.items() if key != "diagnostic_row"}
    if not diagnostic_row or not chronological_row:
        raise ValueError("selected capacity is missing diagnostic or chronological evidence")
    candidates = []
    for diagnostic, chronological in zip(audit.stage_one, audit.stage_two):
        candidates.append({
            "multiplier": float(diagnostic["multiplier"]),
            "diagnostic": dict(diagnostic),
            "chronological": dict(chronological),
            "meets_threshold": bool(diagnostic.get("meets_threshold") and chronological.get("meets_threshold")),
        })
    capacity_audit = {
        "status": "pass",
        "selected": selected,
        "diagnostic": {"status": "pass", "selected_multiplier": float(selected["multiplier"]), "row": diagnostic_row},
        "chronological": {"status": "pass", "selected_multiplier": float(selected["multiplier"]), "row": chronological_row},
        "candidates": candidates,
        "stage_one": [dict(row) for row in audit.stage_one],
        "stage_two": [dict(row) for row in audit.stage_two],
        "training_origins": 500,
        "origin_manifest_sha256": audit.origin_manifest_sha256,
        "source_base_sha256": audit.source_base_sha256,
        "resolved_parameter_sha256": audit.resolved_parameter_sha256,
        "benchmark_receipt_sha256": str(benchmark_receipt_sha256),
        "diagnostic_initial_state": dict(audit.diagnostic_initial_state),
        "full_state_reset_count": int(audit.full_state_reset_count),
        "full_timestamp_start": audit.full_timestamp_start,
        "full_timestamp_end": audit.full_timestamp_end,
        "solver_identity": audit.solver_identity,
        "candidate_multipliers": list(audit.candidate_multipliers),
        "thresholds": dict(audit.thresholds),
        "capacity_scenario_hash": audit.capacity_scenario_hash,
    }
    return {
        "schema_version": FREEZE_SCHEMA,
        "gate0_authorized": True,
        "run_id": str(run_id),
        "capacity_audit": capacity_audit,
        "bess_energy_capacity": float(parameters["bess_energy_capacity"]),
        "trajectory_id": f"{run_id}:capacity_bound_causal",
        "capacity_scenario_hash": audit.capacity_scenario_hash,
        "origin_manifest_sha256": origin_manifest.manifest_sha256,
    }


def _audit_result_payload(audit: CapacityAuditReceipt) -> dict[str, Any]:
    return {"schema_version": AUDIT_SCHEMA, **audit.to_payload()}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_1.json")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Paper\Kitakyushu dataset"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_formal_v4_spec(args.contract)
    if spec.schema_version != SCHEMA_VERSION_V41:
        raise ValueError("capacity certification runner requires the formal-v4.1 contract")
    run_root = args.run_root.resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"capacity certification run root is not fresh: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)

    benchmark_receipt = build_from_data(
        args.data_dir.resolve(),
        spec.benchmark_rule_config or (FRAME_ROOT / "configs" / "standard_ies_formal_v4_rules.yaml"),
        FRAME_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv",
        run_root,
    )
    benchmark_path = run_root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    parameters = dict(yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))["values"])
    for key, value in _ledger(FRAME_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv").items():
        parameters.setdefault(key, value)

    raw, source_metadata = read_kitakyushu_canonical(args.data_dir.resolve(), years=(2015, 2016, 2017, 2018, 2019))
    frame, cleaning = clean_kitakyushu_dataframe(raw)
    data_root = run_root / "data"
    audit_root = run_root / "audit"
    data_root.mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    train_base = None
    for split in ("train", "selection"):
        base = _build_base(frame, split, parameters)
        _save_base(base, data_root / f"base_{split}.npz", {
            str(key): str(value.get("sha256", ""))
            for key, value in source_metadata.get("source_files", {}).items()
            if isinstance(value, dict)
        })
        if split == "train":
            train_base = base
    if train_base is None:
        raise RuntimeError("training base was not built")
    origin_manifest = select_capacity_origins(train_base, {"capacity": spec.capacity})
    origin_path = audit_root / "capacity_origins_v4_1.json"
    origin_manifest.save(origin_path)
    (audit_root / "base_input_receipt.json").write_text(
        json.dumps({"source_metadata": source_metadata, "cleaning": cleaning, "capacity_origin_manifest": str(origin_path), "capacity_origin_count": 500}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audit = run_capacity_audit(
        train_base,
        parameters,
        origin_manifest,
        list(spec.capacity["candidate_multipliers"]),
    )
    _write_once(run_root / "gate0" / "CAPACITY_AUDIT_RESULT.json", _audit_result_payload(audit))
    if audit.status != "pass" or audit.selected is None:
        print(json.dumps({"status": "fail", "selected": None, "run_root": str(run_root)}, ensure_ascii=False))
        return 2
    freeze = build_capacity_freeze_payload(
        audit,
        parameters,
        origin_manifest,
        run_id=run_root.name,
        benchmark_receipt_sha256=sha256_file(run_root / "gate0" / "benchmark" / "FORMAL_V4_BENCHMARK_RECEIPT.json"),
    )
    _write_once(run_root / "gate0" / "CAPACITY_FREEZE.json", freeze)
    print(json.dumps({"status": "pass", "selected": freeze["capacity_audit"]["selected"], "run_root": str(run_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AUDIT_SCHEMA", "FREEZE_SCHEMA", "build_capacity_freeze_payload", "main"]
