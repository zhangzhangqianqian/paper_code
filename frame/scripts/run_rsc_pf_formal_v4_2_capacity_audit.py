"""Build the train-only capacity evidence used by formal-v4.2 Gate 0."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from scripts.build_rsc_pf_formal_v4_benchmark import build_from_data  # noqa: E402
from scripts.build_rsc_pf_formal_v4_data import _build_base, _save_base  # noqa: E402
from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_contract import FormalV42Contract, load_formal_v4_2_contract  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate0 import CAPACITY_SCHEMA  # noqa: E402
from src.joint_dispatch.formal_v4_access import FormalV4AccessController  # noqa: E402
from src.joint_dispatch.formal_v4_capacity import run_capacity_audit, select_capacity_origins  # noqa: E402
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries  # noqa: E402
from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical  # noqa: E402
from src.scheduling.dispatch_lp import solve_dispatch_lp  # noqa: E402


class CapacityEvidenceError(RuntimeError):
    """Raised when no capacity candidate passes both frozen audit stages."""


def capacity_years_for_split(contract: FormalV42Contract, split: str) -> tuple[int, ...]:
    if split == "train":
        return tuple(int(year) for year in contract.train_years)
    if split == "selection":
        return (int(contract.selection_year),)
    raise CapacityEvidenceError(f"forbidden capacity split: {split}")


def _validate_base_years(base: FormalV4BaseSeries, expected: tuple[int, ...], split: str) -> None:
    years = tuple(sorted(set(int(year) for year in pd.DatetimeIndex(base.timestamps).year)))
    if years != expected or base.split != split:
        raise CapacityEvidenceError(f"{split} base years mismatch: {years} != {expected}")


def _ledger(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "parameter_id" not in reader.fieldnames or "value" not in reader.fieldnames:
            raise CapacityEvidenceError("parameter ledger is missing parameter_id/value")
        return {
            str(row["parameter_id"]): float(row["value"])
            for row in reader
            if row.get("parameter_id") and row.get("value") not in (None, "")
        }


def _read_split(
    data_dir: Path,
    split: str,
    contract: FormalV42Contract,
    controller: FormalV4AccessController,
) -> tuple[Any, Mapping[str, Any], Any]:
    years = capacity_years_for_split(contract, split)

    def guard(source_kind: str, year: int, archive: Path, member: str) -> None:
        controller.guard_archive_member(
            archive,
            member,
            split=split,
            purpose=f"formal-v4.2-{split}-capacity-read",
            caller="run_rsc_pf_formal_v4_2_capacity_audit",
            years=(year,),
        )

    def record(event: dict[str, object]) -> None:
        controller.record_archive_event(
            str(event["container_path"]),
            str(event["member_name"]),
            split=split,
            purpose=f"formal-v4.2-{split}-capacity-read",
            caller="run_rsc_pf_formal_v4_2_capacity_audit",
            years=(int(event["year"]),),
            container_sha256=str(event["container_sha256"]),
            member_sha256=str(event["member_sha256"]),
        )

    raw, metadata = read_kitakyushu_canonical(
        data_dir,
        years=years,
        audit_sink=record,
        access_guard=guard,
    )
    frame, cleaning = clean_kitakyushu_dataframe(raw)
    return frame, metadata, cleaning


def _source_hashes(metadata: Mapping[str, Any]) -> dict[str, str]:
    sources = metadata.get("source_files", {})
    if not isinstance(sources, Mapping):
        return {}
    return {
        str(name): str(row.get("sha256", ""))
        for name, row in sources.items()
        if isinstance(row, Mapping)
    }


def build_capacity_evidence_from_bases(
    train_base: FormalV4BaseSeries,
    selection_base: FormalV4BaseSeries,
    parameters: Mapping[str, Any],
    contract: FormalV42Contract,
    run_root: str | Path,
    source_manifest_sha256: str,
    *,
    solver: Any = solve_dispatch_lp,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    _validate_base_years(train_base, tuple(contract.train_years), "train")
    _validate_base_years(selection_base, (contract.selection_year,), "selection")
    capacity = dict(contract.capacity)
    candidates = tuple(float(value) for value in capacity["candidate_multipliers"])
    thresholds = {
        "cooling_shortage_energy_ratio_max": float(capacity["cooling_shortage_energy_ratio_max"]),
        "cooling_shortage_hour_rate_max": float(capacity["cooling_shortage_hour_rate_max"]),
    }
    origin_manifest = select_capacity_origins(train_base, {"capacity": capacity})
    origin_payload = {
        "schema": "formal-v4.2-capacity-origin-manifest-v1",
        "run_id": root.name,
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "evaluation_year_accessed": False,
        "origin_manifest": origin_manifest.to_payload(),
    }
    write_once_json(root / "audit" / "CAPACITY_ORIGINS.json", origin_payload)
    audit = run_capacity_audit(
        train_base,
        parameters,
        origin_manifest,
        candidates,
        cooling_shortage_energy_ratio_max=thresholds["cooling_shortage_energy_ratio_max"],
        cooling_shortage_hour_rate_max=thresholds["cooling_shortage_hour_rate_max"],
        solver=solver,
    )
    audit_payload = {
        "schema": "formal-v4.2-capacity-audit-result-v1",
        "run_id": root.name,
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "fit_years": list(contract.train_years),
        "selection_influenced_capacity": False,
        "evaluation_year_accessed": False,
        **audit.to_payload(),
    }
    write_once_json(root / "gate0" / "CAPACITY_AUDIT_RESULT.json", audit_payload)
    if audit.status != "pass" or audit.selected is None:
        stage_one_passed = any(bool(row.get("meets_threshold")) for row in audit.stage_one)
        stage_two_passed = any(bool(row.get("meets_threshold")) for row in audit.stage_two)
        failed_stage = "diagnostic" if not stage_one_passed else "chronological"
        if stage_one_passed and stage_two_passed:
            failed_stage = "paired diagnostic/chronological"
        raise CapacityEvidenceError(f"capacity {failed_stage} audit did not produce a common passing multiplier")
    selected_multiplier = float(audit.selected["multiplier"])
    matching_stage_one = next((row for row in audit.stage_one if float(row["multiplier"]) == selected_multiplier), None)
    matching_stage_two = next((row for row in audit.stage_two if float(row["multiplier"]) == selected_multiplier), None)
    if not matching_stage_one or not matching_stage_two or not matching_stage_one.get("meets_threshold") or not matching_stage_two.get("meets_threshold"):
        raise CapacityEvidenceError("capacity diagnostic and chronological receipts disagree")
    freeze = {
        "schema": CAPACITY_SCHEMA,
        "run_id": root.name,
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "status": "pass",
        "fit_years": list(contract.train_years),
        "selection_years_materialized": [contract.selection_year],
        "selection_influenced_capacity": False,
        "evaluation_year_accessed": False,
        "selected": dict(audit.selected),
        "candidate_multipliers": list(audit.candidate_multipliers),
        "thresholds": dict(audit.thresholds),
        "diagnostic_row": dict(matching_stage_one),
        "chronological_row": dict(matching_stage_two),
        "origin_manifest_sha256": origin_manifest.manifest_sha256,
        "source_base_sha256": audit.source_base_sha256,
        "resolved_parameter_sha256": audit.resolved_parameter_sha256,
        "capacity_scenario_hash": audit.capacity_scenario_hash,
        "solver_identity": audit.solver_identity,
    }
    write_once_json(root / "gate0" / "CAPACITY_FREEZE.json", freeze)
    return freeze


def build_capacity_evidence(
    contract: FormalV42Contract,
    run_root: str | Path,
    data_dir: str | Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_dir).resolve()
    if not data_path.is_dir():
        raise CapacityEvidenceError(f"Kitakyushu data directory is missing: {data_path}")
    controller = FormalV4AccessController(allow_evaluation=False)
    rules_path = FRAME_ROOT / "configs" / "standard_ies_formal_v4_rules.yaml"
    ledger_path = FRAME_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv"
    build_from_data(data_path, rules_path, ledger_path, root, access_controller=controller)
    benchmark_path = root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    parameters = dict(benchmark["values"])
    for key, value in _ledger(ledger_path).items():
        parameters.setdefault(key, value)
    bases: dict[str, FormalV4BaseSeries] = {}
    input_receipt: dict[str, Any] = {}
    for split in ("train", "selection"):
        frame, metadata, cleaning = _read_split(data_path, split, contract, controller)
        base = _build_base(frame, split, parameters)
        _validate_base_years(base, capacity_years_for_split(contract, split), split)
        _save_base(base, root / "data" / f"base_{split}.npz", _source_hashes(metadata))
        bases[split] = base
        input_receipt[split] = {
            "years": list(capacity_years_for_split(contract, split)),
            "rows": len(base.timestamps),
            "source_files": _source_hashes(metadata),
            "cleaning": cleaning,
        }
    write_once_json(root / "audit" / "BASE_INPUT_RECEIPT.json", {
        "schema": "formal-v4.2-base-input-receipt-v1",
        "run_id": root.name,
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "splits": input_receipt,
        "evaluation_year_accessed": False,
    })
    write_once_json(root / "protocol" / "DATA_ACCESS_RECEIPT.json", {
        "schema": "formal-v4.2-data-access-v1",
        "run_id": root.name,
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "events": controller.build_data_access_receipt()["events"],
        "evaluation_year_accessed": False,
    })
    write_once_json(root / "protocol" / "ARCHIVE_ACCESS_RECEIPT.json", {
        "schema": "formal-v4.2-archive-access-v1",
        "run_id": root.name,
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "events": controller.build_archive_access_receipt()["events"],
        "evaluation_year_accessed": False,
    })
    return build_capacity_evidence_from_bases(
        bases["train"],
        bases["selection"],
        parameters,
        contract,
        root,
        source_manifest_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Paper\Kitakyushu dataset"))
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    contract = load_formal_v4_2_contract(args.contract)
    try:
        result = build_capacity_evidence(contract, args.run_root, args.data_dir, args.source_manifest_sha256)
    except Exception as exc:
        print(json.dumps({"status": "fail", "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "pass", "selected": result["selected"], "run_root": str(args.run_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CapacityEvidenceError",
    "build_capacity_evidence",
    "build_capacity_evidence_from_bases",
    "capacity_years_for_split",
    "main",
]
