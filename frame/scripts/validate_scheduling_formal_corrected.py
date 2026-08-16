"""Validate the corrected Stage 10.14 scheduling result directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


S_COLUMNS = {
    "origin", "horizon_objective", "planned_cost_first_step", "realized_cost",
    "planning_deviation_cost", "planned_carbon_first_step", "realized_carbon_first_step",
    "planned_curtailment_first_step", "realized_curtailment_first_step", "regret",
    "oracle_realized_cost", "oracle_realized_carbon", "oracle_unserved_electricity",
    "oracle_unserved_cooling", "oracle_unserved_heating", "oracle_soc",
}
R_COLUMNS = {
    "origin", "grid_mae_executed", "gas_mae_executed", "grid_upward_energy",
    "grid_downward_energy", "gas_upward_energy", "gas_downward_energy", "total_imbalance_cost",
}


def _finite_numeric(frame: pd.DataFrame, columns: set[str]) -> bool:
    numeric = frame[list(columns - {"origin"})].apply(pd.to_numeric, errors="coerce")
    return bool(np.isfinite(numeric.to_numpy(dtype=float)).all())


def validate_run_table(frame: pd.DataFrame, track: str, expected_origins: set[str], tolerance: float = 1e-9) -> dict[str, object]:
    required = S_COLUMNS if track == "simulated_dispatch" else R_COLUMNS
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{track} rows missing columns: {missing}")
    origins = set(frame["origin"].astype(str))
    if origins != expected_origins:
        raise ValueError(f"{track} origins are not exactly aligned")
    if len(frame) != len(expected_origins):
        raise ValueError(f"{track} expected {len(expected_origins)} rows, got {len(frame)}")
    if not _finite_numeric(frame, required):
        raise ValueError(f"{track} rows contain NaN/Inf")
    if track == "simulated_dispatch":
        deviation = frame["realized_cost"] - frame["planned_cost_first_step"]
        if not np.allclose(deviation, frame["planning_deviation_cost"], atol=tolerance, rtol=0.0):
            raise ValueError("planning_deviation_cost identity failed")
        regret = frame["realized_cost"] - frame["oracle_realized_cost"]
        if not np.allclose(regret, frame["regret"], atol=tolerance, rtol=0.0):
            raise ValueError("regret identity failed")
        carbon_delta = frame["realized_carbon_first_step"] - frame["oracle_realized_carbon"]
        if "carbon_delta_vs_oracle" in frame and not np.allclose(carbon_delta, frame["carbon_delta_vs_oracle"], atol=tolerance, rtol=0.0):
            raise ValueError("carbon_delta_vs_oracle identity failed")
    return {"rows": int(len(frame)), "columns": sorted(frame.columns), "finite": True}


def _expected_run_dirs(contract: Mapping[str, object], root: Path) -> list[tuple[str, str, int, str, Path]]:
    models = [str(value) for value in contract["models"]]
    seeds = [int(value) for value in contract["seeds"]]
    scenarios = [str(value) for value in contract["simulated_scenarios"]]
    result: list[tuple[str, str, int, str, Path]] = []
    for model in models:
        for seed in seeds:
            result.append(("real_replay", model, seed, "real_replay", root / "real_replay" / model / f"seed_{seed}" / "real_replay"))
    for model in models:
        for seed in seeds:
            for scenario in scenarios:
                result.append(("simulated_dispatch", model, seed, scenario, root / "simulated_dispatch" / model / f"seed_{seed}" / scenario))
    return result


def validate_formal_directory(formal_dir: Path, contract: Mapping[str, object]) -> dict[str, object]:
    root_manifest_path = formal_dir / "formal_scheduling_manifest.json"
    if not root_manifest_path.exists():
        raise FileNotFoundError(root_manifest_path)
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    if root_manifest.get("status") != "complete":
        raise ValueError(f"formal root status is not complete: {root_manifest.get('status')}")
    if root_manifest.get("schema_version") != "scheduling-formal-root-v3":
        raise ValueError("formal root schema is not scheduling-formal-root-v3")
    expected_origins: set[str] | None = None
    expected_run_dirs = _expected_run_dirs(contract, formal_dir)
    run_reports: list[dict[str, object]] = []
    failures: list[str] = []
    for track, model, seed, scenario, run_dir in _expected_run_dirs(contract, formal_dir):
        try:
            manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "scheduling-formal-run-v3" or manifest.get("status") != "complete":
                raise ValueError("invalid run manifest schema/status")
            frame = pd.read_csv(run_dir / "rows.csv")
            origins = set(frame["origin"].astype(str))
            if expected_origins is None:
                expected_origins = origins
            report = validate_run_table(frame, track, expected_origins)
            report.update({"track": track, "model": model, "seed": seed, "scenario": scenario})
            run_reports.append(report)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{track}/{model}/{seed}/{scenario}: {exc}")
    if expected_origins is None:
        failures.append("no model runs discovered")
    expected_oracles = [str(value) for value in contract["simulated_scenarios"]]
    oracle_reports = []
    for scenario in expected_oracles:
        oracle_dir = formal_dir / "oracle" / scenario
        try:
            manifest = json.loads((oracle_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "scheduling-oracle-v1" or manifest.get("status") != "complete":
                raise ValueError("invalid oracle manifest schema/status")
            table = pd.read_csv(oracle_dir / "rows.csv")
            oracle_origins = set(table["origin"].astype(str))
            if expected_origins is not None and oracle_origins != expected_origins:
                raise ValueError("oracle origins are not aligned")
            oracle_reports.append({"scenario": scenario, "rows": int(len(table))})
        except Exception as exc:  # noqa: BLE001
            failures.append(f"oracle/{scenario}: {exc}")
    expected_model_count = len(expected_run_dirs)
    expected_oracle_count = len(expected_oracles)
    result = {
        "schema_version": "scheduling-formal-acceptance-v1",
        "status": "passed" if not failures else "failed",
        "model_run_count": len(run_reports),
        "expected_model_run_count": expected_model_count,
        "oracle_run_count": len(oracle_reports),
        "expected_oracle_run_count": expected_oracle_count,
        "origin_count": len(expected_origins or set()),
        "runs": run_reports,
        "oracles": oracle_reports,
        "failures": failures,
    }
    if len(run_reports) != result["expected_model_run_count"] or len(oracle_reports) != result["expected_oracle_run_count"]:
        result["status"] = "failed"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.resolve().read_text(encoding="utf-8"))
    report = validate_formal_directory(args.formal_dir.resolve(), contract)
    output = args.formal_dir.resolve() / "formal_acceptance_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
