"""Read-only loading and source-data materialization for Stage 10.14."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FormalRun:
    track: str
    model: str
    seed: int
    scenario: str
    directory: Path
    rows: pd.DataFrame


@dataclass(frozen=True)
class FormalSchedulingData:
    runs: tuple[FormalRun, ...]
    oracles: Mapping[str, pd.DataFrame]
    origins: tuple[str, ...]

    def runs_for(self, track: str | None = None, scenario: str | None = None) -> tuple[FormalRun, ...]:
        return tuple(run for run in self.runs if (track is None or run.track == track) and (scenario is None or run.scenario == scenario))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_run(directory: Path, track: str, model: str, seed: int, scenario: str) -> FormalRun:
    manifest_path = directory / "run_manifest.json"
    rows_path = directory / "rows.csv"
    if not manifest_path.exists() or not rows_path.exists():
        raise FileNotFoundError(f"incomplete run: {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "scheduling-formal-run-v3" or manifest.get("status") != "complete":
        raise ValueError(f"invalid corrected run manifest: {directory}")
    for name, digest in manifest.get("file_sha256", {}).items():
        path = directory / name
        if not path.exists() or _sha256(path) != digest:
            raise ValueError(f"hash mismatch: {path}")
    frame = pd.read_csv(rows_path)
    if frame.empty or "origin" not in frame.columns:
        raise ValueError(f"empty or malformed rows: {rows_path}")
    # Formal rows already carry provenance columns.  Validate them rather
    # than inserting duplicates; this keeps the loader compatible with the
    # corrected runner's self-describing CSV schema.
    expected = {"track": track, "model": model, "seed": int(seed), "scenario": scenario}
    for column, value in expected.items():
        if column in frame.columns:
            observed = frame[column]
            if not bool((observed == value).all()):
                raise ValueError(f"provenance mismatch in {rows_path}: {column}")
        else:
            frame.insert(min(len(frame.columns), ["track", "model", "seed", "scenario"].index(column)), column, value)
    return FormalRun(track, model, int(seed), scenario, directory, frame)


def _expected_paths(formal_dir: Path, contract: Mapping[str, object]) -> list[tuple[str, str, int, str, Path]]:
    models = [str(v) for v in contract["models"]]
    seeds = [int(v) for v in contract["seeds"]]
    scenarios = [str(v) for v in contract["simulated_scenarios"]]
    result = []
    for model in models:
        for seed in seeds:
            result.append(("real_replay", model, seed, "real_replay", formal_dir / "real_replay" / model / f"seed_{seed}" / "real_replay"))
    for model in models:
        for seed in seeds:
            for scenario in scenarios:
                result.append(("simulated_dispatch", model, seed, scenario, formal_dir / "simulated_dispatch" / model / f"seed_{seed}" / scenario))
    return result


def load_formal_runs(formal_dir: Path, contract: Mapping[str, object]) -> FormalSchedulingData:
    """Load and validate corrected R/S/oracle outputs without mutation."""

    formal_dir = Path(formal_dir)
    runs = tuple(_load_run(path, track, model, seed, scenario) for track, model, seed, scenario, path in _expected_paths(formal_dir, contract))
    origin_sets = [set(run.rows["origin"].astype(str)) for run in runs]
    if not origin_sets or any(values != origin_sets[0] for values in origin_sets[1:]):
        raise ValueError("formal runs do not have an identical origin set")
    origins = tuple(sorted(origin_sets[0]))
    oracles: dict[str, pd.DataFrame] = {}
    for scenario in (str(v) for v in contract["simulated_scenarios"]):
        directory = formal_dir / "oracle" / scenario
        manifest = directory / "run_manifest.json"
        rows = directory / "rows.csv"
        if not manifest.exists() or not rows.exists():
            raise FileNotFoundError(f"missing oracle cache: {directory}")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "scheduling-oracle-v1" or payload.get("status") != "complete":
            raise ValueError(f"invalid oracle manifest: {directory}")
        table = pd.read_csv(rows)
        if set(table["origin"].astype(str)) != set(origins):
            raise ValueError(f"oracle origins are not aligned: {directory}")
        oracles[scenario] = table
    return FormalSchedulingData(runs=runs, oracles=oracles, origins=origins)


def _daily(frame: pd.DataFrame, value_columns: list[str], sum_columns: set[str]) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["origin"]).dt.strftime("%Y-%m-%d")
    aggregations = {
        column: ("sum" if column in sum_columns else "mean")
        for column in value_columns
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column])
    }
    grouped = data.groupby(["track", "model", "seed", "scenario", "date"], as_index=False).agg(aggregations)
    counts = data.groupby(["track", "model", "seed", "scenario", "date"], as_index=False).size().rename(columns={"size": "hour_count"})
    return grouped.merge(counts, on=["track", "model", "seed", "scenario", "date"], how="left")


def materialize_source_data(data: FormalSchedulingData, output_dir: Path) -> dict[str, Path]:
    """Create deterministic CSV source data used by tables, statistics and figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    all_rows = pd.concat([run.rows for run in data.runs], ignore_index=True)
    r_hourly = all_rows[all_rows["track"] == "real_replay"].copy()
    s_hourly = all_rows[all_rows["track"] == "simulated_dispatch"].copy()
    s_core = s_hourly[s_hourly["scenario"] == "core"].copy()
    # ``seed`` is a grouping key, not a quantity to average.  Excluding it
    # also keeps the daily scenario table free of a duplicate seed column.
    s_numeric_cols = [
        column
        for column in s_hourly.select_dtypes(include=[np.number]).columns.tolist()
        if column != "seed"
    ]
    for name, frame in {
        "r_hourly_executed.csv": r_hourly,
        "s_hourly_core.csv": s_core,
        "s_daily_scenarios_by_seed.csv": _daily(s_hourly, s_numeric_cols, {"grid_upward_energy", "grid_downward_energy", "gas_upward_energy", "gas_downward_energy", "total_imbalance_cost", "realized_cost", "planning_deviation_cost", "regret", "realized_carbon_first_step", "realized_curtailment_first_step"}),
        "s_equipment_hourly_core.csv": s_core,
    }.items():
        path = output_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = path
    r_value_cols = [c for c in r_hourly.columns if c not in {"track", "model", "seed", "scenario", "origin", "season"}]
    s_value_cols = [c for c in s_core.columns if c not in {"track", "model", "seed", "scenario", "origin"}]
    for name, frame, cols, sums in (
        ("r_daily_by_seed.csv", r_hourly, r_value_cols, {"grid_upward_energy", "grid_downward_energy", "gas_upward_energy", "gas_downward_energy", "total_imbalance_cost"}),
        ("s_daily_core_by_seed.csv", s_core, s_value_cols, {"realized_cost", "planning_deviation_cost", "regret", "realized_carbon_first_step", "realized_curtailment_first_step"}),
    ):
        path = output_dir / name
        _daily(frame, cols, sums).to_csv(path, index=False, encoding="utf-8-sig")
        outputs[name] = path
    oracle = pd.concat([table.assign(scenario=scenario) for scenario, table in data.oracles.items()], ignore_index=True)
    oracle_path = output_dir / "s_oracle_hourly.csv"
    oracle.to_csv(oracle_path, index=False, encoding="utf-8-sig")
    outputs["s_oracle_hourly.csv"] = oracle_path
    exclusions = []
    for date, group in s_core.groupby(pd.to_datetime(s_core["origin"]).dt.strftime("%Y-%m-%d")):
        if len(group["origin"].unique()) < 24:
            exclusions.append({"date": date, "hour_count": int(len(group["origin"].unique())), "reason": "partial_final_day_excluded_from_day_inference"})
    exclusion_path = output_dir / "result_exclusions.csv"
    pd.DataFrame(exclusions).to_csv(exclusion_path, index=False, encoding="utf-8-sig")
    outputs["result_exclusions.csv"] = exclusion_path
    audit_path = output_dir / "input_audit.json"
    audit_path.write_text(json.dumps({"schema_version": "source-data-v1", "run_count": len(data.runs), "origin_count": len(data.origins), "tracks_separate": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["input_audit.json"] = audit_path
    return outputs
