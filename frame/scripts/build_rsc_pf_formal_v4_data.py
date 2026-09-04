"""Build the pre-Gate0 causal base archives for formal v4.

This command intentionally stops before solving an LP or materializing device
histories.  Gate 0 chooses the capacity scenario first; only then may
``materialize_state_windows`` create train/selection windows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_protocol_v4 import SCHEMA_VERSION_V41, load_formal_v4_spec  # noqa: E402
from src.joint_dispatch.formal_v4_capacity import select_capacity_origins  # noqa: E402
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries, FormalV4Normalization, materialize_state_windows  # noqa: E402
from src.joint_dispatch.formal_v4_history import SettledTrajectory, generate_settled_device_trajectory  # noqa: E402
from src.joint_dispatch.formal_v4_artifacts import build_artifact_manifest, build_normalization_receipt, sha256_file  # noqa: E402
from src.joint_dispatch.contract import DISPATCH_ORDER  # noqa: E402
from src.scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp  # noqa: E402
from src.kitakyushu_pipeline import clean_kitakyushu_dataframe, read_kitakyushu_canonical  # noqa: E402
from src.scheduling.renewables import pv_available, wt_available  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ledger(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "parameter_id" not in rows.fieldnames or "value" not in rows.fieldnames:
            raise ValueError("parameter ledger must contain parameter_id and value columns")
        return {str(row["parameter_id"]): float(row["value"]) for row in rows if row.get("value") not in (None, "")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_1.json")
    parser.add_argument("--mode", choices=("base", "materialize-state"), required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "selection", "evaluation"), default=("train", "selection"))
    parser.add_argument("--kitakyushu-data-dir", type=Path, default=Path(r"D:\Paper\Kitakyushu dataset"))
    parser.add_argument("--audit-origins", type=int, default=None, help="legacy override; formal-v4.1 always uses the frozen 500-origin manifest")
    parser.add_argument("--capacity-receipt", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--benchmark-path", type=Path, default=None)
    return parser


def _years_for_split(split: str) -> tuple[int, ...]:
    return {"train": (2015, 2016, 2017, 2018), "selection": (2019,), "evaluation": (2020,)}[split]


def _build_base(frame: pd.DataFrame, split: str, parameters: dict[str, float]) -> FormalV4BaseSeries:
    frame = frame.loc[pd.to_datetime(frame["timestamp"]).dt.year.isin(_years_for_split(split))].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"no canonical rows for {split}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise").to_numpy(dtype="datetime64[ns]")
    load_and_exog = frame[["electricity", "cooling", "heating", "gas", "temperature", "humidity", "solar_irradiance", "wind_speed", "wind_direction", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend"]].to_numpy(dtype=np.float64)
    profile_parameters = {
        "pv_rated_capacity": float(parameters.get("pv_capacity", 1.0)),
        "pv_reference_irradiance": float(parameters["pv_reference_irradiance"]),
        "pv_conversion_efficiency": float(parameters["pv_conversion_efficiency"]),
        "pv_reference_temperature": float(parameters["pv_reference_temperature"]),
        "pv_temperature_coefficient": float(parameters["pv_temperature_coefficient"]),
        "wt_rated_capacity": float(parameters.get("wt_capacity", 1.0)),
        "wt_cut_in_speed": float(parameters["wt_cut_in_speed"]),
        "wt_rated_speed": float(parameters["wt_rated_speed"]),
        "wt_cut_out_speed": float(parameters["wt_cut_out_speed"]),
    }
    realized = np.column_stack((pv_available(frame, profile_parameters), wt_available(frame, profile_parameters)))
    forecast = np.vstack((realized[:1], realized[:-1]))
    carbon = float(parameters.get("carbon_price_default", parameters.get("carbon_price", 0.0)))
    prices = np.tile(np.asarray([parameters.get("grid_energy_price", 1.0), parameters.get("gas_energy_price", 1.0), carbon], dtype=np.float64), (len(frame), 1))
    return FormalV4BaseSeries(load_and_exog, forecast, realized, prices, timestamps, split)


def _save_base(base: FormalV4BaseSeries, path: Path, source_hashes: dict[str, str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal-v4 base archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, load_and_exog=base.load_and_exog, renewable_forecast=base.renewable_forecast,
        renewable_realized=base.renewable_realized, prices_and_weights=base.prices_and_weights,
        timestamps=base.timestamps.astype("datetime64[ns]"), split=np.asarray(base.split),
    )
    metadata = {"schema_version": "formal-v4-base-series-v1", "split": base.split, "source_hashes": source_hashes, "rows": len(base.timestamps)}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _load_base(path: Path) -> FormalV4BaseSeries:
    with np.load(path, allow_pickle=False) as payload:
        return FormalV4BaseSeries(
            payload["load_and_exog"], payload["renewable_forecast"],
            payload["renewable_realized"], payload["prices_and_weights"],
            payload["timestamps"], str(np.asarray(payload["split"]).item()),
        )


def _resolve_benchmark_path(spec, *, run_root: Path | None, benchmark_path: Path | None) -> Path:
    if benchmark_path is not None:
        resolved = benchmark_path.resolve()
    elif spec.schema_version == SCHEMA_VERSION_V41:
        if run_root is None:
            raise ValueError("formal-v4.1 data building requires an explicit --run-root")
        resolved = (run_root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml").resolve()
    else:
        resolved = Path(spec.paths["benchmark_path"]).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"resolved benchmark does not exist: {resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if spec.schema_version == SCHEMA_VERSION_V41:
        if not isinstance(payload, dict) or payload.get("schema_version") != "standard-ies-benchmark-v4.1":
            raise ValueError("formal-v4.1 requires a generated standard-ies-benchmark-v4.1 benchmark")
        if tuple(payload.get("source_years", ())) != (2015, 2016, 2017, 2018):
            raise ValueError("formal-v4.1 benchmark must be bounded to 2015-2018")
    return resolved


def _causal_trajectory(base: FormalV4BaseSeries, parameters: dict[str, float], capacity_receipt: Path) -> SettledTrajectory:
    """Generate the causal trajectory through the canonical settled transition."""

    return generate_settled_device_trajectory(
        base,
        parameters,
        capacity_receipt=capacity_receipt,
        trajectory_id=f"formal_v4_1_causal_settled_{base.split}",
    )


def _save_materialized(split, path: Path, normalization: FormalV4Normalization | None, metadata: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal-v4 materialized archive: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "load_history": split.load_history, "exog_history": split.exog_history,
        "renewable_history": split.renewable_history, "device_history": split.device_history,
        "activity_history": split.activity_history, "forecast_target": split.forecast_target,
        "rigid_demand": split.rigid_demand, "renewable_forecast": split.renewable_forecast,
        "renewable_realized": split.renewable_realized, "prices_and_weights": split.prices_and_weights,
        "initial_soc": split.initial_soc, "previous_chp": split.previous_chp,
        "target_times": split.target_times, "trajectory_ids": split.trajectory_ids,
        "state_hashes": split.state_hashes, "split": np.asarray(split.split),
        "history_source": np.asarray(split.history_source),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    if normalization is not None:
        for name in ("load_mean", "load_scale", "exog_mean", "exog_scale", "device_mean", "device_scale", "activity_mean", "activity_scale", "scheduler_mean", "scheduler_scale"):
            payload[f"normalization_{name}"] = getattr(normalization, name)
        payload["normalization_fitted_split"] = np.asarray(normalization.fitted_split)
    np.savez_compressed(path, **payload)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> int:
    args = _parser().parse_args()
    spec = load_formal_v4_spec(args.contract)
    run_root = args.run_root.resolve() if args.run_root is not None else None
    benchmark_path = _resolve_benchmark_path(spec, run_root=run_root, benchmark_path=args.benchmark_path)
    if args.mode == "materialize-state":
        if run_root is None:
            raise ValueError("formal-v4.1 materialization requires an explicit --run-root")
        if args.capacity_receipt is None:
            raise PermissionError("--capacity-receipt is required for state materialization")
        receipt = json.loads(args.capacity_receipt.read_text(encoding="utf-8"))
        if receipt.get("gate0_authorized") is not True:
            raise PermissionError("capacity receipt is not Gate 0 authorized")
        data_root = run_root / "data" if spec.schema_version == SCHEMA_VERSION_V41 else Path(spec.paths["data_root"])
        bases = [_load_base(data_root / f"base_{split}.npz") for split in args.splits]
        if not bases:
            raise ValueError("at least one split is required")
        # Build the trajectory on a single chronological stream so selection
        # starts from the carried state at the end of training.
        combined = FormalV4BaseSeries(
            np.concatenate([item.load_and_exog for item in bases]),
            np.concatenate([item.renewable_forecast for item in bases]),
            np.concatenate([item.renewable_realized for item in bases]),
            np.concatenate([item.prices_and_weights for item in bases]),
            np.concatenate([item.timestamps for item in bases]), "train",
        )
        benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
        parameters = dict(benchmark["values"])
        for key, value in _ledger(FRAME_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv").items():
            parameters.setdefault(key, value)
        trajectory = _causal_trajectory(combined, parameters, args.capacity_receipt)
        by_split = {}
        offset = 0
        for base in bases:
            piece = FormalV4BaseSeries(
                base.load_and_exog, base.renewable_forecast, base.renewable_realized,
                base.prices_and_weights, base.timestamps, base.split,
            )
            by_split[base.split] = materialize_state_windows(
                piece, trajectory.settled_dispatch[offset:offset + len(piece.timestamps)], capacity_receipt=args.capacity_receipt, split=base.split,
                settled_mask=trajectory.settled_mask[offset:offset + len(piece.timestamps)],
                trajectory_hash=trajectory.trajectory_sha256,
                bess_energy_capacity=float(receipt["bess_energy_capacity"]),
            )
            offset += len(piece.timestamps)
        normalization = FormalV4Normalization.fit(by_split["train"]) if "train" in by_split else None
        output_root = run_root / "data"
        for split, windows in by_split.items():
            _save_materialized(windows, output_root / f"{split}.npz", normalization if split == "train" else None, {"schema_version": "formal-v4-materialized-v1", "capacity_receipt": str(args.capacity_receipt), "split": split, "rows": len(windows), "trajectory_id": trajectory.trajectory_id, "trajectory_audit": trajectory.audit.to_payload()})
        if "train" in by_split and "selection" in by_split:
            capacity_hash = sha256_file(args.capacity_receipt)
            train_hash = sha256_file(output_root / "train.npz")
            normalization_payload = build_normalization_receipt(
                normalization,
                train_archive_sha256=train_hash,
                capacity_receipt_sha256=capacity_hash,
                train_timestamps=by_split["train"].target_times,
            )
            normalization_path = run_root / "NORMALIZATION_RECEIPT.json"
            if normalization_path.exists():
                raise FileExistsError(f"refusing to overwrite normalization receipt: {normalization_path}")
            normalization_path.write_text(json.dumps(normalization_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            artifact_manifest = build_artifact_manifest(
                run_root,
                [output_root / "train.npz", output_root / "selection.npz", normalization_path],
                lineage={"capacity_receipt_sha256": capacity_hash, "capacity_scenario_hash": str(receipt.get("capacity_scenario_hash", ""))},
            )
            artifact_manifest.save(run_root / "ARTIFACT_MANIFEST.json")
        print(json.dumps({"status": "pass", "mode": args.mode, "splits": list(by_split), "rows": {key: len(value) for key, value in by_split.items()}, "output_root": str(output_root)}, ensure_ascii=False))
        return 0
    years = tuple(sorted({year for split in args.splits for year in _years_for_split(split)}))
    raw, source_metadata = read_kitakyushu_canonical(args.kitakyushu_data_dir, years=years)
    frame, cleaning = clean_kitakyushu_dataframe(raw)
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    parameters = dict(benchmark["values"])
    # The ledger records both final constants and derivation ratios (for
    # example, ``chp_electric_capacity=0.35`` means 0.35 times training P95,
    # not a 0.35-kW dispatch capacity).  The frozen benchmark contains the
    # resolved dispatch values, so it must take precedence; the ledger only
    # supplies auxiliary profile constants absent from the benchmark.
    for key, value in _ledger(FRAME_ROOT / "configs" / "scheduling_parameter_ledger_v2.csv").items():
        parameters.setdefault(key, value)
    hashes = {str(key): str(value.get("sha256", "")) for key, value in source_metadata.get("source_files", {}).items() if isinstance(value, dict)}
    output_root = run_root / "data" if run_root is not None else Path(spec.paths["data_root"])
    train_base: FormalV4BaseSeries | None = None
    for split in args.splits:
        base = _build_base(frame, split, parameters)
        _save_base(base, output_root / f"base_{split}.npz", hashes)
        if split == "train":
            train_base = base
    audit_root = run_root / "audit" if run_root is not None else Path(spec.paths["audit_root"])
    audit_root.mkdir(parents=True, exist_ok=True)
    origin_count = 0
    origin_manifest_path = None
    if train_base is not None:
        manifest = select_capacity_origins(train_base, {"capacity": spec.capacity})
        origin_manifest_path = audit_root / "capacity_origins_v4_1.json"
        manifest.save(origin_manifest_path)
        origin_count = len(manifest.origin_indices)
    np.savez_compressed(audit_root / "capacity_inputs.npz", split=np.asarray(args.splits), rows=np.asarray([len(frame)]), audit_origins=np.asarray(origin_count))
    (audit_root / "base_input_receipt.json").write_text(json.dumps({"source_metadata": source_metadata, "cleaning": cleaning, "source_hashes": hashes, "capacity_origin_manifest": str(origin_manifest_path) if origin_manifest_path else None, "capacity_origin_count": origin_count}, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "mode": "base", "splits": list(args.splits), "output_root": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
