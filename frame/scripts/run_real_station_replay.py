"""Stage 10.6: replay real-station purchase nominations and imbalance settlement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.data import build_scheduling_frame  # noqa: E402
from src.scheduling.real_replay import build_energy_nomination, settle_real_replay  # noqa: E402
from src.scheduling.imbalance_settlement import summarize_replay  # noqa: E402


def _load_run(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    manifest_path = path.parent / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run_manifest.json: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
        origins = np.asarray(payload["target_times"], dtype="datetime64[ns]")
    if prediction.ndim != 3 or prediction.shape[1:] != (4, 4):
        raise ValueError(f"Expected [N,4,4], got {prediction.shape}: {path}")
    if len(origins) != len(prediction):
        raise ValueError(f"Origin/prediction length mismatch: {path}")
    return prediction, origins, manifest


def _actual_windows(frame: pd.DataFrame, origins: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    actual = frame.copy()
    actual["timestamp"] = pd.to_datetime(actual["timestamp"])
    actual = actual.sort_values("timestamp").drop_duplicates("timestamp")
    timestamps = actual["timestamp"].to_numpy(dtype="datetime64[ns]")
    grid_values = actual["actual_grid_import"].to_numpy(dtype=np.float64)
    gas_values = actual["gas"].to_numpy(dtype=np.float64)
    origin_values = np.asarray(origins, dtype="datetime64[ns]")
    starts = np.searchsorted(timestamps, origin_values)
    offsets = np.arange(horizon, dtype=np.int64)[None, :]
    positions = starts[:, None] + offsets
    in_bounds = (starts < len(timestamps)) & (positions[:, -1] < len(timestamps))
    safe_positions = np.clip(positions, 0, max(0, len(timestamps) - 1))
    contiguous = np.zeros(len(origin_values), dtype=bool)
    if len(origin_values):
        matched = timestamps[safe_positions]
        expected = origin_values[:, None] + offsets.astype("timedelta64[h]")
        contiguous = np.all(matched == expected, axis=1)
    valid = in_bounds & contiguous
    valid_origins = origin_values[valid]
    valid_positions = safe_positions[valid]
    return {
        "origins": np.asarray(valid_origins, dtype="datetime64[ns]"),
        "actual_grid_import": grid_values[valid_positions],
        "gas": gas_values[valid_positions],
    }


def _renewable_windows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        predictions = np.asarray(payload["predictions"], dtype=np.float64)
        origins = np.asarray(payload["origin_times"], dtype="datetime64[ns]")
    if predictions.ndim != 3 or predictions.shape[1:] != (4, 2):
        raise ValueError(f"Expected renewable [N,4,2], got {predictions.shape}")
    return predictions, origins


def _season(timestamp: pd.Timestamp) -> str:
    if timestamp.month in (12, 1, 2):
        return "winter"
    if timestamp.month in (3, 4, 5):
        return "spring"
    if timestamp.month in (6, 7, 8):
        return "summer"
    return "autumn"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--renewable-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--grid-upward-price", type=float, default=1.15)
    parser.add_argument("--grid-downward-price", type=float, default=0.85)
    parser.add_argument("--gas-upward-price", type=float, default=0.70)
    parser.add_argument("--gas-downward-price", type=float, default=0.50)
    args = parser.parse_args()
    if args.horizon != 4:
        raise ValueError("The frozen scheduling contract requires a four-hour horizon")

    renewable, renewable_origins = _renewable_windows(args.renewable_file)
    renewable_map = {origin: renewable[i] for i, origin in enumerate(renewable_origins)}
    frame = build_scheduling_frame(args.data_dir, years=(2021,))
    prices = {
        "grid_upward": args.grid_upward_price,
        "grid_downward": args.grid_downward_price,
        "gas_upward": args.gas_upward_price,
        "gas_downward": args.gas_downward_price,
    }
    rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    for prediction_path in sorted(args.prediction_dir.rglob("predictions_test.npz")):
        prediction, origins, manifest = _load_run(prediction_path)
        protocol = str(manifest.get("protocol", "unknown"))
        # This stage is the formal 2021 R-track replay; non-2021 runs are not silently mixed in.
        if not np.all(pd.DatetimeIndex(origins).year == 2021):
            continue
        valid_idx = [i for i, origin in enumerate(origins) if origin in renewable_map]
        if not valid_idx:
            continue
        selected_origins = origins[valid_idx]
        actual = _actual_windows(frame.data, selected_origins, args.horizon)
        actual_map = {origin: i for i, origin in enumerate(actual["origins"])}
        keep = [i for i, origin in enumerate(selected_origins) if origin in actual_map]
        if not keep:
            continue
        selected_origins = selected_origins[keep]
        predictions = prediction[np.asarray(valid_idx)[keep]]
        pv = np.asarray([renewable_map[origin] for origin in selected_origins], dtype=np.float64)
        actual_idx = [actual_map[origin] for origin in selected_origins]
        actual_values = {
            "actual_grid_import": actual["actual_grid_import"][actual_idx],
            "gas": actual["gas"][actual_idx],
        }
        result = settle_real_replay(
            build_energy_nomination(predictions, pv, selected_origins), actual_values, prices
        )
        model = prediction_path.relative_to(args.prediction_dir).parts[0]
        candidate = prediction_path.relative_to(args.prediction_dir).parts[1] if len(prediction_path.relative_to(args.prediction_dir).parts) > 2 else "default"
        seed = prediction_path.parent.name
        metrics = summarize_replay(result)
        summary.append({"model": model, "candidate": candidate, "seed": seed, "protocol": protocol, "n_windows": len(selected_origins), **metrics})
        for origin, grid_err, gas_err in zip(selected_origins, result.grid_error, result.gas_error):
            rows.append({"model": model, "candidate": candidate, "seed": seed, "protocol": protocol, "origin": str(origin), "season": _season(pd.Timestamp(origin)), "grid_mae_window": float(np.mean(np.abs(grid_err))), "gas_mae_window": float(np.mean(np.abs(gas_err)))})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "real_replay_by_window.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(args.output_dir / "real_replay_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "stage": "10.6",
        "status": "pass" if summary else "fail",
        "track": "real_replay",
        "test_year": 2021,
        "horizon": args.horizon,
        "price_units": "normalized cost units per dataset-native energy unit",
        "prices": prices,
        "actual_source": "Power.zip actual_grid_import plus canonical station-side gas",
        "future_actuals_used_only_after_forecast": True,
        "run_count": len(summary),
        "output_files": ["real_replay_by_window.csv", "real_replay_summary.csv"],
    }
    (args.output_dir / "real_replay_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
