"""Stage 10.12: freeze formal scheduling inputs before reading 2021 outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.data import build_scheduling_frame  # noqa: E402
from src.scheduling.parameter_audit import read_parameter_ledger  # noqa: E402
from src.scheduling.renewables import pv_available  # noqa: E402


SEEDS = [2026, 2027, 2028, 2029, 2030]

# Normalized evaluation coefficients.  They are not reported as local
# electricity/gas tariffs; the contract hash freezes them for reproducibility.
REAL_SETTLEMENT_PRICES = {
    "grid_upward": 1.15,
    "grid_downward": 0.85,
    "gas_upward": 0.70,
    "gas_downward": 0.50,
}


def _fit_real_pv_scale(
    data_dir: Path,
    ledger_path: Path,
    source_years: tuple[int, ...],
) -> float:
    """Freeze the R-track PV scale using training years only.

    The real station file contains measured PV output, whereas the transparent
    weather profile has unit rated capacity.  The scale is therefore a derived
    parameter, not a tunable test-year quantity, and must be recorded in the
    frozen contract before any 2021 replay is read.
    """

    ledger = read_parameter_ledger(ledger_path)
    values = {record.parameter_id: float(record.value) for record in ledger.records}
    profile_parameters = {
        "pv_rated_capacity": 1.0,
        "pv_reference_irradiance": values["pv_reference_irradiance"],
        "pv_conversion_efficiency": values["pv_conversion_efficiency"],
        "pv_reference_temperature": values["pv_reference_temperature"],
        "pv_temperature_coefficient": values["pv_temperature_coefficient"],
    }
    frame = build_scheduling_frame(data_dir, years=source_years)
    timestamps = pd.to_datetime(frame.data["timestamp"]).dt.year
    train = frame.data.loc[timestamps.isin(source_years)].copy()
    profile = pv_available(train, profile_parameters)
    actual = pd.to_numeric(train["actual_pv"], errors="raise").to_numpy(dtype=np.float64)
    mask = profile > 1e-8
    if not np.any(mask):
        raise ValueError("training period has no positive normalized PV profile")
    scale = float(np.median(actual[mask] / profile[mask]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid training-only real PV scale: {scale}")
    return scale


def _model_sources(repo_root: Path) -> dict[str, dict[str, object]]:
    return {
        "scheme2r": {"root": "frame/reports/stage7r_4_kitakyushu_formal/full/A4/H4", "candidate": "H4"},
        "dynamic_symmetric": {"root": "frame/reports/stage7r_7_joint_baselines_formal/full/dynamic_symmetric/H1", "candidate": "H1"},
        "hard_share": {"root": "frame/reports/stage7r_7_joint_baselines_formal/full/hard_share/H2", "candidate": "H2"},
        "mmoe_lite": {"root": "frame/reports/stage7r_5_kitakyushu_formal/full/mmoe_lite", "candidate": "external_fixed"},
        "ple_lite": {"root": "frame/reports/stage7r_7_joint_baselines_formal/full/ple_lite/ple_lite_fixed_v1", "candidate": "ple_lite_fixed_v1"},
    }


def _check_sources(repo_root: Path, sources: dict[str, dict[str, object]]) -> tuple[int, dict[str, list[str]]]:
    origin_count: int | None = None
    checked: dict[str, list[str]] = {}
    for model, info in sources.items():
        root = repo_root / Path(str(info["root"]))
        checked[model] = []
        for seed in SEEDS:
            run_dir = root / f"seed_{seed}"
            prediction_path = run_dir / "predictions_test.npz"
            manifest_path = run_dir / "run_manifest.json"
            if not prediction_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(f"Missing formal prediction artifacts for {model}, seed {seed}: {run_dir}")
            with np.load(prediction_path, allow_pickle=False) as payload:
                prediction = np.asarray(payload["prediction"])
                origins = np.asarray(payload["target_times"])
            if prediction.shape[1:] != (4, 4) or len(origins) != prediction.shape[0]:
                raise ValueError(f"Invalid formal forecast shape for {model}, seed {seed}: {prediction.shape}")
            if origin_count is None:
                origin_count = int(len(origins))
            elif len(origins) != origin_count:
                raise ValueError("Formal models do not share the same 2021 origin count")
            checked[model].append(str(prediction_path.relative_to(repo_root)))
    if origin_count is None:
        raise ValueError("No formal prediction runs found")
    return origin_count, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--ledger", default="frame/configs/scheduling_parameter_ledger_v2.csv", type=Path)
    parser.add_argument("--preflight", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    sources = _model_sources(repo_root)
    origin_count, checked = _check_sources(repo_root, sources)
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("formal_dispatch_allowed") is not True:
        raise RuntimeError("Formal dispatch cannot be frozen before a passing preflight")
    benchmark_hash = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    ledger_hash = hashlib.sha256(args.ledger.read_bytes()).hexdigest()
    recorded_hash = args.benchmark.with_suffix(args.benchmark.suffix + ".sha256").read_text(encoding="utf-8").strip()
    if benchmark_hash != recorded_hash:
        raise RuntimeError("Benchmark SHA-256 does not match its sidecar")
    source_years = (2015, 2016, 2017, 2018, 2019)
    data_dir = args.data_dir.resolve()
    real_pv_scale = _fit_real_pv_scale(data_dir, args.ledger.resolve(), source_years)
    contract = {
        "schema_version": "scheduling-formal-contract-v2",
        "dataset": "kitakyushu_energy_station",
        "test_year": 2021,
        "history_hours": 24,
        "horizon_hours": 4,
        "task_order": ["electricity", "cooling", "heating", "gas"],
        "models": list(sources),
        "seeds": SEEDS,
        "prediction_sources": sources,
        "origin_count": origin_count,
        "tracks": ["real_replay", "simulated_dispatch"],
        "simulated_scenarios": ["core", "no_bess", "no_chp", "no_renewables", "single_hour", "carbon_price_sensitivity"],
        "gas_main_balance": False,
        "ordinary_future_actual_allowed": False,
        "parameter_freeze": {"benchmark_path": str(args.benchmark), "benchmark_sha256": benchmark_hash, "ledger_path": str(args.ledger), "ledger_sha256": ledger_hash, "source_years": list(source_years)},
        "real_track": {
            "pv_scale_train_only": real_pv_scale,
            "pv_scale_source_years": list(source_years),
            "pv_scale_method": "median(actual_pv / unit_rated_weather_profile) over positive training-profile hours",
            "settlement_price_units": "normalized cost units per dataset-native energy unit",
            "settlement_prices": REAL_SETTLEMENT_PRICES,
        },
        "preflight": {"path": str(args.preflight), "formal_dispatch_allowed": True},
        "data_directory": str(data_dir),
        "checked_prediction_artifacts": checked,
        "no_test_tuning_after_freeze": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"stage": "10.12", "status": "pass", "output": str(args.output), "sha256": digest, "origin_count": origin_count, "models": list(sources), "seeds": SEEDS}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
