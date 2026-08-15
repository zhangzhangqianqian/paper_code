"""Stage 10.12: freeze formal scheduling inputs before reading 2021 outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SEEDS = [2026, 2027, 2028, 2029, 2030]


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
        "parameter_freeze": {"benchmark_path": str(args.benchmark), "benchmark_sha256": benchmark_hash, "ledger_path": str(args.ledger), "ledger_sha256": ledger_hash, "source_years": [2015, 2016, 2017, 2018, 2019]},
        "preflight": {"path": str(args.preflight), "formal_dispatch_allowed": True},
        "data_directory": str(args.data_dir),
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
