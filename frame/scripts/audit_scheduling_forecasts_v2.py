"""阶段10.5：审计已有正式预测能否进入双轨调度接口。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.forecast_adapter import load_forecast_run  # noqa: E402


KNOWN_RUN_ROOTS = (
    "stage7r_3_kitakyushu_formal",
    "stage7r_7_joint_baselines_formal",
)
EXPECTED_MODELS = {
    "scheme2r",
    "dynamic_symmetric",
    "hard_share",
    "mmoe_lite",
    "ple_lite",
    "stl_matched",
}


def _discover(root: Path) -> list[tuple[str, str, str, int, Path]]:
    records = []
    for run_root in (root / name for name in KNOWN_RUN_ROOTS):
        if not run_root.exists():
            continue
        for prediction in run_root.rglob("predictions_test.npz"):
            parts = prediction.parts
            try:
                seed_index = next(index for index, part in enumerate(parts) if part.startswith("seed_"))
            except StopIteration:
                continue
            if seed_index < 2:
                continue
            protocol = parts[seed_index - 3]
            model = parts[seed_index - 2]
            candidate = parts[seed_index - 1]
            try:
                seed = int(parts[seed_index][5:])
            except ValueError:
                continue
            if model not in EXPECTED_MODELS:
                continue
            records.append((protocol, model, candidate, seed, prediction.parent))
    return sorted(set(records))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    records = _discover(args.prediction_root)
    rows = []
    failures = []
    for protocol, model, candidate, seed, run_dir in records:
        try:
            run = load_forecast_run(model, candidate, seed, run_dir.parents[2])
            rows.append(
                {
                    "protocol": protocol,
                    "model": model,
                    "candidate": candidate,
                    "seed": seed,
                    "manifest_protocol": run.protocol,
                    "sample_count": int(run.prediction.shape[0]),
                    "prediction_shape": str(list(run.prediction.shape)),
                    "origin_start": str(run.origin_times[0]),
                    "origin_end": str(run.origin_times[-1]),
                    "source_path": str(run.source_path),
                }
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            failures.append({"protocol": protocol, "model": model, "candidate": candidate, "seed": seed, "error": str(exc)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "scheduling_forecast_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(list(rows[0]) if rows else ["model", "candidate", "seed", "error"]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "stage": "10.5",
        "status": "pass" if rows and not failures else "fail",
        "run_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "test_predictions_are_existing_formal_outputs": True,
    }
    (args.output_dir / "forecast_audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
