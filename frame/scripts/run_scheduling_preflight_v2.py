"""Stage 10.11: preflight gate before any formal 2021 dispatch run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def _check(condition: bool, detail: object) -> dict[str, object]:
    return {"status": "pass" if condition else "fail", "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    checks: dict[str, object] = {}
    smoke_path = args.smoke_dir / "smoke_manifest.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else {}
    checks["smoke"] = _check(
        smoke.get("status") == "pass"
        and smoke.get("origin_count") == 96
        and smoke.get("ordinary_forecast_uses_future_actual") is False
        and float(smoke.get("max_s_balance_residual", float("inf"))) <= 1e-7
        and float(smoke.get("max_simultaneous_charge_discharge", float("inf"))) <= 1e-8,
        smoke,
    )
    sha_path = args.benchmark.with_suffix(args.benchmark.suffix + ".sha256")
    digest = hashlib.sha256(args.benchmark.read_bytes()).hexdigest() if args.benchmark.exists() else ""
    recorded = sha_path.read_text(encoding="utf-8").strip() if sha_path.exists() else ""
    checks["benchmark_hash"] = _check(bool(digest) and digest == recorded, {"computed": digest, "recorded": recorded})
    zip_names = (
        "Electricity load & Heating load & Cooling load & Hot water load.zip",
        "Gas usage.zip",
        "Weather data.zip",
        "Power.zip",
    )
    checks["required_data"] = _check(all((args.data_dir / name).exists() for name in zip_names), {name: (args.data_dir / name).exists() for name in zip_names})
    try:
        scipy_version = importlib.metadata.version("scipy")
        from packaging.version import Version

        scipy_ok = Version("1.13") <= Version(scipy_version) < Version("1.14")
    except Exception as exc:  # pragma: no cover - environment dependent
        scipy_version = str(exc)
        scipy_ok = False
    checks["scipy_highs_dependency"] = _check(scipy_ok, scipy_version)
    try:
        git_status = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception as exc:  # pragma: no cover - environment dependent
        git_status = [str(exc)]
    checks["git_status"] = {"status": "warning" if git_status else "pass", "dirty_entries": len(git_status)}
    checks["output_directory"] = _check(not args.output_dir.exists() or not any(args.output_dir.iterdir()), str(args.output_dir))
    allowed = all(item.get("status") == "pass" for key, item in checks.items() if key != "git_status")
    manifest = {
        "stage": "10.11",
        "formal_dispatch_allowed": allowed,
        "checks": checks,
        "test_year_tuning_allowed": False,
        "test_year": 2021,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scheduling_preflight_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
