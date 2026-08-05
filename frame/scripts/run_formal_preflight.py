"""Run the non-training admission checks before Stage 6-R."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.stage6_contract import load_stage6_selection_contract  # noqa: E402
from scripts.generate_formal_run_matrix import build_formal_run_matrix  # noqa: E402


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return not result.stdout.strip()


def run_preflight(
    data_dir: Path,
    freeze_path: Path,
    contract_path: Path,
    phase: str = "stage6r",
    stage7_contract_path: Path | None = None,
) -> Dict[str, Any]:
    contract = load_stage6_selection_contract(contract_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8")) if freeze_path.exists() else None
    checks: Dict[str, Any] = {
        "stage6_contract": "pass",
        "freeze_present": freeze is not None,
        "data_directory": data_dir.is_dir(),
        "required_zip_files": [],
        "matrix": "pending",
        "git_clean": _git_clean(),
        "test_suite": "not_run",
        "stage7_contract": "not_required" if phase == "stage6r" else "pending",
    }
    for name in (
        "Electricity load & Heating load & Cooling load & Hot water load.zip",
        "Gas usage.zip",
        "Weather data.zip",
    ):
        checks["required_zip_files"].append({"name": name, "exists": (data_dir / name).is_file()})
    if freeze is not None:
        rows = build_formal_run_matrix(freeze, {"contract": "stage7.0"})
        checks["matrix"] = {
            "status": "pass",
            "effective_runs": len(rows),
            "duplicate_count": len(rows) - len({
                (r["stage"], r["protocol"], r["model"], r["candidate_id"], r["seed"], r["execution"])
                for r in rows
            }),
        }
    if phase == "stage7r":
        if stage7_contract_path is None or not stage7_contract_path.exists():
            checks["stage7_contract"] = "missing"
        else:
            stage7_contract = json.loads(
                stage7_contract_path.read_text(encoding="utf-8")
            )
            checks["stage7_contract"] = (
                "pass"
                if stage7_contract.get("contract_status") == "ready_for_stage7_smoke"
                else "stale_or_legacy"
            )
    usage = shutil.disk_usage(REPOSITORY_ROOT)
    checks["disk_free_gb"] = round(usage.free / (1024 ** 3), 2)
    stage6_data_ready = bool(
        checks["stage6_contract"] == "pass"
        and checks["data_directory"]
        and all(item["exists"] for item in checks["required_zip_files"])
        and checks["git_clean"]
    )
    stage7_ready = bool(
        stage6_data_ready
        and checks["freeze_present"]
        and checks["matrix"] != "pending"
        and checks["matrix"]["duplicate_count"] == 0
        and checks["stage7_contract"] == "pass"
    )
    checks["phase"] = phase
    checks["formal_training_allowed"] = stage6_data_ready if phase == "stage6r" else stage7_ready
    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="执行正式训练前非训练准入检查")
    parser.add_argument("--data-dir", default="D:/Paper/Kitakyushu dataset")
    parser.add_argument("--freeze-config", default="frame/reports/stage6_6_kitakyushu/stage6_selected_config.json")
    parser.add_argument("--contract", default="frame/configs/stage6_selection_contract.json")
    parser.add_argument(
        "--stage7-contract", default="frame/configs/stage7_contract.json"
    )
    parser.add_argument("--phase", choices=("stage6r", "stage7r"), default="stage6r")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = run_preflight(
        _resolve(args.data_dir),
        _resolve(args.freeze_config),
        _resolve(args.contract),
        args.phase,
        _resolve(args.stage7_contract),
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if checks["formal_training_allowed"]:
        return
    print("formal_training_allowed=false; do not start Stage 6-R", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
