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


def _check_stage6r_smoke(smoke_root: Path) -> Dict[str, Any]:
    """Validate the four short smoke outputs without touching training data."""

    expectations = {
        "stage6r_2_smoke": ("stage6_2_manifest.json", "candidate_run_count", {24}),
        "stage6r_3_smoke": ("stage6_3_manifest.json", "candidate_run_count", {2, 3}),
        "stage6r_4_smoke": ("stage6_4_manifest.json", "analyzed_run_count", None),
        "stage6r_5_smoke": ("stage6_5_manifest.json", "summary_row_count", None),
    }
    entries: Dict[str, Any] = {}
    passed = True
    for name, (manifest_name, count_key, allowed_counts) in expectations.items():
        root = smoke_root / name
        manifest_path = root / manifest_name
        entry: Dict[str, Any] = {
            "directory": str(root),
            "manifest": str(manifest_path),
            "exists": manifest_path.is_file(),
        }
        if not manifest_path.is_file():
            entry["status"] = "missing"
            passed = False
            entries[name] = entry
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entry["status"] = "invalid_manifest"
            passed = False
            entries[name] = entry
            continue
        count = manifest.get(count_key)
        test_accessed = manifest.get("test_set_accessed")
        forbidden = [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and path.name in {"metrics_test.json", "predictions_test.npz"}
        ]
        entry.update(
            {
                "count_key": count_key,
                "count": count,
                "test_set_accessed": test_accessed,
                "forbidden_test_artifacts": forbidden,
            }
        )
        count_ok = (
            isinstance(count, int)
            and (
                count in allowed_counts
                if allowed_counts is not None
                else count >= 1
            )
        )
        entry["status"] = (
            "pass"
            if count_ok
            and test_accessed is False
            and not forbidden
            else "failed"
        )
        passed = passed and entry["status"] == "pass"
        entries[name] = entry
    return {"status": "pass" if passed else "failed", "entries": entries}


def run_preflight(
    data_dir: Path,
    freeze_path: Path,
    contract_path: Path,
    phase: str = "stage6r",
    stage7_contract_path: Path | None = None,
    smoke_root: Path | None = None,
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
        "smoke": {"status": "not_required"},
    }
    for name in (
        "Electricity load & Heating load & Cooling load & Hot water load.zip",
        "Gas usage.zip",
        "Weather data.zip",
    ):
        checks["required_zip_files"].append({"name": name, "exists": (data_dir / name).is_file()})
    # Stage 6-R creates the freeze; an old freeze must not be used to
    # validate or block the Stage 6-R run.  Matrix validation belongs to the
    # post-freeze Stage 7-R preflight.
    if phase == "stage6r":
        checks["matrix"] = "not_required"
    elif freeze is not None:
        try:
            rows = build_formal_run_matrix(freeze, {"contract": "stage7.0"})
            checks["matrix"] = {
                "status": "pass",
                "effective_runs": len(rows),
                "duplicate_count": len(rows) - len({
                    (r["stage"], r["protocol"], r["model"], r["candidate_id"], r["seed"], r["execution"])
                    for r in rows
                }),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # A legacy or incomplete freeze must block formal training without
            # crashing preflight; report why a fresh Stage 6-R freeze is needed.
            checks["matrix"] = {
                "status": "failed",
                "error": str(exc),
                "duplicate_count": None,
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
    if phase == "stage6r":
        checks["smoke"] = _check_stage6r_smoke(
            smoke_root or (REPOSITORY_ROOT / "frame" / "reports")
        )
    usage = shutil.disk_usage(REPOSITORY_ROOT)
    checks["disk_free_gb"] = round(usage.free / (1024 ** 3), 2)
    stage6_data_ready = bool(
        checks["stage6_contract"] == "pass"
        and checks["data_directory"]
        and all(item["exists"] for item in checks["required_zip_files"])
        and checks["git_clean"]
        and (
            phase != "stage6r"
            or checks["smoke"].get("status") == "pass"
        )
    )
    stage7_ready = bool(
        stage6_data_ready
        and checks["freeze_present"]
        and isinstance(checks["matrix"], dict)
        and checks["matrix"].get("status") == "pass"
        and checks["matrix"].get("duplicate_count") == 0
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
    parser.add_argument("--smoke-root", default="frame/reports")
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
        _resolve(args.smoke_root),
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if checks["formal_training_allowed"]:
        return
    blocked_phase = "Stage 6-R" if args.phase == "stage6r" else "Stage 7-R"
    print(
        f"formal_training_allowed=false; do not start {blocked_phase}",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
