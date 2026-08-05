"""Orchestrate the isolated Stage 7-R reproducible formal revision."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_stage7_3 import build_run_plan  # noqa: E402
from run_stage7_4 import build_ablation_run_plan  # noqa: E402
from run_stage7_5 import build_external_run_plan  # noqa: E402
from run_stage7_stl_reference import build_stl_run_plan  # noqa: E402
from src.data_pipeline import save_json  # noqa: E402


REVISION_VERSION = "stage7R.1"


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _primary_is_scheme2r(freeze: Mapping[str, object] | None) -> bool:
    if not isinstance(freeze, Mapping):
        return True
    primary = freeze.get("primary_model")
    return isinstance(primary, Mapping) and primary.get("model") == "scheme2r"


def build_revision_plan(
    freeze: Mapping[str, object] | None = None,
) -> Tuple[Dict[str, object], ...]:
    """Build materialized child groups from the frozen primary model."""

    include_a4 = not _primary_is_scheme2r(freeze)
    return (
        {
            "group": "stage7_3",
            "script": "run_stage7_3.py",
            "manifest": "stage7_3_manifest.json",
            "run_count": len(build_run_plan()),
        },
        {
            "group": "stage7_4",
            "script": "run_stage7_4.py",
            "manifest": "stage7_4_manifest.json",
            "run_count": len(build_ablation_run_plan(include_a4=include_a4)),
        },
        {
            "group": "stage7_5",
            "script": "run_stage7_5.py",
            "manifest": "stage7_5_manifest.json",
            "run_count": len(build_external_run_plan()),
        },
        {
            "group": "stl_reference",
            "script": "run_stage7_stl_reference.py",
            "manifest": "stage7r_stl_manifest.json",
            "run_count": len(build_stl_run_plan()),
        },
    )


def _validate_contract(contract: Mapping[str, object]) -> None:
    if contract.get("revision_version") != REVISION_VERSION:
        raise ValueError("Stage 7-R requires the stage7R.1 contract")
    if contract.get("dataset") != "kitakyushu_energy_station":
        raise ValueError("Stage 7-R is frozen for Kitakyushu Energy Station")
    matrix = contract.get("run_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("Stage 7-R contract run matrix is missing")
    if matrix.get("stage7_3_main_internal") != 20:
        raise ValueError("Stage 7-R Stage 7.3 must contain 20 runs")
    if matrix.get("stage7_5_external") != 44:
        raise ValueError("Stage 7-R Stage 7.5 must contain 44 runs including loads-only control")
    if matrix.get("stage7r_matched_stl") != 10:
        raise ValueError("Stage 7-R matched STL must contain 10 runs")
    stage7_4 = matrix.get("stage7_4_ablations")
    if stage7_4 not in (40, 50):
        raise ValueError("Stage 7-R Stage 7.4 must contain 40 or 50 runs")
    materialized = 20 + int(stage7_4) + 44 + 10
    reused = int(matrix.get("reused_a4_run_count", -1))
    formal = int(matrix.get("formal_run_count", -1))
    if int(matrix.get("materialized_run_count", -1)) != materialized:
        raise ValueError("Stage 7-R materialized run count is inconsistent")
    if formal != materialized + reused:
        raise ValueError("Stage 7-R effective run count is inconsistent")
    if int(matrix.get("deterministic_run_count", -1)) != 4:
        raise ValueError("Stage 7-R deterministic run count must be 4")
    if int(matrix.get("trained_run_count", -1)) != materialized - 4:
        raise ValueError("Stage 7-R trained run count is inconsistent")
    seed_control = contract.get("seed_control")
    if not isinstance(seed_control, Mapping) or not all(seed_control.values()):
        raise ValueError("Stage 7-R strict seed controls are incomplete")
    legacy = contract.get("legacy_results")
    if not isinstance(legacy, Mapping) or legacy.get("preserve_read_only") is not True:
        raise ValueError("Stage 7-R must preserve legacy results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated Stage 7-R reproducible revision"
    )
    parser.add_argument("--kitakyushu-data-dir", default="D:/Paper/Kitakyushu dataset")
    parser.add_argument(
        "--contract",
        default="frame/configs/stage7r_contract.json",
        help="Stage 7.0 contract generated from the new Stage 6-R freeze",
    )
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json",
        help="Stage 6-R freeze consumed by all Stage 7 child scripts",
    )
    parser.add_argument(
        "--revision-contract",
        default="frame/configs/stage7r_reproducibility_contract.json",
    )
    parser.add_argument(
        "--output-root",
        default="frame/reports/stage7r_kitakyushu_reproducible",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    contract_path = _resolve(args.revision_contract)
    contract = _read_json(contract_path)
    _validate_contract(contract)
    freeze = _read_json(_resolve(args.freeze_config))
    plan = build_revision_plan(freeze)
    run_count = sum(int(item["run_count"]) for item in plan)
    primary_is_scheme2r = _primary_is_scheme2r(freeze)
    reused_a4 = 10 if primary_is_scheme2r else 0
    formal_run_count = run_count + reused_a4
    trained_run_count = run_count - 4
    matrix = contract["run_matrix"]
    expected_stage7_4 = next(
        int(item["run_count"]) for item in plan if item["group"] == "stage7_4"
    )
    if (
        matrix.get("stage7_4_ablations") != expected_stage7_4
        or matrix.get("materialized_run_count") != run_count
        or matrix.get("formal_run_count") != formal_run_count
        or matrix.get("trained_run_count") != trained_run_count
        or matrix.get("reused_a4_run_count") != reused_a4
    ):
        raise ValueError(
            "Stage 7-R revision contract does not match the supplied Stage 6 freeze"
        )
    output_root = _resolve(args.output_root)

    commands: List[Dict[str, object]] = []
    for item in plan:
        output_dir = output_root / str(item["group"])
        command = [
            sys.executable,
            str(SCRIPTS_DIR / str(item["script"])),
            "--kitakyushu-data-dir",
            str(_resolve(args.kitakyushu_data_dir)),
            "--output-dir",
            str(output_dir),
            "--contract",
            str(_resolve(args.contract)),
            "--freeze-config",
            str(_resolve(args.freeze_config)),
        ]
        if args.resume:
            command.append("--resume")
        elif args.force:
            command.append("--force")
        commands.append(
            {
                **item,
                "output_dir": str(output_dir),
                "command": command,
            }
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "stage": "7-R",
                    "revision_version": REVISION_VERSION,
                    "status": "dry_run",
                    "formal_run_count": formal_run_count,
                    "materialized_run_count": run_count,
                    "trained_run_count": trained_run_count,
                    "deterministic_run_count": 4,
                    "reused_a4_run_count": reused_a4,
                    "legacy_results_preserved": True,
                    "output_root": str(output_root),
                    "groups": commands,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    manifest_path = output_root / "stage7r_execution_manifest.json"
    if manifest_path.exists() and not (args.resume or args.force):
        raise FileExistsError(
            f"manifest exists; use --resume or --force: {manifest_path}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    completed_groups: List[Dict[str, object]] = []
    failed_group = None
    for item in commands:
        print(
            json.dumps(
                {
                    "stage": "7-R",
                    "status": "group_started",
                    "group": item["group"],
                    "run_count": item["run_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        completed = subprocess.run(
            item["command"],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        child_manifest_path = Path(str(item["output_dir"])) / str(item["manifest"])
        if completed.returncode != 0 or not child_manifest_path.exists():
            failed_group = {
                "group": item["group"],
                "return_code": completed.returncode,
                "manifest": str(child_manifest_path),
            }
            break
        child_manifest = _read_json(child_manifest_path)
        if (
            child_manifest.get("status") != "passed"
            or child_manifest.get("run_count_completed") != item["run_count"]
            or child_manifest.get("run_count_failed") != 0
            or child_manifest.get("strict_seed_control") is not True
        ):
            failed_group = {
                "group": item["group"],
                "return_code": completed.returncode,
                "manifest": str(child_manifest_path),
                "manifest_status": child_manifest.get("status"),
            }
            break
        completed_groups.append(
            {
                "group": item["group"],
                "run_count": item["run_count"],
                "manifest": str(child_manifest_path),
            }
        )

    manifest = {
        "stage": "7-R",
        "revision_version": REVISION_VERSION,
        "status": "passed" if failed_group is None and len(completed_groups) == 4 else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision_contract": str(contract_path),
        "output_root": str(output_root),
        "formal_run_count_expected": formal_run_count,
        "materialized_run_count_expected": run_count,
        "trained_run_count_expected": trained_run_count,
        "reused_a4_run_count_expected": reused_a4,
        "completed_groups": completed_groups,
        "failed_group": failed_group,
        "legacy_results_preserved": True,
        "test_used_for_selection": False,
        "next_step": "run_stage7r_acceptance.py",
    }
    save_json(manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if manifest["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
