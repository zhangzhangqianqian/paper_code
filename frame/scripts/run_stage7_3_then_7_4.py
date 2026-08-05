"""Run Stage 7.3 and then Stage 7.4 without manual intervention.

The two training stages remain independent.  This launcher only enforces the
order: Stage 7.4 is started after Stage 7.3 has a zero exit code and a passed
root manifest.  No Stage 7.3 prediction path is passed to Stage 7.4.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STAGE73_SCRIPT = Path(__file__).with_name("run_stage7_3.py")
STAGE74_SCRIPT = Path(__file__).with_name("run_stage7_4.py")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_path(output_dir: Path, stage: str) -> Path:
    return output_dir / f"stage{stage.replace('.', '_')}_manifest.json"


def validate_stage_manifest(
    output_dir: str | Path,
    stage: str,
) -> Dict[str, object]:
    """Validate a completed root manifest and return its JSON object."""

    root = _resolve(output_dir)
    manifest_path = _manifest_path(root, stage)
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Stage {stage} manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Stage {stage} manifest is not a JSON object")
    if manifest.get("status") != "passed":
        raise ValueError(f"Stage {stage} manifest is not passed: {manifest_path}")
    expected = manifest.get("run_count_expected")
    completed = manifest.get("run_count_completed")
    failed = manifest.get("run_count_failed")
    if expected is None or completed != expected or failed != 0:
        raise ValueError(
            f"Stage {stage} manifest is incomplete: expected={expected}, "
            f"completed={completed}, failed={failed}"
        )
    return manifest


def build_stage_command(
    script: Path,
    *,
    kitakyushu_data_dir: str,
    contract: str,
    freeze_config: str,
    output_dir: str,
    dry_run: bool = False,
    resume: bool = False,
) -> List[str]:
    """Build a child command while keeping Stage 7.3/7.4 paths separate."""

    command = [
        sys.executable,
        str(script),
        "--kitakyushu-data-dir",
        kitakyushu_data_dir,
        "--contract",
        contract,
        "--freeze-config",
        freeze_config,
        "--output-dir",
        output_dir,
    ]
    if dry_run:
        command.append("--dry-run")
    if resume:
        command.append("--resume")
    return command


def _run_child(
    command: Sequence[str],
    *,
    label: str,
    log_handle,
) -> int:
    """Run a child process, teeing output to the console and log."""

    print(f"[{_utc_now()}] START {label}")
    log_handle.write(f"[{_utc_now()}] START {label}\n")
    log_handle.flush()
    process = subprocess.Popen(
        list(command),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        log_handle.write(line)
        log_handle.flush()
    return_code = process.wait()
    message = f"[{_utc_now()}] END {label} exit_code={return_code}\n"
    print(message, end="")
    log_handle.write(message)
    log_handle.flush()
    return return_code


def _record_stage(
    records: List[Dict[str, object]],
    *,
    label: str,
    command: Sequence[str],
    output_dir: str,
    return_code: int,
    manifest: Mapping[str, object] | None,
    error: str | None = None,
) -> None:
    records.append(
        {
            "stage": label,
            "command": list(command),
            "output_dir": str(_resolve(output_dir)),
            "return_code": int(return_code),
            "manifest_status": None if manifest is None else manifest.get("status"),
            "run_count_expected": (
                None if manifest is None else manifest.get("run_count_expected")
            ),
            "run_count_completed": (
                None if manifest is None else manifest.get("run_count_completed")
            ),
            "run_count_failed": (
                None if manifest is None else manifest.get("run_count_failed")
            ),
            "error": error,
        }
    )


def should_start_stage74(stage73_passed: bool) -> bool:
    """Return whether Stage 7.4 may be scheduled after Stage 7.3."""

    return bool(stage73_passed)


def _run_one_stage(
    *,
    label: str,
    stage_number: str,
    command: Sequence[str],
    output_dir: str,
    log_handle,
    records: List[Dict[str, object]],
    validate_manifest_after_run: bool,
) -> bool:
    return_code = _run_child(command, label=label, log_handle=log_handle)
    manifest: Mapping[str, object] | None = None
    error: str | None = None
    if return_code == 0 and validate_manifest_after_run:
        try:
            manifest = validate_stage_manifest(output_dir, stage_number)
        except (FileNotFoundError, ValueError) as exc:
            error = str(exc)
    passed = return_code == 0 and (not validate_manifest_after_run or error is None)
    _record_stage(
        records,
        label=label,
        command=command,
        output_dir=output_dir,
        return_code=return_code,
        manifest=manifest,
        error=error,
    )
    if not passed:
        reason = error or f"child exit code {return_code}"
        print(f"[{_utc_now()}] STOP after {label}: {reason}")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 7.3, validate it, then run Stage 7.4"
    )
    parser.add_argument(
        "--kitakyushu-data-dir",
        default="D:/Paper/Kitakyushu dataset",
    )
    parser.add_argument("--contract", default="frame/configs/stage7r_contract.json")
    parser.add_argument(
        "--freeze-config",
        default="frame/reports/stage6r_6_kitakyushu/stage6_selected_config.json",
    )
    parser.add_argument(
        "--stage7-3-output-dir",
        default="frame/reports/stage7r_3_kitakyushu_formal",
    )
    parser.add_argument(
        "--stage7-4-output-dir",
        default="frame/reports/stage7r_4_kitakyushu_formal",
    )
    parser.add_argument(
        "--orchestration-dir",
        default="frame/reports/stage7r_3_4_kitakyushu_orchestration",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-7-3", action="store_true")
    parser.add_argument("--resume-7-4", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run and (args.resume_7_3 or args.resume_7_4):
        raise ValueError("--dry-run cannot be combined with resume flags")
    if _resolve(args.stage7_3_output_dir) == _resolve(args.stage7_4_output_dir):
        raise ValueError("Stage 7.3 and Stage 7.4 output directories must differ")

    orchestration_dir = _resolve(args.orchestration_dir)
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    log_path = orchestration_dir / "stage7_3_4_orchestration.log"
    manifest_path = orchestration_dir / "stage7_3_4_orchestration_manifest.json"
    records: List[Dict[str, object]] = []
    started_at = _utc_now()
    stage73_command = build_stage_command(
        STAGE73_SCRIPT,
        kitakyushu_data_dir=args.kitakyushu_data_dir,
        contract=args.contract,
        freeze_config=args.freeze_config,
        output_dir=args.stage7_3_output_dir,
        dry_run=args.dry_run,
        resume=args.resume_7_3,
    )
    stage74_command = build_stage_command(
        STAGE74_SCRIPT,
        kitakyushu_data_dir=args.kitakyushu_data_dir,
        contract=args.contract,
        freeze_config=args.freeze_config,
        output_dir=args.stage7_4_output_dir,
        dry_run=args.dry_run,
        resume=args.resume_7_4,
    )

    with log_path.open("a", encoding="utf-8") as log_handle:
        stage73_ok = _run_one_stage(
            label="stage7.3",
            stage_number="7.3",
            command=stage73_command,
            output_dir=args.stage7_3_output_dir,
            log_handle=log_handle,
            records=records,
            validate_manifest_after_run=not args.dry_run,
        )
        if should_start_stage74(stage73_ok):
            _run_one_stage(
                label="stage7.4",
                stage_number="7.4",
                command=stage74_command,
                output_dir=args.stage7_4_output_dir,
                log_handle=log_handle,
                records=records,
                validate_manifest_after_run=not args.dry_run,
            )

    finished_at = _utc_now()
    success = len(records) == 2 and all(
        int(record["return_code"]) == 0 and record["error"] is None
        for record in records
    )
    summary = {
        "stage": "7.3_then_7.4",
        "status": "passed" if success else "failed",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "dry_run": bool(args.dry_run),
        "stage73_output_dir": str(_resolve(args.stage7_3_output_dir)),
        "stage74_output_dir": str(_resolve(args.stage7_4_output_dir)),
        "stage74_started_only_after_stage73_pass": True,
        "stage73_predictions_passed_to_stage74": False,
        "records": records,
        "log_file": str(log_path),
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
