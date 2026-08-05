"""建立正式实验前 R0 基线清单。

该脚本只做只读审计，不移动、删除或修改实验结果，也不读取测试指标。
它记录当前 Git 版本、工作区状态、reports 忽略状态、旧结果目录清单以及
Kitakyushu 源数据压缩包的 SHA256，供后续 R1--R8 修复和正式实验追溯。

示例：

    python frame/scripts/audit_formal_readiness.py `
        --kitakyushu-data-dir "D:\\Paper\\Kitakyushu dataset" `
        --output "frame\\reports\\r0_baseline_20260805\\formal_readiness_manifest.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_git(root: Path, *args: str) -> dict[str, Any]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return ()
    return (item for item in path.rglob("*") if item.is_file())


def _directory_inventory(path: Path) -> dict[str, Any]:
    files = list(_iter_files(path))
    total_bytes = sum(item.stat().st_size for item in files)
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": int(total_bytes),
        "total_megabytes": round(total_bytes / (1024 * 1024), 3),
    }


def _report_inventory(reports_root: Path) -> list[dict[str, Any]]:
    if not reports_root.exists():
        return []
    inventories: list[dict[str, Any]] = []
    for directory in sorted(
        (item for item in reports_root.iterdir() if item.is_dir()),
        key=lambda item: item.name,
    ):
        item = _directory_inventory(directory)
        item["name"] = directory.name
        inventories.append(item)
    return inventories


def _source_archives(data_root: Path, hash_data: bool) -> list[dict[str, Any]]:
    if not data_root.exists():
        return []
    archives: list[dict[str, Any]] = []
    for path in sorted(data_root.glob("*.zip"), key=lambda item: item.name.lower()):
        record: dict[str, Any] = {
            "name": path.name,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "sha256_status": "computed" if hash_data else "not_computed",
        }
        if hash_data:
            record["sha256"] = _sha256(path)
        archives.append(record)
    return archives


def _ignored(root: Path, relative_path: str) -> dict[str, Any]:
    result = _run_git(root, "check-ignore", "-q", "--no-index", relative_path)
    return {
        "path": relative_path,
        "ignored": result["returncode"] == 0,
        "check_returncode": result["returncode"],
    }


def build_manifest(root: Path, data_root: Path, hash_data: bool) -> dict[str, Any]:
    status = _run_git(root, "status", "--short")
    branch = _run_git(root, "branch", "--show-current")
    commit = _run_git(root, "rev-parse", "HEAD")
    root_status = _run_git(root, "status", "--porcelain=v1", "--branch")

    reports_root = root / "frame" / "reports"
    report_directories = _report_inventory(reports_root)
    return {
        "manifest_version": "r0-formal-readiness-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Read-only baseline before formal-experiment repair; no training or test metrics read.",
        "formal_training_blocked": True,
        "blocked_until": "R0-R8 acceptance gates pass",
        "repository": {
            "root": str(root),
            "branch": branch["stdout"] or None,
            "commit": commit["stdout"] or None,
            "status_porcelain_branch": root_status["stdout"],
            "status_short": status["stdout"].splitlines() if status["stdout"] else [],
            "worktree_clean": not bool(status["stdout"]),
            "git_commands": {
                "branch_returncode": branch["returncode"],
                "commit_returncode": commit["returncode"],
                "status_returncode": status["returncode"],
            },
        },
        "scope_policy": {
            "files_moved": False,
            "files_deleted": False,
            "legacy_results_archived": False,
            "test_metrics_read": False,
            "test_predictions_read": False,
            "old_results_are_legacy_only": True,
        },
        "git_ignore_checks": [
            _ignored(root, "frame/reports"),
            # Check a representative generated raw-data file.  The directory
            # itself is not matched by ``frame/data/raw/*`` in .gitignore.
            _ignored(root, "frame/data/raw/__r0_probe__.csv"),
        ],
        "legacy_reports": {
            "root": str(reports_root),
            "directories": report_directories,
            "total_bytes": sum(item["total_bytes"] for item in report_directories),
        },
        "source_data": {
            "root": str(data_root),
            "archives": _source_archives(data_root, hash_data),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
        },
        "next_step": "Review this manifest, then begin R1 canonical contract repair; do not start Stage 6-R or Stage 7-R yet.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="repository root; defaults to the parent of frame/",
    )
    parser.add_argument(
        "--kitakyushu-data-dir",
        type=Path,
        default=Path(r"D:\Paper\Kitakyushu dataset"),
        help="directory containing the Kitakyushu source ZIP files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"frame\reports\r0_baseline_20260805\formal_readiness_manifest.json"),
        help="JSON output path, resolved relative to repository root when relative",
    )
    parser.add_argument(
        "--no-hash-data",
        action="store_true",
        help="skip source ZIP SHA256 calculation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = (args.repo_root or _repository_root()).resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    data_root = args.kitakyushu_data_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root, data_root, hash_data=not args.no_hash_data)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "manifest": str(output),
        "worktree_clean": manifest["repository"]["worktree_clean"],
        "reports_ignored": manifest["git_ignore_checks"][0]["ignored"],
        "legacy_report_directories": len(manifest["legacy_reports"]["directories"]),
        "source_archives_hashed": len([
            item for item in manifest["source_data"]["archives"]
            if item["sha256_status"] == "computed"
        ]),
        "formal_training_blocked": manifest["formal_training_blocked"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
