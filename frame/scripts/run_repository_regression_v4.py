"""Run and record the complete repository regression suite.

The runner is working-directory independent: it always executes from the Git
root, uses the configured PyTorch interpreter, and writes one immutable receipt
for the exact command and output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
DEFAULT_PYTHON = Path(r"D:\anaconda\envs\pytorch\python.exe")


def _git_head() -> str:
    result = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def _summary(output: str) -> dict[str, int]:
    """Extract pytest's terminal counts without trusting collection order."""

    counts = {"passed": 0, "failed": 0, "skipped": 0, "xfailed": 0, "xpassed": 0, "errors": 0}
    for key, pattern in (
        ("passed", r"(\d+) passed"), ("failed", r"(\d+) failed"),
        ("skipped", r"(\d+) skipped"), ("xfailed", r"(\d+) xfailed"),
        ("xpassed", r"(\d+) xpassed"), ("errors", r"(\d+) error(?:s)?"),
    ):
        matches = re.findall(pattern, output)
        if matches:
            counts[key] = int(matches[-1])
    return counts


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def run_regression(*, output_path: str | Path, python_executable: str | Path = DEFAULT_PYTHON, pytest_args: Sequence[str] = ()) -> dict[str, Any]:
    destination = Path(output_path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite regression receipt: {destination}")
    executable = Path(python_executable).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    basetemp = destination.parent / f"{destination.stem}_pytest_tmp"
    command = [str(executable), "-B", "-m", "pytest", str(FRAME_ROOT / "tests"), "-q", "--basetemp", str(basetemp), "-p", "no:cacheprovider", *map(str, pytest_args)]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(FRAME_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    process = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    combined = (process.stdout or "") + (process.stderr or "")
    counts = _summary(combined)
    payload = {
        "schema_version": "formal-v4.1-regression-receipt-v1",
        "status": "pass" if process.returncode == 0 else "fail",
        "command": command,
        "working_directory": str(REPO_ROOT),
        "python_executable": str(executable),
        "python_version": platform.python_version(),
        "git_commit": _git_head(),
        "exit_code": int(process.returncode),
        "counts": counts,
        "output_sha256": _sha256(combined.encode("utf-8")),
        "output_tail": combined[-4000:],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    args = parser.parse_args(argv)
    receipt = run_regression(output_path=args.output, python_executable=args.python)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FRAME_ROOT", "REPO_ROOT", "run_regression", "main", "_summary"]
