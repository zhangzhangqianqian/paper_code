"""Preflight gate for the validation-only topology pilot and formal Phase A."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kitakyushu_pipeline import KitakyushuPaths  # noqa: E402
from src.topology_protocol_analysis import register_cross_topology_artifacts  # noqa: E402
from src.topology_protocol_contract import (  # noqa: E402
    assert_phase_access,
    load_topology_contract,
)


def _resolve(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _tracked_worktree_clean(repo_root: Path) -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return not bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return False


def _run_tests(repo_root: Path) -> dict[str, object]:
    modules = (
        "frame.tests.test_topology_protocol_contract",
        "frame.tests.test_topology_regime_audit",
        "frame.tests.test_protocol_statistics",
        "frame.tests.test_formal_model_factory",
        "frame.tests.test_topology_validation_runner",
        "frame.tests.test_topology_protocol_analysis",
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", "unittest", *modules],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {"status": "passed" if result.returncode == 0 else "failed", "returncode": result.returncode}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "error": str(exc)}


def _formal_root(output_dir: Path) -> Path:
    # The preflight report is a sibling of the formal validation root.
    return output_dir.parent / "validation_pilot"


def build_preflight_report(
    contract_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    expected_revision: str | None = None,
    run_tests: bool = True,
) -> dict[str, Any]:
    """Return a complete gate report; ``formal_training_allowed`` is strict."""

    repo = Path(repo_root)
    contract_file = _resolve(contract_path, repo)
    data_root = _resolve(data_dir, repo)
    report_root = _resolve(output_dir, repo)
    checks: dict[str, object] = {}
    try:
        contract = load_topology_contract(contract_file)
        assert_phase_access(contract, phase="A", requested_years=(2017, 2018, 2019, 2020), requested_splits=("train", "validation"))
        checks["contract"] = {"status": "passed", "path": str(contract_file)}
    except Exception as exc:
        contract = None
        checks["contract"] = {"status": "failed", "error": str(exc)}

    try:
        paths = KitakyushuPaths.from_root(data_root)
        paths.validate()
        checks["data"] = {"status": "passed", "required_zips": [str(paths.load_zip.name), str(paths.gas_zip.name), str(paths.weather_zip.name)]}
    except Exception as exc:
        checks["data"] = {"status": "failed", "error": str(exc)}

    base = report_root.parent
    audit_file = base / "data_audit" / "topology_regime_audit.json"
    try:
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        checks["audit"] = {"status": "passed" if audit.get("status") == "passed" and audit.get("latest_audited_year") == 2020 and audit.get("test_year_accessed") is False else "failed", "path": str(audit_file)}
    except (OSError, json.JSONDecodeError) as exc:
        checks["audit"] = {"status": "failed", "error": str(exc)}

    registry_file = base / "cross_topology_artifact_registry.json"
    try:
        registry = register_cross_topology_artifacts(repo, base, require_expected_origin_count=True)
        checks["artifacts"] = {"status": "passed", "count": len(registry["artifacts"]), "path": str(registry_file)}
    except Exception as exc:
        checks["artifacts"] = {"status": "failed", "error": str(exc)}

    smoke_dir = base / "validation_pilot_smoke"
    smoke_file = smoke_dir / "phase_a_manifest.json"
    try:
        smoke = json.loads(smoke_file.read_text(encoding="utf-8"))
        smoke_pass = smoke.get("status") == "passed" and smoke.get("run_count") == 6 and smoke.get("test_set_accessed") is False
        checks["smoke"] = {"status": "passed" if smoke_pass else "failed", "run_count": smoke.get("run_count"), "path": str(smoke_file)}
    except (OSError, json.JSONDecodeError) as exc:
        checks["smoke"] = {"status": "failed", "error": str(exc)}

    current_revision = _git_revision(repo)
    checks["git"] = {"status": "passed" if expected_revision is None or current_revision == expected_revision else "failed", "current_revision": current_revision, "expected_revision": expected_revision, "tracked_worktree_clean": _tracked_worktree_clean(repo)}
    checks["forbidden_years"] = {"status": "passed", "requested_years": [2017, 2018, 2019, 2020], "forbidden_years": [2021]}
    formal_root = _formal_root(report_root)
    existing_mismatch = False
    if formal_root.is_dir():
        existing_mismatch = any(formal_root.rglob("run_manifest.json")) and not (formal_root / "phase_a_manifest.json").is_file()
    checks["formal_output_root"] = {"status": "failed" if existing_mismatch else "passed", "path": str(formal_root), "existing_mismatch": existing_mismatch}
    usage = shutil.disk_usage(report_root.anchor or repo.anchor)
    checks["disk_space"] = {"status": "passed" if usage.free >= 10 * 1024**3 else "failed", "free_gb": round(usage.free / 1024**3, 3)}
    checks["tests"] = _run_tests(repo) if run_tests else {"status": "not_run"}
    mandatory = ("contract", "data", "audit", "artifacts", "smoke", "forbidden_years", "formal_output_root", "disk_space")
    if expected_revision is not None:
        mandatory += ("git",)
    if run_tests:
        mandatory += ("tests",)
    allowed = all(checks.get(name, {}).get("status") == "passed" for name in mandatory)
    report = {"stage": "topology_protocol_pilot_phase_a_preflight", "status": "passed" if allowed else "blocked", "formal_training_allowed": allowed, "checks": checks}
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "phase_a_preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-revision")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    report = build_preflight_report(args.contract, args.data_dir, args.output_dir, expected_revision=args.expected_revision, run_tests=not args.skip_tests)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["formal_training_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
