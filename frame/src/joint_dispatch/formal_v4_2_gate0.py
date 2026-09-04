"""Self-contained prerequisite receipts for the formal-v4.2 Gate 0."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping

import torch

from .formal_v4_2_artifacts import canonical_sha256, sha256_file, write_once_json
from .formal_v4_itransformer import (
    OFFICIAL_BACKBONE_CLASS,
    OFFICIAL_COMMIT,
    OFFICIAL_REPOSITORY,
    verify_itransformer_source_files,
)
from .formal_v4_provenance import build_source_manifest


SOURCE_SCHEMA = "formal-v4.2-source-manifest-v1"
ITRANSFORMER_SCHEMA = "formal-v4.2-itransformer-source-v1"
DIFFOPT_SCHEMA = "formal-v4.2-diffopt-environment-v1"
CAPACITY_SCHEMA = "formal-v4.2-capacity-freeze-v1"
CURRENT_GATE_SCHEMA = "formal-v4.2-current-gate-v1"


class Gate0ReceiptError(ValueError):
    """Raised when prerequisite evidence cannot authorize formal-v4.2."""


@dataclass(frozen=True)
class Gate0InputsV42:
    contract_path: Path
    output_root: Path
    run_id: str
    data_dir: Path
    diffopt_python: Path
    itransformer_source: Path


@dataclass(frozen=True)
class Gate0PrerequisitePaths:
    source_manifest: Path
    itransformer_receipt: Path
    diffopt_receipt: Path
    capacity_receipt: Path

    def __iter__(self) -> Iterator[Path]:
        return iter(
            (
                self.source_manifest,
                self.itransformer_receipt,
                self.diffopt_receipt,
                self.capacity_receipt,
            )
        )


def _read_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise Gate0ReceiptError(f"receipt is missing: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Gate0ReceiptError(f"receipt is not valid JSON: {target}") from exc
    if not isinstance(payload, dict):
        raise Gate0ReceiptError(f"receipt must be a JSON object: {target}")
    return payload


def _require_sha256(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise Gate0ReceiptError(f"{field} is not a lowercase SHA-256 digest")
    return text


def _require_lineage(
    payload: Mapping[str, Any],
    *,
    schema: str,
    run_id: str,
    contract_sha256: str,
) -> None:
    if payload.get("schema") != schema:
        raise Gate0ReceiptError(f"schema mismatch: expected {schema}")
    if payload.get("run_id") != run_id:
        raise Gate0ReceiptError("run_id lineage mismatch")
    if payload.get("contract_sha256") != contract_sha256:
        raise Gate0ReceiptError("contract_sha256 lineage mismatch")
    if payload.get("evaluation_year_accessed") is not False:
        raise Gate0ReceiptError("evaluation-year access is forbidden")


def _git(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Gate0ReceiptError(f"git command failed: {' '.join(args)}") from exc


def _closure_entries(closure_path: Path) -> list[str]:
    if not closure_path.is_file():
        raise Gate0ReceiptError(f"source closure is missing: {closure_path}")
    rows = [
        row.strip()
        for row in closure_path.read_text(encoding="utf-8").splitlines()
        if row.strip() and not row.strip().startswith("#")
    ]
    if not rows or len(rows) != len(set(rows)):
        raise Gate0ReceiptError("source closure must be non-empty and unique")
    return rows


def produce_source_manifest(
    repo_root: str | Path,
    closure_path: str | Path,
    destination: str | Path,
    *,
    run_id: str,
    contract_sha256: str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    closure = Path(closure_path).resolve()
    head = _git(root, "rev-parse", "HEAD")
    manifest = build_source_manifest(root, _closure_entries(closure), head)
    payload: dict[str, Any] = {
        "schema": SOURCE_SCHEMA,
        "run_id": run_id,
        "contract_sha256": _require_sha256(contract_sha256, "contract_sha256"),
        "git_commit": head,
        "entry_count": len(manifest.entries),
        "entries": [entry.to_payload() for entry in manifest.entries],
        "evaluation_year_accessed": False,
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    write_once_json(destination, payload)
    return payload


def validate_source_manifest_receipt(
    path: str | Path,
    *,
    run_id: str,
    contract_sha256: str,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _require_lineage(payload, schema=SOURCE_SCHEMA, run_id=run_id, contract_sha256=contract_sha256)
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries or payload.get("entry_count") != len(entries):
        raise Gate0ReceiptError("source manifest entries are incomplete")
    identity = _require_sha256(payload.get("identity_sha256"), "identity_sha256")
    unsigned = dict(payload)
    unsigned.pop("identity_sha256", None)
    if canonical_sha256(unsigned) != identity:
        raise Gate0ReceiptError("source manifest identity hash mismatch")
    paths: set[str] = set()
    for row in entries:
        if not isinstance(row, Mapping):
            raise Gate0ReceiptError("source manifest entry must be an object")
        relative = str(row.get("path", ""))
        if not relative or relative in paths or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise Gate0ReceiptError("source manifest entry path is invalid")
        paths.add(relative)
        _require_sha256(row.get("sha256"), "source entry sha256")
        if int(row.get("size_bytes", -1)) < 0:
            raise Gate0ReceiptError("source manifest entry size is invalid")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        if _git(root, "rev-parse", "HEAD") != payload.get("git_commit"):
            raise Gate0ReceiptError("source manifest Git commit is stale")
        for row in entries:
            source = (root / str(row["path"])).resolve()
            try:
                source.relative_to(root)
            except ValueError as exc:
                raise Gate0ReceiptError("source manifest path escapes repository") from exc
            if not source.is_file() or source.stat().st_size != int(row["size_bytes"]):
                raise Gate0ReceiptError(f"source manifest file is missing or changed: {row['path']}")
            if sha256_file(source) != row["sha256"]:
                raise Gate0ReceiptError(f"source manifest hash mismatch: {row['path']}")
            if _git(root, "status", "--porcelain=v1", "--", str(row["path"])):
                raise Gate0ReceiptError(f"source manifest file is dirty: {row['path']}")
    return payload


def _source_relative(source_root: Path, repo_root: Path) -> str:
    try:
        return source_root.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise Gate0ReceiptError("iTransformer source root must be inside the repository") from exc


def produce_itransformer_receipt(
    source_root: str | Path,
    destination: str | Path,
    *,
    run_id: str,
    contract_sha256: str,
    source_manifest_sha256: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    repo = Path(repo_root).resolve()
    if not source.is_dir():
        raise Gate0ReceiptError(f"iTransformer source root is missing: {source}")
    commit = _git(source, "rev-parse", "HEAD")
    if commit != OFFICIAL_COMMIT:
        raise Gate0ReceiptError(f"iTransformer commit mismatch: {commit}")
    imported = {
        path.relative_to(source).as_posix(): sha256_file(path)
        for path in sorted(source.rglob("*.py"))
    }
    if not imported:
        raise Gate0ReceiptError("iTransformer source contains no Python files")
    license_path = next(
        (candidate for candidate in (source / "LICENSE", source / "LICENSE.md", source / "LICENSE.txt") if candidate.is_file()),
        None,
    )
    if license_path is None:
        raise Gate0ReceiptError("iTransformer license is missing")
    relative_source = _source_relative(source, repo)
    legacy = {
        "schema_version": "formal-v4.1-itransformer-source-v1",
        "source_root": relative_source,
        "repository": OFFICIAL_REPOSITORY,
        "commit": commit,
        "backbone_class": OFFICIAL_BACKBONE_CLASS,
        "imported_file_hashes": imported,
        "license_file": license_path.relative_to(source).as_posix(),
        "license_sha256": sha256_file(license_path),
        "reproduction_level": "official_backbone_adaptation",
        "verified": True,
    }
    verify_itransformer_source_files(source, legacy, require_license=True)
    payload = {
        "schema": ITRANSFORMER_SCHEMA,
        "run_id": run_id,
        "contract_sha256": _require_sha256(contract_sha256, "contract_sha256"),
        "source_manifest_sha256": _require_sha256(source_manifest_sha256, "source_manifest_sha256"),
        **{key: value for key, value in legacy.items() if key != "schema_version"},
        "evaluation_year_accessed": False,
    }
    write_once_json(destination, payload)
    return payload


def validate_itransformer_receipt(
    path: str | Path,
    *,
    run_id: str,
    contract_sha256: str,
    source_manifest_sha256: str,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _require_lineage(payload, schema=ITRANSFORMER_SCHEMA, run_id=run_id, contract_sha256=contract_sha256)
    if payload.get("source_manifest_sha256") != source_manifest_sha256:
        raise Gate0ReceiptError("iTransformer source-manifest lineage mismatch")
    if payload.get("repository") != OFFICIAL_REPOSITORY or payload.get("commit") != OFFICIAL_COMMIT:
        raise Gate0ReceiptError("iTransformer source is not the frozen official revision")
    if payload.get("backbone_class") != OFFICIAL_BACKBONE_CLASS:
        raise Gate0ReceiptError("iTransformer backbone class mismatch")
    if payload.get("reproduction_level") != "official_backbone_adaptation" or payload.get("verified") is not True:
        raise Gate0ReceiptError("iTransformer adaptation level is invalid")
    imported = payload.get("imported_file_hashes")
    if not isinstance(imported, Mapping) or not imported:
        raise Gate0ReceiptError("iTransformer imported source hashes are missing")
    _require_sha256(payload.get("license_sha256"), "iTransformer license_sha256")
    if source_root is not None:
        legacy = {"schema_version": "formal-v4.1-itransformer-source-v1", **{
            key: payload[key]
            for key in (
                "source_root", "repository", "commit", "backbone_class",
                "imported_file_hashes", "license_file", "license_sha256",
                "reproduction_level", "verified",
            )
        }}
        try:
            verify_itransformer_source_files(source_root, legacy, require_license=True)
        except ValueError as exc:
            raise Gate0ReceiptError(str(exc)) from exc
    return payload


def run_native_diffopt_probe() -> dict[str, Any]:
    try:
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except Exception as exc:
        raise Gate0ReceiptError(f"DiffLP dependency import failed: {type(exc).__name__}: {exc}") from exc
    x = cp.Variable(1)
    q = cp.Parameter(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - q)), [x >= 0])
    if not problem.is_dpp():
        raise Gate0ReceiptError("DiffLP native probe is not DPP compliant")
    layer = CvxpyLayer(problem, parameters=[q], variables=[x])
    value = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    output = layer(value)[0]
    output.sum().backward()
    gradient = float(value.grad.detach().abs().sum()) if value.grad is not None else 0.0
    primal = float(output.detach().reshape(-1)[0])
    return {
        "dpp_passed": True,
        "finite": math.isfinite(primal) and math.isfinite(gradient),
        "primal_value": primal,
        "gradient_norm": gradient,
        "solver": "SCS",
    }


def produce_diffopt_receipt(
    lock_path: str | Path,
    destination: str | Path,
    *,
    run_id: str,
    contract_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    lock_target = Path(lock_path).resolve()
    lock = _read_json(lock_target)
    expected_python = Path(str(lock.get("python_executable", ""))).resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise Gate0ReceiptError(f"DiffLP interpreter mismatch: {actual_python} != {expected_python}")
    requirements = lock_target.with_suffix(".in")
    if not requirements.is_file() or sha256_file(requirements) != lock.get("requirements_sha256"):
        raise Gate0ReceiptError("DiffLP requirements hash mismatch")
    packages = {}
    for package in ("numpy", "torch", "cvxpy", "cvxpylayers", "diffcp", "ecos"):
        try:
            packages[package] = torch.__version__ if package == "torch" else metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise Gate0ReceiptError(f"DiffLP dependency is missing: {package}") from exc
    expected_packages = lock.get("packages")
    if not isinstance(expected_packages, Mapping) or any(packages.get(key) != str(value) for key, value in expected_packages.items()):
        raise Gate0ReceiptError("DiffLP installed package versions do not match the frozen lock")
    probe = run_native_diffopt_probe()
    if probe.get("dpp_passed") is not True or probe.get("finite") is not True:
        raise Gate0ReceiptError("DiffLP native probe is non-finite or not DPP compliant")
    gradient = float(probe.get("gradient_norm", 0.0))
    if not math.isfinite(gradient) or gradient <= 0.0:
        raise Gate0ReceiptError("DiffLP native probe gradient must be finite and nonzero")
    payload = {
        "schema": DIFFOPT_SCHEMA,
        "run_id": run_id,
        "contract_sha256": _require_sha256(contract_sha256, "contract_sha256"),
        "source_manifest_sha256": _require_sha256(source_manifest_sha256, "source_manifest_sha256"),
        "eligible": True,
        "finite": True,
        "dpp_passed": True,
        "gradient_norm": gradient,
        "primal_value": float(probe["primal_value"]),
        "solver": str(probe["solver"]),
        "python_executable": str(actual_python),
        "packages": packages,
        "lock_sha256": sha256_file(lock_target),
        "requirements_sha256": sha256_file(requirements),
        "evaluation_year_accessed": False,
    }
    write_once_json(destination, payload)
    return payload


def validate_diffopt_receipt(
    path: str | Path,
    *,
    run_id: str,
    contract_sha256: str,
    source_manifest_sha256: str,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = _read_json(path)
    _require_lineage(payload, schema=DIFFOPT_SCHEMA, run_id=run_id, contract_sha256=contract_sha256)
    if payload.get("source_manifest_sha256") != source_manifest_sha256:
        raise Gate0ReceiptError("DiffLP source-manifest lineage mismatch")
    gradient = float(payload.get("gradient_norm", 0.0))
    if payload.get("eligible") is not True or payload.get("finite") is not True or payload.get("dpp_passed") is not True:
        raise Gate0ReceiptError("DiffLP environment is not eligible")
    if not math.isfinite(gradient) or gradient <= 0.0:
        raise Gate0ReceiptError("DiffLP gradient is not finite and nonzero")
    if not isinstance(payload.get("packages"), Mapping) or not payload["packages"]:
        raise Gate0ReceiptError("DiffLP package versions are missing")
    _require_sha256(payload.get("lock_sha256"), "DiffLP lock_sha256")
    if lock_path is not None and sha256_file(lock_path) != payload["lock_sha256"]:
        raise Gate0ReceiptError("DiffLP lock hash mismatch")
    return payload


def validate_capacity_receipt(
    path: str | Path,
    *,
    run_id: str,
    contract_sha256: str,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    _require_lineage(payload, schema=CAPACITY_SCHEMA, run_id=run_id, contract_sha256=contract_sha256)
    if payload.get("source_manifest_sha256") != source_manifest_sha256:
        raise Gate0ReceiptError("capacity source-manifest lineage mismatch")
    if payload.get("status") != "pass" or not isinstance(payload.get("selected"), Mapping):
        raise Gate0ReceiptError("capacity audit did not pass")
    if payload.get("fit_years") != [2015, 2016, 2017, 2018]:
        raise Gate0ReceiptError("capacity fit years are not 2015-2018")
    if payload.get("selection_influenced_capacity") is not False:
        raise Gate0ReceiptError("selection data influenced capacity")
    multiplier = float(payload["selected"].get("multiplier", float("nan")))
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise Gate0ReceiptError("capacity multiplier is invalid")
    if not isinstance(payload.get("candidate_multipliers"), list) or not payload["candidate_multipliers"]:
        raise Gate0ReceiptError("capacity candidate grid is missing")
    if not isinstance(payload.get("thresholds"), Mapping) or not payload["thresholds"]:
        raise Gate0ReceiptError("capacity thresholds are missing")
    _require_sha256(payload.get("capacity_scenario_hash"), "capacity_scenario_hash")
    return payload


def validate_prerequisites(
    paths: Gate0PrerequisitePaths,
    *,
    run_id: str,
    contract_sha256: str,
    repo_root: str | Path | None = None,
    itransformer_source: str | Path | None = None,
    lock_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    source = validate_source_manifest_receipt(
        paths.source_manifest,
        run_id=run_id,
        contract_sha256=contract_sha256,
        repo_root=repo_root,
    )
    source_hash = sha256_file(paths.source_manifest)
    itransformer = validate_itransformer_receipt(
        paths.itransformer_receipt,
        run_id=run_id,
        contract_sha256=contract_sha256,
        source_manifest_sha256=source_hash,
        source_root=itransformer_source,
    )
    diffopt = validate_diffopt_receipt(
        paths.diffopt_receipt,
        run_id=run_id,
        contract_sha256=contract_sha256,
        source_manifest_sha256=source_hash,
        lock_path=lock_path,
    )
    capacity = validate_capacity_receipt(
        paths.capacity_receipt,
        run_id=run_id,
        contract_sha256=contract_sha256,
        source_manifest_sha256=source_hash,
    )
    return {
        "source_manifest": {"passed": True, "path": str(paths.source_manifest), "sha256": source_hash, "entry_count": source["entry_count"]},
        "itransformer_receipt": {"passed": True, "path": str(paths.itransformer_receipt), "sha256": sha256_file(paths.itransformer_receipt), "commit": itransformer["commit"]},
        "diffopt_receipt": {"passed": True, "path": str(paths.diffopt_receipt), "sha256": sha256_file(paths.diffopt_receipt), "gradient_norm": diffopt["gradient_norm"]},
        "capacity_receipt": {"passed": True, "path": str(paths.capacity_receipt), "sha256": sha256_file(paths.capacity_receipt), "selected": capacity["selected"]},
    }


__all__ = [
    "CAPACITY_SCHEMA",
    "CURRENT_GATE_SCHEMA",
    "DIFFOPT_SCHEMA",
    "Gate0InputsV42",
    "Gate0PrerequisitePaths",
    "Gate0ReceiptError",
    "ITRANSFORMER_SCHEMA",
    "SOURCE_SCHEMA",
    "produce_diffopt_receipt",
    "produce_itransformer_receipt",
    "produce_source_manifest",
    "run_native_diffopt_probe",
    "validate_capacity_receipt",
    "validate_diffopt_receipt",
    "validate_itransformer_receipt",
    "validate_prerequisites",
    "validate_source_manifest_receipt",
]
