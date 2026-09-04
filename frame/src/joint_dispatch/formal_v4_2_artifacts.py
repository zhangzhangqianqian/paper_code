"""Immutable artifacts and resumable method-row receipts for formal-v4.2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


RUN_SCHEMA = "formal-v4.2-run-v1"
ROW_SCHEMA = "formal-v4.2-method-row-v1"
FAILURE_SCHEMA = "formal-v4.2-method-failure-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^formal_v4_2_[A-Za-z0-9][A-Za-z0-9_-]*$")
_REQUIRED_COMPLETE = {
    "contract_sha256",
    "source_manifest_sha256",
    "data_sha256",
    "checkpoint_sha256",
    "runtime_seconds",
    "status",
}


class LineageError(ValueError):
    """Raised when an artifact cannot belong to the requested lineage."""


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    compact = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    encoded = canonical_bytes(payload)
    if target.exists():
        if target.read_bytes() == encoded:
            return hashlib.sha256(encoded).hexdigest()
        raise FileExistsError(f"immutable artifact differs: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.writing")
    if temporary.exists():
        raise FileExistsError(f"stale artifact write exists: {temporary}")
    temporary.write_bytes(encoded)
    temporary.replace(target)
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, field: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise LineageError(f"{field} must be a lowercase SHA-256 digest")
    return text


def verify_lineage(payload: Mapping[str, Any], expected: Mapping[str, str]) -> None:
    for field, required in expected.items():
        actual = payload.get(field)
        if actual != required:
            raise LineageError(f"{field} lineage mismatch: {actual!r} != {required!r}")


def create_v42_run_root(
    output_root: str | Path,
    run_id: str,
    contract_sha256: str,
) -> Path:
    if not _RUN_ID.fullmatch(str(run_id)):
        raise ValueError("formal-v4.2 run id is invalid")
    contract_hash = _require_sha256(contract_sha256, "contract_sha256")
    base = Path(output_root).resolve()
    root = (base / run_id).resolve()
    try:
        root.relative_to(base)
    except ValueError as exc:
        raise ValueError("formal-v4.2 run root escapes output root") from exc
    write_once_json(
        root / "protocol" / "RUN.json",
        {
            "schema": RUN_SCHEMA,
            "run_id": run_id,
            "contract_sha256": contract_hash,
        },
    )
    return root


@dataclass(frozen=True, order=True)
class MethodSeedKey:
    method_id: str
    seed: int | None

    def __post_init__(self) -> None:
        if not self.method_id.strip():
            raise ValueError("method_id must be non-empty")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed must be non-negative or None")

    @property
    def directory_name(self) -> str:
        method = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.method_id).strip("_")
        seed = "deterministic" if self.seed is None else f"seed_{int(self.seed)}"
        return f"{method}/{seed}"


class ArtifactStore:
    def __init__(self, root: str | Path, *, contract_sha256: str) -> None:
        self.root = Path(root).resolve()
        self.contract_sha256 = _require_sha256(contract_sha256, "contract_sha256")

    def row_root(self, key: MethodSeedKey) -> Path:
        return self.root / "rows" / key.directory_name

    def complete_path(self, key: MethodSeedKey) -> Path:
        return self.row_root(key) / "COMPLETE.json"

    def failure_path(self, key: MethodSeedKey) -> Path:
        return self.row_root(key) / "FAILURE.json"

    def complete(self, key: MethodSeedKey, payload: Mapping[str, Any]) -> str:
        missing = sorted(_REQUIRED_COMPLETE - set(payload))
        if missing:
            raise LineageError(f"complete receipt is missing {missing[0]}")
        if payload.get("status") != "complete":
            raise LineageError("complete receipt status must be complete")
        if payload.get("contract_sha256") != self.contract_sha256:
            raise LineageError("contract_sha256 lineage mismatch")
        for field in (
            "contract_sha256",
            "source_manifest_sha256",
            "data_sha256",
            "checkpoint_sha256",
        ):
            _require_sha256(payload[field], field)
        runtime = float(payload["runtime_seconds"])
        if not (runtime >= 0.0 and runtime < float("inf")):
            raise LineageError("runtime_seconds must be finite and non-negative")
        receipt = {
            **dict(payload),
            "schema": ROW_SCHEMA,
            "method_id": key.method_id,
            "seed": key.seed,
        }
        return write_once_json(self.complete_path(key), receipt)

    def failed(
        self,
        key: MethodSeedKey,
        exception: BaseException,
        lineage: Mapping[str, Any],
    ) -> str:
        source_hash = _require_sha256(
            lineage.get("source_manifest_sha256", ""),
            "source_manifest_sha256",
        )
        data_hash = _require_sha256(lineage.get("data_sha256", ""), "data_sha256")
        receipt = {
            "schema": FAILURE_SCHEMA,
            "status": "failed",
            "method_id": key.method_id,
            "seed": key.seed,
            "contract_sha256": self.contract_sha256,
            "source_manifest_sha256": source_hash,
            "data_sha256": data_hash,
            "exception_class": type(exception).__name__,
            "exception_message": str(exception),
        }
        return write_once_json(self.failure_path(key), receipt)

    def completed_rows(self) -> set[MethodSeedKey]:
        completed: set[MethodSeedKey] = set()
        for path in sorted((self.root / "rows").glob("*/*/COMPLETE.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            verify_lineage(payload, {"contract_sha256": self.contract_sha256})
            if payload.get("schema") != ROW_SCHEMA or payload.get("status") != "complete":
                raise LineageError(f"invalid complete receipt: {path}")
            completed.add(MethodSeedKey(str(payload["method_id"]), payload.get("seed")))
        return completed


def write_failure_receipt(
    path: str | Path,
    *,
    stage: str,
    exception: BaseException,
    lineage: Mapping[str, Any],
) -> str:
    return write_once_json(
        path,
        {
            "schema": "formal-v4.2-stage-failure-v1",
            "status": "failed",
            "stage": str(stage),
            "exception_class": type(exception).__name__,
            "exception_message": str(exception),
            "lineage": dict(lineage),
        },
    )


__all__ = [
    "ArtifactStore",
    "LineageError",
    "MethodSeedKey",
    "canonical_sha256",
    "create_v42_run_root",
    "sha256_file",
    "verify_lineage",
    "write_failure_receipt",
    "write_once_json",
]
