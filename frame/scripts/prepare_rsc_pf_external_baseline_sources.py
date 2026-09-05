"""Freeze auditable source receipts for the selected external baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_registry import load_external_registry  # noqa: E402

CONFIG_SCHEMA = "rsc-pf-external-implementation-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_CONFIG_KEYS = {
    "schema_version", "registry_path", "source_root", "output_root", "lookback", "horizon",
    "task_count", "task_order", "exog_dim", "dispatch_dim", "status_dim", "seeds", "training",
    "normalization_fit_split", "validation_only", "test_set_accessed",
}
REQUIRED_TRAINING_KEYS = {
    "batch_size", "learning_rate", "weight_decay", "gradient_clip", "min_epochs", "max_epochs", "patience",
}
OPTIONAL_RUNTIME_KEYS = {
    "data_protocol", "data_root", "train_file", "validation_file", "pilot_file", "historical_device_dim",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_external_implementation_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = dict(_read_json(source))
    unknown = sorted(set(payload) - REQUIRED_CONFIG_KEYS - OPTIONAL_RUNTIME_KEYS)
    missing = sorted(REQUIRED_CONFIG_KEYS - set(payload))
    if unknown or missing:
        raise ValueError(f"implementation config keys invalid; unknown={unknown}, missing={missing}")
    if payload["schema_version"] != CONFIG_SCHEMA:
        raise ValueError(f"schema_version must equal {CONFIG_SCHEMA!r}")
    if payload["lookback"] != 24 or payload["horizon"] != 4 or payload["task_count"] != 4:
        raise ValueError("lookback, horizon, and task_count must be 24, 4, and 4")
    if payload["task_order"] != ["electricity", "cooling", "heating", "gas"]:
        raise ValueError("task_order must match the canonical four-task order")
    if payload["exog_dim"] != 12 or payload["dispatch_dim"] != 21 or payload["status_dim"] != 6:
        raise ValueError("external implementation dimensions do not match the frozen contract")
    if payload["seeds"] != [2026, 2027, 2028, 2029, 2030]:
        raise ValueError("seeds must match the frozen five-seed protocol")
    training = payload["training"]
    if not isinstance(training, Mapping):
        raise ValueError("training must be an object")
    unknown_training = sorted(set(training) - REQUIRED_TRAINING_KEYS)
    missing_training = sorted(REQUIRED_TRAINING_KEYS - set(training))
    if unknown_training or missing_training:
        raise ValueError(f"training keys invalid; unknown={unknown_training}, missing={missing_training}")
    if training["batch_size"] != 256 or training["min_epochs"] != 20 or training["max_epochs"] != 30 or training["patience"] != 5:
        raise ValueError("training epoch/batch settings do not match the frozen protocol")
    if payload["normalization_fit_split"] != "train":
        raise ValueError("normalization must be fit on train only")
    if payload["validation_only"] is not True or payload["test_set_accessed"] is not False:
        raise ValueError("source preparation must be validation-only and test-set-free")
    return payload


def _forbidden_test_path(value: str) -> bool:
    normalized = value.replace("\\", "/").lower()
    return any(token in normalized for token in ("/test_set/", "/sealed_test/", "2021_test", "test-set"))


def _method_receipt(method: Any) -> dict[str, Any]:
    evidence_hash = method.evidence_sha256.lower()
    if not SHA256_RE.fullmatch(evidence_hash):
        raise ValueError(f"{method.method_id}: evidence_sha256 must be 64 lowercase hex characters")
    if not method.primary_source_url.startswith(("https://", "http://")):
        raise ValueError(f"{method.method_id}: primary_source_url must be an http(s) URL")
    if not method.license_route.strip():
        raise ValueError(f"{method.method_id}: license_route is required")
    source_kind = "official_code" if method.official_code_url else "paper_equations"
    if source_kind == "paper_equations" and method.license_route.lower() in {"forbidden", "license_forbidden", "no_lawful_route"}:
        raise ValueError(f"{method.method_id}: paper-equation reproduction has no lawful evidence route")
    return {
        "slot": method.slot,
        "method_id": method.method_id,
        "candidate_id": method.candidate_id,
        "paper_title": method.paper_title,
        "paper_year": method.paper_year,
        "doi": method.doi,
        "stable_id": method.stable_id,
        "primary_source_url": method.primary_source_url,
        "official_code_url": method.official_code_url,
        "source_kind": source_kind,
        "license_route": method.license_route,
        "reproduction_level": method.reproduction_level,
        "evidence_sha256": evidence_hash,
        "implementation_ready": method.implementation_ready,
        "test_set_accessed": False,
    }


def prepare_sources(registry_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    registry_file = Path(registry_path)
    output = Path(output_root)
    if _forbidden_test_path(str(registry_file)) or _forbidden_test_path(str(output)):
        raise ValueError("source receipt path may not refer to a sealed test-set path")
    registry = load_external_registry(registry_file)
    if len(registry.methods) != 3:
        raise ValueError("exactly three frozen external methods are required")
    methods = [_method_receipt(method) for method in registry.methods]
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "rsc-pf-external-source-receipt-v1",
        "registry_path": str(registry_file),
        "registry_sha256": _sha256(registry_file),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_only": True,
        "test_set_accessed": False,
        "implementation_ready": all(item["implementation_ready"] for item in methods),
        "methods": methods,
    }
    receipt_path = output / "source_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"receipt_path": str(receipt_path), "registry_sha256": receipt["registry_sha256"],
            "method_ids": [item["method_id"] for item in methods],
            "implementation_ready": receipt["implementation_ready"], "test_set_accessed": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.config:
        load_external_implementation_config(args.config)
    result = prepare_sources(args.registry, args.output_root)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
