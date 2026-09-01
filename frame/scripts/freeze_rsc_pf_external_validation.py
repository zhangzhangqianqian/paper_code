"""Freeze the validation-only external-baseline handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_baseline_training import METHODS, SEEDS, _METHOD_SAFE, _forbidden_test_path  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def freeze_external_validation(output_root: Path, registry_path: Path) -> dict[str, Any]:
    """Require all validation evidence and write an immutable handoff receipt."""

    root = Path(output_root)
    registry = Path(registry_path)
    if _forbidden_test_path(root) or _forbidden_test_path(registry):
        raise ValueError("freeze paths may not contain a sealed test-set path")
    manifest_path = root / "external_validation_manifest.json"
    source_receipt_path = root / "sources" / "source_receipt.json"
    resource_path = root / "lp_resource_gate.json"
    for path in (manifest_path, source_receipt_path, resource_path, registry):
        if not path.is_file():
            raise FileNotFoundError(f"required freeze artifact is missing: {path}")
    manifest = _read_json(manifest_path)
    source_receipt = _read_json(source_receipt_path)
    resource = _read_json(resource_path)
    if manifest.get("test_set_accessed") is not False or source_receipt.get("test_set_accessed") is not False:
        raise ValueError("freeze evidence is not test-set-free")
    if resource.get("gate_passed") is not True or resource.get("test_set_accessed") is not False:
        raise ValueError("LP resource gate has not passed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(METHODS) * len(SEEDS):
        raise ValueError("freeze requires exactly 15 method-seed evaluation entries")
    seen: set[tuple[str, int]] = set()
    artifact_hashes: dict[str, str] = {}
    feasibility: dict[str, dict[str, float]] = {}
    optimizer_roles = {
        "iTransformer-PTO": "none at inference",
        "DecisionFocused-Online": "exact optimizer at inference",
        "DigitalTwins-Policy": "none at inference",
    }
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("evaluation manifest entry must be an object")
        method = str(entry.get("method_id"))
        seed = int(entry.get("seed", -1))
        if method not in METHODS or seed not in SEEDS or (method, seed) in seen:
            raise ValueError("evaluation manifest has an invalid or duplicate method-seed entry")
        seen.add((method, seed))
        if entry.get("test_set_accessed") is not False:
            raise ValueError("evaluation entry accessed test data")
        receipt_path = Path(str(entry.get("checkpoint", ""))).parent / "evaluation" / "evaluation_receipt.json"
        if _forbidden_test_path(receipt_path) or not receipt_path.is_file():
            raise FileNotFoundError(f"evaluation receipt is missing: {receipt_path}")
        receipt = _read_json(receipt_path)
        if receipt.get("test_set_accessed") is not False or not _finite(receipt.get("metrics")):
            raise ValueError(f"evaluation receipt is invalid: {receipt_path}")
        checkpoint = Path(str(receipt.get("checkpoint", "")))
        if _forbidden_test_path(checkpoint) or not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
        digest = _sha256(checkpoint)
        if digest != str(receipt.get("checkpoint_sha256", "")).lower():
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        artifact_hashes[f"{method}/seed_{seed}"] = digest
        role = str(receipt.get("optimizer_role"))
        if role != optimizer_roles[method]:
            raise ValueError(f"optimizer role mismatch for {method}/seed_{seed}")
        metrics = receipt["metrics"]
        feasibility.setdefault(method, {"min_rate": 1.0, "max_shortage": 0.0})
        feasibility[method]["min_rate"] = min(feasibility[method]["min_rate"], float(metrics["feasibility_rate"]))
        feasibility[method]["max_shortage"] = max(feasibility[method]["max_shortage"], float(metrics["shortage_mean"]))
        expected_lp = 0 if method == "DigitalTwins-Policy" else int(metrics["windows"])
        if int(receipt.get("exact_lp_calls", -1)) != expected_lp:
            raise ValueError(f"LP accounting mismatch for {method}/seed_{seed}")
    if seen != {(method, seed) for method in METHODS for seed in SEEDS}:
        raise ValueError("freeze does not cover every frozen method and seed")
    source_methods = {str(item.get("method_id")) for item in source_receipt.get("methods", [])}
    if source_methods != set(METHODS):
        raise ValueError("source receipt does not cover all methods")
    source_hash = _sha256(source_receipt_path)
    registry_hash = _sha256(registry)
    payload = {
        "schema_version": "rsc-pf-external-validation-freeze-v1",
        "status": "frozen",
        "frozen_at_utc": "2026-09-01",
        "registry_path": str(registry),
        "registry_sha256": registry_hash,
        "source_receipt": str(source_receipt_path),
        "source_receipt_sha256": source_hash,
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": _sha256(manifest_path),
        "resource_gate": resource,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "artifact_hashes": artifact_hashes,
        "optimizer_roles": optimizer_roles,
        "feasibility_summary": feasibility,
        "test_set_accessed": False,
        "metric_definitions": manifest.get("metric_definitions", {}),
        "handoff_scope": "validation-only external-baseline evidence; no sealed test split and no manuscript rewrite",
    }
    freeze_path = root / "validation_freeze_receipt.json"
    freeze_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    handoff = root / "implementation_to_test_handoff.md"
    handoff.write_text(
        "# External-baseline validation handoff\n\n"
        "Status: frozen validation evidence for the three external comparison methods.\n\n"
        "- Methods: iTransformer-PTO, DecisionFocused-Online, DigitalTwins-Policy\n"
        "- Seeds: 2026--2030 (five per method)\n"
        "- Split: train/validation only; the sealed 2021 test split was not accessed.\n"
        "- iTransformer-PTO and DecisionFocused-Online use one exact LP per validation window; DigitalTwins-Policy uses zero exact LP calls at inference.\n"
        "- The next permitted step is the results-table/figure phase. Do not change the frozen model, source registry, split definitions, or optimizer-role accounting.\n\n"
        f"Machine-readable receipt: `{freeze_path}`\n",
        encoding="utf-8",
    )
    return {"freeze_receipt": str(freeze_path), "handoff": str(handoff), "status": "frozen", "test_set_accessed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze external-baseline validation evidence")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_external_validation(args.output_root, args.registry), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

