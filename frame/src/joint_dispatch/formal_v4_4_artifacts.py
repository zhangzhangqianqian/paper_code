"""Write-once Pilot artifacts and independent metric recomputation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from .formal_v4_4_contract import FormalV44Contract
from .formal_v4_4_metrics import compute_forecast_metrics_v44


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value): return _jsonable(asdict(value))
    if isinstance(value, Mapping): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    return value


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def _ensure_finite(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values(): _ensure_finite(child)
    elif isinstance(value, (list, tuple)):
        for child in value: _ensure_finite(child)
    elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
        if not np.isfinite(value).all(): raise ValueError("artifact payload must be finite")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        raise ValueError("artifact payload must be finite")


def write_json_once(path: str | Path, payload: Mapping[str, Any]) -> str:
    """Write exactly once; even an identical second write is rejected."""
    _ensure_finite(payload); encoded = canonical_bytes(payload); target = Path(path)
    if target.exists(): raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True); temporary = target.with_name(f".{target.name}.writing")
    if temporary.exists(): raise FileExistsError(temporary)
    temporary.write_bytes(encoded); temporary.replace(target)
    return hashlib.sha256(encoded).hexdigest()


def write_npz_once(path: str | Path, arrays: Mapping[str, Any]) -> str:
    target = Path(path)
    if target.exists(): raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = {str(key): np.asarray(value) for key, value in arrays.items()}
    for value in normalized.values():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all(): raise ValueError("NPZ arrays must be finite")
    temporary = target.with_name(f".{target.name}.writing")
    np.savez_compressed(temporary, **normalized)
    generated = Path(str(temporary) + ".npz") if not temporary.name.endswith(".npz") else temporary
    generated.replace(target)
    return sha256_file(target)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def validate_lineage_v44(payload: Mapping[str, Any]) -> None:
    lineage = payload.get("lineage", payload)
    if not isinstance(lineage, Mapping): raise ValueError("v4.4 lineage must be an object")
    required = ("source_manifest_sha256", "train_data_sha256", "selection_data_sha256", "contract_sha256")
    missing = [key for key in required if not isinstance(lineage.get(key), str) or not _SHA256.fullmatch(lineage[key])]
    if missing: raise ValueError(f"v4.4 lineage missing valid {missing[0]}")
    if "accessed_years" in payload and 2020 in [int(value) for value in payload["accessed_years"]]: raise ValueError("evaluation year was accessed before Gate 2")


def recompute_pilot_receipt(report_dir: str | Path, contract: FormalV44Contract) -> dict[str, Any]:
    """Recompute stored Pilot metrics without importing the Pilot runner."""
    root = Path(report_dir)
    arrays_path = root / "PILOT_ARRAYS.npz"
    metrics_path = root / "PILOT_METRICS.json"
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as data:
            metrics_obj = compute_forecast_metrics_v44(data["prediction"], data["target"], data["probability"], data["prior_probability"], data["regimes"], data["times"])
        metrics = _jsonable(metrics_obj)
    elif metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("Pilot metrics or arrays are missing")
    if not isinstance(metrics, Mapping): raise ValueError("stored Pilot metrics must be an object")
    contract.validate(); _ensure_finite(metrics)
    return {"metrics": dict(metrics), "metrics_sha256": canonical_sha256(metrics)}


__all__ = ["canonical_sha256", "recompute_pilot_receipt", "sha256_file", "validate_lineage_v44", "write_json_once", "write_npz_once"]
