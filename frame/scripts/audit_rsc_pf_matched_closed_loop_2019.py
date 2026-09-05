"""Independent auditor for matched closed-loop row artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
import sys

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.closed_loop_metrics import summarize_closed_loop

METHOD_ORDER = ("RSC-PF", "iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy")
SEEDS = (2026, 2027, 2028, 2029, 2030)
EXPECTED_GATE1_RATIO = 1.0353263112927777


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_manifest(manifest: Mapping[str, Any], *, require_full: bool | None = None) -> dict[str, Any]:
    """Validate protocol identity and the method/seed matrix without I/O."""

    if manifest.get("schema") != "rsc-pf-matched-closed-loop-manifest-v1":
        raise ValueError("matched closed-loop manifest schema is invalid")
    if int(manifest.get("selection_year", -1)) != 2019 or int(manifest.get("evaluation_year", -1)) != 2020:
        raise ValueError("manifest year boundary is invalid")
    if manifest.get("gate1_authorized") is not False or manifest.get("formal_candidate") is not False:
        raise ValueError("manifest must preserve failed Gate-1 status")
    if manifest.get("test_set_accessed") is not False or manifest.get("evaluation_year_accessed") is not False:
        raise ValueError("manifest records unauthorized future access")
    failure = manifest.get("gate1_failure", {})
    if failure.get("criterion") != "electricity_wape_ratio" or float(failure.get("observed", float("nan"))) != EXPECTED_GATE1_RATIO or float(failure.get("limit", float("nan"))) != 1.02:
        raise ValueError("Gate-1 failure evidence is not preserved")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("manifest rows must be a list")
    smoke = bool(manifest.get("smoke", False))
    full = bool(not smoke) if require_full is None else bool(require_full)
    expected_count = 16 if full else len(rows)
    if full and len(rows) != expected_count:
        raise ValueError("full manifest must contain exactly 16 model rows")
    expected: set[tuple[str, int]] = {("RSC-PF", 2026)} | {(method, seed) for method in METHOD_ORDER[1:] for seed in SEEDS}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("manifest row must be an object")
        method, seed = str(row.get("method_id")), int(row.get("seed", -1))
        key = (method, seed)
        if key in seen:
            raise ValueError("duplicate method/seed row")
        seen.add(key)
        if method not in METHOD_ORDER:
            raise ValueError(f"unknown method in manifest: {method}")
        expected_role = "exact optimizer at inference" if method in {"iTransformer-PTO", "DecisionFocused-Online"} else "none at inference"
        if str(row.get("optimizer_role")) != expected_role:
            raise ValueError(f"optimizer role mismatch for {method}")
        origins = int(manifest.get("origins", -1))
        expected_calls = origins if method in {"iTransformer-PTO", "DecisionFocused-Online"} else 0
        if int(row.get("inference_lp_calls", -1)) != expected_calls:
            raise ValueError(f"inference LP call count mismatch for {method}")
    if full and seen != expected:
        raise ValueError("manifest does not contain the frozen 16-row method/seed matrix")
    reference = manifest.get("reference")
    if not isinstance(reference, Mapping) or reference.get("method_id") != "Perfect-Information-MPC":
        raise ValueError("manifest is missing the separate Perfect-Information-MPC reference")
    origins = int(manifest.get("origins", -1))
    if int(reference.get("reference_lp_calls", -1)) != origins:
        raise ValueError("reference LP call count does not equal origin count")
    if full and origins != 8709:
        raise ValueError("full manifest must contain exactly 8,709 origins")
    return {"rows": len(rows), "origins": origins, "full_matrix": bool(full), "methods": sorted(seen)}


def _audit_row(row: Mapping[str, Any], root: Path, *, atol: float = 1.0e-8, thermal_active_scales: Mapping[str, float] | None = None) -> dict[str, Any]:
    row_dir = Path(str(row["path"]))
    if not row_dir.is_absolute():
        row_dir = FRAME_ROOT / row_dir
    npz_path, receipt_path = row_dir / "rollout.npz", row_dir / "receipt.json"
    if not npz_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"row artifact is incomplete: {row_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("test_set_accessed") is not False or receipt.get("evaluation_year_accessed") is not False:
        raise ValueError(f"row {row['method_id']} records future access")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    summary = summarize_closed_loop(arrays, thermal_active_scales=thermal_active_scales)
    # Compare only the scalar summary values that are deterministic and finite;
    # NaN fields (for an empty warm-up latency vector) are intentionally skipped.
    recorded = receipt.get("metrics", {})
    for section, value in summary.items():
        if section not in recorded or not isinstance(value, Mapping):
            continue
        for name, scalar in value.items():
            if isinstance(scalar, (int, float)) and np.isfinite(float(scalar)) and name in recorded[section] and np.isfinite(float(recorded[section][name])) and not np.isclose(float(scalar), float(recorded[section][name]), atol=atol, rtol=0.0):
                raise ValueError(f"row metric mismatch for {row['method_id']}:{section}.{name}")
    return {"method_id": row["method_id"], "seed": int(row["seed"]), "rollout_sha256": _sha256(npz_path), "rows": int(len(arrays["times"])), "metrics_recomputed": summary}


def _audit_legacy_replay_provenance(root: Path, rsc_row: Mapping[str, Any], *, origins: int) -> dict[str, Any]:
    """Require the frozen legacy replay receipt and first-origin equality."""

    provenance = root / "provenance"
    receipt_path = provenance / "RSC_PF_LEGACY_REPLAY.json"
    replay_path = provenance / "rsc_pf_legacy_replay.npz"
    if not receipt_path.is_file() or not replay_path.is_file():
        raise FileNotFoundError("full matched run is missing the RSC-PF legacy replay provenance")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "rsc-pf-legacy-replay-v1":
        raise ValueError("legacy replay receipt schema is invalid")
    if int(receipt.get("rows", -1)) != origins:
        raise ValueError("legacy replay row count does not match the full run")
    if receipt.get("scheduler_soc_source") != "materialized_origin" or receipt.get("evaluation_year_accessed") is not False:
        raise ValueError("legacy replay receipt has invalid SOC or year provenance")
    checks = receipt.get("checks", {})
    if not isinstance(checks, Mapping) or not all(bool(value) for value in checks.values()):
        raise ValueError("legacy replay receipt does not contain all passing checks")
    row_dir = Path(str(rsc_row["path"]))
    if not row_dir.is_absolute():
        row_dir = FRAME_ROOT / row_dir
    with np.load(row_dir / "rollout.npz", allow_pickle=False) as payload:
        current = {name: np.asarray(payload[name]) for name in payload.files}
    with np.load(replay_path, allow_pickle=False) as payload:
        legacy = {name: np.asarray(payload[name]) for name in payload.files}
    aliases = {
        "forecast": "forecast_nominal",
        "scheduler_demand": "scheduler_demand",
        "planned_dispatch": "planned_dispatch",
        "settled_dispatch": "settled_dispatch",
    }
    first_checks: dict[str, bool] = {}
    for current_name, legacy_name in aliases.items():
        first_checks[current_name] = bool(np.allclose(current[current_name][0], legacy[legacy_name][0], atol=1.0e-5, rtol=1.0e-6))
    first_checks["times"] = bool(np.array_equal(current["times"][0:1].astype("datetime64[ns]"), legacy["times"][0:1].astype("datetime64[ns]")))
    first_checks["state_hash"] = str(current["state_hashes"][0]) == str(legacy["state_hashes"][0])
    if not all(first_checks.values()):
        raise ValueError(f"full RSC-PF first-origin replay mismatch: {first_checks}")
    return {"receipt": receipt, "first_origin_checks": first_checks, "replay_sha256": _sha256(replay_path)}


def audit_artifacts(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    protocol = audit_manifest(manifest)
    root = path.parent
    scales = manifest.get("thermal_active_scales")
    if not isinstance(scales, Mapping) or any(float(scales.get(name, 0.0)) <= 0.0 for name in ("cooling", "heating")):
        raise ValueError("manifest is missing positive frozen thermal active scales")
    rows = [_audit_row(row, root, thermal_active_scales=scales) for row in manifest["rows"]]
    reference = manifest["reference"]
    ref_dir = Path(str(reference["path"]))
    if not ref_dir.is_absolute():
        ref_dir = FRAME_ROOT / ref_dir
    ref_npz = ref_dir / "perfect_information_mpc.npz"
    if not ref_npz.is_file():
        raise FileNotFoundError(ref_npz)
    with np.load(ref_npz, allow_pickle=False) as payload:
        ref_arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if len(ref_arrays.get("times", ())) != protocol["origins"]:
        raise ValueError("reference origin count mismatch")
    legacy = None
    if protocol["full_matrix"]:
        rsc_rows = [row for row in manifest["rows"] if str(row.get("method_id")) == "RSC-PF" and int(row.get("seed", -1)) == 2026]
        if len(rsc_rows) != 1:
            raise ValueError("full manifest must contain one RSC-PF seed-2026 row")
        legacy = _audit_legacy_replay_provenance(root, rsc_rows[0], origins=protocol["origins"])
    projection_path = root / "RESOURCE_PROJECTION.json"
    projection = json.loads(projection_path.read_text(encoding="utf-8")) if projection_path.is_file() else None
    result = {"schema": "rsc-pf-matched-closed-loop-audit-v1", "manifest": protocol, "rows": rows, "reference": {"method_id": "Perfect-Information-MPC", "rows": protocol["origins"], "rollout_sha256": _sha256(ref_npz)}, "legacy_replay": legacy, "resource_projection": projection, "gate1_authorized": False, "formal_candidate": False, "test_set_accessed": False, "evaluation_year_accessed": False, "status": "PASS"}
    output = root / "MATCHED_CLOSED_LOOP_AUDIT.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = audit_artifacts(args.manifest)
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "test_set_accessed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
