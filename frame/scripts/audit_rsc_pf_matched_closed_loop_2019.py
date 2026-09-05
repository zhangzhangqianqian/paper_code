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


def _audit_row(row: Mapping[str, Any], root: Path, *, atol: float = 1.0e-8) -> dict[str, Any]:
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
    summary = summarize_closed_loop(arrays)
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


def audit_artifacts(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    protocol = audit_manifest(manifest)
    root = path.parent
    rows = [_audit_row(row, root) for row in manifest["rows"]]
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
    result = {"schema": "rsc-pf-matched-closed-loop-audit-v1", "manifest": protocol, "rows": rows, "reference": {"method_id": "Perfect-Information-MPC", "rows": protocol["origins"], "rollout_sha256": _sha256(ref_npz)}, "gate1_authorized": False, "formal_candidate": False, "test_set_accessed": False, "evaluation_year_accessed": False, "status": "PASS"}
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
