"""Build a read-only inventory before the complete formal experiment.

This task does not train, evaluate, modify, or delete any model artifact.  It
only classifies the current implementation and evidence lineage so a later
task cannot confuse a contract or a smoke probe with a completed result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


FRAME_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_METHODS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "State-Conditioned-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
    "Seasonal-Naive-PTO",
    "Perfect-Information-MPC",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> Iterable[Path]:
    if not path.exists():
        return ()
    return (item for item in path.rglob("*") if item.is_file())


def _contains(path: Path, needle: str) -> bool:
    if not path.is_file():
        return False
    try:
        return needle in path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _existing_files(frame_root: Path, relative_paths: Iterable[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for relative in relative_paths:
        path = frame_root / relative
        record: dict[str, str] = {"path": relative, "status": "missing"}
        if path.is_file():
            record["status"] = "present"
            record["sha256"] = _sha256(path)
        records.append(record)
    return records


def _method_path_tokens(method_id: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in {
            method_id,
            method_id.replace(" ", "_"),
            method_id.replace(" ", ""),
            method_id.replace("-", "_"),
        }
        if token
    )


CHECKPOINT_PATTERNS = {
    "RSC-PF": (("J_joint.pt",), ("STAGE_J_JOINT.pt",)),
    "Decoupled-RSC-PF": (("J_decoupled.pt",), ("STAGE_J_DECOUPLED.pt",)),
    "Direct-Policy": (("Direct_Policy",),),
    "Scheme2R-PTO": (("Scheme2R_PTO",),),
    "State-Conditioned-PTO": (("State_Conditioned_PTO",),),
    "Official iTransformer-PTO": (("iTransformer_PTO",),),
    "Differentiable-LP": (("Differentiable_LP",),),
    "Seasonal-Naive-PTO": (("Seasonal_Naive_PTO",),),
    "Perfect-Information-MPC": (("Perfect_Information_MPC",),),
}


def _checkpoint_evidence(reports_root: Path, method_id: str) -> list[str]:
    patterns = CHECKPOINT_PATTERNS[method_id]
    found: list[str] = []
    for path in _iter_files(reports_root):
        if path.suffix.lower() not in {".pt", ".pth", ".ckpt"}:
            continue
        text = str(path)
        if any(all(token in text for token in pattern) for pattern in patterns) and not any(year in text for year in ("2020", "2021")):
            found.append(str(path.relative_to(reports_root)))
    return sorted(found)


def _formal_result_present(reports_root: Path, method_id: str) -> bool:
    """Return true only for an independently audited complete result row."""

    for audit_path in reports_root.rglob("GATE2_AUDIT.json") if reports_root.exists() else ():
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") not in {"PASS", "pass", "passed"}:
            continue
        rows = payload.get("rows", ())
        if isinstance(rows, Mapping):
            rows = rows.values()
        if any(str(row.get("method_id")) == method_id for row in rows if isinstance(row, Mapping)):
            return True
    return False


def _training_function_present(frame_root: Path, method_id: str) -> bool:
    if method_id in {"Seasonal-Naive-PTO", "Perfect-Information-MPC"}:
        return False
    sources = (
        frame_root / "src" / "joint_dispatch" / "formal_v4_2_gate2_training.py",
        frame_root / "src" / "joint_dispatch" / "formal_v4_2_methods.py",
        frame_root / "src" / "joint_dispatch" / "formal_v4_method_adapter.py",
        frame_root / "src" / "joint_dispatch" / "formal_v4_baselines.py",
    )
    needles = {
        "RSC-PF": ("train_joint", "RSC-PF"),
        "Decoupled-RSC-PF": ("train_joint", "Decoupled-RSC-PF"),
        "Direct-Policy": ("train_direct_policy", "Direct-Policy"),
        "Scheme2R-PTO": ("train_scheme2r", "Scheme2R-PTO"),
        "State-Conditioned-PTO": ("State-Conditioned-PTO",),
        "Official iTransformer-PTO": ("train_official_itransformer_pto",),
        "Differentiable-LP": ("train_differentiable_lp",),
        "Seasonal-Naive-PTO": ("Seasonal-Naive-PTO",),
        "Perfect-Information-MPC": ("Perfect-Information-MPC",),
    }[method_id]
    return any(_contains(path, needle) for path in sources for needle in needles)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _matched_v2_inventory(frame_root: Path) -> dict[str, Any]:
    reports_root = frame_root / "reports" / "rsc_pf_matched_closed_loop_2019" / "v2"
    manifest = _load_json(reports_root / "MATCHED_CLOSED_LOOP_MANIFEST.json")
    audit = _load_json(reports_root / "MATCHED_CLOSED_LOOP_AUDIT.json")
    rows = manifest.get("rows", [])
    rsc_seeds = sorted(
        int(row["seed"])
        for row in rows
        if isinstance(row, Mapping) and row.get("method_id") == "RSC-PF" and "seed" in row
    )
    return {
        "path": str(reports_root),
        "manifest_present": bool(manifest),
        "audit_present": bool(audit),
        "row_count": len(rows) if isinstance(rows, list) else 0,
        "rsc_pf_seeds": rsc_seeds,
        "origins": manifest.get("origins"),
        "audit_status": audit.get("status"),
        "formal_candidate": manifest.get("formal_candidate"),
        "gate1_authorized": manifest.get("gate1_authorized"),
        "evaluation_year_accessed": manifest.get("evaluation_year_accessed"),
        "test_set_accessed": manifest.get("test_set_accessed"),
    }


def build_inventory(frame_root: Path) -> dict[str, Any]:
    frame_root = Path(frame_root).resolve()
    reports_root = frame_root / "reports"
    diffopt_files = _existing_files(
        frame_root,
        (
            "src/joint_dispatch/formal_v4_diffopt.py",
            "src/joint_dispatch/formal_v4_2_gate2_training.py",
            "scripts/run_rsc_pf_formal_v4_diffopt_gate.py",
            "requirements/formal_v4_diffopt.lock",
        ),
    )
    itransformer_root = frame_root / "third_party" / "iTransformer_source"
    methods: dict[str, dict[str, Any]] = {}
    for method_id in PRIMARY_METHODS:
        training_present = _training_function_present(frame_root, method_id)
        checkpoint_paths = _checkpoint_evidence(reports_root, method_id)
        checkpoint_present = bool(checkpoint_paths)
        formal_result = _formal_result_present(reports_root, method_id)
        methods[method_id] = {
            "training_function_present": training_present,
            "checkpoint_available": checkpoint_present,
            "checkpoint_evidence": checkpoint_paths[:20],
            "formal_result_present": formal_result,
            "native_gradient_gate_present": method_id == "Differentiable-LP" and all(
                record["status"] == "present" for record in diffopt_files[:3]
            ),
            "status": (
                "formal_result"
                if formal_result
                else "checkpoint_available"
                if checkpoint_present
                else "missing_formal_result"
                if training_present
                else "implementation_missing"
            ),
        }
    return {
        "schema": "rsc-pf-complete-formal-inventory-v1",
        "read_only": True,
        "frame_root": str(frame_root),
        "primary_methods": list(PRIMARY_METHODS),
        "methods": methods,
        "itransformer_source": {
            "path": str(itransformer_root),
            "exists": itransformer_root.is_dir(),
            "file_count": sum(1 for _ in _iter_files(itransformer_root)),
        },
        "difflp_evidence_files": diffopt_files,
        "matched_v2": _matched_v2_inventory(frame_root),
        "future_access": {
            "evaluation_year_accessed": False,
            "excluded_year_accessed": False,
            "test_set_accessed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-root", type=Path, default=FRAME_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_inventory(args.frame_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "schema": payload["schema"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
