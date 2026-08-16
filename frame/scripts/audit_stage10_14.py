"""Final read-only audit for the Stage 10.14 scheduling evidence package.

The audit does not retrain or re-run dispatch.  It verifies the corrected
formal-run manifest, the analysis manifest hashes, required tables/figures,
and the separation of the real-replay and simulated-dispatch tracks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REQUIRED_TABLES = {
    "table_r_main_model_comparison.csv",
    "table_s_core_model_comparison.csv",
    "table_r_seasonal_comparison.csv",
    "table_s_scenario_effects.csv",
    "table_s_equipment_dispatch.csv",
    "table_s_solver_reliability.csv",
    "table_s_feasibility_audit.csv",
    "table_r_pairwise_significance.csv",
    "table_s_core_pairwise_significance.csv",
    "claim_evidence_map.csv",
}
REQUIRED_FIGURE_PREFIXES = (
    "fig_r1_real_replay_comparison",
    "fig_s1_core_cost_regret_carbon",
    "fig_s2_scenario_effects",
    "fig_s3_typical_day_dispatch",
    "fig_s4_solver_and_feasibility",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(formal_dir: Path, analysis_dir: Path, output_path: Path) -> dict[str, object]:
    formal_dir = formal_dir.resolve()
    analysis_dir = analysis_dir.resolve()
    checks: dict[str, object] = {}
    errors: list[str] = []

    root_manifest_path = formal_dir / "formal_scheduling_manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8")) if root_manifest_path.exists() else {}
    checks["formal_root_manifest"] = root_manifest.get("schema_version") == "scheduling-formal-root-v3" and root_manifest.get("status") == "complete"
    if not checks["formal_root_manifest"]:
        errors.append("formal root manifest is missing or not scheduling-formal-root-v3 complete")
    checks["formal_directory_name"] = formal_dir.name == "formal_corrected"
    if not checks["formal_directory_name"]:
        errors.append("analysis must use a directory named formal_corrected to prevent legacy-output confusion")

    analysis_manifest_path = analysis_dir / "scheduling_analysis_manifest.json"
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8")) if analysis_manifest_path.exists() else {}
    checks["analysis_manifest"] = analysis_manifest.get("schema_version") == "scheduling-analysis-v2" and analysis_manifest.get("status") == "complete"
    checks["tracks_kept_separate"] = analysis_manifest.get("tracks_kept_separate") is True
    checks["no_test_tuning"] = analysis_manifest.get("test_set_used_for_tuning") is False
    if not checks["analysis_manifest"]:
        errors.append("analysis manifest is missing or incomplete")
    if not checks["tracks_kept_separate"]:
        errors.append("real-replay and simulated-dispatch tracks are not marked separate")
    if not checks["no_test_tuning"]:
        errors.append("analysis manifest does not certify that test data were not used for tuning")

    table_dir = analysis_dir / "tables"
    missing_tables = sorted(name for name in REQUIRED_TABLES if not (table_dir / name).exists())
    checks["required_tables"] = not missing_tables
    if missing_tables:
        errors.append(f"missing tables: {', '.join(missing_tables)}")
    figure_dir = analysis_dir / "figures"
    missing_figures = [prefix for prefix in REQUIRED_FIGURE_PREFIXES if not any(figure_dir.glob(prefix + ".*"))]
    checks["required_figures"] = not missing_figures
    if missing_figures:
        errors.append(f"missing figure families: {', '.join(missing_figures)}")

    output_hashes = analysis_manifest.get("output_sha256", {})
    hash_errors = []
    for relative, expected in output_hashes.items():
        path = analysis_dir / relative
        if not path.exists() or _sha256(path) != expected:
            hash_errors.append(relative)
    checks["analysis_hashes"] = not hash_errors
    if hash_errors:
        errors.append(f"analysis output hash mismatch: {', '.join(hash_errors)}")

    report = {
        "schema_version": "stage10.14-final-audit-v1",
        "status": "pass" if not errors else "fail",
        "formal_dir": str(formal_dir),
        "analysis_dir": str(analysis_dir),
        "checks": checks,
        "errors": errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", required=True, type=Path)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.formal_dir, args.analysis_dir, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
