"""Run the auditable Stage 10.14 source-data, table and figure pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduling.analysis_io import load_formal_runs  # noqa: E402
from src.scheduling.analysis_tables import generate_tables  # noqa: E402
from src.scheduling.figures import generate_figures  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_analysis_manifest(formal_dir: Path, contract_path: Path, output_dir: Path, outputs: list[Path], run_count: int, oracle_count: int) -> dict[str, object]:
    return {
        "schema_version": "scheduling-analysis-v2",
        "status": "complete",
        "formal_dir": str(formal_dir),
        "analysis_contract": str(contract_path),
        "run_count": int(run_count),
        "oracle_count": int(oracle_count),
        "test_set_used_for_tuning": False,
        "tracks_kept_separate": True,
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 2026,
        "multiple_comparison": "benjamini_hochberg",
        "python": platform.python_version(),
        "output_sha256": {str(path.relative_to(output_dir)): _sha256(path) for path in outputs if path.is_file()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", required=True, type=Path)
    parser.add_argument("--analysis-contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    formal_dir = args.formal_dir.resolve()
    contract_path = args.analysis_contract.resolve()
    output_dir = args.output_dir.resolve()
    if formal_dir.name != "formal_corrected":
        raise ValueError("analysis refuses legacy scheduling output; use formal_corrected")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if args.dry_run:
        print(json.dumps({"stage": "10.14", "status": "dry_run", "formal_dir": str(formal_dir), "output_dir": str(output_dir), "expected_model_runs": contract["expected_runs"]["total_model_runs"], "expected_oracle_runs": contract["expected_runs"]["oracle_scenario_runs"]}, ensure_ascii=False, indent=2))
        return 0
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"analysis output is non-empty; use a new directory or --resume: {output_dir}")
    data = load_formal_runs(formal_dir, contract)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_data"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    generate_tables(data, table_dir)
    generate_figures(source_dir, figure_dir)
    files = [path for path in output_dir.rglob("*") if path.is_file() and path.name != "scheduling_analysis_manifest.json"]
    manifest = build_analysis_manifest(formal_dir, contract_path, output_dir, files, len(data.runs), len(data.oracles))
    (output_dir / "scheduling_analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "10.14", "status": "complete", "run_count": len(data.runs), "oracle_count": len(data.oracles), "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
