"""Produce one fresh, training-free formal-v4.1 Gate 0 evidence root.

The orchestrator is deliberately fail-closed.  It never imports or launches a
Gate 1 runner; a failed producer leaves its partial root for audit and no
authorization marker is written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent

STAGE_ORDER: tuple[str, ...] = (
    "benchmark_capacity_data_materialization",
    "trajectory_access_archive_receipts",
    "objective_teacher_receipts",
    "curriculum_gradient_receipts",
    "method_adapter_receipt",
    "itransformer_source_receipt",
    "diffopt_gate",
    "resource_projection",
    "receipt_inventory",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fresh_root(reports_root: Path) -> Path:
    reports_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("formal_v4_1_gate0_%Y%m%d_%H%M%S_%f")
    candidate = reports_root / stamp
    if candidate.exists():
        raise FileExistsError(f"fresh Gate 0 run root already exists: {candidate}")
    return candidate


def _run(command: Sequence[str], *, stage: str, root: Path) -> None:
    # The capacity producer requires the run-root path to be nonexistent on
    # entry, so the first-stage log must live beside (not inside) that root.
    log_path = root.parent / f"{root.name}.orchestrator.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{stage}] {' '.join(command)}\n")
        completed = subprocess.run(command, cwd=str(REPO_ROOT), text=True, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"stage {stage} failed with exit code {completed.returncode}")


def _inventory(root: Path) -> dict[str, Any]:
    required = (
        "gate0/benchmark/STANDARD_IES_BENCHMARK.yaml",
        "gate0/benchmark/FORMAL_V4_BENCHMARK_RECEIPT.json",
        "gate0/CAPACITY_FREEZE.json",
        "gate0/CAPACITY_AUDIT_RESULT.json",
        "data/base_train.npz",
        "data/base_selection.npz",
        "data/train.npz",
        "data/selection.npz",
        "data/trajectory.npz",
        "NORMALIZATION_RECEIPT.json",
        "ARTIFACT_MANIFEST.json",
        "C_REF_RECEIPT.json",
        "protocol/DATA_ACCESS_RECEIPT.json",
        "protocol/ARCHIVE_ACCESS_RECEIPT.json",
        "protocol/TRAJECTORY_RECEIPT.json",
        "protocol/TEACHER_ALIGNMENT_RECEIPT.json",
        "protocol/CURRICULUM_RECEIPT.json",
        "protocol/GRADIENT_RECEIPT.json",
        "protocol/METHOD_ADAPTER_RECEIPT.json",
        "protocol/ITRANSFORMER_SOURCE_RECEIPT.json",
        "protocol/DIFFERENTIABLE_LP_GATE.json",
        "protocol/RESOURCE_PROJECTION.json",
        "audit/REGRESSION_RECEIPT.json",
    )
    missing = [path for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("Gate 0 receipt inventory is incomplete: " + ", ".join(missing))
    payload = {
        "schema_version": "formal-v4.1-receipt-inventory-v1",
        "protocol_id": "formal-v4.1-gate0-evidence-orchestration-v1",
        "stage_order": list(STAGE_ORDER),
        "files": {path: _sha256(root / path) for path in required},
        "test_set_accessed": False,
    }
    destination = root / "audit" / "RECEIPT_INVENTORY.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def run_evidence_orchestrator(*, reports_root: Path, diffopt_python: Path, main_python: Path, data_dir: Path, contract: Path | None = None, source_root: Path | None = None, stage_runner: Callable[[str, Path], None] | None = None) -> dict[str, Any]:
    """Run the fixed producer sequence and return the fresh root summary."""

    root = _fresh_root(reports_root.resolve())
    contract_path = (contract or FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_1.json").resolve()
    source_path = (source_root or FRAME_ROOT / "third_party" / "iTransformer_source").resolve()
    python = str(main_python.resolve())
    diff_python = str(diffopt_python.resolve())

    def invoke(stage: str, command: Sequence[str]) -> None:
        if stage_runner is not None:
            stage_runner(stage, root)
        else:
            _run(command, stage=stage, root=root)

    try:
        invoke("benchmark_capacity_data_materialization", [python, str(FRAME_ROOT / "scripts" / "run_rsc_pf_formal_v4_capacity_audit.py"), "--contract", str(contract_path), "--run-root", str(root), "--data-dir", str(data_dir.resolve())])
        benchmark = root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
        capacity = root / "gate0" / "CAPACITY_FREEZE.json"
        invoke("trajectory_access_archive_receipts", [python, str(FRAME_ROOT / "scripts" / "build_rsc_pf_formal_v4_data.py"), "--contract", str(contract_path), "--mode", "materialize-state", "--splits", "train", "selection", "--run-root", str(root), "--benchmark-path", str(benchmark), "--capacity-receipt", str(capacity)])
        invoke("objective_teacher_receipts", [python, str(FRAME_ROOT / "scripts" / "build_rsc_pf_formal_v4_objective_evidence.py"), "--run-root", str(root), "--benchmark-path", str(benchmark), "--capacity-receipt", str(capacity), "--train-archive", str(root / "data" / "train.npz"), "--selection-archive", str(root / "data" / "selection.npz")])
        invoke("curriculum_gradient_receipts", [python, str(FRAME_ROOT / "scripts" / "build_rsc_pf_formal_v4_training_evidence.py"), "--run-root", str(root), "--benchmark-path", str(benchmark), "--train-archive", str(root / "data" / "train.npz"), "--c-ref", str(root / "C_REF_RECEIPT.json")])
        invoke("method_adapter_receipt", [python, str(FRAME_ROOT / "scripts" / "build_rsc_pf_formal_v4_adapter_evidence.py"), "--run-root", str(root), "--benchmark-path", str(benchmark), "--train-archive", str(root / "data" / "train.npz")])
        invoke("itransformer_source_receipt", [python, str(FRAME_ROOT / "scripts" / "prepare_rsc_pf_official_itransformer_v4.py"), "--source-root", str(source_path), "--output-receipt", str(root / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json")])
        invoke("diffopt_gate", [diff_python, str(FRAME_ROOT / "scripts" / "run_rsc_pf_formal_v4_diffopt_gate.py"), "--run-root", str(root), "--benchmark-path", str(benchmark), "--train-archive", str(root / "data" / "train.npz"), "--output-receipt", str(root / "protocol" / "DIFFERENTIABLE_LP_GATE.json"), "--window-count", "100"])
        invoke("resource_projection", [diff_python, str(FRAME_ROOT / "scripts" / "run_rsc_pf_formal_v4_resource_gate.py"), "--run-root", str(root), "--train-archive", str(root / "data" / "train.npz"), "--output-receipt", str(root / "protocol" / "RESOURCE_PROJECTION.json"), "--sample-count", "500"])
        if stage_runner is None:
            # Full regression is intentionally the final producer and writes a
            # receipt only after pytest returns zero.  It does not train or
            # open evaluation data.
            regression_log = root / "audit" / "REGRESSION_PYTEST.txt"
            regression_log.parent.mkdir(parents=True, exist_ok=True)
            with regression_log.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run([python, "-m", "pytest", str(FRAME_ROOT / "tests"), "-q", "-p", "no:cacheprovider"], cwd=str(REPO_ROOT), text=True, stdout=log_handle, stderr=subprocess.STDOUT)
            if completed.returncode != 0:
                raise RuntimeError(f"stage receipt_inventory regression failed with exit code {completed.returncode}")
            (root / "audit" / "REGRESSION_RECEIPT.json").write_text(json.dumps({"schema_version": "formal-v4.1-regression-receipt-v1", "status": "pass", "command": "pytest frame/tests -q -p no:cacheprovider", "test_set_accessed": False, "log_sha256": _sha256(regression_log)}, ensure_ascii=False, indent=2), encoding="utf-8")
        if stage_runner is not None:
            stage_runner("receipt_inventory", root)
        inventory = _inventory(root)
        return {"status": "pass", "run_root": str(root), "stage_order": list(STAGE_ORDER), "inventory_sha256": _sha256(root / "audit" / "RECEIPT_INVENTORY.json"), "test_set_accessed": False}
    except Exception as exc:
        root.mkdir(parents=True, exist_ok=True)
        (root / "ORCHESTRATOR_FAILURE.json").write_text(json.dumps({"status": "fail", "error_type": type(exc).__name__, "reason": str(exc), "stage_order": list(STAGE_ORDER), "test_set_accessed": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diffopt-python", type=Path, required=True)
    parser.add_argument("--main-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--reports-root", type=Path, default=FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_1")
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\Paper\Kitakyushu dataset"))
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_1.json")
    parser.add_argument("--source-root", type=Path, default=FRAME_ROOT / "third_party" / "iTransformer_source")
    args = parser.parse_args(argv)
    summary = run_evidence_orchestrator(reports_root=args.reports_root, diffopt_python=args.diffopt_python, main_python=args.main_python, data_dir=args.data_dir, contract=args.contract, source_root=args.source_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["STAGE_ORDER", "main", "run_evidence_orchestrator"]
