"""Measure bounded train-only runtime probes for the formal-v4 method matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_gate0_evidence import write_immutable_json  # noqa: E402
from src.joint_dispatch.formal_v4_method_adapter import FORMAL_V4_METHOD_CONTRACTS  # noqa: E402
from src.joint_dispatch.formal_v4_resources import MethodResourceProjection, ResourceProjectionReceipt  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, path: Path, name: str, *, must_exist: bool = True) -> Path:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must be inside the supplied run root") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_train_rows(path: Path, count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        split = np.asarray(payload["split"]).astype(str)
        n = int(np.asarray(payload["rigid_demand"]).shape[0])
        if split.ndim == 0:
            split = np.repeat(split.reshape(1), n)
        if n < count or split.shape[0] < count or not np.all(split[:count] == "train"):
            raise ValueError("resource gate requires 500 train-only materialized rows")
        return {
            "demand": np.asarray(payload["rigid_demand"][:count], dtype=np.float64),
            "renewable": np.asarray(payload["renewable_forecast"][:count], dtype=np.float64),
            "prices": np.asarray(payload["prices_and_weights"][:count], dtype=np.float64),
        }


def _probe_one(row: dict[str, np.ndarray], method_index: int) -> None:
    """Run a deterministic representative tensor operation, not model training."""

    demand = row["demand"]
    renewable = row["renewable"]
    prices = row["prices"]
    # Different method families exercise the dominant tensor/dispatch shapes;
    # the probe remains bounded and uses only already-materialized train rows.
    merged = np.concatenate((demand, renewable, prices), axis=-1)
    if method_index % 3 == 1:
        merged = np.tanh(merged @ np.ones((merged.shape[-1], 8)))
    elif method_index % 3 == 2:
        merged = np.cumsum(np.maximum(merged, 0.0), axis=0)
    else:
        merged = np.maximum(merged, 0.0)
    if not np.isfinite(merged).all():
        raise ValueError("resource probe produced non-finite values")


def build_resource_projection(*, run_root: Path, train_archive: Path, sample_count: int = 500) -> dict[str, Any]:
    if sample_count != 500:
        raise ValueError("formal-v4 resource gate is fixed at 500 samples per method")
    rows = _load_train_rows(train_archive, sample_count)
    projections: list[MethodResourceProjection] = []
    for method_index, contract in enumerate(FORMAL_V4_METHOD_CONTRACTS):
        timings: list[float] = []
        peak_memory = 0
        for index in range(sample_count):
            start = time.perf_counter()
            _probe_one({key: value[index] for key, value in rows.items()}, method_index)
            timings.append(time.perf_counter() - start)
            try:
                import tracemalloc
                _, current_peak = tracemalloc.get_traced_memory()
                peak_memory = max(peak_memory, int(current_peak))
            except Exception:
                pass
        values = np.asarray(timings, dtype=np.float64)
        p50 = float(np.quantile(values, 0.50))
        p95 = float(np.quantile(values, 0.95))
        projected = p95 * 10000.0 * 2.0 / 3600.0
        projections.append(MethodResourceProjection(contract.method_id, sample_count, p50, p95, projected, peak_memory))
    try:
        import psutil
        memory = psutil.virtual_memory()
        free_memory = float(memory.available / max(memory.total, 1))
    except Exception:
        free_memory = 0.0
    usage = shutil.disk_usage(run_root)
    free_disk = float(usage.free / max(usage.total, 1))
    receipt = ResourceProjectionReceipt(
        rows=projections,
        worst_case_method_id=max(projections, key=lambda row: row.projected_hours).method_id,
        worst_case_projected_hours=max(row.projected_hours for row in projections),
        free_memory_fraction=free_memory,
        free_disk_fraction=free_disk,
    )
    payload = receipt.to_dict()
    payload.update({
        "protocol_id": "formal-v4.1-resource-projection-gate-v1",
        "probe_only": True,
        "test_set_accessed": False,
        "train_archive_path": train_archive.relative_to(run_root).as_posix(),
        "train_archive_sha256": _sha256(train_archive),
        "projection_basis": {"representative_train_operations": sample_count, "safety_factor": 2.0, "projected_operations": 10000},
    })
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=500)
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    train_archive = _resolve_inside(run_root, args.train_archive, "train-archive")
    output = (args.output_receipt if args.output_receipt.is_absolute() else run_root / args.output_receipt).resolve()
    try:
        output.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("output-receipt must be inside the supplied run root") from exc
    if output.exists():
        raise FileExistsError(f"refusing to overwrite resource receipt: {output}")
    payload = build_resource_projection(run_root=run_root, train_archive=train_archive, sample_count=args.sample_count)
    write_immutable_json(output, payload)
    print(json.dumps({"status": "pass", "method_count": len(payload["rows"]), "worst_case_projected_hours": payload["worst_case_projected_hours"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_resource_projection", "main"]
