"""Compute paired uncertainty intervals for the common RSC-PF validation comparison.

The common evaluator stores one chronological value per validation window and
seed.  This script treats the five seeds as the independent model replicates
and preserves serial dependence within the 8,780 paired windows using the
repository's moving-block bootstrap.  It reports effect estimates and 95% CIs;
it deliberately does not manufacture p-values from a validation-only result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.evaluation import paired_moving_block_bootstrap  # noqa: E402


SEEDS = (2026, 2027, 2028, 2029, 2030)
RSC_METHOD = "RSC-PF"
BASELINES = ("iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy")
SAFE_METHOD = {
    RSC_METHOD: "RSC_PF",
    "iTransformer-PTO": "iTransformer_PTO",
    "DecisionFocused-Online": "DecisionFocused_Online",
    "DigitalTwins-Policy": "DigitalTwins_Policy",
}
DISPATCH_METRICS = {
    "operating_cost": "lower_is_better",
    "physical_carbon": "lower_is_better",
    "penalized_objective": "lower_is_better",
    "regret_vs_oracle": "lower_is_better",
    "shortage": "lower_is_better",
    "feasible": "higher_is_better",
}
EXPECTED_WINDOWS = 8780


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _test_like(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    return any(part in {"test", "test_set", "sealed_test"} for part in parts) or any(
        token in normalized for token in ("2021_test", "test-set", "sealed-test")
    )


def _finite_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != EXPECTED_WINDOWS:
        raise ValueError(f"{name} must have shape ({EXPECTED_WINDOWS},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _metrics_path(report_root: Path, method: str, seed: int) -> Path:
    if method not in SAFE_METHOD or int(seed) not in SEEDS:
        raise ValueError(f"unknown method/seed: {method}, {seed}")
    path = report_root / "validation" / SAFE_METHOD[method] / f"seed_{int(seed)}" / "evaluation" / "validation_metrics.npz"
    if _test_like(path):
        raise ValueError(f"statistics cannot read a sealed test-set path: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _receipt_path(report_root: Path, method: str, seed: int) -> Path:
    path = _metrics_path(report_root, method, seed).with_name("evaluation_receipt.json")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_series(report_root: Path, method: str, seed: int) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, Path]:
    metrics_path = _metrics_path(report_root, method, seed)
    receipt_path = _receipt_path(report_root, method, seed)
    receipt = _read_json(receipt_path)
    if receipt.get("method_id") != method:
        raise ValueError(f"receipt method mismatch: {receipt_path}")
    if int(receipt.get("seed", -1)) != int(seed) or receipt.get("stage") != "validation":
        raise ValueError(f"receipt seed/stage mismatch: {receipt_path}")
    if receipt.get("test_set_accessed") is not False:
        raise ValueError(f"receipt is not validation-only: {receipt_path}")
    if int(receipt.get("metrics", {}).get("windows", -1)) != EXPECTED_WINDOWS:
        raise ValueError(f"receipt window count changed: {receipt_path}")
    with np.load(metrics_path, allow_pickle=False) as archive:
        series = {name: _finite_array(archive[name], f"{method}/{seed}/{name}") for name in DISPATCH_METRICS}
    return series, receipt, metrics_path, receipt_path


def _interpret(observed: float, lower: float, upper: float, direction: str) -> str:
    if lower > 0.0 or upper < 0.0:
        sign = "higher" if observed > 0.0 else "lower"
        if direction == "higher_is_better":
            quality = "better" if observed > 0.0 else "worse"
        else:
            quality = "better" if observed < 0.0 else "worse"
        return f"RSC-PF {quality}; CI excludes zero ({sign} difference)"
    return "difference inconclusive; 95% CI includes zero"


def analyze_common_statistics(
    report_root: Path | str,
    *,
    output_root: Path | str | None = None,
    block_hours: int = 168,
    replicates: int = 2000,
    alpha: float = 0.05,
    seed: int = 2026,
) -> dict[str, Any]:
    """Return and persist paired dispatch-metric uncertainty results."""

    root = Path(report_root)
    out = Path(output_root) if output_root is not None else root
    if _test_like(root) or _test_like(out):
        raise ValueError("statistics report roots may not point to a sealed test-set path")
    if block_hours <= 0 or replicates <= 0 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid bootstrap settings")
    loaded: dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, Any], Path, Path]] = {}
    for method in (RSC_METHOD, *BASELINES):
        for model_seed in SEEDS:
            loaded[(method, model_seed)] = _load_series(root, method, model_seed)

    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for (method, model_seed), (_, _, metrics_path, receipt_path) in loaded.items():
        source_hashes[str(metrics_path)] = _sha256(metrics_path)
        source_hashes[str(receipt_path)] = _sha256(receipt_path)
    for baseline in BASELINES:
        for metric, direction in DISPATCH_METRICS.items():
            rsc_matrix = np.asarray([loaded[(RSC_METHOD, model_seed)][0][metric] for model_seed in SEEDS], dtype=np.float64)
            baseline_matrix = np.asarray([loaded[(baseline, model_seed)][0][metric] for model_seed in SEEDS], dtype=np.float64)
            result = paired_moving_block_bootstrap(
                rsc_matrix,
                baseline_matrix,
                block_hours=block_hours,
                replicates=replicates,
                alpha=alpha,
                seed=seed,
            )
            rows.append({
                "comparison_id": f"{RSC_METHOD}_minus_{baseline}_{metric}",
                "method_a": RSC_METHOD,
                "method_b": baseline,
                "metric": metric,
                "direction": direction,
                "observed_difference_a_minus_b": result["observed_difference"],
                "ci_lower": result["ci_lower"],
                "ci_upper": result["ci_upper"],
                "alpha": result["alpha"],
                "replicates": result["replicates"],
                "block_hours": result["block_hours"],
                "dependence_model": result["dependence_model"],
                "n_seeds": len(SEEDS),
                "windows_per_seed": EXPECTED_WINDOWS,
                "independent_unit": "random seed/model replicate",
                "paired_unit": "same chronological validation window across methods",
                "interpretation": _interpret(float(result["observed_difference"]), float(result["ci_lower"]), float(result["ci_upper"]), direction),
                "test_set_accessed": False,
            })
    fieldnames = list(rows[0].keys())
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "common_validation_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": "rsc-pf-common-validation-statistics-v1",
        "stage": "validation",
        "methods": [RSC_METHOD, *BASELINES],
        "seeds": list(SEEDS),
        "windows_per_seed": EXPECTED_WINDOWS,
        "metric_scope": "dispatch metrics saved by the common open-loop four-hour validation evaluator",
        "bootstrap": {
            "method": "paired_contiguous_moving_block",
            "block_hours": int(block_hours),
            "replicates": int(replicates),
            "alpha": float(alpha),
            "seed": int(seed),
        },
        "independent_unit": "random seed/model replicate (n=5)",
        "paired_unit": "same chronological validation window across methods; serial dependence retained within blocks",
        "rows": rows,
        "source_hashes": source_hashes,
        "test_set_accessed": False,
        "limitations": [
            "This is validation-only evidence; no sealed test performance is claimed.",
            "Intervals cover dispatch metrics whose per-window arrays were saved by the common evaluator; forecast-error intervals are not included in this receipt.",
            "No p-values are reported because the primary purpose is effect-size and uncertainty reporting under paired temporal dependence.",
        ],
    }
    json_path = out / "common_validation_statistics.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["csv"] = str(csv_path)
    report["json"] = str(json_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-root",
        type=Path,
        default=FRAME_ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--block-hours", type=int, default=168)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = analyze_common_statistics(
        args.report_root,
        output_root=args.output_root,
        block_hours=args.block_hours,
        replicates=args.replicates,
        alpha=args.alpha,
        seed=args.seed,
    )
    print(json.dumps({"csv": report["csv"], "json": report["json"], "rows": len(report["rows"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
