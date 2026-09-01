"""Evaluate RSC-PF on the same open-loop validation-window protocol as baselines.

This module is intentionally separate from the existing closed-loop v2 runner.
The frozen external-baseline receipts use a four-hour window evaluation, so the
RSC-PF checkpoint is evaluated on the identical windows and metric definitions
before any method-level comparison table is assembled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from scripts.evaluate_rsc_pf_external_baselines import _dispatch_metrics  # noqa: E402
from src.joint_dispatch.data import load_joint_split  # noqa: E402
from src.joint_dispatch.evaluation import evaluate_forecast  # noqa: E402
from src.joint_dispatch.external_baseline_training import SEEDS, _forbidden_test_path  # noqa: E402


METHOD_ID = "RSC-PF"
SOURCE_METHOD = "Warm-Start-Joint"
COMMON_METHODS = (METHOD_ID, "iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy")
METRIC_KEYS = (
    "forecast_mae_mean",
    "forecast_rmse_mean",
    "forecast_wape_mean",
    "operating_cost_mean",
    "physical_carbon_mean",
    "penalized_objective_mean",
    "regret_vs_oracle_mean",
    "shortage_mean",
    "feasibility_rate",
)


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


def _test_like(path: str | Path) -> bool:
    """Reject explicit test/sealed-test path components, including leaf paths."""

    normalized = str(path).replace("\\", "/").lower()
    parts = [part for part in normalized.split("/") if part]
    return any(part in {"test", "test_set", "sealed_test"} for part in parts) or any(
        token in normalized for token in ("2021_test", "test-set", "sealed-test")
    )


def _resolve(value: str | Path, root: Path = FRAME_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_contract(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    return _read_json(Path(config))


def _paths(config: Mapping[str, Any], *, data_root: Path | None, output_root: Path | None) -> dict[str, Path]:
    paths = dict(config.get("paths", {}))
    resolved_data = data_root if data_root is not None else _resolve(paths.get("data_root", "reports/joint_forecast_dispatch_v1/data"))
    resolved_output = output_root if output_root is not None else FRAME_ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation"
    benchmark = _resolve(paths.get("benchmark_path", "D:/Paper/standard_ies_benchmark_v1.yaml"))
    checkpoint_root = _resolve(paths.get("output_root", "reports/joint_forecast_dispatch_v2")) / "validation" / SOURCE_METHOD
    for name, path in {
        "data_root": resolved_data,
        "output_root": resolved_output,
        "benchmark_path": benchmark,
        "checkpoint_root": checkpoint_root,
    }.items():
        if _test_like(path) or _forbidden_test_path(path):
            raise ValueError(f"{name} points to a sealed test-set path: {path}")
    return {
        "data_root": Path(resolved_data),
        "output_root": Path(resolved_output),
        "benchmark_path": Path(benchmark),
        "checkpoint_root": Path(checkpoint_root),
    }


def _model_from_normalization(normalization: Any, benchmark: Mapping[str, Any]):
    import torch
    from src.joint_dispatch.model import JointForecastDispatchModel

    return JointForecastDispatchModel(
        task_mean=torch.from_numpy(normalization.load_mean),
        task_scale=torch.from_numpy(normalization.load_scale),
        physical_feature_mean=torch.from_numpy(np.concatenate((normalization.load_mean, normalization.scheduler_mean))),
        physical_feature_scale=torch.from_numpy(np.concatenate((normalization.load_scale, normalization.scheduler_scale))),
        decoder_parameters=benchmark["values"],
        dropout=0.0,
    )


def _load_checkpoint(model: Any, checkpoint: Path, receipt_path: Path, *, seed: int) -> dict[str, Any]:
    import torch

    if _test_like(checkpoint) or not checkpoint.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"RSC-PF checkpoint/receipt is missing: {checkpoint}")
    receipt = _read_json(receipt_path)
    if receipt.get("test_set_accessed") is not False or receipt.get("method") != SOURCE_METHOD:
        raise ValueError(f"RSC-PF checkpoint receipt is not validation-only: {receipt_path}")
    if int(receipt.get("seed", -1)) != int(seed) or receipt.get("complete") is not True:
        raise ValueError(f"RSC-PF checkpoint receipt has wrong seed/status: {receipt_path}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, Mapping) else None
    if state is None and isinstance(payload, Mapping):
        state = payload.get("model_state_dict")
    if state is None:
        raise ValueError(f"checkpoint has no model state dict: {checkpoint}")
    model.load_state_dict(state, strict=True)
    return {
        "training_receipt": str(receipt_path),
        "training_receipt_sha256": _sha256(receipt_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint": str(checkpoint),
    }


def evaluate_rsc_pf_validation(
    seed: int,
    config: str | Path | Mapping[str, Any],
    *,
    data_root: Path | None = None,
    output_root: Path | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Evaluate one Warm-Start-Joint RSC-PF checkpoint on validation windows."""

    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    contract = _load_contract(config)
    paths = _paths(contract, data_root=data_root, output_root=output_root)
    train, normalization, _ = load_joint_split(paths["data_root"] / "train.npz")
    validation, _, _ = load_joint_split(paths["data_root"] / "validation.npz")
    if train.split != "train" or validation.split != "validation":
        raise ValueError("common evaluation requires train and validation artifacts")
    if len(validation) != 8780:
        raise ValueError(f"frozen validation window count changed: {len(validation)}")
    if normalization is None or normalization.fitted_split != "train":
        raise ValueError("normalization must be fitted on train")
    import yaml

    benchmark = yaml.safe_load(paths["benchmark_path"].read_text(encoding="utf-8"))
    checkpoint = paths["checkpoint_root"] / f"seed_{int(seed)}" / "best_checkpoint.pt"
    training_receipt = checkpoint.parent / "validation_receipt.json"
    model = _model_from_normalization(normalization, benchmark)
    provenance = _load_checkpoint(model, checkpoint, training_receipt, seed=int(seed))
    normalized = normalization.transform(validation)
    import torch

    predictions: list[np.ndarray] = []
    dispatches: list[np.ndarray] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation), int(batch_size)):
            stop = min(len(validation), start + int(batch_size))
            out = model(
                load_history=torch.from_numpy(normalized.load_history[start:stop]),
                exog_history=torch.from_numpy(normalized.exog_history[start:stop]),
                device_history=torch.from_numpy(normalized.device_history[start:stop]),
                device_status=torch.from_numpy(normalized.device_status[start:stop]),
                scheduler_context=torch.from_numpy(validation.scheduler_context[start:stop]),
                previous_chp=torch.from_numpy(validation.previous_chp[start:stop]),
            )
            predictions.append(out.forecast_physical.detach().cpu().numpy().astype(np.float32))
            dispatches.append(out.dispatch.detach().cpu().numpy().astype(np.float32))
    prediction = np.concatenate(predictions, axis=0)
    dispatch = np.concatenate(dispatches, axis=0)
    if prediction.shape != (len(validation), 4, 4) or dispatch.shape != (len(validation), 4, 21):
        raise ValueError("RSC-PF output shape does not match the frozen common contract")
    if not np.isfinite(prediction).all() or not np.isfinite(dispatch).all():
        raise ValueError("RSC-PF produced non-finite outputs")
    forecast_table = evaluate_forecast(prediction, validation.forecast_target)
    metrics, per_window = _dispatch_metrics(dispatch, validation, lp_calls=0)
    metrics.update({
        "forecast_mae_mean": float(np.nanmean(forecast_table.mae)),
        "forecast_rmse_mean": float(np.nanmean(forecast_table.rmse)),
        "forecast_wape_mean": float(np.nanmean(forecast_table.wape)),
        "forecast_mae_by_task_horizon": forecast_table.mae.tolist(),
        "forecast_rmse_by_task_horizon": forecast_table.rmse.tolist(),
        "forecast_wape_by_task_horizon": forecast_table.wape.tolist(),
        "lp_success_rate": 1.0,
        "evaluation_seconds": float(time.perf_counter() - started),
        "evaluation_windows_per_second": float(len(validation) / max(time.perf_counter() - started, 1.0e-12)),
    })
    destination = paths["output_root"] / "validation" / "RSC_PF" / f"seed_{int(seed)}" / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "validation_metrics.npz", **per_window)
    receipt = {
        "schema_version": "rsc-pf-common-validation-receipt-v1",
        "method_id": METHOD_ID,
        "source_method": SOURCE_METHOD,
        "seed": int(seed),
        "stage": "validation",
        "checkpoint": provenance["checkpoint"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "training_receipt": provenance["training_receipt"],
        "training_receipt_sha256": provenance["training_receipt_sha256"],
        "metrics": metrics,
        "optimizer_role": "none at inference",
        "exact_lp_calls": 0,
        "test_set_accessed": False,
        "split_manifest": {
            "train": str(paths["data_root"] / "train.npz"),
            "validation": str(paths["data_root"] / "validation.npz"),
        },
        "metric_protocol": "open-loop four-hour validation-window metrics under the external-baseline definitions",
    }
    (destination / "evaluation_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def _external_manifest_path(output_root: Path) -> Path:
    path = output_root / "external_validation_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_common_validation_manifest(output_root: Path) -> Path:
    """Combine frozen external receipts with the RSC-PF receipts only."""

    root = Path(output_root)
    external_path = _external_manifest_path(root)
    external = _read_json(external_path)
    if external.get("test_set_accessed") is not False:
        raise ValueError("external manifest is not test-set-free")
    entries = [dict(item) for item in external.get("entries", [])]
    for seed in SEEDS:
        path = root / "validation" / "RSC_PF" / f"seed_{seed}" / "evaluation" / "evaluation_receipt.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        receipt = _read_json(path)
        if receipt.get("test_set_accessed") is not False or receipt.get("method_id") != METHOD_ID:
            raise ValueError(f"invalid RSC-PF receipt: {path}")
        entries.append(receipt)
    expected = {(method, seed) for method in COMMON_METHODS for seed in SEEDS}
    observed = {(str(item.get("method_id")), int(item.get("seed", -1))) for item in entries}
    if observed != expected:
        raise ValueError(f"common manifest coverage mismatch: observed={sorted(observed)}")
    payload = {
        "schema_version": "rsc-pf-common-validation-manifest-v1",
        "stage": "validation",
        "methods": list(COMMON_METHODS),
        "seeds": list(SEEDS),
        "windows_per_seed": 8780,
        "metric_protocol": "open-loop four-hour validation-window metrics; realized-demand shortage evaluation",
        "metric_units": {
            "operating_cost_mean": "normalized source-index units summed over four-hour windows",
            "physical_carbon_mean": "physical carbon-index units summed over four-hour windows",
            "regret_vs_oracle_mean": "first-step penalized objective minus oracle objective",
            "feasibility_rate": "fraction of windows with zero realized-demand shortage",
        },
        "optimizer_roles": {
            "RSC-PF": "none at inference",
            "iTransformer-PTO": "none at inference",
            "DecisionFocused-Online": "exact optimizer at inference",
            "DigitalTwins-Policy": "none at inference",
        },
        "entries": entries,
        "test_set_accessed": False,
        "external_manifest": str(external_path),
    }
    destination = root / "common_validation_manifest.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def aggregate_common_validation_table(manifest_path: Path, output_root: Path) -> Path:
    """Aggregate method means/SDs without selecting a seed or recomputing metrics."""

    manifest = _read_json(Path(manifest_path))
    if manifest.get("test_set_accessed") is not False:
        raise ValueError("common manifest is not test-set-free")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("common manifest entries must be a list")
    expected = {(method, seed) for method in COMMON_METHODS for seed in SEEDS}
    observed = {(str(item.get("method_id")), int(item.get("seed", -1))) for item in entries}
    if observed != expected:
        raise ValueError("common table requires exactly four methods and five seeds")
    grouped: dict[str, list[dict[str, Any]]] = {method: [] for method in COMMON_METHODS}
    for entry in entries:
        if entry.get("test_set_accessed") is not False:
            raise ValueError("common table contains a test-set receipt")
        metrics = entry.get("metrics")
        if not isinstance(metrics, Mapping) or not all(key in metrics and _finite(metrics[key]) for key in METRIC_KEYS):
            raise ValueError(f"entry has missing/non-finite common metrics: {entry.get('method_id')}/{entry.get('seed')}")
        if int(metrics.get("windows", -1)) != 8780:
            raise ValueError("common table entry has wrong validation-window count")
        grouped[str(entry["method_id"])].append(entry)
    role_map = {
        "RSC-PF": "none at inference",
        "iTransformer-PTO": "none at inference",
        "DecisionFocused-Online": "exact optimizer at inference",
        "DigitalTwins-Policy": "none at inference",
    }
    rows: list[dict[str, Any]] = []
    for method in COMMON_METHODS:
        group = grouped[method]
        row: dict[str, Any] = {
            "method_id": method,
            "seed_count": len(group),
            "windows_per_seed": 8780,
            "optimizer_role": role_map[method],
            "exact_lp_calls_total": int(sum(int(item.get("exact_lp_calls", -1)) for item in group)),
        }
        for key in METRIC_KEYS:
            values = np.asarray([float(item["metrics"][key]) for item in group], dtype=np.float64)
            label = key[:-5] if key.endswith("_mean") else key
            row[f"{label}_mean"] = float(values.mean())
            row[f"{label}_sd"] = float(values.std(ddof=1))
        row["test_set_accessed"] = False
        rows.append(row)
    destination = Path(output_root) / "common_validation_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_contract_v2.json")
    parser.add_argument("--output-root", type=Path, default=FRAME_ROOT / "reports" / "rsc_pf_external_baselines_v1" / "implementation")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--build-table", action="store_true")
    args = parser.parse_args(argv)
    seeds = list(SEEDS) if args.all_seeds else [args.seed or SEEDS[0]]
    for seed in seeds:
        evaluate_rsc_pf_validation(seed, args.config, data_root=args.data_root, output_root=args.output_root)
    manifest_path = None
    table_path = None
    if args.build_table:
        manifest_path = build_common_validation_manifest(args.output_root)
        table_path = aggregate_common_validation_table(manifest_path, args.output_root)
    print(json.dumps({"status": "complete", "seeds": seeds, "manifest": str(manifest_path) if manifest_path else None, "table": str(table_path) if table_path else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
