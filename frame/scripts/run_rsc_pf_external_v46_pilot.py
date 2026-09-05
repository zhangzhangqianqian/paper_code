"""Evaluate frozen external-baseline checkpoints on the 2019 selection_full pilot.

The validation checkpoints are frozen before this command is run.  This script
only evaluates those checkpoints on the predeclared pilot artifact; it never
opens a sealed test path and never changes model parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_baseline_training import (  # noqa: E402
    METHODS,
    SEEDS,
    _METHOD_SAFE,
    _build,
    _config_paths,
    _decoder_parameters,
    _load_data,
    _split_manifest,
    load_external_checkpoint,
)
from src.joint_dispatch.external_v46_data import (  # noqa: E402
    ExternalV46Normalization,
    build_external_v46_oracle,
    load_external_v46_split,
)
from scripts.evaluate_rsc_pf_external_baselines import (  # noqa: E402
    _dispatch_metrics,
    _lp_parameters,
    _predict,
)
from src.joint_dispatch.evaluation import evaluate_forecast  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pilot(paths: Mapping[str, Path], train) -> tuple[Any, ExternalV46Normalization]:
    pilot_path = paths["data_root"] / paths["pilot_file"]
    normalized = str(pilot_path).replace("\\", "/").lower()
    if any(token in normalized for token in ("/test/", "sealed_test", "2021_test", "test-set")):
        raise ValueError("pilot path is sealed-test-like")
    if not pilot_path.is_file():
        raise FileNotFoundError(pilot_path)
    pilot = load_external_v46_split(pilot_path, "pilot")
    cache = paths["output_root"] / "data_cache" / "pilot_oracle.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as payload:
            oracle = np.asarray(payload["oracle"], dtype=np.float32)
    else:
        oracle = build_external_v46_oracle(pilot, _lp_parameters())
        np.savez_compressed(cache, oracle=oracle)
    if oracle.shape != (len(pilot),):
        raise ValueError("pilot oracle cache has the wrong shape")
    # The batch contract keeps teacher labels present even for inference-only
    # evaluation. They are never read by _predict, but zero placeholders make
    # the absence of a training label explicit and auditable.
    pilot = replace(
        pilot,
        teacher_dispatch=np.zeros((len(pilot), 4, 21), dtype=np.float32),
        oracle_first_step_objective=oracle,
    )
    return pilot, ExternalV46Normalization.fit(train)


def evaluate_pilot(method_id: str, seed: int, config: Mapping[str, Any]) -> dict[str, Any]:
    if method_id not in METHODS:
        raise ValueError(f"unknown external method: {method_id}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    paths = _config_paths(config)
    train, _validation, normalization = _load_data(paths)
    pilot, pilot_normalization = _load_pilot(paths, train)
    # The independently fitted object must match the train-fitted statistics
    # used by the frozen checkpoint; reusing the returned value avoids fitting
    # any statistic on the pilot itself.
    normalization = normalization if isinstance(normalization, ExternalV46Normalization) else pilot_normalization
    checkpoint = paths["output_root"] / "validation" / _METHOD_SAFE[method_id] / f"seed_{int(seed)}" / "best_checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    provenance = load_external_checkpoint(checkpoint, expected_method=method_id, expected_seed=seed)
    model = _build(method_id)
    model.load_state_dict(provenance["model_state_dict"])
    started = time.perf_counter()
    prediction, direct_dispatch = _predict(method_id, model, pilot, normalization, int(config.get("evaluation_batch_size", 256)))
    forecast_table = evaluate_forecast(prediction, pilot.forecast_target)
    if method_id in {"iTransformer-PTO", "DecisionFocused-Online"}:
        from src.joint_dispatch.pto import PTOForecasts, solve_pto_windows

        cache = solve_pto_windows(PTOForecasts(method_id, prediction, pilot.forecast_target), pilot, _lp_parameters())
        dispatch = cache.dispatch
        lp_calls = cache.offline_exact_lp_calls
        lp_success_rate = float(cache.success.mean())
    else:
        dispatch = np.asarray(direct_dispatch, dtype=np.float32)
        lp_calls = 0
        lp_success_rate = 1.0
    metrics, per_window = _dispatch_metrics(dispatch, pilot, lp_calls=lp_calls)
    elapsed = float(time.perf_counter() - started)
    metrics.update({
        "forecast_mae_mean": float(np.nanmean(forecast_table.mae)),
        "forecast_rmse_mean": float(np.nanmean(forecast_table.rmse)),
        "forecast_wape_mean": float(np.nanmean(forecast_table.wape)),
        "forecast_mae_by_task_horizon": forecast_table.mae.tolist(),
        "forecast_rmse_by_task_horizon": forecast_table.rmse.tolist(),
        "forecast_wape_by_task_horizon": forecast_table.wape.tolist(),
        "lp_success_rate": lp_success_rate,
        "evaluation_seconds": elapsed,
        "evaluation_windows_per_second": float(len(pilot) / max(elapsed, 1.0e-12)),
    })
    evaluation_dir = checkpoint.parent / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(evaluation_dir / "pilot_metrics.npz", **per_window)
    split_manifest = _split_manifest(paths)
    split_manifest["pilot"] = str(paths["data_root"] / paths["pilot_file"])
    receipt = {
        "schema_version": "rsc-pf-external-pilot-receipt-v1",
        "method_id": method_id,
        "seed": int(seed),
        "stage": "pilot",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "metrics": metrics,
        "optimizer_role": "exact optimizer at inference" if method_id == "DecisionFocused-Online" else "none at inference",
        "exact_lp_calls": int(lp_calls),
        "test_set_accessed": False,
        "pilot_file": str(paths["data_root"] / paths["pilot_file"]),
        "pilot_windows": int(len(pilot)),
        "pilot_oracle_cache": str(paths["output_root"] / "data_cache" / "pilot_oracle.npz"),
        "split_manifest": split_manifest,
    }
    (evaluation_dir / "pilot_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate frozen external baselines on selection_full pilot")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--all-methods", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--config", type=Path, default=FRAME_ROOT / "configs" / "rsc_pf_external_baseline_implementation_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    methods = METHODS if args.all_methods else ((args.method,) if args.method else METHODS)
    seeds = SEEDS if args.all_seeds else ((args.seed,) if args.seed else SEEDS)
    if args.all_methods and not args.all_seeds:
        raise ValueError("pilot evaluation requires --all-seeds when --all-methods is used")
    results = [evaluate_pilot(method, int(seed), config) for method in methods for seed in seeds]
    print(json.dumps({"evaluated": len(results), "pilot_windows": results[0]["pilot_windows"] if results else 0, "test_set_accessed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

