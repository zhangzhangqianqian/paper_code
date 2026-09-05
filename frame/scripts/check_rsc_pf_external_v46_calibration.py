"""Audit the short v4.6 external-baseline calibration gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.external_baseline_training import (
    METHODS,
    _METHOD_SAFE,
    _batches,
    _build,
    _config_paths,
    _decoder_parameters,
    _external_lp_parameters,
    _load_data,
    load_external_checkpoint,
)
from src.joint_dispatch.external_baseline_losses import physical_feasibility_penalty
from src.joint_dispatch.pto import PTOForecasts, solve_pto_windows
from scripts.evaluate_rsc_pf_external_baselines import _dispatch_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_calibration(config_path: Path, *, limit: int = 16) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = _config_paths(config)
    source_receipt = paths["source_root"] / "source_receipt.json"
    if not source_receipt.is_file():
        raise FileNotFoundError(source_receipt)
    train, validation, normalization = _load_data(paths)
    validation = validation.take(np.arange(min(int(limit), len(validation)), dtype=np.int64))
    entries: list[dict[str, Any]] = []
    for method in METHODS:
        checkpoint = paths["output_root"] / "calibration" / _METHOD_SAFE[method] / "seed_2026" / "best_checkpoint.pt"
        receipt_path = checkpoint.parent / "training_receipt.json"
        provenance = load_external_checkpoint(checkpoint, expected_method=method, expected_seed=2026)
        training_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        model = _build(method)
        model.load_state_dict(provenance["model_state_dict"])
        model.eval()
        forecasts: list[np.ndarray] = []
        direct_dispatch: list[np.ndarray] = []
        decoder_penalty: list[float] = []
        with torch.no_grad():
            for batch in _batches(
                validation, normalization=normalization, batch_size=256, seed=0,
                shuffle=False, normalized=method != "DigitalTwins-Policy",
            ):
                if method == "iTransformer-PTO":
                    forecasts.append(model(batch.load_history, batch.exog_history).cpu().numpy())
                elif method == "DecisionFocused-Online":
                    forecasts.append(model(batch).forecast.cpu().numpy())
                else:
                    output = model(batch)
                    forecasts.append(output.forecast_physical.cpu().numpy())
                    direct_dispatch.append(output.dispatch.cpu().numpy())
                    physical_features = torch.cat((output.forecast_physical, batch.scheduler_context), dim=-1)
                    decoder_penalty.append(float(physical_feasibility_penalty(output.dispatch, physical_features, _decoder_parameters()).item()))
        prediction = np.concatenate(forecasts, axis=0)
        if method != "DigitalTwins-Policy":
            prediction = prediction * normalization.load_scale.reshape(1, 1, -1) + normalization.load_mean.reshape(1, 1, -1)
            cache = solve_pto_windows(PTOForecasts(method, prediction, validation.forecast_target), validation, _external_lp_parameters())
            dispatch = cache.dispatch
            lp_calls = cache.offline_exact_lp_calls
            lp_success_rate = float(cache.success.mean())
        else:
            dispatch = np.concatenate(direct_dispatch, axis=0)
            lp_calls = 0
            lp_success_rate = 1.0
        metrics, _raw = _dispatch_metrics(dispatch, validation, lp_calls=lp_calls)
        structural_penalty = max(decoder_penalty) if decoder_penalty else 0.0
        entry = {
            "method_id": method,
            "seed": 2026,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "training_receipt_sha256": _sha256(receipt_path),
            "training_source_receipt_sha256": training_receipt.get("source_receipt_sha256"),
            "lp_success_rate": lp_success_rate,
            "exact_lp_calls": int(lp_calls),
            "windows": int(len(validation)),
            "structural_decoder_penalty_max": float(structural_penalty),
            "metrics": metrics,
            "test_set_accessed": False,
            "finite_outputs": bool(np.isfinite(prediction).all() and np.isfinite(dispatch).all()),
        }
        entry["gate_passed"] = bool(
            entry["finite_outputs"] and np.isfinite(float(training_receipt["best_validation_loss"]))
            and lp_success_rate >= 1.0 and structural_penalty <= 1.0e-4
            and (lp_calls == len(validation) if method != "DigitalTwins-Policy" else lp_calls == 0)
            and training_receipt.get("test_set_accessed") is False
        )
        entries.append(entry)
    payload = {
        "schema_version": "rsc-pf-external-v46-calibration-gate-v1",
        "config": str(config_path),
        "source_receipt": str(source_receipt),
        "source_receipt_sha256": _sha256(source_receipt),
        "limit": int(len(validation)),
        "entries": entries,
        "gate_passed": all(bool(entry["gate_passed"]) for entry in entries),
        "test_set_accessed": False,
    }
    destination = paths["output_root"] / "CALIBRATION_GATE.json"
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit v4.6 external baseline calibration")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    payload = check_calibration(args.config, limit=args.limit)
    print(json.dumps({"gate_passed": payload["gate_passed"], "output": str(Path(payload["config"]).parent)}, ensure_ascii=False))
    return 0 if payload["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
