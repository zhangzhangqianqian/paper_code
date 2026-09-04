"""Freeze and probe the formal-v4.1 nine-method adapter matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_gate0_evidence import write_immutable_json  # noqa: E402
from src.joint_dispatch.formal_v4_method_adapter import FORMAL_V4_METHOD_CONTRACTS, build_formal_v4_method_adapter  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _parameters(path: Path) -> dict[str, float]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("values"), Mapping):
        raise ValueError("benchmark must contain values")
    return {str(key): float(value) for key, value in payload["values"].items()}


def _window(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "load_history": np.asarray(payload["load_history"][0], dtype=np.float64),
            "exog_history": np.asarray(payload["exog_history"][0], dtype=np.float64),
            "device_history": np.asarray(payload["device_history"][0], dtype=np.float64),
            "activity_history": np.asarray(payload["activity_history"][0], dtype=np.float64),
            "renewable_forecast": np.asarray(payload["renewable_forecast"][0], dtype=np.float64),
            "prices_and_weights": np.asarray(payload["prices_and_weights"][0], dtype=np.float64),
            "initial_soc": np.asarray(payload["initial_soc"][0], dtype=np.float64),
            "previous_chp": np.asarray(payload["previous_chp"][0], dtype=np.float64),
            "forecast_target": np.asarray(payload["forecast_target"][0], dtype=np.float64),
        }


def build_method_adapter_receipt(*, benchmark_path: Path, train_archive_path: Path, source_commit: str | None = None) -> dict[str, Any]:
    parameters = _parameters(benchmark_path)
    window = _window(train_archive_path)
    rows = []
    contract_only = {"Official iTransformer-PTO", "Differentiable-LP", "Perfect-Information-MPC"}
    for contract in FORMAL_V4_METHOD_CONTRACTS:
        row = {
            "method_id": contract.method_id,
            "role": contract.role,
            "deployable": contract.deployable,
            "produces_forecast": contract.produces_forecast,
            "forecast_metrics_applicable": contract.forecast_metrics_applicable,
            "online_optimizer_calls_per_window": contract.online_optimizer_calls_per_window,
            "expected_forecast_shape": list(contract.expected_forecast_shape) if contract.expected_forecast_shape is not None else None,
            "expected_dispatch_shape": list(contract.expected_dispatch_shape),
            "probe_status": "contract_only" if contract.method_id in contract_only else "not_run",
        }
        if contract.method_id not in contract_only:
            adapter = build_formal_v4_method_adapter(contract.method_id, parameters)
            result = adapter.predict_and_dispatch(window, {})
            dispatch = np.asarray(result["dispatch"])
            if dispatch.shape != contract.expected_dispatch_shape or not np.isfinite(dispatch).all():
                raise ValueError(f"adapter probe returned an invalid dispatch for {contract.method_id}")
            if contract.produces_forecast:
                forecast = np.asarray(result["forecast"])
                if forecast.shape != contract.expected_forecast_shape or not np.isfinite(forecast).all():
                    raise ValueError(f"adapter probe returned an invalid forecast for {contract.method_id}")
            if int(result["optimizer_calls"]) != contract.online_optimizer_calls_per_window:
                raise ValueError(f"adapter probe optimizer count mismatch for {contract.method_id}")
            row["probe_status"] = "passed"
        rows.append(row)
    return {
        "schema_version": "formal-v4.1-method-adapter-receipt-v1",
        "protocol_id": "formal-v4.1-method-adapter-probe-v1",
        "source_commit": source_commit or _git_commit(),
        "probe_only": True,
        "test_set_accessed": False,
        "benchmark_sha256": _sha256(benchmark_path),
        "train_archive_sha256": _sha256(train_archive_path),
        "methods": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_method_adapter_receipt(
        benchmark_path=args.benchmark_path.resolve(),
        train_archive_path=args.train_archive.resolve(),
    )
    write_immutable_json(args.run_root.resolve() / "protocol" / "METHOD_ADAPTER_RECEIPT.json", payload)
    print(json.dumps({"status": "pass", "method_count": len(payload["methods"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_method_adapter_receipt", "main"]
