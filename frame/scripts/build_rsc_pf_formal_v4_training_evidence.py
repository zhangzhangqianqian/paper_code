"""Build formal-v4.1 curriculum and gradient-boundary probe receipts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_models import RSCPFModel  # noqa: E402
from src.joint_dispatch.formal_v4_objective import formal_v4_curriculum_weights  # noqa: E402
from src.joint_dispatch.formal_v4_training import probe_gradient_boundary  # noqa: E402
from src.joint_dispatch.formal_v4_gate0_evidence import write_immutable_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_curriculum_receipt(*, source_commit: str | None = None) -> dict[str, Any]:
    rows = []
    for epoch in (0, 9, 18):
        weights = formal_v4_curriculum_weights(epoch=epoch, ramp_epochs=18, imitation_final=0.0, decision_start=0.05)
        rows.append({"epoch": epoch, "forecast": float(weights.forecast), "imitation": float(weights.imitation), "decision": float(weights.decision)})
    return {
        "schema_version": "formal-v4.1-curriculum-receipt-v1",
        "protocol_id": "formal-v4.1-curriculum-probe-v1",
        "source_commit": source_commit or _git_commit(),
        "probe_only": True,
        "test_set_accessed": False,
        "step_weights": [0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
        "forecast_task_weights": [1.0, 1.0, 1.0, 0.25],
        "gas_forecast_weight": 0.25,
        "rigid_task_weights": [1.0, 1.0, 1.0],
        "epochs": rows,
    }


def _batch_from_train(path: Path, *, batch_size: int = 2) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as payload:
        if len(payload["load_history"]) < batch_size:
            raise ValueError("train archive is too small for gradient probe")
        indices = slice(0, batch_size)
        def normalized(name: str) -> np.ndarray:
            statistic_prefix = {"load_history": "load", "exog_history": "exog", "device_history": "device", "activity_history": "activity"}.get(name, name)
            mean = payload["normalization_load_mean"] if name == "forecast_target" else payload[f"normalization_{statistic_prefix}_mean"]
            scale = payload["normalization_load_scale"] if name == "forecast_target" else payload[f"normalization_{statistic_prefix}_scale"]
            return (payload[name][indices] - mean) / np.maximum(scale, 1.0e-6)
        context = np.concatenate((
            payload["renewable_forecast"][indices],
            payload["prices_and_weights"][indices],
            np.repeat(payload["initial_soc"][indices, None, :], 4, axis=1),
        ), axis=-1)
        return {
            "load_history": torch.as_tensor(normalized("load_history"), dtype=torch.float32),
            "exog_history": torch.as_tensor(normalized("exog_history"), dtype=torch.float32),
            "device_history": torch.as_tensor(normalized("device_history"), dtype=torch.float32),
            "activity_history": torch.as_tensor(payload["activity_history"][indices], dtype=torch.float32),
            "scheduler_context": torch.as_tensor(context, dtype=torch.float32),
            "previous_chp": torch.as_tensor(payload["previous_chp"][indices], dtype=torch.float32),
            "target_normalized": torch.as_tensor(normalized("forecast_target"), dtype=torch.float32),
            "target_physical": torch.as_tensor(payload["forecast_target"][indices], dtype=torch.float32),
            "realized_renewables": torch.as_tensor(payload["renewable_realized"][indices], dtype=torch.float32),
            "initial_soc": torch.as_tensor(payload["initial_soc"][indices], dtype=torch.float32),
        }


def build_gradient_receipt(
    batch: Mapping[str, torch.Tensor],
    parameters: Mapping[str, Any],
    *,
    c_ref: float,
    seed: int = 2026,
    source_commit: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    # Bind every probe model to the exact benchmark decoder parameters used by
    # the materialized train archive.  Leaving the default decoder parameters
    # here can reject a valid archive trajectory at the previous-CHP capacity
    # boundary (and would make the gradient receipt non-representative).
    source = RSCPFModel(decoder_parameters=parameters, dropout=0.0)
    joint = RSCPFModel(decoder_parameters=parameters, dropout=0.0)
    decoupled = RSCPFModel(decoder_parameters=parameters, dropout=0.0)
    joint.load_state_dict(source.state_dict())
    decoupled.load_state_dict(source.state_dict())
    hashes = {"joint": _model_hash(joint), "decoupled": _model_hash(decoupled)}
    joint_probe = probe_gradient_boundary(joint, batch, parameters, mode="joint", c_ref=c_ref)
    decoupled_probe = probe_gradient_boundary(decoupled, batch, parameters, mode="decoupled", c_ref=c_ref)
    return {
        "schema_version": "formal-v4.1-gradient-receipt-v1",
        "protocol_id": "formal-v4.1-gradient-probe-v1",
        "source_commit": source_commit or _git_commit(),
        "probe_only": True,
        "test_set_accessed": False,
        "seed": int(seed),
        "initial_model_hashes": hashes,
        "joint": asdict(joint_probe),
        "decoupled": asdict(decoupled_probe),
        "checkpoint_created": False,
    }


def _parameters(path: Path) -> dict[str, float]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("values"), Mapping):
        raise ValueError("benchmark must contain values")
    return {str(key): float(value) for key, value in payload["values"].items()}


def build_training_evidence(*, run_root: Path, benchmark_path: Path, train_archive_path: Path, c_ref_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    c_ref = json.loads(c_ref_path.read_text(encoding="utf-8"))
    curriculum = build_curriculum_receipt()
    batch = _batch_from_train(train_archive_path)
    gradient = build_gradient_receipt(batch, _parameters(benchmark_path), c_ref=float(c_ref["c_ref"]))
    write_immutable_json(run_root / "protocol" / "CURRICULUM_RECEIPT.json", curriculum)
    write_immutable_json(run_root / "protocol" / "GRADIENT_RECEIPT.json", gradient)
    return curriculum, gradient


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--c-ref", type=Path, required=True)
    args = parser.parse_args(argv)
    curriculum, gradient = build_training_evidence(
        run_root=args.run_root.resolve(),
        benchmark_path=args.benchmark_path.resolve(),
        train_archive_path=args.train_archive.resolve(),
        c_ref_path=args.c_ref.resolve(),
    )
    print(json.dumps({"status": "pass", "curriculum_epochs": len(curriculum["epochs"]), "joint_gradient": gradient["joint"]["forecaster_decision_gradient_norm"], "decoupled_gradient": gradient["decoupled"]["forecaster_decision_gradient_norm"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_curriculum_receipt", "build_gradient_receipt", "build_training_evidence", "main"]
