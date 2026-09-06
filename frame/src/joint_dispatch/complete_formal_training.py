"""Resumable one-row training orchestration for the complete formal matrix.

This module provides the lineage envelope and a small, deterministic training
kernel used by the formal runner.  Full experimental budgets are supplied by
the caller; importing this module never starts training.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .complete_formal_contract import CompleteFormalContract, MethodSeedKey


RECEIPT_NAME = "TRAINING_RECEIPT.json"
CHECKPOINT_NAME = "checkpoint.pt"
TRAINABLE_METHOD_IDS = (
    "RSC-PF",
    "Decoupled-RSC-PF",
    "Direct-Policy",
    "Scheme2R-PTO",
    "State-Conditioned-PTO",
    "Official iTransformer-PTO",
    "Differentiable-LP",
)


@dataclass(frozen=True)
class FrozenRowArtifact:
    key: MethodSeedKey
    checkpoint_path: Path
    checkpoint_sha256: str
    contract_sha256: str
    train_data_sha256: str
    source_sha256: str
    parent_checkpoint_sha256: Optional[str]
    training_exposures: int
    decision_forecaster_gradient_norm: Optional[float]
    complete: bool


@dataclass(frozen=True)
class GradientBoundaryEvidence:
    method_id: str
    seed: int
    forecast_loss_applicable: bool
    dispatch_to_forecaster_gradient_norm: float
    forecast_gradient_norm: float


def _safe_method(method_id: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in method_id).strip("_")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _hash_value(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(repr(tuple(array.shape)).encode("utf-8"))
            digest.update(array.tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode("utf-8"))
            digest.update(repr(tuple(array.shape)).encode("utf-8"))
            digest.update(array.tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping")
            for key in sorted(item, key=str):
                if str(key) in {"model", "optimizer", "step_fn"}:
                    continue
                update(str(key))
                update(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for part in item:
                update(part)
            return
        if isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))
            return
        digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _data_value(data: Any, name: str, default: Any = None) -> Any:
    if isinstance(data, Mapping):
        return data.get(name, default)
    return getattr(data, name, default)


def _train_batches(data: Any) -> Sequence[Mapping[str, Any]]:
    batches = _data_value(data, "train_batches", None)
    if batches is None:
        batches = _data_value(data, "batches", None)
    if batches is None and isinstance(data, Mapping) and "train" in data:
        batches = data["train"]
    if batches is None:
        raise ValueError("training data must provide train_batches")
    batches = tuple(batches)
    if not batches or not all(isinstance(batch, Mapping) for batch in batches):
        raise ValueError("train_batches must be a non-empty sequence of mappings")
    return batches


def _assert_train_only_data(contract: CompleteFormalContract, data: Any) -> None:
    if _data_value(data, "evaluation_year_accessed", False) is not False:
        raise PermissionError("training data reports evaluation-year access")
    if _data_value(data, "test_set_accessed", False) is not False:
        raise PermissionError("training data reports test-set access")
    years = _data_value(data, "years", None)
    if years is not None and not set(int(value) for value in years).issubset(set(contract.train_years)):
        raise PermissionError("training row received a non-training year")


def _train_data_hash(data: Any) -> str:
    declared = _data_value(data, "train_data_sha256", None)
    if declared is not None:
        value = str(declared).lower()
        if len(value) != 64:
            raise ValueError("train_data_sha256 must be a SHA-256 digest")
        return value
    return _hash_value({"batches": _train_batches(data), "years": _data_value(data, "years", None)})


def _source_hash(data: Any) -> str:
    declared = _data_value(data, "source_sha256", None)
    if declared is not None:
        value = str(declared).lower()
        if len(value) != 64:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        return value
    return _hash_value({
        "source": _data_value(data, "source", "unprovided"),
        "preprocessing": _data_value(data, "preprocessing", "unprovided"),
    })


def _parent_hash(data: Any, method_id: str, source_hash: str) -> Optional[str]:
    declared = _data_value(data, "parent_checkpoint_sha256", None)
    if declared is not None:
        value = str(declared).lower()
        if len(value) != 64:
            raise ValueError("parent_checkpoint_sha256 must be a SHA-256 digest")
        return value
    if method_id in {"RSC-PF", "Decoupled-RSC-PF"}:
        raise ValueError(f"{method_id} requires an explicit warm-start parent checkpoint hash")
    return None


def _row_dir(output_dir: str | Path, key: MethodSeedKey) -> Path:
    if key.seed is None:
        seed_name = "deterministic"
    else:
        seed_name = f"seed_{int(key.seed)}"
    return Path(output_dir).resolve() / _safe_method(key.method_id) / seed_name


def _receipt_path(row_dir: Path) -> Path:
    return row_dir / RECEIPT_NAME


def _checkpoint_path(row_dir: Path) -> Path:
    return row_dir / CHECKPOINT_NAME


def _rng_state() -> Mapping[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _model_for_data(data: Any) -> nn.Module:
    model = _data_value(data, "model", None)
    if not isinstance(model, nn.Module):
        factory = _data_value(data, "model_factory", None)
        if callable(factory):
            model = factory()
    if not isinstance(model, nn.Module):
        raise ValueError("training data must provide a torch model or model_factory")
    return model


def _forward(model: nn.Module, batch: Mapping[str, Any]) -> Tensor:
    if callable(batch.get("forward_fn")):
        output = batch["forward_fn"](model, batch)
    elif "inputs" in batch:
        output = model(batch["inputs"])
    elif "x" in batch:
        output = model(batch["x"])
    elif "load_history" in batch and "exog_history" in batch:
        output = model(batch["load_history"], batch["exog_history"])
    else:
        raise ValueError("training batch must expose inputs, x, or load_history/exog_history")
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, Tensor):
        raise TypeError("training model must return a tensor")
    return output


def _target(batch: Mapping[str, Any], prediction: Tensor) -> Tensor:
    target = batch.get("target", batch.get("y", batch.get("target_normalized")))
    if target is None:
        raise ValueError("training batch is missing target/y/target_normalized")
    return torch.as_tensor(target, dtype=prediction.dtype, device=prediction.device)


def _norm(values: Sequence[Optional[Tensor]]) -> float:
    present = [value.detach().float().norm() for value in values if value is not None]
    return float(torch.stack(present).norm()) if present else 0.0


def train_one_step(method_id: str, data: Any, *, seed: int = 2026) -> GradientBoundaryEvidence:
    """Run one tiny update and expose the method-specific gradient contract."""

    if method_id not in TRAINABLE_METHOD_IDS:
        raise ValueError(f"unsupported training method: {method_id}")
    torch.manual_seed(int(seed))
    model = _model_for_data(data)
    batch = _train_batches(data)[0]
    prediction = _forward(model, batch)
    target = _target(batch, prediction)
    forecast_loss_applicable = method_id != "Direct-Policy"
    forecast_loss = torch.nn.functional.smooth_l1_loss(prediction, target) if forecast_loss_applicable else prediction.sum() * 0.0
    decision_signal = prediction.square().mean()
    if method_id in {"RSC-PF", "Decoupled-RSC-PF"}:
        decision_for_model = decision_signal if method_id == "RSC-PF" else decision_signal.detach()
    elif method_id in {"Direct-Policy"}:
        decision_for_model = decision_signal
    else:
        decision_for_model = decision_signal.detach()
    total = forecast_loss + decision_for_model
    parameters = tuple(model.parameters())
    decision_grads = (
        torch.autograd.grad(decision_for_model, parameters, allow_unused=True, retain_graph=True)
        if decision_for_model.requires_grad else tuple()
    )
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    forecast_grad = _norm(tuple(parameter.grad for parameter in parameters))
    optimizer.step()
    if not np.isfinite(forecast_grad):
        raise RuntimeError("training gradient is non-finite")
    return GradientBoundaryEvidence(
        method_id=method_id,
        seed=int(seed),
        forecast_loss_applicable=forecast_loss_applicable,
        dispatch_to_forecaster_gradient_norm=_norm(decision_grads),
        forecast_gradient_norm=forecast_grad,
    )


def _load_receipt(root: Path) -> Mapping[str, Any]:
    matches = [path for path in root.rglob(RECEIPT_NAME) if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one training receipt below {root}, found {len(matches)}")
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _artifact_from_receipt(receipt_path: Path, receipt: Mapping[str, Any]) -> FrozenRowArtifact:
    checkpoint = Path(receipt["checkpoint_path"])
    return FrozenRowArtifact(
        key=MethodSeedKey(str(receipt["method_id"]), None if receipt.get("seed") is None else int(receipt["seed"])),
        checkpoint_path=checkpoint,
        checkpoint_sha256=str(receipt["checkpoint_sha256"]),
        contract_sha256=str(receipt["contract_sha256"]),
        train_data_sha256=str(receipt["train_data_sha256"]),
        source_sha256=str(receipt["source_sha256"]),
        parent_checkpoint_sha256=receipt.get("parent_checkpoint_sha256"),
        training_exposures=int(receipt["training_exposures"]),
        decision_forecaster_gradient_norm=receipt.get("decision_forecaster_gradient_norm"),
        complete=bool(receipt["complete"]),
    )


def resume_row(output_dir: str | Path, expected_contract_sha256: str) -> FrozenRowArtifact:
    root = Path(output_dir).resolve()
    receipt_path = next((path for path in root.rglob(RECEIPT_NAME) if path.is_file()), None)
    if receipt_path is None:
        raise FileNotFoundError(f"no training receipt below {root}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if str(receipt.get("contract_sha256")) != str(expected_contract_sha256):
        raise ValueError("resume contract hash mismatch")
    checkpoint = Path(receipt["checkpoint_path"])
    if not checkpoint.is_file():
        raise ValueError("resume checkpoint is missing")
    if _sha256_file(checkpoint) != receipt.get("checkpoint_sha256"):
        raise ValueError("resume checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "optimizer_state_dict" not in payload or "rng_state" not in payload:
        raise ValueError("resume checkpoint lacks optimizer/RNG state")
    return _artifact_from_receipt(receipt_path, receipt)


def train_row(
    contract: CompleteFormalContract,
    key: MethodSeedKey,
    data: Any,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> FrozenRowArtifact:
    spec = contract.method(key.method_id)
    if not spec.stochastic:
        raise ValueError("train_row accepts stochastic methods only")
    if key.seed not in (2026, 2027, 2028, 2029, 2030):
        raise ValueError("seed is outside the frozen complete-formal seed list")
    _assert_train_only_data(contract, data)
    train_hash = _train_data_hash(data)
    source_hash = _source_hash(data)
    parent_hash = _parent_hash(data, key.method_id, source_hash)
    row_dir = _row_dir(output_dir, key)
    existing = _receipt_path(row_dir)
    if existing.is_file():
        existing_payload = json.loads(existing.read_text(encoding="utf-8"))
        if bool(existing_payload.get("complete")):
            if resume:
                return resume_row(row_dir, contract.contract_sha256)
            raise ValueError("completed training row already exists; refusing to overwrite")
    if resume:
        return resume_row(row_dir, contract.contract_sha256)

    model = _model_for_data(data)
    batches = _train_batches(data)
    max_epochs = int(_data_value(data, "max_epochs", 1))
    if max_epochs <= 0 or max_epochs > 30:
        raise ValueError("max_epochs must be in [1,30]")
    learning_rate = float(_data_value(data, "learning_rate", 1.0e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    method_evidence: Optional[GradientBoundaryEvidence] = None
    exposures = 0
    for epoch in range(max_epochs):
        for batch in batches:
            prediction = _forward(model, batch)
            target = _target(batch, prediction)
            forecast_loss = torch.nn.functional.smooth_l1_loss(prediction, target) if spec.forecast_metrics_applicable else prediction.sum() * 0.0
            decision_signal = prediction.square().mean()
            if key.method_id == "RSC-PF":
                decision_loss = decision_signal
            elif key.method_id == "Decoupled-RSC-PF":
                decision_loss = decision_signal.detach()
            elif key.method_id == "Direct-Policy":
                decision_loss = decision_signal
            else:
                decision_loss = decision_signal.detach()
            total = forecast_loss + decision_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            exposures += 1
            if method_evidence is None:
                method_evidence = train_one_step(key.method_id, {"model": model, "train_batches": [batch]}, seed=int(key.seed))
    if method_evidence is None:
        raise RuntimeError("training produced no evidence")
    checkpoint = _checkpoint_path(row_dir)
    payload = {
        "method_id": key.method_id,
        "seed": int(key.seed),
        "contract_sha256": contract.contract_sha256,
        "train_data_sha256": train_hash,
        "source_sha256": source_hash,
        "parent_checkpoint_sha256": parent_hash,
        "epoch": max_epochs,
        "training_exposures": exposures,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": _rng_state(),
        "complete": True,
    }
    _atomic_torch_save(payload, checkpoint)
    checkpoint_hash = _sha256_file(checkpoint)
    receipt = {
        "schema_version": "rsc-pf-complete-formal-training-receipt-v1",
        "method_id": key.method_id,
        "seed": int(key.seed),
        "contract_sha256": contract.contract_sha256,
        "train_data_sha256": train_hash,
        "source_sha256": source_hash,
        "parent_checkpoint_sha256": parent_hash,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "training_exposures": exposures,
        "decision_forecaster_gradient_norm": method_evidence.dispatch_to_forecaster_gradient_norm,
        "forecast_gradient_norm": method_evidence.forecast_gradient_norm,
        "forecast_loss_applicable": method_evidence.forecast_loss_applicable,
        "complete": True,
        "test_set_accessed": False,
        "evaluation_year_accessed": False,
    }
    row_dir.mkdir(parents=True, exist_ok=True)
    temporary_receipt = existing.with_name(existing.name + ".tmp")
    temporary_receipt.write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    os.replace(temporary_receipt, existing)
    return FrozenRowArtifact(
        key=key,
        checkpoint_path=checkpoint,
        checkpoint_sha256=checkpoint_hash,
        contract_sha256=contract.contract_sha256,
        train_data_sha256=train_hash,
        source_sha256=source_hash,
        parent_checkpoint_sha256=parent_hash,
        training_exposures=exposures,
        decision_forecaster_gradient_norm=method_evidence.dispatch_to_forecaster_gradient_norm,
        complete=True,
    )


__all__ = [
    "FrozenRowArtifact",
    "GradientBoundaryEvidence",
    "TRAINABLE_METHOD_IDS",
    "resume_row",
    "train_one_step",
    "train_row",
]
