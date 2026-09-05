"""Resumable calibration and validation for the frozen external baselines.

The runner is deliberately validation-only.  It loads the already generated
causal train/validation windows, fits normalization on train, records every
provenance field needed for a later evaluation, and refuses test-shaped paths.
The DecisionFocused-Online adapter uses the source-verified differentiable
surrogate for neural validation; its exact optimizer role remains an
evaluation-time accounting field, as required by the paper.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .external_baseline_data import (
    ExternalBaselineBatch,
    ExternalNormalization,
    fit_external_normalization,
)
from .external_v46_data import (
    ExternalV46Normalization,
    ExternalV46Split,
    build_external_v46_oracle,
    build_external_v46_teacher,
    load_external_v46_split,
)
from .external_baseline_losses import (
    audit_external_gradients,
    decision_focused_loss,
    forecast_loss,
    physical_feasibility_penalty,
    policy_imitation_loss,
    verified_decision_focused_surrogate,
)
from .external_baselines import build_external_baseline
from .data import JointWindowSplit, load_joint_split


METHODS = ("iTransformer-PTO", "DecisionFocused-Online", "DigitalTwins-Policy")
SEEDS = (2026, 2027, 2028, 2029, 2030)
_METHOD_SAFE = {method: re.sub(r"[^A-Za-z0-9]+", "_", method).strip("_") for method in METHODS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _forbidden_test_path(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/").lower()
    return any(token in normalized for token in ("/test/", "/test_set/", "/sealed_test/", "2021_test", "test-set"))


def _config_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    root = _repo_root()
    values = {
        "data_root": config.get("data_root", "reports/joint_forecast_dispatch_v1/data"),
        "output_root": config.get("output_root", "reports/rsc_pf_external_baselines_v1/implementation"),
        "source_root": config.get("source_root", "reports/rsc_pf_external_baselines_v1/implementation/sources"),
        "registry_path": config.get("registry_path", "configs/rsc_pf_external_baselines_v1.json"),
        "implementation_config": config.get("implementation_config", "configs/rsc_pf_external_baseline_implementation_v1.json"),
    }
    paths = {name: _resolve(value, root) for name, value in values.items()}
    paths["train_file"] = Path(str(config.get("train_file", "train.npz")))
    paths["validation_file"] = Path(str(config.get("validation_file", "validation.npz")))
    paths["pilot_file"] = Path(str(config.get("pilot_file", "selection_full.npz")))
    paths["data_protocol"] = Path(str(config.get("data_protocol", "joint_v1")))
    for name in ("data_root", "source_root", "output_root", "registry_path", "implementation_config"):
        if _forbidden_test_path(paths[name]):
            raise ValueError(f"{name} points to a sealed test-set path")
    return paths


def _validate_method_seed(method_id: str, seed: int) -> None:
    if method_id not in METHODS:
        raise ValueError(f"unknown external method: {method_id}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")


def _read_source_provenance(paths: Mapping[str, Path]) -> tuple[str, str, dict[str, Any]]:
    receipt_path = paths["source_root"] / "source_receipt.json"
    registry_path = paths["registry_path"]
    if not receipt_path.is_file():
        raise FileNotFoundError(f"source receipt is missing: {receipt_path}")
    if not registry_path.is_file():
        raise FileNotFoundError(f"external registry is missing: {registry_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("test_set_accessed") is not False or receipt.get("validation_only") is not True:
        raise ValueError("source receipt is not validation-only")
    registry_hash = _sha256(registry_path)
    if str(receipt.get("registry_sha256", "")).lower() != registry_hash.lower():
        raise ValueError("source receipt registry hash does not match the frozen registry")
    methods = {str(item.get("method_id")): item for item in receipt.get("methods", [])}
    if set(methods) != set(METHODS):
        raise ValueError("source receipt does not cover all frozen external methods")
    return registry_hash, _sha256(receipt_path), receipt


def _external_lp_parameters() -> dict[str, float]:
    values = _decoder_parameters()
    values.update({
        "bess_throughput_cost": 1.0e-6,
        "grid_energy_price": 1.0,
        "gas_energy_price": 0.6,
        "grid_emission_factor": 0.5,
        "gas_emission_factor": 0.25,
        "carbon_price_default": 0.0,
        "unserved_penalty": 100.0,
    })
    return values


def _split_manifest(paths: Mapping[str, Path]) -> dict[str, str]:
    return {
        "train": str(paths["data_root"] / paths["train_file"]),
        "validation": str(paths["data_root"] / paths["validation_file"]),
    }


def _load_data(paths: Mapping[str, Path]) -> tuple[Any, Any, Any]:
    train_path = paths["data_root"] / paths["train_file"]
    validation_path = paths["data_root"] / paths["validation_file"]
    for path in (train_path, validation_path):
        if _forbidden_test_path(path):
            raise ValueError("external training attempted to access a test path")
        if not path.is_file():
            raise FileNotFoundError(f"required split artifact is missing: {path}")
    if str(paths.get("data_protocol", Path("joint_v1"))) == "formal_v46":
        train = load_external_v46_split(train_path, "train")
        validation = load_external_v46_split(validation_path, "validation")
        cache = paths["output_root"] / "data_cache" / "v46_labels.npz"
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as payload:
                train_teacher = payload["train_teacher"]
                validation_teacher = payload["validation_teacher"]
                validation_oracle = payload["validation_oracle"]
        else:
            train_teacher = build_external_v46_teacher(train, _external_lp_parameters())
            validation_teacher = build_external_v46_teacher(validation, _external_lp_parameters())
            validation_oracle = build_external_v46_oracle(validation, _external_lp_parameters())
            np.savez_compressed(
                cache, train_teacher=train_teacher, validation_teacher=validation_teacher,
                validation_oracle=validation_oracle,
            )
        train = replace_external_labels(train, train_teacher, np.zeros((len(train),), dtype=np.float32))
        validation = replace_external_labels(validation, validation_teacher, validation_oracle)
        return train, validation, ExternalV46Normalization.fit(train)
    train, _stored_norm, _train_meta = load_joint_split(train_path)
    validation, _validation_norm, _validation_meta = load_joint_split(validation_path)
    if train.split != "train" or validation.split != "validation":
        raise ValueError("external runner requires train and validation split artifacts")
    normalization = fit_external_normalization(train)
    return train, validation, normalization


def replace_external_labels(split: ExternalV46Split, teacher: np.ndarray, oracle: np.ndarray) -> ExternalV46Split:
    if teacher.shape != (len(split), 4, 21) or oracle.shape != (len(split),):
        raise ValueError("external labels do not match the v4.6 split")
    return replace_dataclass(split, teacher_dispatch=teacher, oracle_first_step_objective=oracle)


def replace_dataclass(split: ExternalV46Split, **changes: Any) -> ExternalV46Split:
    return ExternalV46Split(
        load_history=changes.get("load_history", split.load_history),
        exog_history=changes.get("exog_history", split.exog_history),
        device_history=changes.get("device_history", split.device_history),
        device_status=changes.get("device_status", split.device_status),
        scheduler_context=changes.get("scheduler_context", split.scheduler_context),
        previous_chp=changes.get("previous_chp", split.previous_chp),
        forecast_target=changes.get("forecast_target", split.forecast_target),
        renewable_realized=changes.get("renewable_realized", split.renewable_realized),
        target_times=changes.get("target_times", split.target_times), split=split.split,
        teacher_dispatch=changes.get("teacher_dispatch", split.teacher_dispatch),
        oracle_first_step_objective=changes.get("oracle_first_step_objective", split.oracle_first_step_objective),
        source_path=split.source_path,
    )


def _take_split(split: JointWindowSplit, limit: int | None) -> JointWindowSplit:
    if limit is None:
        return split
    limit = int(limit)
    if limit <= 0:
        raise ValueError("smoke_limit must be positive")
    return split.take(np.arange(min(limit, len(split)), dtype=np.int64))


def _batches(
    split: JointWindowSplit,
    *,
    normalization: ExternalNormalization,
    batch_size: int,
    seed: int,
    shuffle: bool,
    normalized: bool,
) -> Iterable[ExternalBaselineBatch]:
    working = normalization.transform(split) if normalized else split
    indices = np.arange(len(working), dtype=np.int64)
    if shuffle:
        np.random.default_rng(int(seed)).shuffle(indices)
    for start in range(0, len(indices), int(batch_size)):
        yield ExternalBaselineBatch.from_split(working, indices[start : start + batch_size])


def _decoder_parameters() -> dict[str, float]:
    # Values are the frozen Standard-IES benchmark parameters used to create
    # the causal windows, not a separately tuned policy setting.
    return {
        "grid_import_capacity": 1930.5,
        "chp_electric_capacity": 450.45,
        "chp_heat_capacity": 579.15,
        "gas_boiler_capacity": 1148.2812,
        "electric_chiller_capacity": 2166.102,
        "absorption_chiller_capacity": 2166.102,
        "bess_power_capacity": 257.4,
        "bess_energy_capacity": 1029.6,
        "chp_electric_efficiency": 0.35,
        "chp_heat_efficiency": 0.45,
        "gas_boiler_efficiency": 0.9,
        "electric_chiller_cop": 3.5,
        "absorption_chiller_cop": 0.75,
        "bess_roundtrip_efficiency": 0.9,
        "chp_ramp_fraction": 0.5,
    }


def _build(method_id: str) -> nn.Module:
    if method_id == "DigitalTwins-Policy":
        return build_external_baseline(method_id, {"decoder_parameters": _decoder_parameters()})
    return build_external_baseline(method_id)


def _loss_for_batch(
    method_id: str,
    model: nn.Module,
    batch: ExternalBaselineBatch,
    normalization: ExternalNormalization,
    *,
    training: bool,
) -> Tensor:
    if method_id == "iTransformer-PTO":
        prediction = model(batch.load_history, batch.exog_history)
        return forecast_loss(prediction, batch.forecast_target, normalization)
    if method_id == "DecisionFocused-Online":
        output = model(batch)  # type: ignore[operator]
        params = {
            "surrogate_smoothing": 1.0e-3,
            "surrogate_feasibility_penalty": 1.0e-2,
            "surrogate_cost_scale": 1.0,
        }
        if training:
            return decision_focused_loss(output, batch, params, surrogate=verified_decision_focused_surrogate)
        return verified_decision_focused_surrogate(output, batch, params)
    if method_id == "DigitalTwins-Policy":
        output = model(batch)  # type: ignore[operator]
        forecast_term = torch.nn.functional.smooth_l1_loss(output.forecast_physical, batch.forecast_target)
        imitation = policy_imitation_loss(output.dispatch, batch.teacher_dispatch)
        physical_features = torch.cat((output.forecast_physical, batch.scheduler_context), dim=-1)
        feasibility = physical_feasibility_penalty(output.dispatch, physical_features, _decoder_parameters())
        return 0.1 * forecast_term + imitation + 0.1 * feasibility
    raise AssertionError(method_id)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _effective_settings(config: Mapping[str, Any], smoke_limit: int | None) -> dict[str, Any]:
    raw = config.get("training", {})
    if not isinstance(raw, Mapping):
        raise ValueError("training settings must be a mapping")
    settings = {
        "batch_size": int(raw.get("batch_size", 256)),
        "learning_rate": float(raw.get("learning_rate", 1e-3)),
        "weight_decay": float(raw.get("weight_decay", 1e-4)),
        "gradient_clip": float(raw.get("gradient_clip", 1.0)),
        "min_epochs": int(raw.get("min_epochs", 20)),
        "max_epochs": int(raw.get("max_epochs", 30)),
        "patience": int(raw.get("patience", 5)),
    }
    for name in ("batch_size", "min_epochs", "max_epochs", "patience"):
        if settings[name] <= 0:
            raise ValueError(f"training.{name} must be positive")
    if settings["max_epochs"] < settings["min_epochs"]:
        raise ValueError("max_epochs must be at least min_epochs")
    if smoke_limit is not None:
        settings["min_epochs"] = 1
        settings["max_epochs"] = int(config.get("smoke_epochs", 2))
        settings["patience"] = min(settings["patience"], 1)
    return settings


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    method_id: str,
    seed: int,
    epoch: int,
    best_validation_loss: float,
    history: list[dict[str, Any]],
    gradient_receipts: list[dict[str, Any]],
    config_hash: str,
    registry_hash: str,
    source_receipt_hash: str,
    stage: str,
    data_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": "rsc-pf-external-checkpoint-v1",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "method_id": method_id,
        "seed": int(seed),
        "epoch": int(epoch),
        "best_validation_loss": float(best_validation_loss),
        "history": history,
        "gradient_receipts": gradient_receipts,
        "config_sha256": config_hash,
        "registry_sha256": registry_hash,
        "source_receipt_sha256": source_receipt_hash,
        "stage": stage,
        "split_manifest": _split_manifest(data_paths),
        "test_set_accessed": False,
        "rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
    }


def load_external_checkpoint(path: str | Path, *, expected_method: str, expected_seed: int) -> dict[str, Any]:
    """Load a provenance-checked checkpoint without opening any test artifact."""

    _validate_method_seed(expected_method, expected_seed)
    source = Path(path)
    if _forbidden_test_path(source):
        raise ValueError("checkpoint path may not contain a sealed test-set path")
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "rsc-pf-external-checkpoint-v1":
        raise ValueError("checkpoint schema is invalid")
    if payload.get("method_id") != expected_method or int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError("checkpoint provenance does not match method/seed")
    if payload.get("test_set_accessed") is not False:
        raise ValueError("checkpoint is not test-set-free")
    for name, value in dict(payload.get("split_manifest", {})).items():
        if _forbidden_test_path(value) or name not in {"train", "validation"}:
            raise ValueError("checkpoint split manifest contains a test path")
    if not isinstance(payload.get("model_state_dict"), Mapping) or not isinstance(payload.get("optimizer_state_dict"), Mapping):
        raise ValueError("checkpoint state dictionaries are missing")
    if not np.isfinite(float(payload.get("best_validation_loss", float("nan")))):
        raise ValueError("checkpoint validation loss is not finite")
    return dict(payload)


def _run(
    method_id: str,
    seed: int,
    config: Mapping[str, Any],
    *,
    stage: str,
    smoke_limit: int | None = None,
    resume: bool = False,
) -> Path:
    _validate_method_seed(method_id, seed)
    if stage not in {"calibration", "validation"}:
        raise ValueError("stage must be calibration or validation")
    if stage == "calibration" and int(seed) != 2026:
        raise ValueError("calibration is frozen to seed 2026")
    paths = _config_paths(config)
    registry_hash, source_receipt_hash, source_receipt = _read_source_provenance(paths)
    train, validation, normalization = _load_data(paths)
    train = _take_split(train, smoke_limit)
    validation = _take_split(validation, smoke_limit)
    settings = _effective_settings(config, smoke_limit)
    config_for_hash = dict(config)
    config_for_hash.pop("resume", None)
    config_for_hash["stage"] = stage
    config_for_hash["method_id"] = method_id
    config_for_hash["seed"] = int(seed)
    config_for_hash["smoke_limit"] = smoke_limit
    config_hash = _canonical_hash(config_for_hash)
    run_dir = paths["output_root"] / stage / _METHOD_SAFE[method_id] / f"seed_{int(seed)}"
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(f"output directory exists; pass resume explicitly: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    model = _build(method_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    history: list[dict[str, Any]] = []
    gradient_receipts: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    start_epoch = 0
    last_path = run_dir / "last_checkpoint.pt"
    best_path = run_dir / "best_checkpoint.pt"
    if resume:
        payload = load_external_checkpoint(last_path, expected_method=method_id, expected_seed=seed)
        if payload.get("config_sha256") != config_hash or payload.get("registry_sha256") != registry_hash:
            raise ValueError("resume checkpoint hash does not match the current source/config")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload.get("history", []))
        gradient_receipts = list(payload.get("gradient_receipts", []))
        best_validation_loss = float(payload["best_validation_loss"])
        start_epoch = int(payload["epoch"]) + 1
    else:
        _set_seed(seed)
    stale = 0
    for epoch in range(start_epoch, settings["max_epochs"]):
        model.train()
        train_values: list[float] = []
        last_gradient: dict[str, Any] | None = None
        for batch in _batches(train, normalization=normalization, batch_size=settings["batch_size"], seed=seed + epoch, shuffle=True, normalized=method_id != "DigitalTwins-Policy"):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(method_id, model, batch, normalization, training=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {method_id} training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            last_gradient = audit_external_gradients(loss, model)
            optimizer.step()
            train_values.append(float(loss.detach().item()))
        model.eval()
        validation_values: list[float] = []
        with torch.enable_grad():
            for batch in _batches(validation, normalization=normalization, batch_size=settings["batch_size"], seed=seed, shuffle=False, normalized=method_id != "DigitalTwins-Policy"):
                value = _loss_for_batch(method_id, model, batch, normalization, training=False)
                if not torch.isfinite(value):
                    raise RuntimeError(f"non-finite {method_id} validation loss at epoch {epoch}")
                validation_values.append(float(value.detach().item()))
        train_loss = float(np.mean(train_values)) if train_values else float("nan")
        validation_loss = float(np.mean(validation_values)) if validation_values else float("nan")
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise RuntimeError("empty or non-finite training/validation history")
        if last_gradient is None:
            raise RuntimeError("no gradient receipt was produced")
        gradient_receipts.append({"epoch": epoch, **last_gradient})
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        payload = _checkpoint_payload(
            model=model, optimizer=optimizer, method_id=method_id, seed=seed, epoch=epoch,
            best_validation_loss=min(best_validation_loss, validation_loss), history=history,
            gradient_receipts=gradient_receipts, config_hash=config_hash, registry_hash=registry_hash,
            source_receipt_hash=source_receipt_hash, stage=stage, data_paths=paths,
        )
        torch.save(payload, last_path)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            payload["best_validation_loss"] = best_validation_loss
            torch.save(payload, best_path)
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= settings["min_epochs"] and stale >= settings["patience"]:
            break
    if not best_path.is_file():
        raise RuntimeError(f"{method_id}/seed_{seed} produced no best checkpoint")
    receipt = {
        "schema_version": "rsc-pf-external-training-receipt-v1",
        "stage": stage,
        "method_id": method_id,
        "seed": int(seed),
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "epochs_completed": len(history),
        "best_validation_loss": float(best_validation_loss),
        "history": history,
        "gradient_receipts": gradient_receipts,
        "source_receipt_sha256": source_receipt_hash,
        "registry_sha256": registry_hash,
        "config_sha256": config_hash,
        "split_manifest": _split_manifest(paths),
        "normalization_fit_split": "train",
        "test_set_accessed": False,
        "source_method_receipt": next(item for item in source_receipt["methods"] if item["method_id"] == method_id),
        "optimizer_role": "exact optimizer at inference" if method_id == "DecisionFocused-Online" else "none at inference",
    }
    (run_dir / "training_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return best_path


def run_external_calibration(method_id: str, seed: int, config: Mapping[str, Any]) -> Path:
    """Run the explicitly small calibration gate (seed 2026 only)."""

    return _run(method_id, seed, config, stage="calibration", smoke_limit=config.get("smoke_limit"), resume=bool(config.get("resume", False)))


def run_external_validation(method_id: str, seed: int, config: Mapping[str, Any]) -> Path:
    """Run one formal validation seed after calibration has passed."""

    _validate_method_seed(method_id, seed)
    paths = _config_paths(config)
    if not (paths["output_root"] / "calibration" / _METHOD_SAFE[method_id] / "seed_2026" / "training_receipt.json").is_file():
        raise RuntimeError(f"calibration gate is incomplete for {method_id}")
    return _run(method_id, seed, config, stage="validation", smoke_limit=config.get("smoke_limit"), resume=bool(config.get("resume", False)))


__all__ = ["load_external_checkpoint", "run_external_calibration", "run_external_validation"]
