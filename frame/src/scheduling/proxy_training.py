"""Deterministic CPU training and checkpoint management for the scheduling proxy."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .proxy_contract import FEATURE_ORDER, LABEL_ORDER, ProxyContract, contract_sha256, file_sha256
from .proxy_dataset import LabeledProxySplit, ProxyDataset, ProxyNormalizationStats, scenario_id_digest
from .proxy_model import (
    FeasibleSchedulingProxy,
    ProxyModelConfig,
    SchedulingProxy,
    build_proxy_model,
    load_model_checkpoint,
    save_model_checkpoint,
)
from .proxy_physics import feasible_proxy_loss, physics_aware_loss
from .proxy_decoder import decode_feasible_dispatch


_PROVENANCE_METADATA_FIELDS = (
    "contract_sha256", "benchmark_sha256", "source_type", "generator_version",
    "feature_order", "label_order", "selection_split", "test_split_used_for_selection",
    "seed", "train_split", "train_seed", "validation_split", "validation_seed",
    "train_scenario_id_digest", "validation_scenario_id_digest",
)


def _validate_provenance_metadata(metadata: Mapping[str, Any], artifact_name: str) -> None:
    """Require the complete, cross-artifact provenance schema."""

    missing = [name for name in _PROVENANCE_METADATA_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"{artifact_name} provenance is incomplete: missing {missing}")
    for name in ("contract_sha256", "benchmark_sha256", "train_scenario_id_digest", "validation_scenario_id_digest"):
        value = str(metadata[name]).lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{artifact_name} {name} is not a SHA-256 digest")
    if metadata["source_type"] != "pure_simulation":
        raise ValueError(f"{artifact_name} source_type mismatch")
    if not str(metadata["generator_version"]):
        raise ValueError(f"{artifact_name} generator_version is missing")
    if tuple(metadata["feature_order"]) != FEATURE_ORDER or tuple(metadata["label_order"]) != LABEL_ORDER:
        raise ValueError(f"{artifact_name} feature/label order mismatch")
    if metadata["selection_split"] != "validation" or metadata["test_split_used_for_selection"] is not False:
        raise ValueError(f"{artifact_name} selection provenance mismatch")
    if metadata["train_split"] != "train" or metadata["validation_split"] != "validation":
        raise ValueError(f"{artifact_name} split provenance mismatch")
    for name in ("seed", "train_seed", "validation_seed"):
        try:
            value = int(metadata[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{artifact_name} {name} must be an integer") from exc
        if value <= 0 or float(metadata[name]) != float(value):
            raise ValueError(f"{artifact_name} {name} must be a positive integer")


def _validate_v2_metadata(metadata: Mapping[str, Any], artifact_name: str) -> None:
    required = {
        "model_family", "decoder_schema_version", "decoder_dtype", "control_temperature",
        "decision_dim", "decision_groups", "inference_exact_lp_calls",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"{artifact_name} v2 provenance is incomplete: missing {missing}")
    if metadata["model_family"] != "feasible_scheduling_proxy_v2":
        raise ValueError(f"{artifact_name} model family is not v2")
    if metadata["decoder_schema_version"] != "horizon-reachable-feasible-v2" or metadata["decoder_dtype"] != "float64":
        raise ValueError(f"{artifact_name} decoder provenance mismatch")
    if float(metadata["control_temperature"]) != 0.25 or int(metadata["decision_dim"]) != 15:
        raise ValueError(f"{artifact_name} decoder control metadata mismatch")
    expected_groups = {"cooling": [0, 4], "chp": [4, 8], "soc": [8, 11], "renewable_pv": [11, 15]}
    if metadata["decision_groups"] != expected_groups:
        raise ValueError(f"{artifact_name} decoder decision groups mismatch")
    if int(metadata["inference_exact_lp_calls"]) != 0:
        raise ValueError(f"{artifact_name} inference_exact_lp_calls must be zero")


def _cross_check_provenance_metadata(left: Mapping[str, Any], right: Mapping[str, Any], left_name: str, right_name: str) -> None:
    for field in _PROVENANCE_METADATA_FIELDS:
        if field in {"feature_order", "label_order"}:
            if tuple(left[field]) != tuple(right[field]):
                raise ValueError(f"{left_name}/{right_name} provenance {field} mismatch")
        elif left[field] != right[field]:
            raise ValueError(f"{left_name}/{right_name} provenance {field} mismatch")


def _cross_check_v2_metadata(left: Mapping[str, Any], right: Mapping[str, Any], left_name: str, right_name: str) -> None:
    for field in (
        "model_family", "decoder_schema_version", "decoder_dtype", "control_temperature",
        "decision_dim", "decision_groups", "inference_exact_lp_calls",
    ):
        if left.get(field) != right.get(field):
            raise ValueError(f"{left_name}/{right_name} v2 decoder provenance {field} mismatch")


def _strict_epoch(value: Any, field: str, allow_integral_float: bool = False) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer epoch")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    elif allow_integral_float and isinstance(value, (float, np.floating)) and np.isfinite(float(value)) and float(value).is_integer():
        result = int(value)
    else:
        raise ValueError(f"{field} must be an integer epoch")
    if result < 0:
        raise ValueError(f"{field} must be a non-negative epoch")
    return result


def _strict_loss(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{field} must be a finite numeric loss")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative loss")
    return result


def _validate_checkpoint_history_consistency(
    checkpoint: Mapping[str, Any],
    history_payload: Mapping[str, Any],
) -> None:
    """Ensure the selected checkpoint is exactly represented by history."""

    checkpoint_epoch = _strict_epoch(checkpoint.get("epoch"), "checkpoint epoch")
    checkpoint_loss = _strict_loss(checkpoint.get("validation_loss"), "checkpoint validation_loss")
    history = history_payload.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("training history must contain a non-empty list")
    row_epochs: list[int] = []
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ValueError("training history rows must be objects")
        epoch = _strict_epoch(row.get("epoch"), f"history row {index} epoch", allow_integral_float=True)
        if epoch != index:
            raise ValueError("training history epoch values are not coherent")
        _strict_loss(row.get("validation_total"), f"history row {index} validation_total")
        row_epochs.append(epoch)
    history_best_epoch = _strict_epoch(history_payload.get("best_epoch"), "history best_epoch")
    history_best_loss = _strict_loss(history_payload.get("best_validation_loss"), "history best_validation_loss")
    if checkpoint_epoch != history_best_epoch:
        raise ValueError("checkpoint epoch does not match history best_epoch")
    if checkpoint_epoch >= len(history):
        raise ValueError("checkpoint epoch is not present in training history")
    selected_loss = _strict_loss(history[checkpoint_epoch].get("validation_total"), "selected history validation_total")
    if not np.isclose(checkpoint_loss, history_best_loss, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("checkpoint validation_loss does not match history best_validation_loss")
    if not np.isclose(checkpoint_loss, selected_loss, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("checkpoint validation_loss does not match selected history validation_total")
    best_row_loss = min(_strict_loss(row["validation_total"], "history validation_total") for row in history)
    if not np.isclose(history_best_loss, best_row_loss, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("history best_validation_loss is not the selected minimum")


@dataclass(frozen=True)
class TrainingResult:
    model: SchedulingProxy
    stats: ProxyNormalizationStats
    history: list[dict[str, float]]
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path
    history_path: Path
    normalization_path: Path
    metadata: Mapping[str, Any]


def set_deterministic_seed(seed: int) -> None:
    seed = int(seed)
    if seed <= 0:
        raise ValueError("training seed must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def _split_and_stats(
    split: LabeledProxySplit | ProxyDataset,
    stats: ProxyNormalizationStats | None,
    benchmark: Mapping[str, Any] | str | Path | None,
    contract: ProxyContract | None,
) -> tuple[LabeledProxySplit, ProxyNormalizationStats]:
    if isinstance(split, ProxyDataset):
        raw = split.split
        stats = stats or split.stats
    elif isinstance(split, LabeledProxySplit):
        raw = split
    else:
        raise TypeError("train/validation inputs must be LabeledProxySplit or ProxyDataset")
    if stats is None:
        stats = ProxyNormalizationStats.fit(
            raw,
            benchmark=benchmark,
            epsilon=float(contract.normalization.get("epsilon", 1e-6)) if contract is not None else 1e-6,
            scale_floor=float(contract.normalization.get("scale_floor", 1e-3)) if contract is not None else 1e-3,
        )
    return raw, stats


def _validate_training_provenance(
    train_raw: LabeledProxySplit,
    validation_raw: LabeledProxySplit,
    stats: ProxyNormalizationStats,
    contract: ProxyContract | None,
    benchmark: Mapping[str, Any] | str | Path | None,
) -> None:
    train_raw.validate(require_provenance=True)
    validation_raw.validate(require_provenance=True)
    if train_raw.split != "train" or validation_raw.split != "validation":
        raise ValueError("train_proxy requires train.split='train' and validation.split='validation'")
    if np.intersect1d(train_raw.scenario_ids, validation_raw.scenario_ids).size:
        raise ValueError("train and validation scenario IDs must be disjoint")
    expected_contract_hash = contract_sha256(contract) if contract is not None else train_raw.contract_sha256
    expected_benchmark_hash = file_sha256(benchmark) if isinstance(benchmark, (str, Path)) else train_raw.benchmark_sha256
    for raw in (train_raw, validation_raw):
        if raw.contract_sha256.lower() != expected_contract_hash.lower() or raw.benchmark_sha256.lower() != expected_benchmark_hash.lower():
            raise ValueError("train/validation split provenance hash mismatch")
        if raw.source_type != "pure_simulation":
            raise ValueError("training source_type must be pure_simulation")
        if contract is not None and raw.generator_version != contract.generator_version:
            raise ValueError("training generator_version does not match contract")
    if stats.fit_split != "train" or stats.train_seed != train_raw.seed:
        raise ValueError("normalization statistics do not prove the train split provenance")
    if stats.source_type != train_raw.source_type or stats.generator_version != train_raw.generator_version:
        raise ValueError("normalization source/generator does not match train split")
    if stats.benchmark_sha256.lower() != train_raw.benchmark_sha256.lower() or stats.contract_sha256.lower() != train_raw.contract_sha256.lower():
        raise ValueError("normalization hashes do not match train split")
    train_digest = scenario_id_digest(train_raw.scenario_ids)
    validation_digest = scenario_id_digest(validation_raw.scenario_ids)
    if stats.train_split not in ("", "train") or (stats.train_scenario_id_digest and stats.train_scenario_id_digest.lower() != train_digest.lower()):
        raise ValueError("normalization train scenario provenance does not match train split")
    if stats.validation_split not in ("", "validation") or (stats.validation_seed not in (0, int(validation_raw.seed))):
        raise ValueError("normalization validation provenance does not match validation split")
    if stats.validation_scenario_id_digest and stats.validation_scenario_id_digest.lower() != validation_digest.lower():
        raise ValueError("normalization validation scenario provenance does not match validation split")
    if stats.selection_split not in ("", "validation") or stats.test_split_used_for_selection not in (None, False):
        raise ValueError("normalization selection provenance is invalid")
    expected_feature_order = contract.feature_order if contract is not None else FEATURE_ORDER
    expected_label_order = contract.label_order if contract is not None else LABEL_ORDER
    if tuple(stats.feature_order) != tuple(expected_feature_order) or tuple(stats.label_order) != tuple(expected_label_order):
        raise ValueError("normalization feature order does not match the proxy contract")


def _loader(split: LabeledProxySplit, stats: ProxyNormalizationStats, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    dataset = ProxyDataset(split, stats)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle, generator=generator, num_workers=0)


def _run_epoch(
    model: SchedulingProxy | FeasibleSchedulingProxy,
    loader: DataLoader,
    parameters: Mapping[str, Any],
    stats: ProxyNormalizationStats,
    weights: Mapping[str, float],
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    v2 = isinstance(model, FeasibleSchedulingProxy)
    component_names = ("total", "coordinate", "renewable_split", "objective", "carbon", "slack") if v2 else ("total", "dispatch", "balance", "conversion", "soc", "cost", "carbon", "gas_prior")
    totals: dict[str, float] = {name: 0.0 for name in component_names}
    count = 0
    scale = torch.from_numpy(stats.label_scale.astype(np.float32))
    input_mean = torch.from_numpy(stats.input_mean.astype(np.float32))
    input_scale = torch.from_numpy(stats.input_scale.astype(np.float32))
    for batch in loader:
        inputs = batch["inputs"].to("cpu")
        raw_inputs = batch["raw_inputs"].to("cpu")
        target = batch["target"].to("cpu")
        raw_target = batch["raw_target"].to("cpu")
        teacher_cost = batch["teacher_cost"].to("cpu")
        teacher_carbon = batch["teacher_carbon"].to("cpu")
        gas_mask = batch["gas_prior_mask"].to("cpu")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            if v2:
                # The primary v2 scheduler path masks the synthetic gas prior
                # before both normalization and decoding.  It remains context
                # only and never enters a physical equation.
                physical_inputs = raw_inputs.clone()
                physical_inputs[..., 3] = 0.0
                normalized_inputs = (physical_inputs - input_mean) / input_scale
                logits = model.forward_logits(normalized_inputs)
                prediction = decode_feasible_dispatch(logits, physical_inputs, parameters)
                carbon_price = physical_inputs[..., 8].mean(dim=1)
                teacher_objective = teacher_cost + carbon_price * teacher_carbon
                total, components = feasible_proxy_loss(
                    prediction,
                    raw_target,
                    physical_inputs,
                    parameters,
                    teacher_objective=teacher_objective,
                    teacher_carbon=teacher_carbon,
                    weights=weights,
                )
            else:
                prediction = model(inputs)
                total, components = physics_aware_loss(
                    prediction,
                    target,
                    raw_inputs,
                    parameters,
                    teacher_cost=teacher_cost,
                    teacher_carbon=teacher_carbon,
                    gas_prior_mask=gas_mask,
                    label_scale=scale,
                    loss_weights=weights,
                )
            if training:
                total.backward()
                clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
                optimizer.step()
        batch_size = int(inputs.shape[0])
        count += batch_size
        for name in totals:
            totals[name] += float(components[name].detach().cpu()) * batch_size
    if count <= 0:
        raise ValueError("cannot train on an empty split")
    return {name: value / count for name, value in totals.items()}


def train_proxy(
    train_split: LabeledProxySplit | ProxyDataset,
    validation_split: LabeledProxySplit | ProxyDataset,
    output_dir: str | Path,
    contract: ProxyContract | None = None,
    benchmark: Mapping[str, Any] | str | Path | None = None,
    stats: ProxyNormalizationStats | None = None,
    max_epochs: int | None = None,
    patience: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
    learning_rate: float | None = None,
    weight_decay: float | None = None,
) -> TrainingResult:
    """Train with validation-only checkpoint selection and reload the best model."""

    train_raw, stats = _split_and_stats(train_split, stats, benchmark, contract)
    validation_raw, _ = _split_and_stats(validation_split, stats, benchmark, contract)
    _validate_training_provenance(train_raw, validation_raw, stats, contract, benchmark)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if contract is not None:
        training_cfg = contract.training
        max_epochs = int(max_epochs if max_epochs is not None else training_cfg["max_epochs"])
        patience = int(patience if patience is not None else training_cfg["patience"])
        batch_size = int(batch_size if batch_size is not None else training_cfg["batch_size"])
        seed = int(seed if seed is not None else training_cfg["seed"])
        learning_rate = float(learning_rate if learning_rate is not None else training_cfg["learning_rate"])
        weight_decay = float(weight_decay if weight_decay is not None else training_cfg["weight_decay"])
        gradient_clip_norm = float(training_cfg["gradient_clip_norm"])
        weights = dict(contract.loss_weights)
    else:
        max_epochs = int(max_epochs if max_epochs is not None else 30)
        patience = int(patience if patience is not None else 5)
        batch_size = int(batch_size if batch_size is not None else 32)
        seed = int(seed if seed is not None else 2026)
        learning_rate = float(learning_rate if learning_rate is not None else 1e-3)
        weight_decay = float(weight_decay if weight_decay is not None else 1e-5)
        gradient_clip_norm = 1.0
        weights = {"dispatch": 1.0, "balance": 0.2, "conversion": 0.1, "soc": 0.1, "cost": 0.05, "carbon": 0.05, "gas_prior": 0.05}
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0:
        raise ValueError("max_epochs, patience and batch_size must be positive")
    train_digest = scenario_id_digest(train_raw.scenario_ids)
    validation_digest = scenario_id_digest(validation_raw.scenario_ids)
    if stats.seed not in (0, int(seed)):
        raise ValueError("normalization training seed provenance does not match requested seed")
    stats = replace(
        stats,
        selection_split="validation",
        test_split_used_for_selection=False,
        seed=int(seed),
        train_split="train",
        validation_split="validation",
        validation_seed=int(validation_raw.seed),
        train_scenario_id_digest=train_digest,
        validation_scenario_id_digest=validation_digest,
    )
    set_deterministic_seed(seed)
    model = build_proxy_model(contract)
    model.to("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    parameters: Mapping[str, Any]
    if benchmark is None:
        # The loss needs only the scalar conversion/cost parameters.  The
        # benchmark is required for the normal pipeline; this fallback keeps
        # small unit tests self-contained when they pass a hand-written map.
        parameters = {}
    else:
        from .synthetic_scenarios import load_benchmark
        benchmark_data = load_benchmark(benchmark) if isinstance(benchmark, (str, Path)) else benchmark
        parameters = benchmark_data["values"] if isinstance(benchmark_data, Mapping) and "values" in benchmark_data else benchmark_data
    train_loader = _loader(train_raw, stats, batch_size, seed, shuffle=True)
    validation_loader = _loader(validation_raw, stats, batch_size, seed + 1, shuffle=False)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"
    normalization_path = output_dir / "normalization_stats.npz"
    stats.save(normalization_path)
    best_loss = float("inf")
    best_epoch = -1
    wait = 0
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        train_metrics = _run_epoch(model, train_loader, parameters, stats, weights, optimizer, gradient_clip_norm)
        with torch.no_grad():
            val_metrics = _run_epoch(model, validation_loader, parameters, stats, weights, None, gradient_clip_norm)
        row = {"epoch": float(epoch), **{f"train_{name}": value for name, value in train_metrics.items()}, **{f"validation_{name}": value for name, value in val_metrics.items()}}
        history.append(row)
        val_loss = float(val_metrics["total"])
        if val_loss < best_loss - 1e-12:
            best_loss = val_loss
            best_epoch = epoch
            wait = 0
            metadata = {
                "selection_split": "validation",
                "test_split_used_for_selection": False,
                "seed": seed,
                "contract_sha256": train_raw.contract_sha256,
                "benchmark_sha256": train_raw.benchmark_sha256,
                "feature_order": list(stats.feature_order),
                "label_order": list(stats.label_order),
                "source_type": train_raw.source_type,
                "generator_version": train_raw.generator_version,
                "train_split": train_raw.split,
                "train_seed": int(train_raw.seed),
                "validation_split": validation_raw.split,
                "validation_seed": int(validation_raw.seed),
                "train_scenario_id_digest": scenario_id_digest(train_raw.scenario_ids),
                "validation_scenario_id_digest": scenario_id_digest(validation_raw.scenario_ids),
            }
            if isinstance(model, FeasibleSchedulingProxy):
                metadata.update({
                    "model_family": "feasible_scheduling_proxy_v2",
                    "decoder_schema_version": "horizon-reachable-feasible-v2",
                    "decoder_dtype": "float64",
                    "control_temperature": 0.25,
                    "decision_dim": 15,
                    "decision_groups": {"cooling": [0, 4], "chp": [4, 8], "soc": [8, 11], "renewable_pv": [11, 15]},
                    "inference_exact_lp_calls": 0,
                })
            save_model_checkpoint(checkpoint_path, model, optimizer=None, epoch=epoch, validation_loss=best_loss, metadata=metadata)
        else:
            wait += 1
            if wait >= patience:
                break
    if best_epoch < 0 or not checkpoint_path.exists():
        raise RuntimeError("training did not produce a validation checkpoint")
    history_payload = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selection_split": "validation",
        "test_split_used_for_selection": False,
        "seed": int(seed),
        "contract_sha256": train_raw.contract_sha256,
        "benchmark_sha256": train_raw.benchmark_sha256,
        "source_type": train_raw.source_type,
        "generator_version": train_raw.generator_version,
        "feature_order": list(stats.feature_order),
        "label_order": list(stats.label_order),
        "train_split": train_raw.split,
        "train_seed": int(train_raw.seed),
        "validation_split": validation_raw.split,
        "validation_seed": int(validation_raw.seed),
        "train_scenario_id_digest": scenario_id_digest(train_raw.scenario_ids),
        "validation_scenario_id_digest": scenario_id_digest(validation_raw.scenario_ids),
    }
    if isinstance(model, FeasibleSchedulingProxy):
        history_payload.update({
            "model_family": "feasible_scheduling_proxy_v2",
            "decoder_schema_version": "horizon-reachable-feasible-v2",
            "decoder_dtype": "float64",
            "control_temperature": 0.25,
            "decision_dim": 15,
            "decision_groups": {"cooling": [0, 4], "chp": [4, 8], "soc": [8, 11], "renewable_pv": [11, 15]},
            "inference_exact_lp_calls": 0,
        })
    history_path.write_text(json.dumps(history_payload, indent=2, sort_keys=True), encoding="utf-8")
    model, checkpoint = load_model_checkpoint(checkpoint_path, expected_contract=contract)
    return TrainingResult(
        model=model,
        stats=stats,
        history=history,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        normalization_path=normalization_path,
        metadata=dict(checkpoint.get("metadata", {})),
    )


def load_trained_proxy(
    output_dir: str | Path,
    contract: ProxyContract | None = None,
    benchmark_path: str | Path | None = None,
    expected_train_seed: int | None = None,
) -> tuple[SchedulingProxy, ProxyNormalizationStats, Mapping[str, Any]]:
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.json"
    stats_path = output_dir / "normalization_stats.npz"
    # Derive external expectations before loading stats.  A supplied contract
    # carries the frozen benchmark digest, so benchmark_path is optional; if
    # both are supplied they must agree.
    expected_contract_hash = contract_sha256(contract) if contract is not None else None
    contract_benchmark_hash = contract.benchmark_sha256.lower() if contract is not None else None
    supplied_benchmark_hash = file_sha256(benchmark_path) if benchmark_path is not None else None
    if contract_benchmark_hash is not None and supplied_benchmark_hash is not None and contract_benchmark_hash != supplied_benchmark_hash.lower():
        raise ValueError("benchmark path does not match contract benchmark SHA-256")
    expected_benchmark_hash = supplied_benchmark_hash or contract_benchmark_hash
    model, payload = load_model_checkpoint(checkpoint_path, expected_contract=contract)
    stats = ProxyNormalizationStats.load(
        stats_path,
        expected_benchmark_sha256=expected_benchmark_hash,
        expected_contract_sha256=expected_contract_hash,
        expected_fit_split="train",
        expected_train_seed=expected_train_seed,
        expected_source_type=contract.source_type if contract is not None else "pure_simulation",
        expected_generator_version=contract.generator_version if contract is not None else "synthetic-scheduling-v1",
    )
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata is missing")
    _validate_provenance_metadata(metadata, "checkpoint")
    if metadata.get("model_family") == "feasible_scheduling_proxy_v2":
        _validate_v2_metadata(metadata, "checkpoint")
    try:
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("training history is missing or corrupted") from exc
    if not isinstance(history_payload, Mapping):
        raise ValueError("training history is corrupted")
    history_required = {"history", "best_epoch", "best_validation_loss", *_PROVENANCE_METADATA_FIELDS}
    if not history_required.issubset(history_payload):
        raise ValueError("training history provenance is incomplete")
    _validate_provenance_metadata(history_payload, "history")
    if history_payload.get("model_family") == "feasible_scheduling_proxy_v2":
        _validate_v2_metadata(history_payload, "history")
    _validate_checkpoint_history_consistency(payload, history_payload)
    _cross_check_provenance_metadata(metadata, history_payload, "checkpoint", "history")
    if metadata.get("model_family") == "feasible_scheduling_proxy_v2" or history_payload.get("model_family") == "feasible_scheduling_proxy_v2":
        _cross_check_v2_metadata(metadata, history_payload, "checkpoint", "history")

    normalization_metadata = {
        "contract_sha256": stats.contract_sha256,
        "benchmark_sha256": stats.benchmark_sha256,
        "source_type": stats.source_type,
        "generator_version": stats.generator_version,
        "feature_order": list(stats.feature_order),
        "label_order": list(stats.label_order),
        "selection_split": stats.selection_split,
        "test_split_used_for_selection": stats.test_split_used_for_selection,
        "seed": stats.seed,
        "train_split": stats.train_split,
        "train_seed": stats.train_seed,
        "validation_split": stats.validation_split,
        "validation_seed": stats.validation_seed,
        "train_scenario_id_digest": stats.train_scenario_id_digest,
        "validation_scenario_id_digest": stats.validation_scenario_id_digest,
    }
    _validate_provenance_metadata(normalization_metadata, "normalization")
    _cross_check_provenance_metadata(metadata, normalization_metadata, "checkpoint", "normalization")
    _cross_check_provenance_metadata(history_payload, normalization_metadata, "history", "normalization")
    if expected_contract_hash is not None and str(metadata["contract_sha256"]).lower() != expected_contract_hash.lower():
        raise ValueError("checkpoint contract SHA-256 mismatch")
    if expected_benchmark_hash is not None and str(metadata["benchmark_sha256"]).lower() != expected_benchmark_hash.lower():
        raise ValueError("checkpoint benchmark SHA-256 mismatch")
    if expected_train_seed is not None and stats.train_seed != int(expected_train_seed):
        raise ValueError("normalization train seed mismatch")
    if contract is not None:
        if metadata["source_type"] != contract.source_type or metadata["generator_version"] != contract.generator_version:
            raise ValueError("checkpoint contract source/generator mismatch")
        if tuple(metadata["feature_order"]) != tuple(contract.feature_order) or tuple(metadata["label_order"]) != tuple(contract.label_order):
            raise ValueError("checkpoint contract feature/label order mismatch")
    return model, stats, payload


# Aliases for predictable discoverability.
fit_proxy = train_proxy
train_scheduling_proxy = train_proxy
train_model = train_proxy
load_checkpointed_proxy = load_trained_proxy


__all__ = [
    "TrainingResult",
    "set_deterministic_seed",
    "train_proxy",
    "fit_proxy",
    "train_scheduling_proxy",
    "train_model",
    "load_trained_proxy",
    "load_checkpointed_proxy",
]
