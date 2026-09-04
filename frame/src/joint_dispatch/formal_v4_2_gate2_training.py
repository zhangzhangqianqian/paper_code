"""Frozen data manifests and training batches for formal-v4.2 Gate 2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .formal_v4_2_artifacts import canonical_sha256, sha256_file
from .formal_v4_2_checkpoint import save_training_checkpoint
from .formal_v4_2_contract import FormalV42Contract
from .formal_v4_2_data import NormalizationReceiptV42, apply_normalization
from .formal_v4_2_training import StageBudgetV42, run_stage_j, run_stage_p, run_stage_s, seed_everything
from .formal_v4_data import FormalV4WindowSplit
from .formal_v4_models import DirectPolicyModel, RSCPFModel
from .formal_v4_objective import settle_formal_v4_four_hour
from ..models import Scheme2RModel
from ..scheduling.dispatch_lp import DispatchInputs, solve_dispatch_lp
from ..scheduling.dispatch_schema import VARIABLES


_WINDOW_FIELDS = (
    "load_history", "exog_history", "renewable_history", "device_history",
    "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
    "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
    "target_times", "trajectory_ids", "state_hashes",
)


@dataclass(frozen=True)
class Gate2DataBundle:
    train: FormalV4WindowSplit
    calibration: FormalV4WindowSplit
    evaluation: FormalV4WindowSplit
    normalization: NormalizationReceiptV42
    train_manifest_sha256: str
    calibration_manifest_sha256: str
    evaluation_manifest_sha256: str
    normalization_sha256: str


@dataclass(frozen=True)
class Gate2BatchPart:
    batch: Mapping[str, torch.Tensor]
    effective_batch_index: int
    accumulation_weight: float
    indices: np.ndarray


@dataclass(frozen=True)
class TrainedMethodArtifact:
    """One in-memory trained method plus immutable checkpoint evidence."""

    method_id: str
    seed: int
    model: nn.Module
    checkpoint_path: Path
    checkpoint_sha256: str
    training_receipt: Mapping[str, Any]
    stage_s_parent_sha256: str = ""
    decision_forecaster_gradient_norm: float = 0.0


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def load_window_split(path: str | Path) -> FormalV4WindowSplit:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        split = str(np.asarray(payload["split"]).item())
        values = {name: payload[name] for name in _WINDOW_FIELDS}
    return FormalV4WindowSplit(**values, split=split)


def _load_normalization(path: Path) -> NormalizationReceiptV42:
    payload = _load_json(path)
    if payload.get("schema") != "formal-v4.2-normalization-receipt-v1":
        raise ValueError("Gate 2 normalization schema is invalid")
    return NormalizationReceiptV42(
        train_years=tuple(int(value) for value in payload["train_years"]),
        field_mean={key: np.asarray(value, dtype=np.float32) for key, value in payload["field_mean"].items()},
        field_scale={key: np.asarray(value, dtype=np.float32) for key, value in payload["field_scale"].items()},
        zero_scale_mask={key: np.asarray(value, dtype=bool) for key, value in payload["zero_scale_mask"].items()},
        train_data_sha256=str(payload["train_data_sha256"]),
        receipt_sha256=str(payload["receipt_sha256"]),
    )


def _subset(split: FormalV4WindowSplit, indices: np.ndarray) -> FormalV4WindowSplit:
    return replace(split, **{name: getattr(split, name)[indices] for name in _WINDOW_FIELDS})


def _manifest_hash(split: FormalV4WindowSplit, scope: str) -> str:
    return canonical_sha256({
        "scope": scope,
        "count": len(split),
        "target_times": split.target_times.astype("datetime64[ns]").astype(np.int64).tolist(),
        "trajectory_ids": split.trajectory_ids.tolist(),
        "state_hashes": split.state_hashes.tolist(),
    })


def _years(split: FormalV4WindowSplit) -> tuple[int, ...]:
    values = np.datetime_as_string(split.target_times, unit="Y").astype(np.int64)
    return tuple(int(value) for value in np.unique(values))


def load_gate2_data(run_root: str | Path, contract: FormalV42Contract) -> Gate2DataBundle:
    """Load the Gate 1 materialization without sampling formal train/evaluation."""

    contract.validate()
    root = Path(run_root).resolve()
    transition = _load_json(root / "protocol" / "GATE1_TRANSITION.json")
    if transition.get("authorized_gate2") is not True:
        raise PermissionError("Gate 2 requires an authorized Gate 1 transition")
    if transition.get("contract_sha256") != contract.contract_sha256:
        raise PermissionError("Gate 1 contract lineage differs from Gate 2")
    gate1 = root / "gate1"
    evidence_path = gate1 / "GATE1_EVIDENCE.json"
    evidence = _load_json(evidence_path)
    if transition.get("gate1_evidence_sha256") != sha256_file(evidence_path):
        raise PermissionError("Gate 1 evidence hash mismatch")
    if evidence.get("evaluation_year_accessed") is not False:
        raise PermissionError("Gate 1 accessed the evaluation year")
    train_path = gate1 / "TRAIN_WINDOWS.npz"
    selection_path = gate1 / "SELECTION_WINDOWS.npz"
    normalization_path = gate1 / "NORMALIZATION.json"
    if evidence.get("train_windows_sha256") != sha256_file(train_path):
        raise PermissionError("Gate 1 train windows hash mismatch")
    if evidence.get("selection_windows_sha256") != sha256_file(selection_path):
        raise PermissionError("Gate 1 selection windows hash mismatch")
    train = load_window_split(train_path)
    selection = load_window_split(selection_path)
    if _years(train) != contract.train_years:
        raise PermissionError("Gate 2 train manifest does not cover exactly 2015-2018")
    if _years(selection) != (contract.selection_year,):
        raise PermissionError("Gate 2 evaluation manifest is not exactly 2019")
    if len(selection) > 1 and not np.all(np.diff(selection.target_times) > np.timedelta64(0, "s")):
        raise ValueError("Gate 2 evaluation must be chronological")
    origin_path = gate1 / "GATE1_ORIGIN_MANIFEST.json"
    origin_payload = _load_json(origin_path)
    origin_indices = np.asarray(origin_payload["origin_indices"], dtype=np.int64)
    if len(origin_indices) != int(contract.gate2_budget["calibration_origin_count"]):
        raise ValueError("Gate 1 calibration origin count differs from the frozen budget")
    if origin_indices.min(initial=0) < 0 or origin_indices.max(initial=-1) >= len(selection):
        raise ValueError("Gate 1 calibration origin index is outside selection data")
    calibration = _subset(selection, origin_indices)
    normalization = _load_normalization(normalization_path)
    if normalization.train_years != contract.train_years:
        raise PermissionError("Gate 2 normalization is not train-only")
    return Gate2DataBundle(
        train=train,
        calibration=calibration,
        evaluation=selection,
        normalization=normalization,
        train_manifest_sha256=_manifest_hash(train, "all_eligible_2015_2018"),
        calibration_manifest_sha256=sha256_file(origin_path),
        evaluation_manifest_sha256=_manifest_hash(selection, "all_eligible_2019_chronology"),
        normalization_sha256=sha256_file(normalization_path),
    )


def _batch(split: FormalV4WindowSplit, normalization: NormalizationReceiptV42, indices: np.ndarray) -> dict[str, torch.Tensor]:
    normalized = apply_normalization(_subset(split, indices), normalization)
    selected = _subset(split, indices)
    return {
        "load_history": torch.from_numpy(normalized.load_history),
        "exog_history": torch.from_numpy(normalized.exog_history),
        "device_history": torch.from_numpy(normalized.device_history),
        "activity_history": torch.from_numpy(normalized.activity_history),
        "scheduler_context": torch.from_numpy(normalized.scheduler_context),
        "previous_chp": torch.as_tensor(selected.previous_chp, dtype=torch.float64),
        "initial_soc": torch.as_tensor(selected.initial_soc, dtype=torch.float64),
        "target_normalized": torch.from_numpy(normalized.target_normalized),
        "target_physical": torch.as_tensor(selected.forecast_target, dtype=torch.float64),
        "renewable_forecast": torch.as_tensor(selected.renewable_forecast, dtype=torch.float64),
        "realized_renewables": torch.as_tensor(selected.renewable_realized, dtype=torch.float64),
        "prices_and_weights": torch.as_tensor(selected.prices_and_weights, dtype=torch.float64),
    }


def iter_gate2_batches(
    split: FormalV4WindowSplit,
    normalization: NormalizationReceiptV42,
    *,
    effective_batch_size: int = 64,
    micro_batch_size: int | None = None,
) -> Iterator[Gate2BatchPart]:
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    part_size = effective_batch_size if micro_batch_size is None else int(micro_batch_size)
    if part_size <= 0 or effective_batch_size % part_size != 0:
        raise ValueError("micro_batch_size must be a positive divisor of effective_batch_size")
    effective_index = 0
    for start in range(0, len(split), effective_batch_size):
        stop = min(start + effective_batch_size, len(split))
        group = np.arange(start, stop, dtype=np.int64)
        parts = [group[offset:offset + part_size] for offset in range(0, len(group), part_size)]
        for indices in parts:
            yield Gate2BatchPart(
                batch=_batch(split, normalization, indices),
                effective_batch_index=effective_index,
                accumulation_weight=float(len(indices) / len(group)),
                indices=indices,
            )
        effective_index += 1


def full_gate2_batches(
    split: FormalV4WindowSplit,
    normalization: NormalizationReceiptV42,
    *,
    batch_size: int = 64,
) -> list[Mapping[str, torch.Tensor]]:
    return [
        part.batch for part in iter_gate2_batches(
            split, normalization, effective_batch_size=batch_size,
        )
    ]


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=np.float64).mean(axis=(0, 1)).astype(np.float32)
    scale = np.asarray(values, dtype=np.float64).std(axis=(0, 1)).astype(np.float32)
    return mean, np.where(scale < 1.0e-6, 1.0, scale).astype(np.float32)


def build_rsc_model(data: Gate2DataBundle, parameters: Mapping[str, Any]) -> RSCPFModel:
    """Build the frozen state-conditioned model from train-only statistics."""

    split = data.train
    normalization = data.normalization
    renew_mean, renew_scale = _stats(split.renewable_forecast)
    soc_mean, soc_scale = _stats(np.repeat(split.initial_soc[:, None, :], 4, axis=1))
    physical_mean = np.concatenate((
        normalization.field_mean["load"], renew_mean,
        normalization.field_mean["scheduler"], soc_mean,
    ))
    physical_scale = np.concatenate((
        normalization.field_scale["load"], renew_scale,
        normalization.field_scale["scheduler"], soc_scale,
    ))
    return RSCPFModel(
        decoder_parameters=parameters,
        dropout=0.0,
        task_mean=torch.as_tensor(normalization.field_mean["load"]),
        task_scale=torch.as_tensor(normalization.field_scale["load"]),
        physical_feature_mean=torch.as_tensor(physical_mean),
        physical_feature_scale=torch.as_tensor(physical_scale),
        previous_chp_mean=float(np.mean(split.previous_chp)),
        previous_chp_scale=max(float(np.std(split.previous_chp)), 1.0),
    )


def build_direct_policy_model(data: Gate2DataBundle, parameters: Mapping[str, Any]) -> DirectPolicyModel:
    base = build_rsc_model(data, parameters)
    return DirectPolicyModel(
        decoder_parameters=parameters,
        dropout=0.0,
        task_mean=base.core.task_mean.detach().clone(),
        task_scale=base.core.task_scale.detach().clone(),
        physical_feature_mean=base.core.physical_feature_mean.detach().clone(),
        physical_feature_scale=base.core.physical_feature_scale.detach().clone(),
        previous_chp_mean=base.previous_chp_mean.detach().clone(),
        previous_chp_scale=base.previous_chp_scale.detach().clone(),
    )


def _budget(freeze: Mapping[str, Any], override: StageBudgetV42 | None = None) -> StageBudgetV42:
    if override is not None:
        return override
    multiplier = float(
        freeze.get("selected_candidate_value", freeze.get("candidate_value", 1.0))
        if freeze else 1.0
    )
    selected = freeze.get("selected_candidate") if freeze else None
    if isinstance(selected, Mapping):
        multiplier = float(selected.get("candidate_value", multiplier))
    return StageBudgetV42(forecaster_lr=1.0e-5 * multiplier)


def _lineage(data: Gate2DataBundle, freeze: Mapping[str, Any]) -> dict[str, str]:
    contract_hash = str(freeze.get("contract_sha256", ""))
    source_hash = str(freeze.get("source_manifest_sha256", ""))
    if len(contract_hash) != 64:
        contract_hash = "0" * 64
    if len(source_hash) != 64:
        source_hash = "0" * 64
    return {
        "contract_sha256": contract_hash,
        "source_manifest_sha256": source_hash,
        "data_sha256": data.train_manifest_sha256,
        "normalization_sha256": data.normalization_sha256,
    }


def _with_teacher(
    data: Gate2DataBundle,
    teacher_dispatch: np.ndarray,
    *,
    batch_size: int,
) -> list[Mapping[str, torch.Tensor]]:
    teacher = np.asarray(teacher_dispatch, dtype=np.float64)
    if teacher.shape != (len(data.train), 4, len(VARIABLES)) or not np.isfinite(teacher).all():
        raise ValueError("teacher_dispatch must have shape [N,4,21] and be finite")
    batches = full_gate2_batches(data.train, data.normalization, batch_size=batch_size)
    for index, batch in enumerate(batches):
        start = index * batch_size
        batch["teacher_dispatch"] = torch.as_tensor(teacher[start:start + len(batch["load_history"])], dtype=torch.float64)
    return batches


def build_same_information_teacher_dispatch(
    model: RSCPFModel,
    data: Gate2DataBundle,
    parameters: Mapping[str, Any],
    *,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the full train teacher from causal Stage-P forecasts only."""

    forecasts: list[np.ndarray] = []
    model.eval()
    for batch in full_gate2_batches(data.train, data.normalization, batch_size=batch_size):
        with torch.no_grad():
            output = model(**{name: batch[name] for name in (
                "load_history", "exog_history", "device_history", "activity_history",
                "scheduler_context", "previous_chp",
            )})
        forecasts.append(output.forecast_physical.detach().cpu().numpy())
    predicted = np.concatenate(forecasts, axis=0)
    dispatch = np.empty((len(data.train), 4, len(VARIABLES)), dtype=np.float64)
    objective = np.empty(len(data.train), dtype=np.float64)
    for index in range(len(data.train)):
        context = dict(parameters)
        prices = np.asarray(data.train.prices_and_weights[index], dtype=np.float64)
        context["grid_energy_price"] = prices[:, 0]
        context["gas_energy_price"] = prices[:, 1]
        context["carbon_price"] = prices[:, 2]
        solved = solve_dispatch_lp(DispatchInputs(
            demand=predicted[index, :, :3],
            pv_available=data.train.renewable_forecast[index, :, 0],
            wt_available=data.train.renewable_forecast[index, :, 1],
            parameters=context,
            initial_soc=float(data.train.initial_soc[index, 0]),
            previous_chp=float(data.train.previous_chp[index, 0]),
        ))
        if not solved.success:
            raise RuntimeError(f"Gate 2 teacher LP failed at train row {index}: {solved.message}")
        dispatch[index] = np.maximum(
            np.column_stack([solved.values[name] for name in VARIABLES]), 0.0,
        )
        objective[index] = float(solved.objective)
    return dispatch, objective


def _write_training_receipt(directory: Path, payload: Mapping[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "TRAINING_RECEIPT.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite training receipt: {path}")
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _save_artifact(
    *,
    method_id: str,
    seed: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    output_dir: Path,
    lineage: Mapping[str, Any],
    receipt: Mapping[str, Any],
    stage_s_parent_sha256: str = "",
    decision_forecaster_gradient_norm: float = 0.0,
) -> TrainedMethodArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = save_training_checkpoint(
        output_dir / "CHECKPOINT.pt",
        model=model,
        optimizer=optimizer,
        epoch=max(int(epochs) - 1, 0),
        early_stopping={"stage": "gate2", "epochs": int(epochs)},
        lineage=lineage,
    )
    payload = {
        "schema": "formal-v4.2-gate2-training-v1",
        "method_id": method_id,
        "seed": int(seed),
        "epochs": int(epochs),
        "checkpoint_sha256": checkpoint.model_sha256,
        "stage_s_parent_sha256": stage_s_parent_sha256,
        "decision_forecaster_gradient_norm": float(decision_forecaster_gradient_norm),
        **dict(receipt),
    }
    _write_training_receipt(output_dir, payload)
    return TrainedMethodArtifact(
        method_id, int(seed), model, checkpoint.path, checkpoint.model_sha256,
        payload, stage_s_parent_sha256, float(decision_forecaster_gradient_norm),
    )


def train_rsc_family(
    seed: int,
    data: Gate2DataBundle,
    freeze: Mapping[str, Any],
    parameters: Mapping[str, Any],
    output_dir: str | Path,
    *,
    budget: StageBudgetV42 | None = None,
    teacher_dispatch: np.ndarray | None = None,
) -> Mapping[str, TrainedMethodArtifact]:
    """Train the paired RSC rows and their shared State-Conditioned PTO forecast."""

    started = time.perf_counter()
    seed_everything(seed)
    active_budget = _budget(freeze, budget)
    batch_size = 64
    base_batches = full_gate2_batches(data.train, data.normalization, batch_size=batch_size)
    model = build_rsc_model(data, parameters)
    stage_p = run_stage_p(model, {"train": base_batches}, budget=active_budget, seed=seed)
    stage_p_frozen = deepcopy(stage_p.model)
    if teacher_dispatch is None:
        teacher_dispatch, teacher_objective = build_same_information_teacher_dispatch(
            stage_p.model, data, parameters, batch_size=batch_size,
        )
    else:
        teacher_objective = np.ones(len(data.train), dtype=np.float64)
    shared_dir = Path(output_dir) / "_shared" / str(seed)
    shared_dir.mkdir(parents=True, exist_ok=True)
    teacher_path = shared_dir / "TEACHER.npz"
    if teacher_path.exists():
        raise FileExistsError(f"refusing to overwrite teacher: {teacher_path}")
    np.savez_compressed(teacher_path, dispatch=np.asarray(teacher_dispatch), objective=np.asarray(teacher_objective))
    train_batches = _with_teacher(data, teacher_dispatch, batch_size=batch_size)
    stage_s = run_stage_s(stage_p.model, {"train": train_batches}, budget=active_budget, seed=seed)
    root = Path(output_dir)
    lineage = _lineage(data, freeze)
    # Persisting the parent before either branch makes its byte identity auditable.
    parent_checkpoint = save_training_checkpoint(
        shared_dir / "STAGE_S.pt",
        model=stage_s.model,
        optimizer=stage_s.optimizer,
        epoch=max(stage_s.epochs - 1, 0),
        early_stopping={"stage": "S", "seed": int(seed)},
        lineage=lineage,
    )
    c_ref = max(float(np.median(teacher_objective)), 1.0)
    artifacts: dict[str, TrainedMethodArtifact] = {}
    for method_id, mode in (("RSC-PF", "joint"), ("Decoupled-RSC-PF", "decoupled")):
        result = run_stage_j(
            deepcopy(stage_s.model), {"train": train_batches}, mode=mode,
            budget=active_budget, c_ref=c_ref, parameters=parameters, seed=seed,
        )
        artifacts[method_id] = _save_artifact(
            method_id=method_id, seed=seed, model=result.model, optimizer=result.optimizer,
            epochs=result.epochs, output_dir=root / method_id / str(seed), lineage={
                **lineage, "parent_checkpoint_sha256": parent_checkpoint.model_sha256,
            }, stage_s_parent_sha256=parent_checkpoint.model_sha256,
            decision_forecaster_gradient_norm=result.decision_forecaster_gradient_norm,
            receipt={
                "optimizer_steps": result.optimizer_steps,
                "forecast_loss_applicable": True,
                "stage_order": ["P", "teacher", "S", "clone", "J"],
                "loss_history": list(result.loss_history),
                "runtime_seconds": time.perf_counter() - started,
                "test_set_accessed": False,
            },
        )
    # This PTO row intentionally uses the Stage-P state-conditioned predictor,
    # not either Stage-J branch.
    artifacts["State-Conditioned-PTO"] = _save_artifact(
        method_id="State-Conditioned-PTO", seed=seed, model=stage_p_frozen,
        optimizer=stage_p.optimizer, epochs=stage_p.epochs,
        output_dir=root / "State-Conditioned-PTO" / str(seed), lineage=lineage,
        receipt={
            "optimizer_steps": stage_p.optimizer_steps,
            "forecast_loss_applicable": True,
            "stage_order": ["P"],
            "loss_history": list(stage_p.loss_history),
            "runtime_seconds": time.perf_counter() - started,
            "test_set_accessed": False,
        },
    )
    return artifacts


def train_direct_policy(
    seed: int,
    data: Gate2DataBundle,
    freeze: Mapping[str, Any],
    parameters: Mapping[str, Any],
    output_dir: str | Path,
    *,
    budget: StageBudgetV42 | None = None,
    teacher_dispatch: np.ndarray,
) -> TrainedMethodArtifact:
    """Train the decision-only comparator without a forecast head or loss."""

    started = time.perf_counter()
    seed_everything(seed)
    active_budget = _budget(freeze, budget)
    model = build_direct_policy_model(data, parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_budget.scheduler_lr, weight_decay=active_budget.weight_decay)
    batches = _with_teacher(data, teacher_dispatch, batch_size=64)
    history: list[float] = []
    steps = 0
    for _epoch in range(active_budget.max_epochs):
        losses: list[float] = []
        model.train()
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            output = model(**{name: batch[name] for name in (
                "load_history", "exog_history", "device_history", "activity_history",
                "scheduler_context", "previous_chp",
            )})
            teacher = batch["teacher_dispatch"].to(output.dispatch)
            scale = teacher.detach().abs().mean(dim=(0, 1)).clamp_min(1.0)
            imitation = F.smooth_l1_loss(output.dispatch / scale, teacher / scale)
            settled = settle_formal_v4_four_hour(
                output.dispatch, batch["target_physical"].to(output.dispatch)[..., :3],
                batch["realized_renewables"].to(output.dispatch), batch["initial_soc"].to(output.dispatch),
                batch["previous_chp"].to(output.dispatch), parameters,
            )
            decision = settled.per_step_penalized_objective.mean() + settled.constraint_penalty.mean()
            loss = imitation + decision / max(float(np.median(np.abs(teacher_dispatch))), 1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), active_budget.max_grad_norm)
            optimizer.step(); steps += 1
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return _save_artifact(
        method_id="Direct-Policy", seed=seed, model=model, optimizer=optimizer,
        epochs=active_budget.max_epochs, output_dir=Path(output_dir), lineage=_lineage(data, freeze),
        receipt={
            "optimizer_steps": steps,
            "forecast_loss_applicable": False,
            "stage_order": ["decision_only"],
            "loss_history": history,
            "runtime_seconds": time.perf_counter() - started,
            "test_set_accessed": False,
        },
    )


def _train_forecaster(
    model: nn.Module,
    data: Gate2DataBundle,
    *,
    seed: int,
    epochs: int,
    use_exog: bool,
) -> tuple[torch.optim.Optimizer, list[float], int]:
    seed_everything(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    history: list[float] = []
    steps = 0
    for _epoch in range(epochs):
        losses: list[float] = []
        model.train()
        for batch in full_gate2_batches(data.train, data.normalization, batch_size=64):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["load_history"], batch["exog_history"]) if use_exog else model(batch["load_history"])
            target = batch["target_normalized"].to(prediction)
            loss = F.smooth_l1_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); steps += 1
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return optimizer, history, steps


def train_scheme2r_pto(
    seed: int,
    data: Gate2DataBundle,
    freeze: Mapping[str, Any],
    output_dir: str | Path,
    *,
    budget: StageBudgetV42 | None = None,
) -> TrainedMethodArtifact:
    active_budget = _budget(freeze, budget)
    started = time.perf_counter()
    model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0)
    optimizer, history, steps = _train_forecaster(
        model, data, seed=seed, epochs=active_budget.max_epochs, use_exog=True,
    )
    return _save_artifact(
        method_id="Scheme2R-PTO", seed=seed, model=model, optimizer=optimizer,
        epochs=active_budget.max_epochs, output_dir=Path(output_dir), lineage=_lineage(data, freeze),
        receipt={
            "optimizer_steps": steps, "forecast_loss_applicable": True,
            "stage_order": ["forecast"], "loss_history": history,
            "runtime_seconds": time.perf_counter() - started, "test_set_accessed": False,
        },
    )


def train_official_itransformer_pto(
    seed: int,
    data: Gate2DataBundle,
    freeze: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    output_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    receipt_path: str | Path | None = None,
    budget: StageBudgetV42 | None = None,
    model: nn.Module | None = None,
) -> TrainedMethodArtifact:
    from .formal_v4_itransformer import OFFICIAL_COMMIT, OfficialITransformerAdapter

    if source_receipt.get("commit") != OFFICIAL_COMMIT:
        raise ValueError("iTransformer upstream receipt does not match the frozen commit")
    if model is None:
        if source_root is None or receipt_path is None:
            raise ValueError("official iTransformer training requires its source root and receipt path")
        model = OfficialITransformerAdapter(source_root, receipt_path)
    active_budget = _budget(freeze, budget)
    started = time.perf_counter()
    optimizer, history, steps = _train_forecaster(
        model, data, seed=seed, epochs=active_budget.max_epochs, use_exog=False,
    )
    return _save_artifact(
        method_id="Official iTransformer-PTO", seed=seed, model=model, optimizer=optimizer,
        epochs=active_budget.max_epochs, output_dir=Path(output_dir), lineage=_lineage(data, freeze),
        receipt={
            "optimizer_steps": steps, "forecast_loss_applicable": True,
            "stage_order": ["official_backbone_adaptation"], "loss_history": history,
            "upstream_commit": OFFICIAL_COMMIT,
            "method_label": "official_backbone_adaptation",
            "runtime_seconds": time.perf_counter() - started, "test_set_accessed": False,
        },
    )


def train_differentiable_lp(
    seed: int,
    data: Gate2DataBundle,
    freeze: Mapping[str, Any],
    diffopt_receipt: Mapping[str, Any],
    parameters: Mapping[str, Any],
    output_dir: str | Path,
    *,
    budget: StageBudgetV42 | None = None,
    model: nn.Module | None = None,
    layer: nn.Module | None = None,
    micro_batch_size: int = 1,
) -> TrainedMethodArtifact:
    """Train Scheme2R through the differentiable LP with exact exposure accounting."""

    from .formal_v4_diffopt import DifferentiableIESLayer

    if not any(diffopt_receipt.get(name) is True for name in ("eligible", "eligible_for_gate0", "authorized")):
        raise ValueError("Differentiable-LP receipt is not authorized")
    active_budget = _budget(freeze, budget)
    if model is None:
        model = Scheme2RModel(exog_dim=12, task_count=4, lookback=24, horizon=4, dropout=0.0)
    if layer is None:
        layer = DifferentiableIESLayer(parameters)
    seed_everything(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=active_budget.forecaster_lr, weight_decay=active_budget.weight_decay)
    mean = torch.as_tensor(data.normalization.field_mean["load"], dtype=torch.float32)
    scale = torch.as_tensor(data.normalization.field_scale["load"], dtype=torch.float32)
    expected_exposures = len(data.train) * active_budget.max_epochs
    sample_exposures = training_solver_calls = failed_solves = optimizer_steps = 0
    maximum_gradient_norm = 0.0
    history: list[float] = []
    started = time.perf_counter()
    for _epoch in range(active_budget.max_epochs):
        epoch_losses: list[float] = []
        current_effective = None
        for part in iter_gate2_batches(
            data.train, data.normalization, effective_batch_size=64,
            micro_batch_size=micro_batch_size,
        ):
            if current_effective != part.effective_batch_index:
                if current_effective is not None:
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), active_budget.max_grad_norm))
                    if not np.isfinite(grad_norm) or grad_norm <= 0.0:
                        raise RuntimeError("Differentiable-LP accumulated gradient is non-finite or zero")
                    maximum_gradient_norm = max(maximum_gradient_norm, grad_norm)
                    optimizer.step(); optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                current_effective = part.effective_batch_index
            batch = part.batch
            forecast_normalized = model(batch["load_history"], batch["exog_history"])
            forecast_physical = torch.clamp(
                mean.to(forecast_normalized) + scale.to(forecast_normalized) * forecast_normalized,
                min=0.0,
            )
            prices = batch["prices_and_weights"].to(dtype=torch.float64)
            lp_prices = torch.stack((
                prices[..., 0], prices[..., 1],
                torch.zeros_like(prices[..., 0]), prices[..., 2],
            ), dim=-1)
            try:
                dispatch = layer(
                    forecast_physical[..., :3], batch["renewable_forecast"], lp_prices,
                    batch["initial_soc"], batch["previous_chp"],
                )
            except Exception:
                failed_solves += len(part.indices)
                raise
            training_solver_calls += len(part.indices)
            sample_exposures += len(part.indices)
            target_normalized = batch["target_normalized"].to(forecast_normalized)
            forecast_loss = F.smooth_l1_loss(forecast_normalized, target_normalized)
            settled = settle_formal_v4_four_hour(
                dispatch, batch["target_physical"].to(dispatch)[..., :3],
                batch["realized_renewables"].to(dispatch), batch["initial_soc"].to(dispatch),
                batch["previous_chp"].to(dispatch), parameters,
            )
            c_ref = max(float(freeze.get("c_ref", 1.0)), 1.0)
            decision_loss = settled.per_step_penalized_objective.mean() / c_ref + settled.constraint_penalty.mean()
            loss = (forecast_loss + decision_loss) * float(part.accumulation_weight)
            loss.backward()
            epoch_losses.append(float(loss.detach()) / max(float(part.accumulation_weight), 1.0e-12))
        if current_effective is not None:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), active_budget.max_grad_norm))
            if not np.isfinite(grad_norm) or grad_norm <= 0.0:
                raise RuntimeError("Differentiable-LP accumulated gradient is non-finite or zero")
            maximum_gradient_norm = max(maximum_gradient_norm, grad_norm)
            optimizer.step(); optimizer_steps += 1
        history.append(float(np.mean(epoch_losses)))
    if failed_solves or sample_exposures != expected_exposures:
        raise RuntimeError("Differentiable-LP did not complete the frozen sample exposures")
    return _save_artifact(
        method_id="Differentiable-LP", seed=seed, model=model, optimizer=optimizer,
        epochs=active_budget.max_epochs, output_dir=Path(output_dir), lineage=_lineage(data, freeze),
        decision_forecaster_gradient_norm=maximum_gradient_norm,
        receipt={
            "optimizer_steps": optimizer_steps,
            "forecast_loss_applicable": True,
            "stage_order": ["joint_differentiable_lp"],
            "loss_history": history,
            "gradient_norm": maximum_gradient_norm,
            "sample_exposures": sample_exposures,
            "expected_sample_exposures": expected_exposures,
            "failed_solves": failed_solves,
            "effective_batch_size": 64,
            "micro_batch_size": int(micro_batch_size),
            "training_solver_calls": training_solver_calls,
            "runtime_seconds": time.perf_counter() - started,
            "test_set_accessed": False,
        },
    )


__all__ = [
    "Gate2BatchPart",
    "Gate2DataBundle",
    "TrainedMethodArtifact",
    "build_direct_policy_model",
    "build_rsc_model",
    "build_same_information_teacher_dispatch",
    "full_gate2_batches",
    "iter_gate2_batches",
    "load_gate2_data",
    "load_window_split",
    "train_direct_policy",
    "train_differentiable_lp",
    "train_official_itransformer_pto",
    "train_rsc_family",
    "train_scheme2r_pto",
]
