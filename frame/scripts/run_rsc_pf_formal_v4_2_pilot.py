"""Run the bounded, train-only engineering pilot for formal-v4.2."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRAME_ROOT.parent
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_failure_receipt, write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_checkpoint import clone_stage_s_branches, save_training_checkpoint  # noqa: E402
from src.joint_dispatch.formal_v4_2_contract import assert_gate_transition, load_formal_v4_2_contract  # noqa: E402
from src.joint_dispatch.formal_v4_2_data import apply_normalization, fit_train_normalization  # noqa: E402
from src.joint_dispatch.formal_v4_2_gate0 import validate_source_manifest_receipt  # noqa: E402
from src.joint_dispatch.formal_v4_2_rollout import evaluate_chronological_v42  # noqa: E402
from src.joint_dispatch.formal_v4_2_teacher import build_same_information_teacher_v42, save_teacher_overlay  # noqa: E402
from src.joint_dispatch.formal_v4_2_training import StageBudgetV42, run_stage_j, run_stage_j_pair, run_stage_p, run_stage_s  # noqa: E402
from src.joint_dispatch.formal_v4_data import FormalV4BaseSeries, FormalV4WindowSplit, materialize_state_windows  # noqa: E402
from src.joint_dispatch.formal_v4_history import generate_settled_device_trajectory  # noqa: E402
from src.joint_dispatch.formal_v4_models import RSCPFModel  # noqa: E402
from src.joint_dispatch.formal_v4_state import FormalV4ClosedLoopState  # noqa: E402


REQUIRED_CHECKS = (
    "stage_p_loss_decreased", "stage_s_loss_decreased", "stage_j_loss_finite",
    "persistent_optimizer_steps", "stage_s_clone_identical",
    "joint_forecast_decision_gradient_positive",
    "decoupled_forecast_decision_gradient_zero", "first_step_state_carry",
)


def authorize_pilot(receipt: Mapping[str, Any], *, shortage_rate_max: float = 0.80) -> bool:
    """Return true only if every structural and numerical pilot check passes."""

    limit = float(shortage_rate_max)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("shortage_rate_max must be finite and non-negative")
    if not all(receipt.get(name) is True for name in REQUIRED_CHECKS):
        return False
    shortage = float(receipt.get("shortage_rate", np.inf))
    return bool(np.isfinite(shortage) and 0.0 <= shortage <= limit)


def run_pilot(fixture: Any) -> dict[str, Any]:
    """Persist a fixture-backed receipt used only by isolated unit tests."""

    source = dict(fixture) if isinstance(fixture, Mapping) else vars(fixture)
    root = Path(source.get("output_root", Path.cwd())) / str(source.get("run_id", "formal_v4_2_pilot"))
    pilot_dir = root / "pilot"
    if pilot_dir.exists():
        raise FileExistsError(pilot_dir)
    pilot_dir.mkdir(parents=True)
    receipt = dict(source)
    for name in REQUIRED_CHECKS:
        receipt.setdefault(name, False)
    receipt.setdefault("shortage_rate", np.inf)
    receipt.update({
        "schema": "formal-v4.2-pilot-receipt-v1", "paper_eligible": False,
        "evaluation_year_accessed": False, "ranking_generated": False,
        "train_years": [2015, 2016, 2017, 2018], "selection_year_used": False,
        "authorized_gate1": authorize_pilot(receipt, shortage_rate_max=float(source.get("shortage_rate_max", 0.80))),
    })
    write_once_json(pilot_dir / "PILOT_RECEIPT.json", receipt)
    return receipt


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _load_base(path: Path) -> FormalV4BaseSeries:
    with np.load(path, allow_pickle=False) as payload:
        return FormalV4BaseSeries(
            payload["load_and_exog"], payload["renewable_forecast"],
            payload["renewable_realized"], payload["prices_and_weights"],
            payload["timestamps"], str(np.asarray(payload["split"]).item()),
        )


def _continuous_runs(times: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(times) != np.timedelta64(1, "h")) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(times)]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _pilot_base(base: FormalV4BaseSeries, *, segment_hours: int, segments: int) -> FormalV4BaseSeries:
    """Select deterministic seasonal stress segments using train years only."""

    times = base.timestamps
    months = np.asarray([int(str(value)[5:7]) for value in times], dtype=np.int64)
    runs = _continuous_runs(times)
    season_months = ((12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
    score_columns = (2, 0, 1, 0)
    chosen: list[tuple[int, int]] = []
    for wanted_months, column in zip(season_months[:segments], score_columns[:segments]):
        candidates: list[tuple[float, int, int]] = []
        for run_start, run_end in runs:
            if run_end - run_start < segment_hours:
                continue
            eligible = np.arange(run_start + 48, run_end - 4, dtype=np.int64)
            eligible = eligible[np.isin(months[eligible], wanted_months)]
            if eligible.size == 0:
                continue
            peak = int(eligible[np.argmax(base.load_and_exog[eligible, column])])
            start = min(max(peak - 48, run_start), run_end - segment_hours)
            candidates.append((float(base.load_and_exog[peak, column]), int(start), int(start + segment_hours)))
        if not candidates:
            raise ValueError("train data cannot supply every frozen pilot season")
        _, start, end = max(candidates)
        chosen.append((start, end))
    chosen = sorted(set(chosen))
    if len(chosen) != segments:
        raise ValueError("pilot seasonal segments overlap or are incomplete")
    indices = np.concatenate([np.arange(start, end) for start, end in chosen])
    return FormalV4BaseSeries(
        base.load_and_exog[indices], base.renewable_forecast[indices],
        base.renewable_realized[indices], base.prices_and_weights[indices],
        base.timestamps[indices], "train",
    )


_WINDOW_FIELDS = (
    "load_history", "exog_history", "renewable_history", "device_history",
    "activity_history", "forecast_target", "rigid_demand", "renewable_forecast",
    "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp",
    "target_times", "trajectory_ids", "state_hashes",
)


def _subset(split: FormalV4WindowSplit, indices: np.ndarray) -> FormalV4WindowSplit:
    return replace(split, **{name: getattr(split, name)[indices] for name in _WINDOW_FIELDS})


def _balanced_pilot_windows(split: FormalV4WindowSplit, total: int) -> FormalV4WindowSplit:
    groups: list[np.ndarray] = []
    start = 0
    stops = list(np.flatnonzero(np.diff(split.target_times) != np.timedelta64(1, "h")) + 1) + [len(split)]
    for stop in stops:
        groups.append(np.arange(start, int(stop), dtype=np.int64)); start = int(stop)
    per_group, remainder = divmod(total, len(groups))
    selected: list[int] = []
    for index, group in enumerate(groups):
        count = per_group + (1 if index < remainder else 0)
        if len(group) < count:
            raise ValueError("a pilot segment has too few materialized windows")
        selected.extend(group[:count].tolist())
    return _subset(split, np.asarray(selected, dtype=np.int64))


def _scaled_parameters(benchmark: Mapping[str, Any], capacity: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(benchmark["values"])
    multiplier = float(capacity["selected"]["multiplier"])
    for name in ("electric_chiller_capacity", "absorption_chiller_capacity"):
        parameters[name] = float(parameters[name]) * multiplier
    parameters.setdefault("surplus_penalty", 0.1)
    parameters["carbon_price"] = parameters.get("carbon_price_default", 0.0)
    return parameters


def _stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(values, dtype=np.float64).mean(axis=(0, 1)).astype(np.float32)
    scale = np.asarray(values, dtype=np.float64).std(axis=(0, 1)).astype(np.float32)
    return mean, np.where(scale < 1.0e-6, 1.0, scale).astype(np.float32)


def _model(split: FormalV4WindowSplit, normalization: Any, parameters: Mapping[str, Any]) -> RSCPFModel:
    renew_mean, renew_scale = _stats(split.renewable_forecast)
    soc_mean, soc_scale = _stats(np.repeat(split.initial_soc[:, None, :], 4, axis=1))
    physical_mean = np.concatenate((normalization.field_mean["load"], renew_mean, normalization.field_mean["scheduler"], soc_mean))
    physical_scale = np.concatenate((normalization.field_scale["load"], renew_scale, normalization.field_scale["scheduler"], soc_scale))
    return RSCPFModel(
        decoder_parameters=parameters, dropout=0.0,
        task_mean=torch.as_tensor(normalization.field_mean["load"]),
        task_scale=torch.as_tensor(normalization.field_scale["load"]),
        physical_feature_mean=torch.as_tensor(physical_mean),
        physical_feature_scale=torch.as_tensor(physical_scale),
        previous_chp_mean=float(np.mean(split.previous_chp)),
        previous_chp_scale=max(float(np.std(split.previous_chp)), 1.0),
    )


def _batches(split: FormalV4WindowSplit, normalization: Any, batch_size: int, teacher: np.ndarray | None = None) -> list[dict[str, torch.Tensor]]:
    normalized = apply_normalization(split, normalization)
    batches: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(split), batch_size):
        stop = min(start + batch_size, len(split))
        item = {
            "load_history": torch.from_numpy(normalized.load_history[start:stop]),
            "exog_history": torch.from_numpy(normalized.exog_history[start:stop]),
            "device_history": torch.from_numpy(normalized.device_history[start:stop]),
            "activity_history": torch.from_numpy(normalized.activity_history[start:stop]),
            "scheduler_context": torch.from_numpy(normalized.scheduler_context[start:stop]),
            # Preserve the exact capacity endpoint for the strict decoder;
            # float32 rounds 450.45 upward and can look like a bound breach.
            "previous_chp": torch.as_tensor(split.previous_chp[start:stop], dtype=torch.float64),
            "initial_soc": torch.as_tensor(split.initial_soc[start:stop], dtype=torch.float64),
            "target_normalized": torch.from_numpy(normalized.target_normalized[start:stop]),
            "target_physical": torch.as_tensor(split.forecast_target[start:stop], dtype=torch.float64),
            "realized_renewables": torch.as_tensor(split.renewable_realized[start:stop], dtype=torch.float64),
        }
        if teacher is not None:
            item["teacher_dispatch"] = torch.as_tensor(teacher[start:stop], dtype=torch.float64)
        batches.append(item)
    return batches


def _predictor(model: RSCPFModel, normalization: Any):
    def predict(window: FormalV4WindowSplit) -> np.ndarray:
        batch = _batches(window, normalization, 1)[0]
        model.eval()
        with torch.no_grad():
            output = model(**{name: batch[name] for name in (
                "load_history", "exog_history", "device_history", "activity_history",
                "scheduler_context", "previous_chp",
            )})
        return output.forecast_physical[0].detach().cpu().numpy()
    return predict


def _loss_decreased(history: tuple[float, ...]) -> bool:
    return len(history) >= 2 and np.isfinite(history).all() and float(history[-1]) < float(history[0])


def _initial_state(window: FormalV4WindowSplit) -> FormalV4ClosedLoopState:
    return FormalV4ClosedLoopState(
        torch.as_tensor(window.load_history[:1], dtype=torch.float32),
        torch.as_tensor(window.exog_history[:1], dtype=torch.float32),
        torch.as_tensor(window.device_history[:1], dtype=torch.float32),
        torch.as_tensor(window.activity_history[:1], dtype=torch.float32),
        torch.as_tensor(window.initial_soc[:1], dtype=torch.float64),
        torch.as_tensor(window.previous_chp[:1], dtype=torch.float64),
        window.target_times[0], str(window.trajectory_ids[0]),
    )


def _rollout(model: RSCPFModel, split: FormalV4WindowSplit, normalization: Any, parameters: Mapping[str, Any], count: int):
    windows = []
    for index in range(count):
        realized_exog = split.exog_history[index + 1, -1] if index + 1 < len(split) else split.exog_history[index, -1]
        windows.append({
            "forecast_target": split.forecast_target[index], "renewable_forecast": split.renewable_forecast[index],
            "renewable_realized": split.renewable_realized[index], "prices_and_weights": split.prices_and_weights[index],
            "target_time": split.target_times[index], "realized_exog": realized_exog,
        })

    def method(window: Mapping[str, Any], state: FormalV4ClosedLoopState):
        scheduler = np.concatenate((
            np.asarray(window["renewable_forecast"], dtype=np.float32),
            np.asarray(window["prices_and_weights"], dtype=np.float32),
            np.repeat(state.initial_soc.detach().cpu().numpy()[:, None, :], 4, axis=1)[0].astype(np.float32),
        ), axis=-1)[None, ...]
        inputs = {
            "load_history": (state.load_history - torch.as_tensor(normalization.field_mean["load"])) / torch.as_tensor(normalization.field_scale["load"]),
            "exog_history": (state.exog_history - torch.as_tensor(normalization.field_mean["exog"])) / torch.as_tensor(normalization.field_scale["exog"]),
            "device_history": (state.device_history - torch.as_tensor(normalization.field_mean["device"])) / torch.as_tensor(normalization.field_scale["device"]),
            "activity_history": state.activity_history,
            "scheduler_context": torch.from_numpy(scheduler), "previous_chp": state.previous_chp,
        }
        model.eval()
        with torch.no_grad():
            output = model(**inputs)
        return {"dispatch": output.dispatch[0].cpu().numpy(), "forecast": output.forecast_physical[0].cpu().numpy()}

    method.method_id = "RSC-PF-pilot"  # type: ignore[attr-defined]
    rollout_split = _subset(split, np.arange(count, dtype=np.int64))
    result = evaluate_chronological_v42(method, windows, initial_state=_initial_state(rollout_split), parameters=parameters)
    carried = all(
        result.next_states[index].origin == rollout_split.target_times[0] + np.timedelta64(index + 1, "h")
        and np.allclose(result.next_states[index].device_history[0, -1].numpy(), result.settled_dispatch[index, :17])
        for index in range(count)
    )
    demand = np.sum(np.abs(result.forecast_target[:, 0, :3]))
    shortage_rate = float(np.sum(result.shortage_energy) / max(float(demand), 1.0e-12))
    return result, bool(carried), shortage_rate


def run_real_pilot(*, contract_path: Path, output_root: Path, run_id: str) -> dict[str, Any]:
    """Execute real P→teacher→S→clone→J training and 24-step rollout."""

    started = time.perf_counter()
    contract = load_formal_v4_2_contract(contract_path)
    root = output_root.resolve() / run_id
    pilot_dir = root / "pilot"
    if pilot_dir.exists():
        raise FileExistsError(f"pilot directory already exists: {pilot_dir}")
    current = _json(root / "protocol" / "CURRENT_GATE.json")
    evidence_path = root / "gate0" / "GATE0_EVIDENCE.json"
    evidence = _json(evidence_path)
    if current.get("gate") != "gate0" or current.get("next_gate") != "pilot" or current.get("authorized_pilot") is not True:
        raise PermissionError("Pilot requires an authorized Gate 0 transition")
    assert_gate_transition(contract, "gate0", "pilot")
    if current.get("contract_sha256") != contract.contract_sha256 or evidence.get("contract_sha256") != contract.contract_sha256:
        raise PermissionError("Pilot contract lineage differs from Gate 0")
    if current.get("gate_evidence_sha256") != sha256_file(evidence_path):
        raise PermissionError("Gate 0 evidence hash mismatch")
    source_path = root / "protocol" / "SOURCE_MANIFEST.json"
    source = validate_source_manifest_receipt(source_path, run_id=run_id, contract_sha256=contract.contract_sha256, repo_root=REPO_ROOT)
    source_hash = sha256_file(source_path)
    if evidence.get("source_manifest_sha256") != source_hash:
        raise PermissionError("Gate 0 source manifest lineage mismatch")
    pilot_dir.mkdir(parents=True)
    try:
        pilot = contract.payload["pilot"]
        capacity_path = root / "gate0" / "CAPACITY_FREEZE.json"
        capacity = _json(capacity_path)
        benchmark_path = root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
        benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
        parameters = _scaled_parameters(benchmark, capacity)
        base = _load_base(root / "data" / "base_train.npz")
        bounded = _pilot_base(base, segment_hours=int(pilot["segment_hours"]), segments=int(pilot["segments"]))
        trajectory = generate_settled_device_trajectory(bounded, benchmark["values"], capacity_receipt=capacity_path, trajectory_id=f"{run_id}_pilot_train_only")
        materialized = materialize_state_windows(
            bounded, trajectory.settled_dispatch, capacity_receipt=capacity_path,
            bess_energy_capacity=float(parameters["bess_energy_capacity"]), settled_mask=trajectory.settled_mask,
            trajectory_hash=trajectory.trajectory_sha256,
        )
        split = _balanced_pilot_windows(materialized, int(pilot["windows"]))
        normalization = fit_train_normalization(split, contract.train_years)
        write_once_json(pilot_dir / "NORMALIZATION.json", normalization.to_payload())
        np.savez_compressed(pilot_dir / "PILOT_DATA.npz", **{name: getattr(split, name) for name in _WINDOW_FIELDS}, split=np.asarray(split.split), trajectory_sha256=np.asarray(trajectory.trajectory_sha256))
        data_hash = sha256_file(pilot_dir / "PILOT_DATA.npz")
        lineage = {"contract_sha256": contract.contract_sha256, "source_manifest_sha256": source_hash, "data_sha256": data_hash, "normalization_sha256": normalization.receipt_sha256}
        budget = StageBudgetV42(
            max_epochs=int(pilot["stage_epochs"]), minimum_epochs=1, ramp_epochs=max(int(pilot["stage_epochs"]) - 1, 1),
            forecaster_lr=float(pilot["forecaster_learning_rate"]), scheduler_lr=float(pilot["scheduler_learning_rate"]),
        )
        model = _model(split, normalization, parameters)
        forecast_batches = _batches(split, normalization, int(pilot["batch_size"]))
        p_result = run_stage_p(model, {"train": forecast_batches}, budget=budget, seed=int(pilot["seed"]))
        p_checkpoint = save_training_checkpoint(pilot_dir / "STAGE_P.pt", model=p_result.model, optimizer=p_result.optimizer, epoch=p_result.epochs - 1, early_stopping={"pilot": True}, lineage=lineage)
        teacher = build_same_information_teacher_v42(
            _predictor(p_result.model, normalization), split, seed=int(pilot["seed"]), parameters=parameters,
            capacity_sha256=sha256_file(capacity_path), benchmark_sha256=sha256_file(benchmark_path),
            normalization_sha256=normalization.receipt_sha256, stage_p_checkpoint_sha256=p_checkpoint.model_sha256,
            implementation_sha256=sha256_file(FRAME_ROOT / "src" / "joint_dispatch" / "formal_v4_2_teacher.py"),
            source_manifest_sha256=source_hash,
        )
        save_teacher_overlay(pilot_dir / "teacher", teacher)
        train_batches = _batches(split, normalization, int(pilot["batch_size"]), teacher.dispatch)
        s_result = run_stage_s(p_result.model, {"train": train_batches}, budget=budget, seed=int(pilot["seed"]))
        s_checkpoint = save_training_checkpoint(
            pilot_dir / "STAGE_S.pt", model=s_result.model, optimizer=s_result.optimizer, epoch=s_result.epochs - 1,
            early_stopping={"pilot": True}, lineage={**lineage, "teacher_sha256": teacher.arrays_sha256, "parent_checkpoint_sha256": p_checkpoint.model_sha256},
        )
        joint_clone, decoupled_clone = clone_stage_s_branches(s_checkpoint.path, pilot_dir / "STAGE_S_JOINT_CLONE.pt", pilot_dir / "STAGE_S_DECOUPLED_CLONE.pt")
        clone_identical = sha256_file(joint_clone) == sha256_file(decoupled_clone) == s_checkpoint.model_sha256
        c_ref = max(float(np.median(teacher.objective)), 1.0)
        boundary_joint, boundary_decoupled = run_stage_j_pair(s_result, train_batches[0], budget=budget, c_ref=c_ref, parameters=parameters, seed=int(pilot["seed"]))
        joint_result = run_stage_j(deepcopy(s_result.model), {"train": train_batches}, mode="joint", budget=budget, c_ref=c_ref, parameters=parameters, seed=int(pilot["seed"]))
        decoupled_result = run_stage_j(deepcopy(s_result.model), {"train": train_batches}, mode="decoupled", budget=budget, c_ref=c_ref, parameters=parameters, seed=int(pilot["seed"]))
        save_training_checkpoint(pilot_dir / "STAGE_J_JOINT.pt", model=joint_result.model, optimizer=joint_result.optimizer, epoch=joint_result.epochs - 1, early_stopping={"pilot": True, "mode": "joint"}, lineage={**lineage, "parent_checkpoint_sha256": s_checkpoint.model_sha256})
        save_training_checkpoint(pilot_dir / "STAGE_J_DECOUPLED.pt", model=decoupled_result.model, optimizer=decoupled_result.optimizer, epoch=decoupled_result.epochs - 1, early_stopping={"pilot": True, "mode": "decoupled"}, lineage={**lineage, "parent_checkpoint_sha256": s_checkpoint.model_sha256})
        rollout, carried, shortage_rate = _rollout(joint_result.model, materialized, normalization, parameters, int(pilot["rollout_windows"]))
        np.savez_compressed(
            pilot_dir / "ROLLOUT.npz", forecast_target=rollout.forecast_target, forecast_prediction=rollout.forecast_prediction,
            planned_dispatch=rollout.planned_dispatch, settled_dispatch=rollout.settled_dispatch,
            shortage_energy=rollout.shortage_energy, target_times=rollout.target_times,
        )
        checks = {
            "stage_p_loss_decreased": _loss_decreased(p_result.loss_history),
            "stage_s_loss_decreased": _loss_decreased(s_result.loss_history),
            "stage_j_loss_finite": bool(np.isfinite(joint_result.loss_history).all() and np.isfinite(decoupled_result.loss_history).all()),
            "persistent_optimizer_steps": all(value > 1 for value in (p_result.optimizer_steps, s_result.optimizer_steps, joint_result.optimizer_steps, decoupled_result.optimizer_steps)),
            "stage_s_clone_identical": bool(clone_identical),
            "joint_forecast_decision_gradient_positive": boundary_joint.decision_forecaster_gradient_norm > 0.0,
            "decoupled_forecast_decision_gradient_zero": boundary_decoupled.decision_forecaster_gradient_norm == 0.0,
            "first_step_state_carry": carried,
        }
        receipt = {
            "schema": "formal-v4.2-pilot-decision-v1", "run_id": run_id,
            "contract_sha256": contract.contract_sha256, "source_manifest_sha256": source_hash,
            "gate0_evidence_sha256": sha256_file(evidence_path), "data_sha256": data_hash,
            "normalization_sha256": normalization.receipt_sha256, "capacity_sha256": sha256_file(capacity_path),
            "benchmark_sha256": sha256_file(benchmark_path), "seed": int(pilot["seed"]),
            "train_years": list(contract.train_years), "pilot_windows": len(split),
            "rollout_windows": int(pilot["rollout_windows"]), "stage_order": ["P", "teacher", "S", "clone", "J"],
            "loss_summary": {"stage_p": list(p_result.loss_history), "stage_s": list(s_result.loss_history), "stage_j_joint": list(joint_result.loss_history), "stage_j_decoupled": list(decoupled_result.loss_history)},
            "optimizer_steps": {"P": p_result.optimizer_steps, "S": s_result.optimizer_steps, "J_joint": joint_result.optimizer_steps, "J_decoupled": decoupled_result.optimizer_steps},
            "gradient_boundary": {"joint_forecaster": boundary_joint.decision_forecaster_gradient_norm, "joint_scheduler": boundary_joint.scheduler_gradient_norm, "decoupled_forecaster": boundary_decoupled.decision_forecaster_gradient_norm, "decoupled_scheduler": boundary_decoupled.scheduler_gradient_norm},
            "trajectory_audit": trajectory.audit.to_payload(), "checks": checks, **checks,
            "shortage_rate": shortage_rate, "shortage_rate_max": float(pilot["shortage_rate_max"]),
            "runtime_seconds": time.perf_counter() - started, "paper_eligible": False,
            "ranking_generated": False, "selection_year_used": False, "evaluation_year_accessed": False,
        }
        receipt["authorized_gate1"] = authorize_pilot(receipt, shortage_rate_max=float(pilot["shortage_rate_max"]))
        write_once_json(pilot_dir / "PILOT_DECISION.json", receipt)
        if receipt["authorized_gate1"]:
            write_once_json(root / "protocol" / "PILOT_TRANSITION.json", {
                "schema": "formal-v4.2-pilot-transition-v1", "run_id": run_id,
                "contract_sha256": contract.contract_sha256, "pilot_decision_sha256": sha256_file(pilot_dir / "PILOT_DECISION.json"),
                "gate": "pilot", "next_gate": "gate1", "authorized_gate1": True, "evaluation_year_accessed": False,
            })
        return receipt
    except Exception as exc:
        write_failure_receipt(pilot_dir / "PILOT_FAILURE.json", stage="pilot", exception=exc, lineage={"contract_sha256": contract.contract_sha256, "source_manifest_sha256": source_hash})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json")
    parser.add_argument("--output-root", type=Path, default=FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_real_pilot(contract_path=args.contract, output_root=args.output_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"authorized_gate1": False, "run_id": args.run_id, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"authorized_gate1": receipt["authorized_gate1"], "run_id": args.run_id, "shortage_rate": receipt["shortage_rate"]}, ensure_ascii=False))
    return 0 if receipt["authorized_gate1"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["authorize_pilot", "run_pilot", "run_real_pilot", "main"]
