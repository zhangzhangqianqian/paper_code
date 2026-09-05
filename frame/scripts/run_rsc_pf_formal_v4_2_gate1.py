"""Run representative 2019 calibration and freeze formal-v4.2 Gate 1."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
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

from scripts.run_rsc_pf_formal_v4_2_pilot import (  # noqa: E402
    _WINDOW_FIELDS, _batches, _json, _load_base, _model, _predictor,
    _scaled_parameters, _subset,
)
from src.joint_dispatch.formal_v4_2_access import FormalV42AccessController  # noqa: E402
from src.joint_dispatch.formal_v4_2_artifacts import (  # noqa: E402
    canonical_sha256, sha256_file, write_failure_receipt, write_once_json,
)
from src.joint_dispatch.formal_v4_2_checkpoint import save_training_checkpoint  # noqa: E402
from src.joint_dispatch.formal_v4_2_contract import (  # noqa: E402
    FormalV42Contract, assert_gate_transition, load_formal_v4_2_contract,
)
from src.joint_dispatch.formal_v4_2_data import (  # noqa: E402
    Gate1OriginManifestV42, fit_train_normalization, select_gate1_origins,
)
from src.joint_dispatch.formal_v4_2_gate0 import validate_source_manifest_receipt  # noqa: E402
from src.joint_dispatch.formal_v4_2_methods import registered_method_rows  # noqa: E402
from src.joint_dispatch.formal_v4_2_teacher import (  # noqa: E402
    build_same_information_teacher_v42, save_teacher_overlay,
)
from src.joint_dispatch.formal_v4_2_training import (  # noqa: E402
    StageBudgetV42, run_stage_j, run_stage_p, run_stage_s,
)
from src.joint_dispatch.formal_v4_data import FormalV4WindowSplit, materialize_state_windows  # noqa: E402
from src.joint_dispatch.formal_v4_history import generate_settled_device_trajectory  # noqa: E402
from src.joint_dispatch.formal_v4_objective import STEP_WEIGHTS, settle_formal_v4_four_hour  # noqa: E402


class Gate1AuthorizationError(ValueError):
    """Raised when Gate 1 calibration inputs do not satisfy the contract."""


@dataclass(frozen=True)
class Gate1InputV42:
    manifest: Gate1OriginManifestV42
    contract: Any
    normalization_sha256: str
    access_receipt_sha256: str
    candidate_values: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
    candidate_results: tuple[Mapping[str, Any], ...] | None = None
    selected_candidate: Mapping[str, Any] | None = None
    output_path: Path | None = None


@dataclass(frozen=True)
class Gate1FreezeV42:
    payload: Mapping[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    @property
    def authorized_gate2(self) -> bool:
        return bool(self.payload.get("authorized_gate2", False))


def _attr(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _manifest_hash(manifest: Gate1OriginManifestV42) -> str:
    return canonical_sha256(manifest.to_payload())


def _training_freeze(contract: Any, selected: Mapping[str, Any] | None) -> dict[str, Any]:
    training = dict(_attr(contract, "training", {}))
    base_lr = 1.0e-5
    selected_multiplier = float(selected["candidate_value"]) if selected is not None else 1.0
    return {
        "stage_p_max_epochs": int(training.get("stage_p_max_epochs", 30)),
        "stage_s_max_epochs": int(training.get("stage_s_max_epochs", 30)),
        "stage_j_max_epochs": int(training.get("stage_j_max_epochs", 30)),
        "minimum_epochs": int(training.get("minimum_stage_j_epochs", 18)),
        "patience": int(training.get("patience", 5)),
        "validation_interval": int(training.get("validation_interval", 1)),
        "learning_rates": {
            "forecaster_base": base_lr,
            "forecaster_selected": base_lr * selected_multiplier,
            "scheduler": 1.0e-3,
        },
        "candidate_parameter": "stage_j_forecaster_learning_rate_multiplier",
        "candidate_values": [float(value) for value in _attr(contract, "selection", {}).get("gate1_candidate_values", ())],
        "curriculum": {
            "forecast": float(training.get("forecast_weight", 1.0)),
            "imitation_start": float(training.get("imitation_start_weight", 1.0)),
            "imitation_final": float(training.get("imitation_final_weight", 0.0)),
            "decision_start": float(training.get("decision_start_weight", 0.05)),
            "decision_final": float(training.get("decision_final_weight", 1.0)),
            "ramp_epochs": int(training.get("curriculum_ramp_epochs", 18)),
        },
        "rollin_fraction": float(training.get("rollin_start_fraction", 0.40)),
        "model_history_fraction": float(training.get("model_history_fraction", 0.50)),
    }


def run_gate1_calibration(input_data: Gate1InputV42 | Any) -> Gate1FreezeV42:
    """Validate a representative manifest and freeze the complete Gate 2 budget."""

    manifest = _attr(input_data, "manifest")
    if manifest is None:
        raise Gate1AuthorizationError("Gate 1 requires a representative origin manifest")
    activity = _attr(manifest, "activity_fraction", {})
    cooling = float(_attr(manifest, "cooling_active_fraction", activity.get("cooling", 0.0)))
    heating = float(_attr(manifest, "heating_active_fraction", activity.get("heating", 0.0)))
    if cooling < 0.20 or heating < 0.20:
        raise Gate1AuthorizationError("representative manifest has insufficient cooling/heating activity coverage")
    contract = _attr(input_data, "contract")
    try:
        rows = registered_method_rows(contract, gate="gate2")
        contract_hash = str(_attr(contract, "contract_sha256", _attr(contract, "sha256", "")))
        frozen_candidates = tuple(float(value) for value in _attr(contract, "selection", {}).get("gate1_candidate_values", ()))
    except Exception as exc:
        raise Gate1AuthorizationError(f"invalid formal-v4.2 contract: {exc}") from exc
    candidate_values = tuple(float(value) for value in _attr(input_data, "candidate_values", frozen_candidates))
    if candidate_values != frozen_candidates or not candidate_values:
        raise Gate1AuthorizationError("Gate 1 candidates differ from the frozen contract")
    if any(not np.isfinite(value) or value <= 0.0 for value in candidate_values):
        raise Gate1AuthorizationError("Gate 1 candidates must be finite and positive")
    normalization_hash = str(_attr(input_data, "normalization_sha256", ""))
    access_hash = str(_attr(input_data, "access_receipt_sha256", ""))
    if len(normalization_hash) != 64 or len(access_hash) != 64:
        raise Gate1AuthorizationError("Gate 1 requires normalization and access receipt hashes")
    candidate_results = _attr(input_data, "candidate_results", None)
    selected = _attr(input_data, "selected_candidate", None)
    candidate_check = True
    if candidate_results is not None:
        results = tuple(dict(item) for item in candidate_results)
        observed = tuple(float(item["candidate_value"]) for item in results)
        candidate_check = observed == candidate_values and selected is not None and bool(selected.get("eligible", False))
    else:
        results = tuple()
    checks = {
        "activity_coverage": cooling >= 0.20 and heating >= 0.20,
        "contract_methods": len(rows) == 23,
        "normalization_hash": len(normalization_hash) == 64,
        "access_hash": len(access_hash) == 64,
        "candidate_calibration": candidate_check,
        "selection_only": True,
        "no_evaluation_access": True,
    }
    payload: dict[str, Any] = {
        "schema": "formal-v4.2-gate1-freeze-v1",
        "authorized_gate2": all(checks.values()),
        "origin_manifest_sha256": _manifest_hash(manifest),
        "normalization_sha256": normalization_hash,
        "contract_sha256": contract_hash,
        "frozen_training": _training_freeze(contract, selected),
        "frozen_gate2_methods": [asdict(row) for row in rows],
        "checks": checks,
        "access_receipt_sha256": access_hash,
        "selection_year": 2019,
        "activity_fraction": {"cooling": cooling, "heating": heating},
        "evaluation_year_accessed": False,
        "paper_eligible": False,
    }
    if candidate_results is not None:
        payload["candidate_results"] = list(results)
        payload["selected_candidate"] = dict(selected) if selected is not None else None
    output_path = _attr(input_data, "output_path", None)
    if output_path is not None:
        write_once_json(output_path, payload)
    return Gate1FreezeV42(payload)


def _save_split(path: Path, split: FormalV4WindowSplit, trajectory_sha256: str) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{name: getattr(split, name) for name in _WINDOW_FIELDS}, split=np.asarray(split.split), trajectory_sha256=np.asarray(trajectory_sha256))


def _even_indices(length: int, total: int) -> np.ndarray:
    if total <= 0 or total > length:
        raise ValueError("calibration training sample exceeds available windows")
    result = np.rint(np.linspace(0, length - 1, total)).astype(np.int64)
    if len(np.unique(result)) != total:
        raise ValueError("calibration training indices are not unique")
    return result


def _weighted_wape(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight = np.asarray(weights, dtype=np.float64)[:, None, None]
    numerator = np.sum(weight * np.abs(np.asarray(prediction) - np.asarray(target)), axis=(0, 1))
    denominator = np.sum(weight * np.abs(np.asarray(target)), axis=(0, 1))
    return numerator / np.maximum(denominator, 1.0e-12)


def _rigid_demand_target(four_task_target: torch.Tensor) -> torch.Tensor:
    """Separate the three physical balances from the auxiliary gas target."""

    if four_task_target.ndim != 3 or tuple(four_task_target.shape[1:]) != (4, 4):
        raise ValueError("four-task target must have shape [B,4,4]")
    return four_task_target[..., :3]


def _evaluate_candidate(model: torch.nn.Module, split: FormalV4WindowSplit, normalization: Any, parameters: Mapping[str, Any], weights: np.ndarray, *, batch_size: int = 64) -> dict[str, Any]:
    predictions: list[np.ndarray] = []
    dispatches: list[np.ndarray] = []
    controls: list[np.ndarray] = []
    objectives: list[np.ndarray] = []
    shortages: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    model.eval()
    for batch in _batches(split, normalization, batch_size):
        with torch.no_grad():
            output = model(**{name: batch[name] for name in ("load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp")})
            four_task_target = batch["target_physical"].to(output.dispatch)
            settled = settle_formal_v4_four_hour(output.dispatch, _rigid_demand_target(four_task_target), batch["realized_renewables"].to(output.dispatch), batch["initial_soc"].to(output.dispatch), batch["previous_chp"].to(output.dispatch), parameters)
        predictions.append(output.forecast_physical.cpu().numpy())
        dispatches.append(output.dispatch.cpu().numpy())
        controls.append(output.controls.cpu().numpy())
        objectives.append((settled.per_step_penalized_objective * output.dispatch.new_tensor(STEP_WEIGHTS)).sum(dim=1).cpu().numpy())
        shortages.append(settled.normalized_shortage.cpu().numpy())
        residuals.append(settled.constraint_penalty.cpu().numpy())
    prediction = np.concatenate(predictions)
    dispatch = np.concatenate(dispatches)
    control = np.concatenate(controls)
    objective = np.concatenate(objectives)
    shortage = np.concatenate(shortages)
    residual = np.concatenate(residuals)
    normalized_weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
    task_wape = _weighted_wape(prediction, split.forecast_target, weights)
    return {
        "task_wape": task_wape.tolist(),
        "rigid_macro_wape": float(np.mean(task_wape[:3])),
        "gas_wape": float(task_wape[3]),
        "weighted_penalized_objective": float(np.sum(normalized_weights * objective)),
        "weighted_normalized_shortage": float(np.sum(normalized_weights * shortage)),
        "physical_residual_max": float(np.max(np.abs(residual))),
        "control_saturation_fraction": float(np.mean((control < 0.01) | (control > 0.99))),
        "finite": bool(np.isfinite(prediction).all() and np.isfinite(dispatch).all() and np.isfinite(objective).all() and np.isfinite(task_wape).all()),
    }


def _budget(contract: FormalV42Contract, *, candidate_value: float = 1.0) -> StageBudgetV42:
    training = contract.training
    return StageBudgetV42(
        max_epochs=int(training["stage_j_max_epochs"]), minimum_epochs=int(training["minimum_stage_j_epochs"]),
        decision_start=float(training["decision_start_weight"]), decision_final=float(training["decision_final_weight"]),
        ramp_epochs=int(training["curriculum_ramp_epochs"]), forecast_weight=float(training["forecast_weight"]),
        imitation_start=float(training["imitation_start_weight"]), imitation_final=float(training["imitation_final_weight"]),
        forecaster_lr=1.0e-5 * float(candidate_value), scheduler_lr=1.0e-3,
        rollin_fraction=float(training["rollin_start_fraction"]), model_history_fraction=float(training["model_history_fraction"]),
    )


def run_real_gate1(*, contract_path: Path, output_root: Path, run_id: str) -> dict[str, Any]:
    """Run one shared P/S warm start and all frozen Stage-J candidates."""

    started = time.perf_counter()
    contract = load_formal_v4_2_contract(contract_path)
    root = output_root.resolve() / run_id
    gate1_dir = root / "gate1"
    if gate1_dir.exists():
        raise FileExistsError(f"Gate 1 directory already exists: {gate1_dir}")
    transition_path = root / "protocol" / "PILOT_TRANSITION.json"
    pilot_path = root / "pilot" / "PILOT_DECISION.json"
    transition = _json(transition_path)
    pilot = _json(pilot_path)
    assert_gate_transition(contract, "pilot", "gate1")
    if transition.get("authorized_gate1") is not True or pilot.get("authorized_gate1") is not True:
        raise PermissionError("Gate 1 requires an authorized Pilot")
    if transition.get("contract_sha256") != contract.contract_sha256 or pilot.get("contract_sha256") != contract.contract_sha256:
        raise PermissionError("Gate 1 contract lineage differs from Pilot")
    if transition.get("pilot_decision_sha256") != sha256_file(pilot_path):
        raise PermissionError("Pilot decision hash mismatch")
    source_path = root / "protocol" / "SOURCE_MANIFEST.json"
    validate_source_manifest_receipt(source_path, run_id=run_id, contract_sha256=contract.contract_sha256, repo_root=REPO_ROOT)
    source_hash = sha256_file(source_path)
    if pilot.get("source_manifest_sha256") != source_hash:
        raise PermissionError("Gate 1 source manifest differs from Pilot")
    gate1_dir.mkdir(parents=True)
    try:
        controller = FormalV42AccessController(run_root=root, gate="gate1", contract_sha256=contract.contract_sha256)
        train_path = controller.request(root / "data" / "base_train.npz", split="train", years=contract.train_years, caller="run_rsc_pf_formal_v4_2_gate1")
        selection_path = controller.request(root / "data" / "base_selection.npz", split="selection", years=(contract.selection_year,), caller="run_rsc_pf_formal_v4_2_gate1")
        access_path = gate1_dir / "DATA_ACCESS_RECEIPT.json"
        access_receipt = controller.build_receipt()
        write_once_json(access_path, access_receipt)
        if access_receipt["evaluation_year_accessed"]:
            raise PermissionError("Gate 1 accessed the evaluation year")
        capacity_path = root / "gate0" / "CAPACITY_FREEZE.json"
        capacity = _json(capacity_path)
        benchmark_path = root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
        benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
        parameters = _scaled_parameters(benchmark, capacity)
        train_base = _load_base(train_path)
        selection_base = _load_base(selection_path)
        train_trajectory = generate_settled_device_trajectory(train_base, benchmark["values"], capacity_receipt=capacity_path, trajectory_id=f"{run_id}_gate1_train")
        selection_trajectory = generate_settled_device_trajectory(selection_base, benchmark["values"], capacity_receipt=capacity_path, trajectory_id=f"{run_id}_gate1_selection")
        train_windows = materialize_state_windows(train_base, train_trajectory.settled_dispatch, capacity_receipt=capacity_path, bess_energy_capacity=float(parameters["bess_energy_capacity"]), settled_mask=train_trajectory.settled_mask, trajectory_hash=train_trajectory.trajectory_sha256)
        selection_windows = materialize_state_windows(selection_base, selection_trajectory.settled_dispatch, capacity_receipt=capacity_path, bess_energy_capacity=float(parameters["bess_energy_capacity"]), settled_mask=selection_trajectory.settled_mask, trajectory_hash=selection_trajectory.trajectory_sha256)
        _save_split(gate1_dir / "TRAIN_WINDOWS.npz", train_windows, train_trajectory.trajectory_sha256)
        _save_split(gate1_dir / "SELECTION_WINDOWS.npz", selection_windows, selection_trajectory.trajectory_sha256)
        normalization = fit_train_normalization(train_windows, contract.train_years)
        normalization_path = gate1_dir / "NORMALIZATION.json"
        write_once_json(normalization_path, normalization.to_payload())
        manifest = select_gate1_origins(selection_windows, contract.selection["gate1_origin_design"])
        manifest_path = gate1_dir / "GATE1_ORIGIN_MANIFEST.json"
        write_once_json(manifest_path, manifest.to_payload())
        selection_sample = _subset(selection_windows, manifest.origin_indices)
        train_indices = _even_indices(len(train_windows), int(contract.selection["gate1_origin_design"]["total"]))
        train_sample = _subset(train_windows, train_indices)
        train_data_hash = canonical_sha256({"train_windows_sha256": sha256_file(gate1_dir / "TRAIN_WINDOWS.npz"), "selection_windows_sha256": sha256_file(gate1_dir / "SELECTION_WINDOWS.npz"), "train_indices": train_indices.tolist(), "selection_manifest_sha256": _manifest_hash(manifest)})
        lineage = {"contract_sha256": contract.contract_sha256, "source_manifest_sha256": source_hash, "data_sha256": train_data_hash, "normalization_sha256": normalization.receipt_sha256}
        base_budget = _budget(contract)
        model = _model(train_windows, normalization, parameters)
        stage_p = run_stage_p(model, {"train": _batches(train_sample, normalization, 64)}, budget=base_budget, seed=2026)
        stage_p_checkpoint = save_training_checkpoint(gate1_dir / "STAGE_P.pt", model=stage_p.model, optimizer=stage_p.optimizer, epoch=stage_p.epochs - 1, early_stopping={"gate1": True, "stage": "P"}, lineage=lineage)
        stage_p_metrics = _evaluate_candidate(stage_p.model, selection_sample, normalization, parameters, manifest.sampling_weights)
        teacher = build_same_information_teacher_v42(_predictor(stage_p.model, normalization), train_sample, seed=2026, parameters=parameters, capacity_sha256=sha256_file(capacity_path), benchmark_sha256=sha256_file(benchmark_path), normalization_sha256=normalization.receipt_sha256, stage_p_checkpoint_sha256=stage_p_checkpoint.model_sha256, implementation_sha256=sha256_file(FRAME_ROOT / "src" / "joint_dispatch" / "formal_v4_2_teacher.py"), source_manifest_sha256=source_hash)
        save_teacher_overlay(gate1_dir / "teacher", teacher)
        train_batches = _batches(train_sample, normalization, 64, teacher.dispatch)
        stage_s = run_stage_s(stage_p.model, {"train": train_batches}, budget=base_budget, seed=2026)
        stage_s_checkpoint = save_training_checkpoint(gate1_dir / "STAGE_S.pt", model=stage_s.model, optimizer=stage_s.optimizer, epoch=stage_s.epochs - 1, early_stopping={"gate1": True, "stage": "S"}, lineage={**lineage, "teacher_sha256": teacher.arrays_sha256, "parent_checkpoint_sha256": stage_p_checkpoint.model_sha256})
        c_ref = max(float(np.median(teacher.objective)), 1.0)
        guardrails = contract.payload["forecast_guardrails"]
        candidate_results: list[dict[str, Any]] = []
        candidates = tuple(float(value) for value in contract.selection["gate1_candidate_values"])
        for candidate_value in candidates:
            candidate_started = time.perf_counter()
            candidate_budget = _budget(contract, candidate_value=candidate_value)
            result = run_stage_j(deepcopy(stage_s.model), {"train": train_batches}, mode="joint", budget=candidate_budget, c_ref=c_ref, parameters=parameters, seed=2026)
            metrics = _evaluate_candidate(result.model, selection_sample, normalization, parameters, manifest.sampling_weights)
            task = np.asarray(metrics["task_wape"], dtype=np.float64)
            baseline_task = np.asarray(stage_p_metrics["task_wape"], dtype=np.float64)
            guardrail_passed = bool(metrics["rigid_macro_wape"] <= float(guardrails["rigid_macro_wape_ratio_max"]) * max(float(stage_p_metrics["rigid_macro_wape"]), 1.0e-12) and np.all(task[:3] <= float(guardrails["rigid_per_task_wape_ratio_max"]) * np.maximum(baseline_task[:3], 1.0e-12)) and metrics["gas_wape"] <= float(guardrails["gas_wape_ratio_max"]) * max(float(stage_p_metrics["gas_wape"]), 1.0e-12))
            eligible = bool(metrics["finite"] and guardrail_passed and metrics["physical_residual_max"] <= 1.0e-5 and metrics["control_saturation_fraction"] < 0.80 and result.decision_forecaster_gradient_norm > 1.0e-10)
            candidate_dir = gate1_dir / "candidates" / f"multiplier_{candidate_value:g}"
            checkpoint = save_training_checkpoint(candidate_dir / "STAGE_J.pt", model=result.model, optimizer=result.optimizer, epoch=result.epochs - 1, early_stopping={"gate1": True, "stage": "J", "candidate_value": candidate_value}, lineage={**lineage, "teacher_sha256": teacher.arrays_sha256, "parent_checkpoint_sha256": stage_s_checkpoint.model_sha256})
            candidate_results.append({"candidate_value": candidate_value, "forecaster_learning_rate": candidate_budget.forecaster_lr, "scheduler_learning_rate": candidate_budget.scheduler_lr, "seed": 2026, "metrics": metrics, "stage_p_metrics": stage_p_metrics, "guardrail_passed": guardrail_passed, "decision_forecaster_gradient_norm": result.decision_forecaster_gradient_norm, "scheduler_gradient_norm": result.scheduler_gradient_norm, "checkpoint_path": str(checkpoint.path.relative_to(root)), "checkpoint_sha256": checkpoint.model_sha256, "stage_s_parent_sha256": stage_s_checkpoint.model_sha256, "eligible": eligible, "runtime_seconds": time.perf_counter() - candidate_started, "loss_history": list(result.loss_history)})
        eligible_rows = [row for row in candidate_results if row["eligible"]]
        selected = min(eligible_rows, key=lambda row: (row["metrics"]["weighted_penalized_objective"], row["candidate_value"])) if eligible_rows else None
        write_once_json(gate1_dir / "GATE1_CANDIDATES.json", {"schema": "formal-v4.2-gate1-candidates-v1", "candidate_parameter": contract.selection["gate1_candidate_parameter"], "candidate_values": list(candidates), "stage_p_metrics": stage_p_metrics, "results": candidate_results, "selected_candidate_value": None if selected is None else selected["candidate_value"]})
        freeze = run_gate1_calibration(Gate1InputV42(manifest=manifest, contract=contract, normalization_sha256=normalization.receipt_sha256, access_receipt_sha256=sha256_file(access_path), candidate_values=candidates, candidate_results=tuple(candidate_results), selected_candidate=selected, output_path=gate1_dir / "GATE1_FREEZE.json")).to_dict()
        evidence = {**freeze, "run_id": run_id, "source_manifest_sha256": source_hash, "pilot_decision_sha256": sha256_file(pilot_path), "capacity_sha256": sha256_file(capacity_path), "benchmark_sha256": sha256_file(benchmark_path), "train_windows_sha256": sha256_file(gate1_dir / "TRAIN_WINDOWS.npz"), "selection_windows_sha256": sha256_file(gate1_dir / "SELECTION_WINDOWS.npz"), "stage_p_checkpoint_sha256": stage_p_checkpoint.model_sha256, "stage_s_checkpoint_sha256": stage_s_checkpoint.model_sha256, "teacher_sha256": teacher.arrays_sha256, "c_ref": c_ref, "runtime_seconds": time.perf_counter() - started}
        write_once_json(gate1_dir / "GATE1_EVIDENCE.json", evidence)
        if freeze["authorized_gate2"]:
            write_once_json(root / "protocol" / "GATE1_TRANSITION.json", {"schema": "formal-v4.2-gate1-transition-v1", "run_id": run_id, "contract_sha256": contract.contract_sha256, "gate1_freeze_sha256": sha256_file(gate1_dir / "GATE1_FREEZE.json"), "gate1_evidence_sha256": sha256_file(gate1_dir / "GATE1_EVIDENCE.json"), "gate": "gate1", "next_gate": "gate2", "authorized_gate2": True, "evaluation_year_accessed": False})
        return evidence
    except Exception as exc:
        write_failure_receipt(gate1_dir / "GATE1_FAILURE.json", stage="gate1", exception=exc, lineage={"contract_sha256": contract.contract_sha256, "source_manifest_sha256": source_hash})
        raise


def build_gate1_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=FRAME_ROOT / "configs" / "joint_forecast_dispatch_formal_v4_2.json")
    parser.add_argument("--output-root", type=Path, default=FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2")
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_gate1_parser().parse_args(argv)
    try:
        receipt = run_real_gate1(contract_path=args.contract, output_root=args.output_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"authorized_gate2": False, "run_id": args.run_id, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"authorized_gate2": receipt["authorized_gate2"], "run_id": args.run_id, "selected_candidate": None if receipt.get("selected_candidate") is None else receipt["selected_candidate"]["candidate_value"]}, ensure_ascii=False))
    return 0 if receipt["authorized_gate2"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Gate1AuthorizationError", "Gate1FreezeV42", "Gate1InputV42", "build_gate1_parser", "run_gate1_calibration", "run_real_gate1", "main"]
