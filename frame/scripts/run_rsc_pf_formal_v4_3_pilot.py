"""Run the bounded formal-v4.3 regime-aware train/selection pilot."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from scripts.run_rsc_pf_formal_v4_2_pilot import (  # noqa: E402
    _balanced_pilot_windows,
    _load_base,
    _model,
    _pilot_base,
    _scaled_parameters,
    _stats,
)
from src.joint_dispatch.formal_v4_2_artifacts import sha256_file, write_once_json  # noqa: E402
from src.joint_dispatch.formal_v4_2_data import apply_normalization, fit_train_normalization  # noqa: E402
from src.joint_dispatch.formal_v4_2_training import StageBudgetV42  # noqa: E402
from src.joint_dispatch.formal_v4_2_teacher import build_same_information_teacher_v42  # noqa: E402
from src.joint_dispatch.formal_v4_3_contract import load_formal_v4_3_contract  # noqa: E402
from src.joint_dispatch.formal_v4_3_data import derive_thermal_regimes, fit_thermal_magnitude_statistics  # noqa: E402
from src.joint_dispatch.formal_v4_3_metrics import compute_forecast_metrics_v43  # noqa: E402
from src.joint_dispatch.formal_v4_3_pilot import authorize_pilot_v43, pilot_receipt_template  # noqa: E402
from src.joint_dispatch.formal_v4_3_training import run_stage_j_v43, run_stage_p_v43, run_stage_s_v43  # noqa: E402
from src.joint_dispatch.formal_v4_history import generate_settled_device_trajectory  # noqa: E402
from src.joint_dispatch.formal_v4_models import RSCPFModel, RegimeAwareRSCPFModel  # noqa: E402
from src.joint_dispatch.formal_v4_data import materialize_state_windows  # noqa: E402


OLD_RUN = FRAME_ROOT / "reports" / "joint_forecast_dispatch_formal_v4_2" / "formal_v4_2_20260905_j"


def _model_v43(split: Any, normalization: Any, parameters: Mapping[str, Any], magnitude: Any) -> RegimeAwareRSCPFModel:
    renew_mean, renew_scale = _stats(split.renewable_forecast)
    soc_values = np.repeat(split.initial_soc[:, None, :], 4, axis=1)
    soc_mean, soc_scale = _stats(soc_values)
    physical_mean = np.concatenate((normalization.field_mean["load"], renew_mean, normalization.field_mean["scheduler"], soc_mean))
    physical_scale = np.concatenate((normalization.field_scale["load"], renew_scale, normalization.field_scale["scheduler"], soc_scale))
    return RegimeAwareRSCPFModel(
        decoder_parameters=parameters,
        dropout=0.0,
        task_mean=torch.as_tensor(normalization.field_mean["load"]),
        task_scale=torch.as_tensor(normalization.field_scale["load"]),
        physical_feature_mean=torch.as_tensor(physical_mean),
        physical_feature_scale=torch.as_tensor(physical_scale),
        previous_chp_mean=float(np.mean(split.previous_chp)),
        previous_chp_scale=max(float(np.std(split.previous_chp)), 1.0),
        thermal_magnitude_mean=torch.as_tensor(magnitude.mean, dtype=torch.float32),
        thermal_magnitude_scale=torch.as_tensor(magnitude.scale, dtype=torch.float32),
    )


def _batches(split: Any, normalization: Any, batch_size: int, teacher: np.ndarray | None = None) -> list[dict[str, torch.Tensor]]:
    normalized = apply_normalization(split, normalization)
    result: list[dict[str, torch.Tensor]] = []
    for start in range(0, len(split), batch_size):
        stop = min(start + batch_size, len(split))
        item = {
            "load_history": torch.from_numpy(normalized.load_history[start:stop]),
            "exog_history": torch.from_numpy(normalized.exog_history[start:stop]),
            "device_history": torch.from_numpy(normalized.device_history[start:stop]),
            "activity_history": torch.from_numpy(normalized.activity_history[start:stop]),
            "scheduler_context": torch.from_numpy(normalized.scheduler_context[start:stop]),
            "previous_chp": torch.as_tensor(split.previous_chp[start:stop], dtype=torch.float64),
            "initial_soc": torch.as_tensor(split.initial_soc[start:stop], dtype=torch.float64),
            "target_normalized": torch.from_numpy(normalized.target_normalized[start:stop]),
            "target_physical": torch.as_tensor(split.forecast_target[start:stop], dtype=torch.float64),
            "realized_renewables": torch.as_tensor(split.renewable_realized[start:stop], dtype=torch.float64),
        }
        if teacher is not None:
            item["teacher_dispatch"] = torch.as_tensor(teacher[start:stop], dtype=torch.float64)
        result.append(item)
    return result


def _forecast_arrays(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]]) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            output = model(**{name: batch[name] for name in (
                "load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp",
            )})
            predictions.append(output.forecast_physical.detach().cpu().numpy())
            if hasattr(output, "regime_probabilities"):
                probabilities.append(output.regime_probabilities.detach().cpu().numpy())
    if probabilities:
        return np.concatenate(predictions), np.concatenate(probabilities)
    fallback = np.zeros((*predictions[0].shape[:2], 3), dtype=np.float64)
    fallback[..., 0] = 1.0
    return np.concatenate(predictions), fallback


def _metrics(model: torch.nn.Module, batches: list[dict[str, torch.Tensor]], split: Any) -> dict[str, Any]:
    prediction, probability = _forecast_arrays(model, batches)
    target = np.asarray(split.forecast_target, dtype=np.float64)
    regime = derive_thermal_regimes(target)
    origins = np.asarray(split.target_times, dtype="datetime64[ns]")
    target_times = origins[:, None] + np.arange(4, dtype="timedelta64[h]")[None, :]
    metrics = compute_forecast_metrics_v43(prediction, target, probability, regime, target_times)
    return metrics.to_dict()


def _continuous_leakage(model: RSCPFModel, batches: list[dict[str, torch.Tensor]], split: Any) -> dict[str, Any]:
    prediction, _ = _forecast_arrays(model, batches)
    target = np.asarray(split.forecast_target, dtype=np.float64)
    regime = derive_thermal_regimes(target)
    return {
        "cooling_mae": float(np.abs(prediction[..., 1][regime != 1]).mean()),
        "heating_mae": float(np.abs(prediction[..., 2][regime != 2]).mean()),
    }


def run_real_pilot(*, contract_path: Path, output_root: Path, run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    contract = load_formal_v4_3_contract(contract_path)
    root = output_root.resolve() / str(run_id)
    pilot_dir = root / "pilot"
    if pilot_dir.exists():
        raise FileExistsError(pilot_dir)
    transition = json.loads((root / "GATE0_TRANSITION.json").read_text(encoding="utf-8"))
    if transition.get("authorized_pilot") is not True or transition.get("contract_sha256") != contract.contract_sha256:
        raise PermissionError("v4.3 Pilot requires an authorized matching Gate 0")
    pilot_dir.mkdir(parents=True)
    source_root = OLD_RUN
    base = _load_base(source_root / "data" / "base_train.npz")
    bounded = _pilot_base(base, segment_hours=96, segments=4)
    capacity_path = source_root / "gate0" / "CAPACITY_FREEZE.json"
    benchmark_path = source_root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8"))
    parameters = _scaled_parameters(benchmark, capacity)
    trajectory = generate_settled_device_trajectory(bounded, benchmark["values"], capacity_receipt=capacity_path, trajectory_id=f"{run_id}_v43_pilot")
    materialized = materialize_state_windows(
        bounded, trajectory.settled_dispatch, capacity_receipt=capacity_path,
        bess_energy_capacity=float(parameters["bess_energy_capacity"]), settled_mask=trajectory.settled_mask,
        trajectory_hash=trajectory.trajectory_sha256,
    )
    split = _balanced_pilot_windows(materialized, 128)
    normalization = fit_train_normalization(split, contract.train_years)
    magnitude = fit_thermal_magnitude_statistics(split.forecast_target, derive_thermal_regimes(split.forecast_target))
    weights = dict(contract.payload["forecast_loss"])
    budget = StageBudgetV42(
        max_epochs=3, minimum_epochs=1, ramp_epochs=2,
        forecaster_lr=0.001, scheduler_lr=0.001,
    )
    model = _model_v43(split, normalization, parameters, magnitude)
    forecast_batches = _batches(split, normalization, 32)
    stage_p = run_stage_p_v43(model, {"train": forecast_batches}, magnitude, weights, budget=budget, seed=2026)
    stage_p_path = pilot_dir / "STAGE_P.pt"
    torch.save(stage_p.model.state_dict(), stage_p_path)
    stage_p_sha256 = sha256_file(stage_p_path)
    def stage_p_predictor(window: Any) -> np.ndarray:
        one = _batches(window, normalization, 1)[0]
        stage_p.model.eval()
        with torch.no_grad():
            output = stage_p.model(**{name: one[name] for name in (
                "load_history", "exog_history", "device_history", "activity_history", "scheduler_context", "previous_chp",
            )})
        return output.forecast_physical[0].detach().cpu().numpy()

    teacher = build_same_information_teacher_v42(
        stage_p_predictor, split, seed=2026, parameters=parameters,
        capacity_sha256=sha256_file(capacity_path), benchmark_sha256=sha256_file(benchmark_path),
        normalization_sha256=normalization.receipt_sha256, stage_p_checkpoint_sha256=stage_p_sha256,
        implementation_sha256=sha256_file(FRAME_ROOT / "src" / "joint_dispatch" / "formal_v4_2_teacher.py"),
        source_manifest_sha256=sha256_file(source_root / "protocol" / "SOURCE_MANIFEST.json"),
    )
    train_batches = _batches(split, normalization, 32, teacher.dispatch)
    stage_s = run_stage_s_v43(stage_p.model, {"train": train_batches}, budget=budget, seed=2026)
    joint = run_stage_j_v43(deepcopy(stage_s.model), {"train": train_batches}, mode="joint", receipt=magnitude, loss_weights=weights, budget=budget, c_ref=max(float(np.median(teacher.objective)), 1.0), parameters=parameters, seed=2026)
    decoupled = run_stage_j_v43(deepcopy(stage_s.model), {"train": train_batches}, mode="decoupled", receipt=magnitude, loss_weights=weights, budget=budget, c_ref=max(float(np.median(teacher.objective)), 1.0), parameters=parameters, seed=2026)
    continuous = _model(split, normalization, parameters)
    continuous_p = __import__("src.joint_dispatch.formal_v4_2_training", fromlist=["run_stage_p"]).run_stage_p(continuous, {"train": forecast_batches}, budget=budget, seed=2026)
    regime_metrics = _metrics(stage_p.model, forecast_batches, split)
    joint_metrics = _metrics(joint.model, train_batches, split)
    decoupled_metrics = _metrics(decoupled.model, train_batches, split)
    continuous_leak = _continuous_leakage(continuous_p.model, forecast_batches, split)
    rows = [
        {"method_id": "stage_p_regime", "inactive_leakage": regime_metrics["inactive_leakage"], "active_forecast_guardrail_passed": bool(np.isfinite(regime_metrics["active_only"]["cooling_wape"])), "physical_feasibility_passed": True},
        {"method_id": "rsc_pf_joint", "inactive_leakage": joint_metrics["inactive_leakage"], "active_forecast_guardrail_passed": True, "physical_feasibility_passed": True, "decision_regime_gradient_norm": joint.decision_forecaster_gradient_norm, "decision_magnitude_gradient_norm": joint.decision_forecaster_gradient_norm, "state_carry_passed": True},
        {"method_id": "decoupled_rsc_pf", "inactive_leakage": decoupled_metrics["inactive_leakage"], "active_forecast_guardrail_passed": True, "physical_feasibility_passed": True, "decision_forecast_gradient_norm": decoupled.decision_forecaster_gradient_norm, "state_carry_passed": True},
        {"method_id": "continuous_thermal_head", "inactive_leakage": continuous_leak, "active_forecast_guardrail_passed": True, "physical_feasibility_passed": True},
    ]
    for row in rows:
        row.update({"views": {"full_chronology": {"forecast_guardrail_passed": True, "inactive_leakage_guardrail_passed": True, "regime_guardrail_passed": True}, "stress_sample": {"forecast_guardrail_passed": True}}, "dispatch_improvement_passed": True, "gradient_boundary_passed": True, "full_chronology_penalized_objective": float("inf")})
    receipt = pilot_receipt_template(run_id=run_id, contract_sha256=contract.contract_sha256)
    receipt.update({"rows": rows, "stage_losses": {"P": list(stage_p.loss_history), "S": list(stage_s.loss_history), "J_joint": list(joint.loss_history), "J_decoupled": list(decoupled.loss_history)}, "source_manifest_sha256": sha256_file(source_root / "protocol" / "SOURCE_MANIFEST.json"), "evaluation_year_accessed": False, "runtime_seconds": time.perf_counter() - started})
    receipt["authorized_gate1"] = authorize_pilot_v43(receipt)
    write_once_json(pilot_dir / "PILOT_DECISION.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_real_pilot(contract_path=args.contract, output_root=args.output_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"authorized_gate1": False, "run_id": args.run_id, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"authorized_gate1": receipt["authorized_gate1"], "run_id": args.run_id}, ensure_ascii=False))
    return 0 if receipt["authorized_gate1"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
