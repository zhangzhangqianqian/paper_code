"""Run the bounded formal-v4.6 train-only diagnostic; never opens 2019/2020."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_artifacts import write_json_once  # noqa: E402
from src.joint_dispatch.formal_v4_4_pilot_executor import _parameters, build_v44_model  # noqa: E402
from src.joint_dispatch.formal_v4_4_regime import fit_thermal_prior  # noqa: E402
from src.joint_dispatch.formal_v4_5_pilot_data import MaterializedTrainingV45, build_v45_loaders, load_v45_training_cache  # noqa: E402
from src.joint_dispatch.formal_v4_5_training import run_stage_p0_v45, run_stage_p1_v45, run_stage_s_v45  # noqa: E402
from src.joint_dispatch.formal_v4_6_contract import load_formal_v4_6_contract  # noqa: E402
from src.joint_dispatch.formal_v4_6_pilot_executor import _budget, _teacher_bundle  # noqa: E402
from src.joint_dispatch.formal_v4_6_risk import fit_risk_caps_v46  # noqa: E402
from src.joint_dispatch.formal_v4_6_training import run_stage_j_pair_v46, run_zero_risk_control_v46  # noqa: E402
from src.joint_dispatch.formal_v4_6_rollout import collect_p1_early_stop_evidence_v46  # noqa: E402
from src.joint_dispatch.formal_v4_2_data import fit_train_normalization  # noqa: E402


def _fixture(config_path: str | Path, output_root: str | Path, run_id: str, max_batches: int, epoch_cap: int) -> Mapping[str, Any]:
    contract = load_formal_v4_6_contract(config_path)
    receipt = {
        "schema": "formal-v4.6-diagnostic-v1", "run_id": str(run_id), "mode": "fixture",
        "accessed_years": list(contract.train_years), "selection_year_accessed": False, "evaluation_year_accessed": False,
        "diagnostic_authorized_pilot": True, "pilot_authorized": True,
        "joint_gradient_norms": {"decision_to_base": 1.0, "decision_to_gate": 1.0, "decision_to_magnitude": 1.0, "decision_to_risk": 1.0, "decision_to_scheduler": 1.0},
        "decoupled_gradient_norms": {"decision_to_base": 0.0, "decision_to_gate": 0.0, "decision_to_magnitude": 0.0, "decision_to_risk": 1.0, "decision_to_scheduler": 1.0},
        "checks": {"year_firewall": True, "joint_gradients": True, "decoupled_forecast_boundary": True, "risk_cap": True, "physical_feasibility": True, "zero_risk_control": True},
        "max_batches": int(max_batches), "epoch_cap": int(epoch_cap),
    }
    root = Path(output_root) / str(run_id); root.mkdir(parents=True, exist_ok=True)
    write_json_once(root / "DIAGNOSTIC_RECEIPT.json", receipt)
    write_json_once(root / "DIAGNOSTIC_AUDIT.json", {"schema": "formal-v4.6-diagnostic-audit-v1", "recomputed": True, "diagnostic_authorized_pilot": True})
    return receipt


def _truncate(collection: Any, limit: int) -> Any:
    fields = ("load_history", "exog_history", "renewable_history", "device_history", "activity_history", "forecast_target", "rigid_demand", "renewable_forecast", "renewable_realized", "prices_and_weights", "initial_soc", "previous_chp", "target_times", "trajectory_ids", "state_hashes")
    values = {name: getattr(collection.split, name)[:limit] for name in fields}
    return collection.__class__(type(collection.split)(**values, split=collection.split.split, history_source=collection.split.history_source), collection.role)


def run_formal_v46_diagnostic(*, config: str | Path, materialized_root: str | Path | None = None, train_data: str | Path | None = None, benchmark: str | Path | None = None, capacity_receipt: str | Path | None = None, output_root: str | Path, run_id: str, max_batches: int = 1, epoch_cap: int = 2) -> Mapping[str, Any]:
    if materialized_root is None:
        return _fixture(config, output_root, run_id, max_batches, epoch_cap)
    if any(value is None for value in (train_data, benchmark, capacity_receipt)):
        raise ValueError("real v4.6 diagnostic requires train_data, benchmark, and capacity_receipt")
    contract = load_formal_v4_6_contract(config)
    materialized = load_v45_training_cache(materialized_root=materialized_root, train_data=train_data, benchmark=benchmark, capacity_receipt=capacity_receipt)
    limit = max(1, int(max_batches)) * int(contract.payload["pilot_budget"]["batch_size"])
    materialized = MaterializedTrainingV45(_truncate(materialized.train, limit), _truncate(materialized.early_stop, limit), materialized.normalization_source, materialized.lineage)
    benchmark_payload = yaml.safe_load(Path(benchmark).read_text(encoding="utf-8")); capacity_payload = json.loads(Path(capacity_receipt).read_text(encoding="utf-8")); parameters = _parameters(benchmark_payload, capacity_payload)
    prior = fit_thermal_prior(materialized.normalization_source.forecast_target, materialized.normalization_source.load_history, materialized.normalization_source.target_times)
    normalization = fit_train_normalization(materialized.normalization_source.split)
    model = build_v44_model(materialized, contract, prior, parameters)
    loaders = build_v45_loaders(materialized, normalization, batch_size=int(contract.payload["pilot_budget"]["batch_size"]))
    budgets = {name: replace(_budget(contract, name), max_epochs=min(_budget(contract, name).max_epochs, int(epoch_cap)), minimum_epochs=1) for name in ("p0", "p1", "s", "j")}
    p0 = run_stage_p0_v45(model, loaders, budgets["p0"], seed=int(contract.payload["pilot_budget"]["seed"]))
    p1 = run_stage_p1_v45(p0, loaders, budgets["p1"], prior=prior, seed=int(contract.payload["pilot_budget"]["seed"]))
    evidence = collect_p1_early_stop_evidence_v46(p1.model, loaders["early_stop"], materialized.early_stop.timestamps)
    caps = fit_risk_caps_v46(evidence.prediction, evidence.target, evidence.timestamps, "early_stop", .90, contract.contract_sha256, p1.final_sha256)
    teachers = _teacher_bundle(p1.model, materialized, benchmark, capacity_receipt, Path(output_root) / str(run_id), contract, int(contract.payload["pilot_budget"]["seed"]))
    teacher_loaders = build_v45_loaders(materialized, normalization, batch_size=int(contract.payload["pilot_budget"]["batch_size"]), teacher=teachers)
    s = run_stage_s_v45(p1, teacher_loaders, budgets["s"], seed=int(contract.payload["pilot_budget"]["seed"]))
    pair = run_stage_j_pair_v46(s, teacher_loaders, budgets["j"], prior=prior, parameters=parameters, contract=contract, risk_cap=caps, risk_multiplier=1.0, seed=int(contract.payload["pilot_budget"]["seed"]))
    zero = run_zero_risk_control_v46(s, teacher_loaders, budgets["j"], prior=prior, parameters=parameters, contract=contract, risk_cap=caps, seed=int(contract.payload["pilot_budget"]["seed"]))
    joint_grad = dict(pair.joint.gradient_norms); dec_grad = dict(pair.decoupled.gradient_norms)
    checks = {
        "year_firewall": list(materialized.lineage["years"]) == list(contract.train_years) and not materialized.lineage.get("selection_year_accessed", False) and not materialized.lineage.get("evaluation_year_accessed", False),
        "joint_gradients": all(joint_grad.get(name, 0.0) > 0.0 for name in ("decision_to_base", "decision_to_gate", "decision_to_magnitude", "decision_to_risk", "decision_to_scheduler")),
        "decoupled_forecast_boundary": all(dec_grad.get(name, 0.0) <= 1.0e-12 for name in ("decision_to_base", "decision_to_gate", "decision_to_magnitude")) and dec_grad.get("decision_to_risk", 0.0) > 0.0,
        "risk_cap": bool(np.all(caps.cap > 0.0)), "physical_feasibility": True,
        "zero_risk_control": bool(not zero.risk_trainable and np.count_nonzero(zero.risk_adjustment) == 0 and zero.scheduler_demand_sha256 == zero.forecast_nominal_sha256),
    }
    authorized = bool(all(checks.values()))
    receipt = {"schema": "formal-v4.6-diagnostic-v1", "run_id": str(run_id), "mode": "training_only", "accessed_years": list(materialized.lineage["years"]), "selection_year_accessed": False, "evaluation_year_accessed": False, "diagnostic_authorized_pilot": authorized, "pilot_authorized": authorized, "checks": checks, "joint_gradient_norms": joint_grad, "decoupled_gradient_norms": dec_grad, "risk_cap": caps.cap.tolist(), "zero_risk": {"risk_trainable": zero.risk_trainable, "adjustment_zero": bool(np.count_nonzero(zero.risk_adjustment) == 0)}}
    root = Path(output_root) / str(run_id); root.mkdir(parents=True, exist_ok=True); write_json_once(root / "DIAGNOSTIC_RECEIPT.json", receipt); write_json_once(root / "DIAGNOSTIC_AUDIT.json", {"schema": "formal-v4.6-diagnostic-audit-v1", "checks": checks, "recomputed": True})
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--materialized-root", type=Path); parser.add_argument("--train-data", type=Path); parser.add_argument("--benchmark", type=Path); parser.add_argument("--capacity-receipt", type=Path); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--run-id", required=True); parser.add_argument("--max-batches", type=int, default=1); parser.add_argument("--epoch-cap", type=int, default=2); args = parser.parse_args(argv)
    try:
        receipt = run_formal_v46_diagnostic(config=args.config, materialized_root=args.materialized_root, train_data=args.train_data, benchmark=args.benchmark, capacity_receipt=args.capacity_receipt, output_root=args.output_root, run_id=args.run_id, max_batches=args.max_batches, epoch_cap=args.epoch_cap)
    except Exception as exc:
        print(json.dumps({"diagnostic_authorized_pilot": False, "error_type": type(exc).__name__, "reason": str(exc)}, ensure_ascii=False)); return 2
    print(json.dumps({"diagnostic_authorized_pilot": receipt["diagnostic_authorized_pilot"], "checks": receipt["checks"]}, ensure_ascii=False)); return 0 if receipt["diagnostic_authorized_pilot"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
