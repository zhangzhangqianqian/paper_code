"""Independent audit of a formal-v4.4 Pilot run."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

FRAME_ROOT = Path(__file__).resolve().parents[1]
if str(FRAME_ROOT) not in sys.path: sys.path.insert(0, str(FRAME_ROOT))

from src.joint_dispatch.formal_v4_4_artifacts import canonical_sha256, sha256_file, write_json_once  # noqa: E402
from src.joint_dispatch.formal_v4_4_contract import load_formal_v4_4_contract  # noqa: E402
from src.joint_dispatch.formal_v4_4_metrics import compute_forecast_metrics_v44  # noqa: E402
from src.joint_dispatch.formal_v4_4_pilot import EXPECTED_ROWS  # noqa: E402
from src.joint_dispatch.formal_v4_4_pilot_gate import PilotDecisionV44, authorize_pilot_v44  # noqa: E402
from src.joint_dispatch.formal_v4_recourse import settle_first_step_v4_numpy  # noqa: E402
from src.joint_dispatch.formal_v4_4_training import _decision_and_imitation, named_autograd_norms  # noqa: E402
from src.joint_dispatch.formal_v4_2_rollout import calculate_all_residual_families  # noqa: E402
from src.joint_dispatch.formal_v4_4_model import ResidualGatedRSCPFModel  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"JSON object required: {path}")
    return value


def _array_equal(left: Any, right: Any, *, atol: float = 1.0e-8) -> bool:
    return np.asarray(left).shape == np.asarray(right).shape and bool(np.allclose(left, right, rtol=1.0e-7, atol=atol, equal_nan=False))


def _load_rollout(root: Path, method_id: str) -> dict[str, np.ndarray]:
    required = (
        "prediction", "target", "probability", "prior_probability", "regimes", "planned_dispatch",
        "settled_dispatch", "shortage", "physical_residual", "operating_cost", "physical_carbon",
        "penalized_objective", "initial_soc", "previous_chp", "realized_demand", "realized_renewables",
        "realized_prices", "times", "state_hashes",
    )
    path = root / "rollout" / f"{method_id}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as arrays:
        missing = [name for name in required if name not in arrays.files]
        if missing:
            raise ValueError(f"rollout {method_id} is missing {missing[0]}")
        result = {name: np.asarray(arrays[name]) for name in required}
    numeric = [name for name in required if name not in ("times", "state_hashes", "regimes")]
    if any(not np.isfinite(result[name]).all() for name in numeric):
        raise ValueError(f"rollout {method_id} contains non-finite values")
    times = result["times"].astype("datetime64[ns]")
    if times.ndim != 1 or len(times) == 0 or set((times.astype("datetime64[Y]").astype(int) + 1970).tolist()) != {2019}:
        raise ValueError(f"rollout {method_id} is outside the frozen 2019 selection")
    if len(times) > 1 and not np.all(np.diff(times) > np.timedelta64(0, "s")):
        raise ValueError(f"rollout {method_id} is not chronological")
    result["times"] = times
    return result


def _recompute_rollout_physics(row: Mapping[str, np.ndarray], parameters: Mapping[str, Any]) -> tuple[float, float]:
    """Recompute settlement, objective, and residuals from saved row inputs."""

    max_residual = 0.0
    objective = []
    context = dict(parameters)
    for index in range(len(row["times"])):
        prices = row["realized_prices"][index]
        context["grid_energy_price"] = float(prices[0]); context["gas_energy_price"] = float(prices[1]); context["carbon_price"] = float(prices[2])
        outcome = settle_first_step_v4_numpy(
            row["planned_dispatch"][index, 0], row["realized_demand"][index], row["realized_renewables"][index], context,
            initial_soc=float(row["initial_soc"][index, 0]), previous_chp=float(row["previous_chp"][index, 0]),
        )
        if not _array_equal(outcome["realized_dispatch"][0], row["settled_dispatch"][index], atol=2.0e-7):
            raise ValueError("settled dispatch differs from canonical recourse")
        for key, saved in (("shortage", row["shortage"][index]), ("operating_cost", row["operating_cost"][index]), ("physical_carbon", row["physical_carbon"][index]), ("penalized_objective", row["penalized_objective"][index])):
            if not _array_equal(outcome[key][0], saved, atol=2.0e-7):
                raise ValueError(f"{key} differs from canonical recourse")
        state = SimpleNamespace(
            initial_soc=torch.as_tensor([[float(row["initial_soc"][index, 0])]], dtype=torch.float64),
            previous_chp=torch.as_tensor([[float(row["previous_chp"][index, 0])]], dtype=torch.float64),
        )
        residual = calculate_all_residual_families(row["settled_dispatch"][index], state, {
            "demand": row["realized_demand"][index], "renewable": row["realized_renewables"][index],
        }, context)
        vector = np.asarray((
            np.abs(residual.balance).max(initial=0.0), np.abs(residual.capacity).max(initial=0.0), np.abs(residual.conversion).max(initial=0.0),
            np.abs(residual.soc).max(initial=0.0), np.abs(residual.ramp).max(initial=0.0), np.abs(residual.exclusivity).max(initial=0.0),
            np.abs(residual.renewable_accounting).max(initial=0.0), residual.finite.max(initial=0.0),
        ), dtype=np.float64)
        if not _array_equal(vector, row["physical_residual"][index], atol=2.0e-7):
            raise ValueError("physical residual differs from independent recomputation")
        max_residual = max(max_residual, float(vector.max(initial=0.0))); objective.append(float(outcome["penalized_objective"][0]))
    return float(np.mean(objective)), max_residual


def _recompute_gradient_boundary(pilot: Path, receipt: Mapping[str, Any], contract: Any) -> None:
    runtime = _json(pilot / "STAGE_RUNTIME.json")
    with np.load(pilot / "PILOT_BATCH.npz", allow_pickle=False) as arrays:
        batch = {name: np.asarray(arrays[name]) for name in arrays.files}
    if "teacher_dispatch" not in batch:
        raise ValueError("replay batch has no teacher dispatch")
    def tensors() -> dict[str, torch.Tensor]:
        result = {}
        for name, value in batch.items():
            if name == "last_thermal_regime": result[name] = torch.as_tensor(value, dtype=torch.long)
            elif name != "sample_indices": result[name] = torch.as_tensor(value, dtype=torch.float32)
        return result
    params = runtime["parameters"]
    model = ResidualGatedRSCPFModel(
        transition_probability=torch.as_tensor(runtime["transition_probability"], dtype=torch.float32),
        regime_temperature=float(runtime["regime_temperature"]), decoder_parameters=runtime["decoder_parameters"],
        task_mean=torch.as_tensor(runtime["task_mean"]), task_scale=torch.as_tensor(runtime["task_scale"]),
        physical_feature_mean=torch.as_tensor(runtime["physical_feature_mean"]), physical_feature_scale=torch.as_tensor(runtime["physical_feature_scale"]),
        previous_chp_mean=torch.as_tensor(runtime["previous_chp_mean"]), previous_chp_scale=torch.as_tensor(runtime["previous_chp_scale"]), dropout=0.0,
    )
    def probe(name: str, detached: bool) -> dict[str, float]:
        checkpoint = torch.load(pilot / "stages" / f"{name}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.train()
        output = model(detach_forecast_for_dispatch=detached, **tensors())
        decision, _, _ = _decision_and_imitation(model, output, tensors(), c_ref=float(params.get("unserved_penalty", 1.0)), parameters=params, detached=detached)
        return named_autograd_norms(decision, model.v44_parameter_groups(), retain_graph=False)
    joint = probe("J_joint", False); decoupled = probe("J_decoupled", True)
    if not (joint.get("gate", 0.0) > 0.0 and joint.get("magnitude", 0.0) > 0.0 and joint.get("scheduler", 0.0) > 0.0):
        raise ValueError("joint decision gradient does not reach all required groups")
    if not (decoupled.get("base", 0.0) <= float(contract.pilot_thresholds["maximum_decoupled_decision_gradient"]) and decoupled.get("scheduler", 0.0) > 0.0):
        raise ValueError("decoupled decision gradient boundary is not detached")
    saved_joint = receipt.get("joint", {}).get("gradient_norms", {})
    saved_decoupled = receipt.get("decoupled", {}).get("gradient_norms", {})
    checkpoint_joint = torch.load(pilot / "stages" / "J_joint.pt", map_location="cpu", weights_only=False).get("gradient_norms", {})
    checkpoint_decoupled = torch.load(pilot / "stages" / "J_decoupled.pt", map_location="cpu", weights_only=False).get("gradient_norms", {})
    if saved_joint != checkpoint_joint or saved_decoupled != checkpoint_decoupled:
        raise ValueError("receipt gradient norms do not match saved J checkpoints")


def audit_run(run_dir: str | Path, contract_path: str | Path) -> PilotDecisionV44:
    root = Path(run_dir).resolve(); contract = load_formal_v4_4_contract(contract_path); pilot = root / "pilot"
    failures: list[str] = []
    try:
        receipt = _json(pilot / "PILOT_RECEIPT.json"); saved_audit = _json(pilot / "PILOT_AUDIT.json")
        with np.load(pilot / "PILOT_ARRAYS.npz", allow_pickle=False) as arrays:
            required = {name: arrays[name] for name in ("prediction", "target", "probability", "prior_probability", "regimes", "times")}
        metrics = compute_forecast_metrics_v44(required["prediction"], required["target"], required["probability"], required["prior_probability"], required["regimes"], required["times"])
        metrics_dict = asdict(metrics)
        if canonical_sha256(receipt.get("metrics", {})) != canonical_sha256(metrics_dict): failures.append("metrics_recompute")
        if saved_audit.get("metrics_sha256") != canonical_sha256(metrics_dict): failures.append("metrics_hash")
        if saved_audit.get("receipt_sha256") != canonical_sha256(receipt): failures.append("receipt_hash")
        split_path = root / "gate0" / "PILOT_SPLIT.npz"
        if split_path.is_file():
            with np.load(split_path, allow_pickle=False) as split:
                values = [np.asarray(split[name], dtype=np.int64) for name in split.files]
            if any(len(value) == 0 or len(np.unique(value)) != len(value) for value in values): failures.append("split_integrity")
            if len(values) >= 2 and set(values[0].tolist()) & set(values[1].tolist()): failures.append("train_eval_overlap")
        else: failures.append("split_missing")
        if receipt.get("accessed_years") != [2015, 2016, 2017, 2018, 2019]: failures.append("evaluation_access")
        # Production v4.4 runs carry the deep-audit artifacts.  The small
        # injected executor used by unit tests predates those artifacts and is
        # intentionally retained as a shallow orchestration fixture.
        if (pilot / "STAGE_RUNTIME.json").is_file():
            try:
                if tuple(receipt.get("rows", ())) != EXPECTED_ROWS:
                    raise ValueError("frozen row matrix is incomplete")
                rows = {name: _load_rollout(pilot, name) for name in EXPECTED_ROWS}
                joint_row = rows["rsc_pf_joint"]
                if not all(_array_equal(required[name], joint_row[name]) for name in ("prediction", "target", "probability", "prior_probability", "regimes", "times")):
                    raise ValueError("PILOT_ARRAYS is not the saved RSC-PF rollout")
                row_metrics = {
                    name: compute_forecast_metrics_v44(value["prediction"], value["target"], value["probability"], value["prior_probability"], value["regimes"], value["times"])
                    for name, value in rows.items()
                }
                if canonical_sha256(asdict(row_metrics["rsc_pf_joint"])) != canonical_sha256(receipt.get("metrics", {})):
                    raise ValueError("RSC-PF rollout metrics differ from receipt")
                def ratio(num: float, den: float) -> float:
                    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0.0 else float("inf")
                joint_metrics = row_metrics["rsc_pf_joint"]; base_metrics = row_metrics["residual_stage_p1"]
                recomputed_comparisons = {
                    "leakage_ratio": {name: ratio(joint_metrics.inactive_leakage[name], base_metrics.inactive_leakage[name]) for name in ("cooling", "heating")},
                    "active_wape_ratio": {name: ratio(joint_metrics.active_only[name]["wape"], base_metrics.active_only[name]["wape"]) for name in ("cooling", "heating")},
                    "electricity_gas_wape_ratio": {name: ratio(joint_metrics.task[name]["wape"], base_metrics.task[name]["wape"]) for name in ("electricity", "gas")},
                    "four_task_score_ratio": ratio(joint_metrics.four_task_score, base_metrics.four_task_score),
                }
                if canonical_sha256(recomputed_comparisons) != canonical_sha256(receipt.get("comparisons", {})):
                    raise ValueError("rollout comparisons differ from receipt")
                runtime = _json(pilot / "STAGE_RUNTIME.json")
                parameters = runtime.get("parameters")
                if not isinstance(parameters, Mapping):
                    raise ValueError("stage runtime has no decoder parameters")
                for name, row in rows.items():
                    objective, max_residual = _recompute_rollout_physics(row, parameters)
                    recorded_objective = float(np.mean(row["penalized_objective"]))
                    recorded_shortage = float(np.mean(row["shortage"].sum(axis=1)))
                    section = receipt.get("joint", {}) if name == "rsc_pf_joint" else receipt.get("decoupled", {}) if name == "fair_decoupled" else None
                    if section is not None and (not np.isclose(objective, float(section.get("penalized_objective", np.nan)), rtol=1.0e-6, atol=1.0e-5) or not np.isclose(recorded_shortage, float(section.get("shortage", np.nan)), rtol=1.0e-6, atol=1.0e-5)):
                        raise ValueError(f"{name} objective/shortage differs from receipt")
                    if not np.isclose(objective, recorded_objective, rtol=1.0e-6, atol=1.0e-5) or not np.isclose(max_residual, float(np.max(row["physical_residual"])), rtol=1.0e-6, atol=1.0e-6):
                        raise ValueError(f"{name} saved objective/physics is internally inconsistent")
                joint_max = float(np.max(rows["rsc_pf_joint"]["physical_residual"]))
                if not np.isclose(joint_max, float(receipt.get("physics", {}).get("max_residual", np.nan)), rtol=1.0e-6, atol=1.0e-6):
                    raise ValueError("physical residual differs from receipt")
                _recompute_gradient_boundary(pilot, receipt, contract)
            except Exception as exc:
                failures.append(f"deep_audit:{type(exc).__name__}")
        recomputed = authorize_pilot_v44(receipt, contract)
        failures.extend(recomputed.failures)
        if bool(receipt.get("authorized_gate1")) != recomputed.authorized_gate1: failures.append("authorization_disagreement")
        result = PilotDecisionV44(not failures and recomputed.authorized_gate1, recomputed.criteria, tuple(sorted(set(failures))), recomputed.measured, tuple(receipt.get("accessed_years", ())), tuple(receipt.get("rows", ())), str(saved_audit.get("receipt_sha256", "")))
    except Exception as exc:
        result = PilotDecisionV44(False, {}, (f"audit_exception:{type(exc).__name__}",), {}, (), (), "")
    audit_payload = {"schema": "formal-v4.4-pilot-independent-audit-v1", "authorized_gate1": result.authorized_gate1, "failures": list(result.failures), "criteria": dict(result.criteria), "audit_sha256": canonical_sha256({"authorized_gate1": result.authorized_gate1, "failures": list(result.failures), "criteria": dict(result.criteria)})}
    try: write_json_once(pilot / "PILOT_INDEPENDENT_AUDIT.json", audit_payload)
    except FileExistsError: pass
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--contract", type=Path, required=True); args = parser.parse_args(argv)
    result = audit_run(args.run_dir, args.contract); print(json.dumps({"authorized_gate1": result.authorized_gate1, "failures": list(result.failures)}, ensure_ascii=False)); return 0 if result.authorized_gate1 else 2


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["audit_run", "main"]
