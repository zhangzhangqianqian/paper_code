"""Bounded formal-v4.4 Pilot orchestration with explicit stop semantics."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .formal_v4_4_artifacts import canonical_sha256, sha256_file, write_json_once, write_npz_once
from .formal_v4_4_contract import load_formal_v4_4_contract
from .formal_v4_4_metrics import compute_forecast_metrics_v44
from .formal_v4_4_pilot_gate import PilotDecisionV44, authorize_pilot_v44


EXPECTED_YEARS = (2015, 2016, 2017, 2018, 2019)
EXPECTED_ROWS = ("continuous_control", "residual_stage_p1", "rsc_pf_joint", "fair_decoupled", "transition_prior")


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict): raise ValueError(f"expected JSON object: {path}")
    return payload


def _years_from_npz(path: str | Path) -> tuple[int, ...]:
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["timestamps"]).astype("datetime64[Y]").astype(int) + 1970
    return tuple(sorted(set(int(value) for value in times)))


def _validate_gate0_transition(path: str | Path, contract_sha256: str) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("schema") != "formal-v4.4-gate0-transition-v1" or payload.get("contract_sha256") != contract_sha256:
        raise ValueError("Gate0 transition lineage mismatch")
    if payload.get("authorized_pilot") is not True or payload.get("evaluation_year_accessed") is not False:
        raise PermissionError("Gate0 did not authorize Pilot or evaluation access was recorded")
    return payload


def run_pilot_v44(
    *, contract_path: str | Path, gate0_transition: str | Path, source_manifest: str | Path, base_train_data: str | Path,
    base_selection_data: str | Path, benchmark: str | Path, capacity_receipt: str | Path, output_root: str | Path, run_id: str,
    stage_executor: Callable[..., Mapping[str, Any]] | None = None,
) -> PilotDecisionV44:
    """Run the bounded Pilot orchestration; never invokes Gate 1."""
    contract = load_formal_v4_4_contract(contract_path); transition = _validate_gate0_transition(gate0_transition, contract.contract_sha256)
    source_hash = sha256_file(source_manifest); train_hash = sha256_file(base_train_data); selection_hash = sha256_file(base_selection_data)
    gate0_root = Path(gate0_transition).resolve().parent / "gate0"; gate0_receipt = _load_json(gate0_root / "GATE0_RECEIPT.json")
    for field, expected in (("source_manifest_sha256", source_hash), ("train_data_sha256", train_hash), ("selection_data_sha256", selection_hash)):
        if gate0_receipt.get(field) != expected: raise ValueError(f"Gate0 {field} does not match explicit source")
    if _years_from_npz(base_train_data) != tuple(contract.train_years) or _years_from_npz(base_selection_data) != (contract.selection_year,):
        raise ValueError("Pilot source years are outside the frozen 2015-2019 boundary")
    split_path = gate0_root / "PILOT_SPLIT.npz"
    if not split_path.is_file(): raise FileNotFoundError(split_path)
    with np.load(split_path, allow_pickle=False) as split_data:
        split = {name: np.asarray(split_data[name], dtype=np.int64) for name in split_data.files}
    if any(len(value) == 0 or len(np.unique(value)) != len(value) for value in split.values()): raise ValueError("Pilot split contains empty or duplicate indices")
    root = Path(output_root).resolve() / str(run_id)
    if root.exists(): raise FileExistsError(root)
    pilot_dir = root / "pilot"; pilot_dir.mkdir(parents=True)
    if stage_executor is None:
        raise RuntimeError("formal Pilot requires a real stage executor with causal device trajectories and same-information LP labels; no executor was supplied")
    result = dict(stage_executor(train_data=base_train_data, selection_data=base_selection_data, benchmark=benchmark, capacity_receipt=capacity_receipt, split=split, contract=contract))
    required = ("prediction", "target", "probability", "prior_probability", "regimes", "times")
    missing = [name for name in required if name not in result]
    if missing: raise ValueError(f"Pilot stage executor did not return {missing[0]}")
    arrays = {name: np.asarray(result[name]) for name in required}
    write_npz_once(pilot_dir / "PILOT_ARRAYS.npz", arrays)
    metrics = compute_forecast_metrics_v44(arrays["prediction"], arrays["target"], arrays["probability"], arrays["prior_probability"], arrays["regimes"], arrays["times"])
    metrics_dict = asdict(metrics)
    comparisons = result.get("comparisons", {})
    receipt: dict[str, Any] = {
        "schema": "formal-v4.4-pilot-receipt-v1", "run_id": str(run_id), "contract_sha256": contract.contract_sha256,
        "lineage": {"source_manifest_sha256": source_hash, "train_data_sha256": train_hash, "selection_data_sha256": selection_hash, "contract_sha256": contract.contract_sha256},
        "accessed_years": list(EXPECTED_YEARS), "finite_values": [float(metrics.four_task_score)], "comparisons": comparisons,
        "regime": {"transition_balanced_accuracy_gain": float(metrics.transition["balanced_accuracy_gain"]), "macro_f1": metrics.macro_f1, "prior_macro_f1": metrics.prior_macro_f1},
        "metrics": metrics_dict, "joint": dict(result.get("joint", {})), "decoupled": dict(result.get("decoupled", {})), "physics": dict(result.get("physics", {})),
        "rows": list(result.get("rows", EXPECTED_ROWS)), "split_hashes": {key: canonical_sha256({"indices": value.tolist()}) for key, value in split.items()},
    }
    if tuple(receipt["rows"]) != EXPECTED_ROWS: raise ValueError("Pilot rows do not match the frozen stage matrix")
    decision = authorize_pilot_v44(receipt, contract)
    receipt["authorized_gate1"] = decision.authorized_gate1; receipt["failures"] = list(decision.failures); receipt["criteria"] = dict(decision.criteria)
    write_json_once(pilot_dir / "PILOT_RECEIPT.json", receipt)
    audit = {"schema": "formal-v4.4-pilot-audit-v1", "receipt_sha256": canonical_sha256(receipt), "metrics_sha256": canonical_sha256(metrics_dict), "authorized_gate1": decision.authorized_gate1, "failures": list(decision.failures)}
    audit_hash = write_json_once(pilot_dir / "PILOT_AUDIT.json", audit)
    transition_payload = {"schema": "formal-v4.4-pilot-transition-v1", "run_id": str(run_id), "contract_sha256": contract.contract_sha256, "gate0_transition_sha256": sha256_file(gate0_transition), "authorized_gate1": decision.authorized_gate1, "evaluation_year_accessed": False, "audit_sha256": audit_hash}
    write_json_once(root / "PILOT_TRANSITION.json", transition_payload)
    return PilotDecisionV44(decision.authorized_gate1, decision.criteria, decision.failures, decision.measured, EXPECTED_YEARS, tuple(receipt["rows"]), audit_hash)


__all__ = ["EXPECTED_ROWS", "EXPECTED_YEARS", "run_pilot_v44"]
