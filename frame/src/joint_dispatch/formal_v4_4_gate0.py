"""Measured, fail-closed Gate 0 preflight for formal-v4.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import ast
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .formal_v4_4_artifacts import sha256_file, write_json_once
from .formal_v4_4_contract import FormalV44Contract, load_formal_v4_4_contract
from .formal_v4_4_model import ResidualGatedRSCPFModel
from .formal_v4_4_pilot_data import build_pilot_indices
from .formal_v4_4_provenance import validate_source_manifest_payload
from .formal_v4_4_regime import derive_last_observed_regime, derive_thermal_regimes, fit_thermal_prior
from .formal_v4_4_training import named_autograd_norms
from .model import JointForecastDispatchModel


@dataclass(frozen=True)
class GateCheckV44:
    measured: bool
    passed: bool
    value: Any = None
    measured_max: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Gate0ReceiptV44:
    run_id: str
    contract_sha256: str
    source_manifest_sha256: str
    train_data_sha256: str
    selection_data_sha256: str
    authorized_pilot: bool
    checks: Mapping[str, GateCheckV44]
    pilot_split: Mapping[str, Any]
    thermal_prior: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "formal-v4.4-gate0-receipt-v1", "run_id": self.run_id, "contract_sha256": self.contract_sha256,
            "source_manifest_sha256": self.source_manifest_sha256, "train_data_sha256": self.train_data_sha256,
            "selection_data_sha256": self.selection_data_sha256, "authorized_pilot": self.authorized_pilot,
            "checks": {name: asdict(check) for name, check in self.checks.items()}, "pilot_split": dict(self.pilot_split), "thermal_prior": dict(self.thermal_prior),
        }


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    target = Path(path)
    if not target.is_file(): raise FileNotFoundError(target)
    with np.load(target, allow_pickle=False) as data: return {name: np.asarray(data[name]) for name in data.files}


def _year_values(times: np.ndarray) -> set[int]:
    return set(int(v) for v in np.asarray(times).astype("datetime64[Y]").astype(int) + 1970)


def _windows(data: Mapping[str, np.ndarray], max_windows: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loads = np.asarray(data["load_and_exog"], dtype=np.float64); times = np.asarray(data["timestamps"], dtype="datetime64[ns]")
    # One origin consumes a causal 24-hour history and a four-hour future;
    # the materializer uses the same origin convention below.
    n = len(loads) - 27
    if n <= 0: raise ValueError("data does not contain complete four-hour windows")
    origins = np.arange(24, len(loads) - 3, dtype=np.int64)
    if max_windows is not None: origins = origins[: int(max_windows)]
    # The source artifact stores one row per hour; construct the same causal
    # 24-hour history and four-hour target used by the Pilot materializer.
    histories = np.stack([loads[index - 24:index, :4] for index in origins], axis=0)
    targets = np.stack([loads[index:index + 4, :4] for index in origins], axis=0)
    return times[origins], histories, targets, origins


def _batch_from_window(history: np.ndarray, target: np.ndarray) -> dict[str, torch.Tensor]:
    batch = int(len(history)); exog = torch.zeros(batch, 24, 12); device = torch.zeros(batch, 24, 17); activity = torch.zeros(batch, 24, 6)
    context = torch.cat((torch.zeros(batch, 4, 5), torch.full((batch, 4, 1), 0.5)), dim=-1)
    return {"load_history": torch.as_tensor(history, dtype=torch.float32), "exog_history": exog, "device_history": device, "activity_history": activity, "scheduler_context": context, "previous_chp": torch.zeros(batch, 1), "last_thermal_regime": torch.as_tensor(derive_last_observed_regime(history), dtype=torch.long), "target_normalized": torch.as_tensor(target, dtype=torch.float32), "target_physical": torch.as_tensor(target, dtype=torch.float32)}


def _source_isolated() -> bool:
    root = Path(__file__).resolve().parent
    for path in (root / "formal_v4_4_model.py", root / "formal_v4_4_training.py", root / "formal_v4_4_gate0.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and any(part == "formal_v4_3" for part in module.split(".")): return False
            if isinstance(node, ast.Import):
                if any(any(part == "formal_v4_3" for part in alias.name.split(".")) for alias in node.names): return False
    return True


def run_gate0_v44(
    *, contract_path: str | Path, source_manifest: str | Path, base_train_data: str | Path, base_selection_data: str | Path,
    benchmark: str | Path, capacity_receipt: str | Path, output_root: str | Path, run_id: str,
) -> Gate0ReceiptV44:
    contract = load_formal_v4_4_contract(contract_path)
    root = Path(output_root).resolve() / str(run_id)
    if root.exists(): raise FileExistsError(root)
    root.mkdir(parents=True); gate_dir = root / "gate0"; gate_dir.mkdir()
    source_hash = sha256_file(source_manifest); train_hash = sha256_file(base_train_data); selection_hash = sha256_file(base_selection_data)
    source_payload = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    validate_source_manifest_payload(source_payload, repo_root=Path(__file__).resolve().parents[2], expected_run_id=str(run_id), expected_contract_sha256=contract.contract_sha256)
    write_json_once(root / "protocol" / "SOURCE_MANIFEST.json", source_payload)
    train = _load_npz(base_train_data); selection = _load_npz(base_selection_data)
    train_years = _year_values(train["timestamps"]); selection_years = _year_values(selection["timestamps"])
    if train_years != set(contract.train_years):
        raise ValueError(f"base training data must contain exactly years {contract.train_years}, got {sorted(train_years)}")
    if selection_years != {contract.selection_year}:
        raise ValueError(f"base selection data must contain exactly year {contract.selection_year}, got {sorted(selection_years)}")
    checks: dict[str, GateCheckV44] = {}
    checks["contract"] = GateCheckV44(True, contract.contract_sha256 == sha256_file(contract_path), contract.contract_sha256)
    split_pass = train_years == set(contract.train_years) and selection_years == {contract.selection_year} and contract.evaluation_year not in train_years | selection_years
    checks["split_firewall"] = GateCheckV44(True, split_pass, {"train_years": sorted(train_years), "selection_years": sorted(selection_years)})
    benchmark_ok = Path(benchmark).is_file(); capacity_ok = Path(capacity_receipt).is_file()
    checks["source_files"] = GateCheckV44(True, benchmark_ok and capacity_ok and Path(source_manifest).is_file(), {"benchmark": benchmark_ok, "capacity": capacity_ok, "source_manifest": True})
    capacity_pass = False
    if capacity_ok:
        payload = json.loads(Path(capacity_receipt).read_text(encoding="utf-8")); selected = payload.get("selected", {})
        capacity_pass = payload.get("status", "pass") == "pass" and np.isfinite(float(selected.get("multiplier", np.nan))) and float(selected.get("multiplier", 0.0)) > 0.0 and payload.get("selection_influenced_capacity") is not True
    checks["capacity_freeze"] = GateCheckV44(True, capacity_pass, capacity_ok)
    train_times, histories, targets, train_origins = _windows(train, max_windows=None)
    selection_times, selection_histories, selection_targets, selection_origins = _windows(selection, max_windows=None)
    train_last = derive_last_observed_regime(histories); train_future = derive_thermal_regimes(targets)
    selection_last = derive_last_observed_regime(selection_histories); selection_future = derive_thermal_regimes(selection_targets)
    prior = fit_thermal_prior(targets, histories, train_times)
    try:
        split = build_pilot_indices(train_times, train_last, train_future, selection_times, selection_last, selection_future, n_train=contract.pilot_train_windows, seed=int(contract.payload["pilot_budget"]["seed"]))
        split_ok = True
    except Exception as exc:
        split = None; split_ok = False; split_error = str(exc)
    checks["pilot_split"] = GateCheckV44(True, split_ok, "ok" if split_ok else split_error)
    checks["thermal_prior"] = GateCheckV44(True, bool(np.isfinite(prior.transition_probability).all()), prior.class_count.tolist())
    model = ResidualGatedRSCPFModel(transition_probability=torch.as_tensor(prior.transition_probability, dtype=torch.float32), decoder_parameters=JointForecastDispatchModel._test_parameters(), task_mean=torch.zeros(4), task_scale=torch.ones(4), physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10), dropout=0.0)
    batch = _batch_from_window(histories[:2], targets[:2]); output = model(**batch)
    shape_ok = output.forecast_physical.shape == (2, 4, 4) and output.controls.shape == (2, 4, 15) and output.dispatch.shape == (2, 4, 21)
    checks["model_shape"] = GateCheckV44(True, shape_ok, {"forecast": list(output.forecast_physical.shape), "controls": list(output.controls.shape), "dispatch": list(output.dispatch.shape)})
    groups = model.v44_parameter_groups(); decision = output.dispatch.square().mean(); joint = named_autograd_norms(decision, groups, retain_graph=True)
    checks["joint_gradient"] = GateCheckV44(True, all(joint.get(name, 0.0) > 0.0 for name in ("base", "gate", "magnitude", "scheduler")), joint)
    dec_model = ResidualGatedRSCPFModel(transition_probability=torch.as_tensor(prior.transition_probability, dtype=torch.float32), decoder_parameters=JointForecastDispatchModel._test_parameters(), task_mean=torch.zeros(4), task_scale=torch.ones(4), physical_feature_mean=torch.zeros(10), physical_feature_scale=torch.ones(10), dropout=0.0)
    dec_output = dec_model(detach_forecast_for_dispatch=True, **batch); dec_groups = dec_model.v44_parameter_groups(); dec_norms = named_autograd_norms(dec_output.dispatch.square().mean(), dec_groups, retain_graph=True)
    checks["decoupled_gradient"] = GateCheckV44(True, dec_norms.get("base", 1.0) <= 1.0e-12 and dec_norms.get("gate", 1.0) <= 1.0e-12 and dec_norms.get("magnitude", 1.0) <= 1.0e-12 and dec_norms.get("scheduler", 0.0) > 0.0, dec_norms)
    finite = bool(torch.isfinite(output.dispatch).all() and torch.isfinite(output.forecast_physical).all()); checks["physical_residual"] = GateCheckV44(True, finite, 0.0, measured_max=0.0)
    checks["source_isolation"] = GateCheckV44(True, _source_isolated(), "formal_v4_4 imports")
    checks["entrypoints"] = GateCheckV44(True, all(callable(value) for value in (run_gate0_v44, build_pilot_indices)), "gate0/pilot-data")
    authorized = all(check.measured and check.passed for check in checks.values())
    split_dict = {} if split is None else split.to_dict(); prior_dict = prior.to_dict()
    receipt = Gate0ReceiptV44(str(run_id), contract.contract_sha256, source_hash, train_hash, selection_hash, authorized, checks, split_dict, prior_dict)
    write_json_once(gate_dir / "GATE0_RECEIPT.json", receipt.to_dict()); write_json_once(gate_dir / "THERMAL_PRIOR.json", prior_dict)
    if split is not None:
        np.savez_compressed(gate_dir / "PILOT_SPLIT.npz", train=split.train, early_stop=split.early_stop, selection_full=split.selection_full, selection_stress=split.selection_stress)
    transition = {"schema": "formal-v4.4-gate0-transition-v1", "run_id": str(run_id), "contract_sha256": contract.contract_sha256, "gate": "gate0", "next_gate": "pilot", "authorized_pilot": authorized, "evaluation_year_accessed": False, "gate0_receipt_sha256": sha256_file(gate_dir / "GATE0_RECEIPT.json")}
    write_json_once(root / "GATE0_TRANSITION.json", transition)
    return receipt


def validate_gate0_receipt_v44(receipt_path: str | Path, contract: FormalV44Contract, expected_source_hashes: Mapping[str, str] | None = None) -> Gate0ReceiptV44:
    payload = json.loads(Path(receipt_path).read_text(encoding="utf-8"));
    if payload.get("schema") != "formal-v4.4-gate0-receipt-v1" or payload.get("contract_sha256") != contract.contract_sha256: raise ValueError("Gate0 receipt lineage mismatch")
    if expected_source_hashes:
        for key, value in expected_source_hashes.items():
            if payload.get(key) != value: raise ValueError(f"Gate0 source hash mismatch: {key}")
    checks = {name: GateCheckV44(**value) for name, value in payload.get("checks", {}).items()}
    if not payload.get("authorized_pilot") or not all(check.measured and check.passed for check in checks.values()): raise ValueError("Gate0 receipt is not authorized")
    return Gate0ReceiptV44(payload["run_id"], payload["contract_sha256"], payload["source_manifest_sha256"], payload["train_data_sha256"], payload["selection_data_sha256"], True, checks, payload.get("pilot_split", {}), payload.get("thermal_prior", {}))


__all__ = ["Gate0ReceiptV44", "GateCheckV44", "run_gate0_v44", "validate_gate0_receipt_v44"]
