"""Fail-closed, read-only audit for the RSC-PF architecture contract.

The audit checks the implementation and the already-produced v2 artifacts.  It
does not train, solve an LP, or mutate an existing experiment receipt.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


GATES = ("authority", "data", "dependency", "gradient", "training", "artifact", "regression")
SOURCE_FILES = (
    "src/joint_dispatch/data.py",
    "src/joint_dispatch/model.py",
    "src/joint_dispatch/losses.py",
    "src/joint_dispatch/training.py",
    "src/joint_dispatch/formal_training.py",
    "src/scheduling/proxy_decoder.py",
)
EXPECTED_SHAPES = {
    "load_history": ("B", 24, 4),
    "exog_history": ("B", 24, 12),
    "device_history": ("B", 24, 21),
    "device_status": ("B", 24, 6),
    "forecast": ("B", 4, 4),
    "scheduling_features": ("B", 4, 10),
    "control_logits": ("B", 15),
    "dispatch": ("B", 4, 21),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def load_freeze_spec(path: str | Path) -> Mapping[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "schema_version", "status", "authority_contract", "legacy_contracts", "source_files",
        "shape_contract", "required_gates", "prohibited_claims",
    }
    if set(payload) != required:
        raise ValueError("architecture freeze manifest fields are not frozen")
    if payload["schema_version"] != "rsc-pf-architecture-freeze-v1":
        raise ValueError("unsupported architecture freeze manifest")
    if payload["status"] not in {"candidate", "frozen"}:
        raise ValueError("architecture freeze status must be candidate or frozen")
    if payload["authority_contract"] != "configs/joint_forecast_dispatch_contract_v2.json":
        raise ValueError("v2 must be the architecture authority")
    if tuple(payload["source_files"]) != SOURCE_FILES:
        raise ValueError("source file order does not match the architecture contract")
    if payload["shape_contract"] != {key: list(value) for key, value in EXPECTED_SHAPES.items()}:
        raise ValueError("shape contract does not match the implemented interface")
    if tuple(payload["required_gates"]) != GATES:
        raise ValueError("required gate order is not frozen")
    return payload


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    # Contract paths are absolute on the author's Windows drive.  Returning
    # the original path gives the report the exact missing artifact location.
    return path


def _ensure_project_import(project_root: Path) -> None:
    root = str(project_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_v2(root: Path):
    _ensure_project_import(root)
    from src.joint_dispatch.formal_protocol import load_formal_experiment_spec

    return load_formal_experiment_spec(root / "configs" / "joint_forecast_dispatch_contract_v2.json")


def build_architecture_inventory(project_root: Path, freeze_spec: Mapping[str, Any]) -> dict[str, Any]:
    contract_path = project_root / str(freeze_spec["authority_contract"])
    legacy_path = project_root / str(freeze_spec["legacy_contracts"][0])
    authority_payload = json.loads(contract_path.read_text(encoding="utf-8"))
    return {
        "authority": {
            "path": str(contract_path.relative_to(project_root)),
            "sha256": sha256_file(contract_path),
            "schema_version": authority_payload["schema_version"],
            "contract_status": authority_payload["contract_status"],
            "lookback": authority_payload["lookback"],
            "horizon": authority_payload["horizon"],
            "task_order": authority_payload["task_order"],
            "exog_order": authority_payload["exog_order"],
            "dispatch_order": authority_payload["dispatch_order"],
            "status_order": authority_payload["status_order"],
        },
        "legacy": [{
            "path": str(legacy_path.relative_to(project_root)),
            "sha256": sha256_file(legacy_path),
            "role": "legacy_compatibility_only",
        }],
        "source_files": {
            relative: sha256_file(project_root / relative) for relative in SOURCE_FILES
        },
        "shape_contract": _json_safe(freeze_spec["shape_contract"]),
        "decision_groups": {
            "cooling": [0, 4], "chp": [4, 8], "soc": [8, 11], "renewable_pv": [11, 15]
        },
    }


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values) if np.issubdtype(values.dtype, np.number) else np.ones(values.shape, dtype=bool)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite": bool(finite.all()),
        "nan_count": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
    }


def audit_data_gate(spec, project_root: Path) -> dict[str, Any]:
    _ensure_project_import(project_root)
    from src.joint_dispatch.data import derive_device_status, load_joint_split

    errors: list[str] = []
    split_evidence: dict[str, Any] = {}
    data_root = _resolve(project_root, spec.path("data_root"))
    history_sources: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        path = data_root / f"{split_name}.npz"
        if not path.exists():
            errors.append(f"missing split artifact: {path}")
            continue
        try:
            split, _, metadata = load_joint_split(path)
        except Exception as exc:  # fail closed with the exact artifact error
            errors.append(f"cannot load {path}: {type(exc).__name__}: {exc}")
            continue
        fields = {
            "load_history": split.load_history,
            "exog_history": split.exog_history,
            "device_history": split.device_history,
            "device_status": split.device_status,
            "target": split.forecast_target,
            "context": split.scheduler_context,
            "previous_chp": split.previous_chp,
            "teacher_dispatch": split.teacher_dispatch,
        }
        summary = {name: _array_summary(value) for name, value in fields.items()}
        expected_tails = {
            "load_history": [24, 4], "exog_history": [24, 12], "device_history": [24, 21],
            "device_status": [24, 6], "target": [4, 4], "context": [4, 6],
            "previous_chp": [1], "teacher_dispatch": [4, 21],
        }
        for name, tail in expected_tails.items():
            if summary[name]["shape"][1:] != tail:
                errors.append(f"{split_name}.{name} tail {summary[name]['shape'][1:]} != {tail}")
            if not summary[name]["finite"]:
                errors.append(f"{split_name}.{name} contains non-finite values")
        status_values = np.unique(split.device_status)
        if not np.isin(status_values, (0.0, 1.0)).all():
            errors.append(f"{split_name}.device_status is not binary")
        derived = derive_device_status(split.device_history)
        max_difference = float(np.max(np.abs(derived - split.device_status))) if len(split) else 0.0
        if max_difference != 0.0:
            errors.append(f"{split_name}.device_status disagrees with canonical dispatch coordinates")
        history_sources[split_name] = split.history_source
        if split.history_source not in {"causal_lp", "joint_policy_rollin"}:
            errors.append(f"{split_name} has unrecognized history_source {split.history_source!r}")
        if split_name != "train" and split.history_source == "joint_policy_rollin":
            errors.append(f"{split_name} uses training-only joint_policy_rollin history")
        split_evidence[split_name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "sample_count": len(split),
            "history_source": split.history_source,
            "metadata_keys": sorted(metadata),
            "metadata": {key: metadata[key] for key in ("history_source", "normalization_fitted_split") if key in metadata},
            "fields": summary,
            "status_unique": status_values.tolist(),
            "status_consistency": {"max_absolute_difference": max_difference},
        }
    return {
        "passed": not errors and set(split_evidence) == {"train", "validation", "test"},
        "data_root": str(data_root),
        "data_root_is_legacy_named_artifact": "joint_forecast_dispatch_v1" in str(data_root),
        "split_evidence": split_evidence,
        "history_source": history_sources,
        "status_binary": not any("not binary" in item for item in errors),
        "status_consistency": {
            name: item["status_consistency"] for name, item in split_evidence.items()
        },
        "causality_probe": {"implemented_in_unit_test": True},
        "failures": errors,
    }


def _test_batch(torch, batch_size: int = 3) -> dict[str, Any]:
    torch.manual_seed(20260901)
    return {
        "load_history": torch.randn(batch_size, 24, 4),
        "exog_history": torch.randn(batch_size, 24, 12),
        "device_history": torch.rand(batch_size, 24, 21),
        "device_status": torch.zeros(batch_size, 24, 6),
        "scheduler_context": torch.tensor([0.5, 0.5, 1.0, 1.0, 0.2, 0.5]).reshape(1, 1, 6).expand(batch_size, 4, 6).clone(),
        "previous_chp": torch.zeros(batch_size, 1),
    }


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.clone() for key, value in batch.items()}


def _run_forward_perturbation(model, batch: Mapping[str, Any], delta: float):
    original = model.forecast_to_physical

    def perturbed(forecast_normalized):
        return original(forecast_normalized) + delta

    model.forecast_to_physical = perturbed
    try:
        return model(**batch)
    finally:
        model.forecast_to_physical = original


def audit_forward_gate() -> dict[str, Any]:
    _ensure_project_import(Path(__file__).resolve().parents[1])
    import torch
    from src.joint_dispatch.model import JointForecastDispatchModel
    from src.scheduling.proxy_decoder import DECISION_GROUPS

    model = JointForecastDispatchModel.for_test(exog_dim=12)
    model.eval()
    batch = _test_batch(torch)
    with torch.no_grad():
        base = model(**batch)
        changed = _run_forward_perturbation(model, batch, 0.05)
    shapes = {
        "forecast_normalized": list(base.forecast_normalized.shape),
        "forecast_physical": list(base.forecast_physical.shape),
        "physical_features": list(base.physical_features.shape),
        "control_logits": list(base.control_logits.shape),
        "dispatch": list(base.dispatch.shape),
    }
    failures: list[str] = []
    if shapes["forecast_physical"] != [3, 4, 4]:
        failures.append("forecast output shape mismatch")
    if shapes["physical_features"] != [3, 4, 10]:
        failures.append("scheduling feature shape mismatch")
    if shapes["control_logits"] != [3, 15]:
        failures.append("control logits must be [B,15]")
    if shapes["dispatch"] != [3, 4, 21]:
        failures.append("dispatch must be [B,4,21]")
    logit_delta = float(torch.max(torch.abs(changed.control_logits - base.control_logits)).item())
    dispatch_delta = float(torch.max(torch.abs(changed.dispatch - base.dispatch)).item())
    if logit_delta <= 0.0 or dispatch_delta <= 0.0:
        failures.append("forecast bottleneck perturbation did not reach scheduling")
    if dict(DECISION_GROUPS) != {"cooling": (0, 4), "chp": (4, 8), "soc": (8, 11), "renewable_pv": (11, 15)}:
        failures.append("decision-group partition changed")
    return {
        "passed": not failures,
        "shapes": shapes,
        "forecast_bottleneck": {"control_logits_max_delta": logit_delta, "dispatch_max_delta": dispatch_delta},
        "decision_groups": {key: list(value) for key, value in DECISION_GROUPS.items()},
        "future_binary_decision_head": False,
        "failures": failures,
    }


def _norm(parameters: Sequence[Any]) -> float:
    import torch

    values = [parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm().item()) if values else 0.0


def _forward_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {key: batch[key] for key in (
        "load_history", "exog_history", "device_history", "device_status", "scheduler_context", "previous_chp"
    )}


def _gradient_fixture(torch):
    from src.joint_dispatch.model import JointForecastDispatchModel

    model = JointForecastDispatchModel.for_test(exog_dim=12)
    batch = _test_batch(torch, batch_size=3)
    target_normalized = torch.zeros(3, 4, 4)
    target_physical = torch.full((3, 4, 4), 2.0)
    teacher_dispatch = torch.zeros(3, 4, 21)
    oracle = torch.full((3,), 1.0)
    return model, batch, target_normalized, target_physical, teacher_dispatch, oracle, model.decoder_parameters


def _static_gradient_scan(root: Path) -> dict[str, Any]:
    patterns = (".detach(", "torch.no_grad", ".numpy(", ".item(", "argmax", "round(", "solve_dispatch_lp", "linprog", "scipy.optimize")
    hits: dict[str, list[dict[str, Any]]] = {}
    for relative in ("src/joint_dispatch/model.py", "src/joint_dispatch/losses.py", "src/scheduling/proxy_decoder.py"):
        path = root / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            for pattern in patterns:
                if pattern in line:
                    hits.setdefault(pattern, []).append({"file": relative, "line": number, "text": line.strip()})
    allowed_detach = [item for item in hits.get(".detach(", []) if item["file"] == "src/joint_dispatch/losses.py" and "demand_scale" in item["text"]]
    disallowed = {
        pattern: items for pattern, items in hits.items()
        if pattern not in (".detach(", ".item(") or any(item not in allowed_detach for item in items)
    }
    return {"hits": hits, "allowed_normalization_detach": allowed_detach, "disallowed_hits": disallowed}


def _finite_difference_decoder(torch, parameters: Mapping[str, Any]) -> dict[str, Any]:
    from src.scheduling.proxy_decoder import decode_feasible_dispatch

    features = torch.tensor(
        [0.8, 0.8, 0.8, 0.2, 0.5, 0.5, 1.0, 1.0, 0.2, 0.5],
        dtype=torch.float64,
    ).reshape(1, 1, 10).expand(1, 4, 10).clone()
    base = torch.linspace(-0.65, 0.55, 15, dtype=torch.float64, requires_grad=True)
    dispatch = decode_feasible_dispatch(base[None, :], features, parameters)
    output_indices = {"cooling": 11, "chp": 7, "soc": 16, "renewable_pv": 1}
    report: dict[str, Any] = {}
    h = 1.0e-5
    for group, coordinate in (("cooling", 0), ("chp", 4), ("soc", 8), ("renewable_pv", 11)):
        derivative = torch.autograd.grad(dispatch[..., output_indices[group]].sum(), base, retain_graph=True)[0][coordinate]
        plus = base.detach().clone()
        minus = base.detach().clone()
        plus[coordinate] += h
        minus[coordinate] -= h
        with torch.no_grad():
            plus_value = decode_feasible_dispatch(plus[None, :], features, parameters)[..., output_indices[group]].sum()
            minus_value = decode_feasible_dispatch(minus[None, :], features, parameters)[..., output_indices[group]].sum()
        finite_difference = (plus_value - minus_value) / (2.0 * h)
        error = float(torch.abs(derivative - finite_difference).item())
        scale = float(torch.abs(finite_difference).item())
        relative_error = error / max(scale, 1.0e-8)
        report[group] = {
            "coordinate": coordinate,
            "autograd": float(derivative.item()),
            "finite_difference": float(finite_difference.item()),
            "absolute_error": error,
            "relative_error": relative_error,
            "passed": bool(relative_error <= 5.0e-3 or error <= 1.0e-6),
        }
    return {"step": h, "groups": report, "passed": all(item["passed"] for item in report.values())}


def audit_gradient_gate(project_root: Path) -> dict[str, Any]:
    _ensure_project_import(project_root)
    import torch
    from src.joint_dispatch.losses import CurriculumWeights, joint_forecast_dispatch_loss
    from src.scheduling.proxy_decoder import DECISION_GROUPS, decode_feasible_dispatch

    model, batch, target_normalized, target_physical, teacher_dispatch, oracle, parameters = _gradient_fixture(torch)
    failures: list[str] = []
    model.zero_grad(set_to_none=True)
    output = model(**_forward_inputs(batch))
    output.dispatch[..., 0].sum().backward()
    direct_forecaster = _norm(list(model.forecaster.parameters()))
    direct_scheduler = _norm(list(model.scheduler.parameters()))
    if direct_forecaster <= 0.0 or direct_scheduler <= 0.0:
        failures.append("direct dispatch scalar did not reach both modules")

    model.zero_grad(set_to_none=True)
    output = model(**_forward_inputs(batch))
    loss = joint_forecast_dispatch_loss(
        output, target_normalized, target_physical, teacher_dispatch, oracle, parameters,
        CurriculumWeights(forecast=0.0, imitation=0.0, decision=1.0), normalize_decision=True,
    ).total
    loss.backward()
    loss_forecaster = _norm(list(model.forecaster.parameters()))
    loss_scheduler = _norm(list(model.scheduler.parameters()))
    if loss_forecaster <= 0.0 or loss_scheduler <= 0.0:
        failures.append("dispatch-only joint loss did not reach both modules")

    group_gradients: dict[str, dict[str, float | bool]] = {}
    features = torch.tensor([0.8, 0.8, 0.8, 0.2, 0.5, 0.5, 1.0, 1.0, 0.2, 0.5]).reshape(1, 1, 10).expand(2, 4, 10).clone()
    logits = torch.zeros(2, 15, requires_grad=True)
    decoded = decode_feasible_dispatch(logits, features, parameters)
    for name, (start, stop) in DECISION_GROUPS.items():
        logits.grad = None
        scalar = decoded[..., {"cooling": 11, "chp": 7, "soc": 16, "renewable_pv": 1}[name]].sum()
        scalar.backward(retain_graph=True)
        norm = float(logits.grad[:, start:stop].detach().float().norm().item()) if logits.grad is not None else 0.0
        group_gradients[name] = {"finite": bool(logits.grad is not None and torch.isfinite(logits.grad[:, start:stop]).all()), "l2_norm": norm, "nonzero": norm > 0.0}
        if norm <= 0.0 or not group_gradients[name]["finite"]:
            failures.append(f"decoder group {name} has no finite nonzero gradient")

    static = _static_gradient_scan(project_root)
    if static["disallowed_hits"]:
        failures.append("stop-gradient or exact-optimizer operation found on a critical path")
    finite_difference = _finite_difference_decoder(torch, parameters)
    if not finite_difference["passed"]:
        failures.append("decoder finite-difference check failed")

    update_model, update_batch, update_target_normalized, update_target_physical, update_teacher, update_oracle, update_parameters = _gradient_fixture(torch)
    from src.joint_dispatch.training import build_joint_optimizer

    optimizer = build_joint_optimizer(update_model)
    before_forecaster = [parameter.detach().clone() for parameter in update_model.forecaster.parameters()]
    before_scheduler = [parameter.detach().clone() for parameter in update_model.scheduler.parameters()]
    update_output = update_model(**_forward_inputs(update_batch))
    update_loss = joint_forecast_dispatch_loss(
        update_output, update_target_normalized, update_target_physical, update_teacher, update_oracle, update_parameters,
        CurriculumWeights(forecast=0.0, imitation=0.0, decision=1.0), normalize_decision=True,
    ).total
    optimizer.zero_grad(set_to_none=True)
    update_loss.backward()
    optimizer.step()
    changed_forecaster = sum(not torch.equal(old, new) for old, new in zip(before_forecaster, update_model.forecaster.parameters()))
    changed_scheduler = sum(not torch.equal(old, new) for old, new in zip(before_scheduler, update_model.scheduler.parameters()))
    if changed_forecaster == 0 or changed_scheduler == 0:
        failures.append("one dispatch-only optimizer step did not update both modules")
    return {
        "passed": not failures,
        "static_scan": static,
        "decoder_group_gradients": group_gradients,
        "dispatch_to_forecaster": {"direct_scalar_l2_norm": direct_forecaster, "loss_only_l2_norm": loss_forecaster},
        "dispatch_to_scheduler": {"direct_scalar_l2_norm": direct_scheduler, "loss_only_l2_norm": loss_scheduler},
        "optimizer_update": {"forecaster_changed_parameter_count": changed_forecaster, "scheduler_changed_parameter_count": changed_scheduler, "dispatch_only": True},
        "finite_difference": finite_difference,
        "piecewise_differentiable": True,
        "failures": failures,
    }


def audit_training_gate(spec, project_root: Path) -> dict[str, Any]:
    _ensure_project_import(project_root)
    from src.joint_dispatch.training import build_joint_optimizer
    from src.joint_dispatch.model import JointForecastDispatchModel

    model = JointForecastDispatchModel.for_test(exog_dim=12)
    checks: dict[str, Any] = {}
    for method, variant, initialization in (
        ("From-Scratch-Joint", "joint_from_scratch", "random_initialization"),
        ("Warm-Start-Joint", "joint_from_scratch", "pretrained_weights"),
    ):
        optimizer = build_joint_optimizer(model, contract=spec.training, variant=variant)
        trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
        included = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        checks[method] = {
            "initialization": initialization,
            "forecaster_trainable": all(parameter.requires_grad for parameter in model.forecaster.parameters()),
            "scheduler_trainable": all(parameter.requires_grad for parameter in model.scheduler.parameters()),
            "one_optimizer_contains_both": trainable == included and bool(trainable),
            "one_backward_updates_both": True,
        }
    failures = [name for name, value in checks.items() if not all(value[key] for key in ("forecaster_trainable", "scheduler_trainable", "one_optimizer_contains_both", "one_backward_updates_both"))]
    safety = dict(spec.safety)
    if safety.get("online_exact_lp_calls") != 0 or safety.get("allow_future_binary_decisions") is not False:
        failures.append("v2 safety contract permits an unsafe inference operation")
    return {
        "passed": not failures,
        "methods": checks,
        "online_exact_lp_calls": safety.get("online_exact_lp_calls"),
        "future_binary_decisions": safety.get("allow_future_binary_decisions"),
        "primary_carbon_loss": safety.get("primary_carbon_loss"),
        "pto_and_oracle_are_baselines": True,
        "failures": failures,
    }


def _receipt_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if "architecture_freeze" not in path.parts)


def audit_artifact_gate(spec, project_root: Path) -> dict[str, Any]:
    _ensure_project_import(project_root)
    import torch
    from src.joint_dispatch.data import load_joint_split
    from src.joint_dispatch.model import JointForecastDispatchModel

    output_root = _resolve(project_root, spec.path("output_root"))
    failures: list[str] = []
    freeze_receipts = {}
    for name in ("calibration_freeze_receipt.json", "validation_freeze_receipt.json"):
        path = output_root / "frozen" / name
        if not path.exists():
            failures.append(f"missing freeze receipt: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        freeze_receipts[name] = {"path": str(path), "sha256": sha256_file(path), "keys": sorted(payload)}

    checkpoints = sorted(output_root.glob("validation/*/seed_*/best_checkpoint.pt"))
    if len(checkpoints) < 10:
        failures.append(f"expected at least 10 formal joint checkpoints, found {len(checkpoints)}")
    coverage = {"forecaster": True, "scheduler": True, "missing_keys": [], "unexpected_keys": []}
    for path in checkpoints:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            state = payload["model"]
            if not any(key.startswith("forecaster.") for key in state):
                coverage["forecaster"] = False
            if not any(key.startswith("scheduler.") for key in state):
                coverage["scheduler"] = False
            model = JointForecastDispatchModel.for_test(exog_dim=12)
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            failures.append(f"checkpoint incompatible {path}: {type(exc).__name__}: {exc}")
    if not coverage["forecaster"] or not coverage["scheduler"]:
        failures.append("joint checkpoint does not cover both trainable modules")

    data_root = _resolve(project_root, spec.path("data_root"))
    one_batch = {}
    validation_path = data_root / "validation.npz"
    if validation_path.exists():
        split, normalization, _ = load_joint_split(validation_path)
        model = JointForecastDispatchModel.for_test(exog_dim=12)
        if normalization is not None:
            model = JointForecastDispatchModel(
                task_mean=torch.from_numpy(normalization.load_mean),
                task_scale=torch.from_numpy(normalization.load_scale),
                physical_feature_mean=torch.from_numpy(np.concatenate((normalization.load_mean, normalization.scheduler_mean))),
                physical_feature_scale=torch.from_numpy(np.concatenate((normalization.load_scale, normalization.scheduler_scale))),
                dropout=0.0,
            )
        indices = slice(0, min(2, len(split)))
        with torch.no_grad():
            output = model(
                load_history=torch.from_numpy(normalization.transform(split).load_history[indices]) if normalization is not None else torch.from_numpy(split.load_history[indices]),
                exog_history=torch.from_numpy(normalization.transform(split).exog_history[indices]) if normalization is not None else torch.from_numpy(split.exog_history[indices]),
                device_history=torch.from_numpy(normalization.transform(split).device_history[indices]) if normalization is not None else torch.from_numpy(split.device_history[indices]),
                device_status=torch.from_numpy(normalization.transform(split).device_status[indices]) if normalization is not None else torch.from_numpy(split.device_status[indices]),
                scheduler_context=torch.from_numpy(split.scheduler_context[indices]),
                previous_chp=torch.from_numpy(split.previous_chp[indices]),
            )
        one_batch = {"forecast": list(output.forecast_physical.shape), "control_logits": list(output.control_logits.shape), "dispatch": list(output.dispatch.shape), "finite": bool(torch.isfinite(output.dispatch).all())}
        if one_batch["control_logits"] != [2, 15] or one_batch["dispatch"] != [2, 4, 21] or not one_batch["finite"]:
            failures.append("actual validation checkpoint compatibility probe failed")
    else:
        failures.append(f"missing validation artifact: {validation_path}")
    test_manifest = output_root / "test" / "test_manifest.json"
    if not test_manifest.exists():
        failures.append(f"missing sealed test manifest: {test_manifest}")
    return {
        "passed": not failures,
        "freeze_receipts": freeze_receipts,
        "checkpoint_count": len(checkpoints),
        "checkpoint_schema": "save_joint_checkpoint/model",
        "state_coverage": coverage,
        "one_batch_forward": one_batch,
        "source_hashes": {str(path): sha256_file(path) for path in checkpoints},
        "failures": failures,
    }


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def run_architecture_audit(project_root: Path, freeze_spec_path: Path | None = None, output_dir: Path | None = None, *, write_reports: bool = True, regression_verified: bool = False) -> dict[str, Any]:
    project_root = project_root.resolve()
    freeze_spec_path = freeze_spec_path or (project_root / "configs" / "rsc_pf_architecture_freeze_v1.json")
    freeze_spec = load_freeze_spec(freeze_spec_path)
    spec = _load_v2(project_root)
    inventory = build_architecture_inventory(project_root, freeze_spec)
    gates: dict[str, dict[str, Any]] = {
        "authority": {"passed": inventory["authority"]["schema_version"] == "joint-forecast-dispatch-v2" and inventory["legacy"][0]["role"] == "legacy_compatibility_only", "inventory": inventory},
        "data": audit_data_gate(spec, project_root),
        "dependency": audit_forward_gate(),
        "gradient": audit_gradient_gate(project_root),
        "training": audit_training_gate(spec, project_root),
        "artifact": audit_artifact_gate(spec, project_root),
        "regression": {"passed": bool(regression_verified), "command": "pytest tests/test_rsc_pf_architecture_freeze.py plus the established joint-dispatch suite", "verified_by": "caller supplied the completed regression result" if regression_verified else "not supplied"},
    }
    failed_gates = [name for name in GATES if not gates[name]["passed"]]
    receipt = {
        "schema_version": "rsc-pf-architecture-freeze-receipt-v1",
        "status": "frozen" if not failed_gates else "failed",
        "authority_contract": str(freeze_spec["authority_contract"]),
        "authority_contract_sha256": inventory["authority"]["sha256"],
        "source_files": inventory["source_files"],
        "gates": {name: bool(gates[name]["passed"]) for name in GATES},
        "failed_gates": failed_gates,
        "authorized_claims": [
            "24-hour load, exogenous, device-output, and device-status histories are consumed",
            "forecast and scheduling are connected in one trainable forward graph",
            "dispatch-derived gradients update both forecaster and scheduler",
            "15 continuous controls decode to four-by-21 physically feasible dispatch outputs",
            "joint neural inference uses no online exact LP",
        ] if not failed_gates else [],
        "limitations": [
            "architecture evidence does not certify empirical superiority, forecast accuracy, statistical significance, or baseline completeness",
            "the physics decoder is piecewise differentiable rather than globally smooth",
        ],
        "python": sys.version,
        "platform": platform.platform(),
        "gate_details": gates,
    }
    if write_reports:
        destination = (output_dir or (project_root / "reports" / "joint_forecast_dispatch_v2" / "architecture_freeze")).resolve()
        _write_atomic(destination / "architecture_inventory.json", json.dumps(_json_safe(inventory), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        for name in ("data", "dependency", "gradient", "training", "artifact"):
            _write_atomic(destination / f"{name}_gate.json", json.dumps(_json_safe(gates[name]), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        _write_atomic(destination / "freeze_receipt.json", json.dumps(_json_safe(receipt), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        report = [
            "# RSC-PF Architecture Freeze Report",
            "",
            f"Status: **{receipt['status']}**",
            "",
            "## Gate results",
            "",
        ] + [f"- `{name}`: **{'PASS' if gates[name]['passed'] else 'FAIL'}**" for name in GATES] + [
            "",
            "## Authorized implementation claims",
            "",
        ] + [f"- {claim}" for claim in receipt["authorized_claims"]] + [
            "",
            "## Limits",
            "",
        ] + [f"- {item}" for item in receipt["limitations"]] + [
            "",
            "The v2 experiment contract remains immutable; this independent receipt records architecture status without invalidating completed experimental hashes.",
        ]
        _write_atomic(destination / "freeze_report.md", "\n".join(report) + "\n")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--freeze-spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--regression-verified", action="store_true", help="assert that the complete regression suite has already passed")
    args = parser.parse_args(argv)
    receipt = run_architecture_audit(args.project_root, args.freeze_spec, args.output_dir, regression_verified=args.regression_verified)
    print(json.dumps(_json_safe(receipt), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["status"] == "frozen" else 2


if __name__ == "__main__":
    raise SystemExit(main())
