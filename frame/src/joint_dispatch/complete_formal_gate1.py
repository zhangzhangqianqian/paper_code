"""Real Gate 1 orchestration for the frozen complete-v1 experiment.

The complete-v1 contract is intentionally independent from the legacy v4.2
gate files.  This module adapts the already-tested v4.2 data, training and
chronological settlement kernels while stamping every artifact with the
complete-v1 contract and refusing to read the sealed 2020 split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import numpy as np

from .complete_formal_contract import CompleteFormalContract, MethodSeedKey, STOCHASTIC_METHOD_IDS
from .complete_formal_execution import audit_complete_formal
from .complete_formal_gate1_recovery import (
    Gate1RecoveryInspection,
    build_recovery_manifest,
    inspect_recovery_source,
    materialize_candidate,
    restore_differentiable_lp_artifact,
)
from .formal_v4_2_artifacts import sha256_file, write_once_json
from .formal_v4_2_data import NormalizationReceiptV42
from .formal_v4_2_gate2_training import (
    Gate2DataBundle,
    TrainedMethodArtifact,
    _load_normalization,
    _manifest_hash,
    _subset,
    load_window_split,
    train_differentiable_lp,
    train_direct_policy,
    train_official_itransformer_pto,
    train_rsc_family,
    train_scheme2r_pto,
)
from .formal_v4_2_gate2_execution import (
    Gate2RowKey,
    _perfect_information_method,
    _registered_from_artifact,
    execute_gate2_row,
)
from .formal_v4_2_training import StageBudgetV42
from .formal_v4_diffopt import DifferentiableIESLayer
from .formal_v4_method_adapter import build_formal_v4_method_adapter
from .formal_v4_2_methods import build_v42_method


@dataclass(frozen=True)
class Gate1RunConfig:
    contract_path: Path
    gate0_transition_path: Path
    source_run_root: Path
    output_root: Path
    run_id: str
    smoke: bool = False
    resume_from: Path | None = None


@dataclass(frozen=True)
class Gate1DataBundle:
    train: Any
    selection: Any
    normalization: NormalizationReceiptV42
    legacy: Gate2DataBundle
    parameters: Mapping[str, Any]
    lineage: Mapping[str, str]
    source_run_root: Path


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _years(split: Any) -> tuple[int, ...]:
    values = split.target_times.astype("datetime64[Y]").astype(int) + 1970
    return tuple(int(value) for value in np.unique(values))


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def expected_gate1_rows(contract: CompleteFormalContract) -> tuple[MethodSeedKey, ...]:
    """Return the exact 35 stochastic + 2 deterministic Gate 1 rows."""

    rows = contract.expected_rows("gate1")
    if len(rows) != 37:
        raise ValueError("complete-v1 Gate 1 must contain exactly 37 rows")
    return rows


def _validate_gate0(contract: CompleteFormalContract, transition_path: Path) -> dict[str, Any]:
    if not transition_path.is_file():
        raise PermissionError("Gate 1 requires a Gate 0 transition")
    transition = _json(transition_path)
    if transition.get("contract_sha256") != contract.contract_sha256:
        raise PermissionError("Gate 0 transition contract hash differs from complete-v1")
    if transition.get("authorized_gate1_training") is not True:
        raise PermissionError("Gate 0 did not authorize Gate 1 training")
    if transition.get("evaluation_year_accessed") is not False or transition.get("excluded_year_accessed") is not False:
        raise PermissionError("Gate 0 transition reports forbidden year access")
    return transition


def _source_benchmark_matches(source_root: Path, contract: CompleteFormalContract) -> None:
    """Ensure reused v4.2 windows were generated with the frozen capacities."""

    benchmark = source_root / "gate0" / "benchmark" / "STANDARD_IES_BENCHMARK.yaml"
    if not benchmark.is_file():
        raise FileNotFoundError(f"source run benchmark is missing: {benchmark}")
    try:
        import yaml
        payload = yaml.safe_load(benchmark.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - dependency failure is explicit
        raise RuntimeError("Gate 1 requires PyYAML to validate the source benchmark") from exc
    values = payload.get("values", {}) if isinstance(payload, Mapping) else {}
    # The old benchmark receipt predates the complete-v1 ledger and omits a
    # few scalar defaults (for example ``surplus_penalty``).  Window
    # materialization is sensitive to the capacities, efficiencies and ramp;
    # the remaining scalar defaults are taken from the frozen complete-v1
    # contract below when the dispatch kernel is called.
    required = (
        "grid_import_capacity", "chp_electric_capacity", "chp_heat_capacity",
        "gas_boiler_capacity", "electric_chiller_capacity", "absorption_chiller_capacity",
        "bess_power_capacity", "bess_energy_capacity", "pv_capacity", "wt_capacity",
        "chp_electric_efficiency", "chp_heat_efficiency", "gas_boiler_efficiency",
        "electric_chiller_cop", "absorption_chiller_cop", "bess_roundtrip_efficiency",
        "chp_ramp_fraction",
    )
    expected_values = contract.capacity_parameters
    for name in required:
        expected = expected_values[name]
        if name not in values:
            raise PermissionError(f"source benchmark lacks required parameter {name}")
        if not np.isclose(float(values[name]), float(expected), rtol=0.0, atol=1.0e-8):
            raise PermissionError(f"source benchmark parameter differs from complete-v1: {name}")


def load_gate1_data(config: Gate1RunConfig, contract: CompleteFormalContract) -> Gate1DataBundle:
    """Load only the train and 2019 selection windows with strict lineage."""

    contract.validate()
    _validate_gate0(contract, config.gate0_transition_path)
    source_root = config.source_run_root.resolve()
    _source_benchmark_matches(source_root, contract)
    train_path = source_root / "gate1" / "TRAIN_WINDOWS.npz"
    selection_path = source_root / "gate1" / "SELECTION_WINDOWS.npz"
    normalization_path = source_root / "gate1" / "NORMALIZATION.json"
    for path in (train_path, selection_path, normalization_path):
        if not path.is_file():
            raise FileNotFoundError(f"Gate 1 source artifact is missing: {path}")
    train = load_window_split(train_path)
    selection = load_window_split(selection_path)
    if train.split != "train" or selection.split != "selection":
        raise PermissionError("Gate 1 source windows have invalid split labels")
    if _years(train) != contract.train_years:
        raise PermissionError("Gate 1 training windows are not exactly 2015-2018")
    if _years(selection) != (contract.selection_year,):
        raise PermissionError("Gate 1 selection windows are not exactly 2019")
    if len(train) != 34959 or len(selection) != contract.selection_origin_count:
        raise ValueError("Gate 1 source window counts differ from the frozen complete-v1 protocol")
    if not np.all(np.diff(selection.target_times) == np.timedelta64(1, "h")):
        raise ValueError("Gate 1 selection windows are not hourly chronological")
    normalization = _load_normalization(normalization_path)
    if normalization.train_years != contract.train_years:
        raise PermissionError("Gate 1 normalization is not train-only")
    source_manifest_path = source_root / "protocol" / "SOURCE_MANIFEST.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError("Gate 1 source manifest is missing")
    source_hash = sha256_file(source_manifest_path)
    legacy = Gate2DataBundle(
        train=train,
        calibration=selection,
        evaluation=selection,
        normalization=normalization,
        train_manifest_sha256=_manifest_hash(train, "complete_gate1_train_2015_2018"),
        calibration_manifest_sha256=_manifest_hash(selection, "complete_gate1_selection_2019"),
        evaluation_manifest_sha256=_manifest_hash(selection, "complete_gate1_selection_2019"),
        normalization_sha256=sha256_file(normalization_path),
    )
    lineage = {
        "source_manifest_sha256": source_hash,
        "train_manifest_sha256": legacy.train_manifest_sha256,
        "calibration_manifest_sha256": legacy.calibration_manifest_sha256,
        "evaluation_manifest_sha256": legacy.evaluation_manifest_sha256,
        "normalization_sha256": legacy.normalization_sha256,
        "train_windows_sha256": sha256_file(train_path),
        "selection_windows_sha256": sha256_file(selection_path),
    }
    parameters = dict(contract.capacity_parameters)
    return Gate1DataBundle(train, selection, normalization, legacy, parameters, lineage, source_root)


def _smoke_data(data: Gate1DataBundle) -> Gate1DataBundle:
    train = _subset(data.train, np.arange(min(2, len(data.train)), dtype=np.int64))
    selection = _subset(data.selection, np.arange(min(3, len(data.selection)), dtype=np.int64))
    legacy = replace(
        data.legacy, train=train, calibration=selection, evaluation=selection,
        train_manifest_sha256=_manifest_hash(train, "smoke_train"),
        calibration_manifest_sha256=_manifest_hash(selection, "smoke_selection"),
        evaluation_manifest_sha256=_manifest_hash(selection, "smoke_selection"),
    )
    lineage = dict(data.lineage)
    lineage.update({
        "train_manifest_sha256": legacy.train_manifest_sha256,
        "calibration_manifest_sha256": legacy.calibration_manifest_sha256,
        "evaluation_manifest_sha256": legacy.evaluation_manifest_sha256,
    })
    return replace(data, train=train, selection=selection, legacy=legacy, lineage=lineage)


def _budget(
    contract: CompleteFormalContract,
    *,
    multiplier: float = 1.0,
    decision_multiplier: float = 1.0,
    smoke: bool = False,
) -> StageBudgetV42:
    training = contract.payload["training"]
    loss = training["loss"]
    epochs = 1 if smoke else int(training["common"]["max_epochs_per_stage"])
    return StageBudgetV42(
        max_epochs=epochs,
        minimum_epochs=1,
        decision_start=float(loss["decision_weight_start"]),
        decision_final=float(loss["decision_weight_final"]) * float(decision_multiplier),
        ramp_epochs=1 if smoke else int(loss["curriculum_ramp_epochs"]),
        forecast_weight=float(loss["forecast_weight"]),
        imitation_start=float(loss["imitation_weight_start"]),
        imitation_final=float(loss["imitation_weight_final"]),
        forecaster_lr=1.0e-5 * float(multiplier),
        scheduler_lr=1.0e-3,
        weight_decay=1.0e-4,
        max_grad_norm=float(training["common"]["gradient_clip_norm"]),
    )


def _legacy_freeze(
    contract: CompleteFormalContract,
    data: Gate1DataBundle,
    *,
    multiplier: float,
    c_ref: float,
    decision_multiplier: float = 1.0,
) -> dict[str, Any]:
    return {
        "contract_sha256": contract.contract_sha256,
        "source_manifest_sha256": data.lineage["source_manifest_sha256"],
        "selected_candidate_value": float(multiplier),
        "selected_decision_multiplier": float(decision_multiplier),
        "c_ref": float(c_ref),
        "complete_formal_adapter": True,
    }


def _itransformer_receipt(data: Gate1DataBundle, output_dir: Path) -> tuple[dict[str, Any], Path, Path]:
    source_receipt_path = data.source_run_root / "protocol" / "ITRANSFORMER_SOURCE_RECEIPT.json"
    diff_receipt_path = data.source_run_root / "protocol" / "DIFFERENTIABLE_LP_ENVIRONMENT_RECEIPT.json"
    if not source_receipt_path.is_file() or not diff_receipt_path.is_file():
        raise FileNotFoundError("Gate 1 requires the verified iTransformer and DiffLP receipts")
    source = _json(source_receipt_path)
    legacy = {"schema_version": "formal-v4.1-itransformer-source-v1"}
    for name in ("source_root", "repository", "commit", "backbone_class", "imported_file_hashes", "license_file", "license_sha256", "reproduction_level", "verified"):
        if name in source:
            legacy[name] = source[name]
    receipt = output_dir / "ITRANSFORMER_ADAPTER_RECEIPT.json"
    write_once_json(receipt, legacy)
    return source, receipt, diff_receipt_path


def select_gate1_hyperparameters(
    data: Gate1DataBundle,
    contract: CompleteFormalContract,
    output_dir: Path,
    *,
    smoke: bool = False,
    recovery: Gate1RecoveryInspection | None = None,
) -> Mapping[str, Any]:
    """Select the two frozen Gate 1 search parameters on 2019 only.

    Gate 0's resource projection reserves four RSC and four DiffLP trials.
    The selection rows are kept under ``gate1/search`` and are never included
    in the 37-row paper matrix.
    """

    search_dir = output_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        payload = {
            "schema_version": "rsc-pf-complete-formal-gate1-search-v1",
            "status": "smoke-skipped",
            "rsc_selected_decision_multiplier": 1.0,
            "difflp_selected_learning_rate": 1.0e-5,
            "trials": [],
        }
        write_once_json(output_dir / "GATE1_SEARCH.json", payload)
        return payload

    work = data
    source, adapter_receipt, diff_receipt_path = _itransformer_receipt(work, output_dir)
    diff_receipt = _json(diff_receipt_path)
    grids = contract.payload["training"]["search"]
    rsc_grid = tuple(float(value) for value in grids["rsc_pf_decision_multiplier_grid"])
    diff_grid = tuple(float(value) for value in grids["difflp_learning_rate_grid"])
    trials: list[dict[str, Any]] = []
    rsc_candidates: list[tuple[float, float]] = []
    for multiplier in rsc_grid:
        freeze = _legacy_freeze(contract, work, multiplier=1.0, decision_multiplier=multiplier, c_ref=1.0)
        trial_root = search_dir / f"rsc_multiplier_{multiplier:g}"
        key = MethodSeedKey("RSC-PF", 2026)
        evidence = recovery.by_key.get(next(
            candidate for candidate in recovery.by_key
            if candidate.family == "RSC-PF" and candidate.value == multiplier
        )) if recovery is not None else None
        action = "trained"
        if evidence is not None and evidence.state == "reusable-complete":
            destination_row = materialize_candidate(evidence, trial_root)
            row = _json(destination_row / "COMPLETE_GATE1_ROW_RECEIPT.json")
            action = "reused-complete"
        else:
            family = train_rsc_family(
                2026, work.legacy, freeze, work.parameters, trial_root / "rows",
                budget=_budget(contract, decision_multiplier=multiplier),
            )
            row = evaluate_gate1_row(key, family["RSC-PF"], work, trial_root, contract=contract, paper_result=False)
        score = float(row.get("penalized_objective", float("inf")))
        eligible = bool(row.get("finite", False) and row.get("physical_feasible", False))
        trials.append({
            "family": "RSC-PF", "value": multiplier, "score": score, "eligible": eligible,
            "action": action, "training_reused": action == "reused-complete",
            "evaluation_reused": action == "reused-complete",
        })
        if eligible:
            rsc_candidates.append((score, multiplier))
    diff_candidates: list[tuple[float, float]] = []
    for learning_rate in diff_grid:
        multiplier = learning_rate / 1.0e-5
        freeze = _legacy_freeze(contract, work, multiplier=multiplier, c_ref=1.0)
        trial_root = search_dir / f"difflp_lr_{learning_rate:g}"
        key = MethodSeedKey("Differentiable-LP", 2026)
        evidence = recovery.by_key.get(next(
            candidate for candidate in recovery.by_key
            if candidate.family == "Differentiable-LP" and candidate.value == learning_rate
        )) if recovery is not None else None
        action = "trained"
        if evidence is not None and evidence.state == "reusable-complete":
            destination_row = materialize_candidate(evidence, trial_root)
            row = _json(destination_row / "COMPLETE_GATE1_ROW_RECEIPT.json")
            action = "reused-complete"
        elif evidence is not None and evidence.state == "reusable-checkpoint":
            destination_row = materialize_candidate(evidence, trial_root)
            artifact = restore_differentiable_lp_artifact(destination_row, work, contract)
            row = evaluate_gate1_row(key, artifact, work, trial_root, contract=contract, paper_result=False)
            action = "reused-training-reran-evaluation"
        else:
            artifact = train_differentiable_lp(
                2026, work.legacy, freeze, diff_receipt, work.parameters,
                trial_root / "rows" / "Differentiable-LP" / "2026",
                budget=_budget(contract, multiplier=multiplier), layer=DifferentiableIESLayer(work.parameters), micro_batch_size=8,
            )
            row = evaluate_gate1_row(key, artifact, work, trial_root, contract=contract, paper_result=False)
        score = float(row.get("penalized_objective", float("inf")))
        eligible = bool(row.get("finite", False) and row.get("physical_feasible", False))
        trials.append({
            "family": "Differentiable-LP", "value": learning_rate, "score": score, "eligible": eligible,
            "action": action, "training_reused": action != "trained",
            "evaluation_reused": action == "reused-complete",
        })
        if eligible:
            diff_candidates.append((score, learning_rate))
    selected_rsc = min(rsc_candidates, default=(float("inf"), rsc_grid[0]))[1]
    selected_diff = min(diff_candidates, default=(float("inf"), diff_grid[0]))[1]
    payload = {
        "schema_version": "rsc-pf-complete-formal-gate1-search-v1",
        "status": "pass" if rsc_candidates and diff_candidates else "fallback-no-eligible-trial",
        "selection_year": contract.selection_year,
        "selection_origin_count": contract.selection_origin_count,
        "rsc_grid": list(rsc_grid),
        "difflp_grid": list(diff_grid),
        "rsc_selected_decision_multiplier": selected_rsc,
        "difflp_selected_learning_rate": selected_diff,
        "trials": trials,
        "source_manifest_sha256": data.lineage["source_manifest_sha256"],
        "contract_sha256": contract.contract_sha256,
        "recovery_used": recovery is not None,
        "reused_candidate_count": sum(1 for trial in trials if trial.get("training_reused")),
    }
    write_once_json(output_dir / "GATE1_SEARCH.json", payload)
    return payload


def train_gate1_matrix(
    data: Gate1DataBundle,
    contract: CompleteFormalContract,
    output_dir: Path,
    *,
    smoke: bool = False,
    selected_hyperparameters: Mapping[str, Any] | None = None,
) -> Mapping[MethodSeedKey, TrainedMethodArtifact | None]:
    """Train all 35 stochastic rows using the frozen complete-v1 roster."""

    work = _smoke_data(data) if smoke else data
    rows_root = output_dir / "rows"
    rows_root.mkdir(parents=True, exist_ok=True)
    selected = selected_hyperparameters or {}
    rsc_decision_multiplier = float(selected.get("rsc_selected_decision_multiplier", 1.0))
    difflp_lr = float(selected.get("difflp_selected_learning_rate", 1.0e-5))
    budget = _budget(contract, decision_multiplier=rsc_decision_multiplier, smoke=smoke)
    freeze = _legacy_freeze(contract, work, multiplier=1.0, decision_multiplier=rsc_decision_multiplier, c_ref=1.0)
    source, adapter_receipt, diff_receipt_path = _itransformer_receipt(work, output_dir)
    source_receipt = source
    diff_receipt = _json(diff_receipt_path)
    artifacts: dict[MethodSeedKey, TrainedMethodArtifact | None] = {}
    for seed in (2026, 2027, 2028, 2029, 2030):
        family = train_rsc_family(seed, work.legacy, freeze, work.parameters, rows_root, budget=budget)
        teacher_path = rows_root / "_shared" / str(seed) / "TEACHER.npz"
        with np.load(teacher_path, allow_pickle=False) as teacher_payload:
            teacher_dispatch = teacher_payload["dispatch"]
        direct = train_direct_policy(seed, work.legacy, freeze, work.parameters, rows_root / "Direct-Policy" / str(seed), budget=budget, teacher_dispatch=teacher_dispatch)
        scheme = train_scheme2r_pto(seed, work.legacy, freeze, rows_root / "Scheme2R-PTO" / str(seed), budget=budget)
        source_root = Path(str(source.get("source_root", "")))
        if not source_root.is_absolute():
            frame_root = work.source_run_root.parents[2]
            source_root = frame_root / (Path(*source_root.parts[1:]) if source_root.parts and source_root.parts[0].lower() == "frame" else source_root)
        official = train_official_itransformer_pto(
            seed, work.legacy, freeze, source_receipt, rows_root / "Official iTransformer-PTO" / str(seed),
            source_root=source_root, receipt_path=adapter_receipt, budget=budget,
        )
        diff = train_differentiable_lp(
            seed, work.legacy, freeze, diff_receipt, work.parameters,
            rows_root / "Differentiable-LP" / str(seed),
            budget=_budget(contract, multiplier=difflp_lr / 1.0e-5, smoke=smoke),
            layer=DifferentiableIESLayer(work.parameters), micro_batch_size=1 if smoke else 8,
        )
        by_method = {**family, "Direct-Policy": direct, "Scheme2R-PTO": scheme, "Official iTransformer-PTO": official, "Differentiable-LP": diff}
        for method_id in STOCHASTIC_METHOD_IDS:
            artifacts[MethodSeedKey(method_id, seed)] = by_method[method_id]
    return artifacts


def _complete_row_payload(
    raw: Mapping[str, Any],
    key: MethodSeedKey,
    *,
    origin_count: int,
    smoke: bool,
    paper_result: bool | None = None,
) -> dict[str, Any]:
    physical = float(raw.get("physical_residual_max", float("inf")))
    calls = int(raw.get("optimizer_calls", 0))
    return {
        **dict(raw),
        "method_id": key.method_id,
        "seed": key.seed,
        "stage": "gate1",
        "complete": True,
        "synthetic": bool(smoke),
        "paper_result": (not bool(smoke)) if paper_result is None else bool(paper_result),
        "origin_count": int(origin_count),
        "inference_lp_calls": calls,
        "finite": bool(np.isfinite(float(raw.get("penalized_objective", np.nan))) and np.isfinite(physical)),
        "chronological": True,
        "physical_feasible": bool(physical <= 1.0e-5),
        "shortage_total": float(raw.get("shortage_energy", 0.0)),
        "evaluation_year_accessed": False,
        "excluded_year_accessed": False,
        "test_set_accessed": False,
    }


def evaluate_gate1_row(
    key: MethodSeedKey,
    artifact: TrainedMethodArtifact | None,
    data: Gate1DataBundle,
    output_dir: Path,
    *,
    contract: CompleteFormalContract,
    smoke: bool = False,
    paper_result: bool | None = None,
) -> Mapping[str, Any]:
    work = _smoke_data(data) if smoke else data
    lineage = {"contract_sha256": contract.contract_sha256, **work.lineage}
    row_dir = output_dir / "rows" / key.method_id / (str(key.seed) if key.seed is not None else "deterministic")
    if key.method_id == "Perfect-Information-MPC":
        method = _perfect_information_method(work.parameters)
    elif key.method_id == "Seasonal-Naive-PTO":
        adapter = build_formal_v4_method_adapter(
            key.method_id, work.parameters, task_mean=work.normalization.field_mean["load"],
            task_scale=work.normalization.field_scale["load"], normalization=work.normalization,
        )
        method = build_v42_method(key.method_id, parameters=work.parameters, adapter=adapter)
    else:
        if artifact is None:
            raise ValueError(f"trained artifact is missing for {key.method_id}/{key.seed}")
        diff_layer = DifferentiableIESLayer(work.parameters) if key.method_id == "Differentiable-LP" else None
        method = _registered_from_artifact(artifact, work.legacy, work.parameters, diff_layer=diff_layer)
    raw = execute_gate2_row(
        Gate2RowKey(key.method_id, key.seed), method, work.legacy, work.parameters, row_dir, lineage,
    )
    payload = _complete_row_payload(raw, key, origin_count=len(work.selection), smoke=smoke, paper_result=paper_result)
    # The existing v4.2 row receipt remains intact for its own validator; the
    # complete-v1 row is a separate, immutable contract envelope.
    write_once_json(row_dir / "COMPLETE_GATE1_ROW_RECEIPT.json", payload)
    return payload


def evaluate_gate1_matrix(
    data: Gate1DataBundle,
    contract: CompleteFormalContract,
    output_dir: Path,
    artifacts: Mapping[MethodSeedKey, TrainedMethodArtifact | None],
    *,
    smoke: bool = False,
) -> Mapping[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for key in expected_gate1_rows(contract):
        artifact = artifacts.get(key)
        payload = evaluate_gate1_row(key, artifact, data, output_dir, contract=contract, smoke=smoke)
        label = f"{key.method_id}/{key.seed if key.seed is not None else 'deterministic'}"
        rows[label] = payload
    return rows


def run_complete_gate1(config: Gate1RunConfig, *, smoke: bool | None = None) -> Mapping[str, Any]:
    """Run complete Gate 1; importing this module never starts training."""

    smoke_mode = config.smoke if smoke is None else bool(smoke)
    contract = CompleteFormalContract.from_path(config.contract_path)
    expected_gate1_rows(contract)
    gate0 = _validate_gate0(contract, config.gate0_transition_path)
    root = config.output_root.resolve() / config.run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite Gate 1 run: {root}")
    gate1_dir = root / "gate1"
    protocol_dir = root / "protocol"
    gate1_dir.mkdir(parents=True)
    protocol_dir.mkdir(parents=True)
    started = time.perf_counter()
    recovery_manifest_path: Path | None = None
    recovery: Gate1RecoveryInspection | None = None
    try:
        config_for_data = replace(config, smoke=False)
        data = load_gate1_data(config_for_data, contract)
        # Pin the exact inputs beside the complete-v1 receipts.  These are
        # byte-for-byte copies of the validated source-run artifacts; no
        # future/evaluation array is copied or opened.
        shutil.copy2(data.source_run_root / "gate1" / "TRAIN_WINDOWS.npz", gate1_dir / "TRAIN_WINDOWS.npz")
        shutil.copy2(data.source_run_root / "gate1" / "SELECTION_WINDOWS.npz", gate1_dir / "SELECTION_WINDOWS.npz")
        shutil.copy2(data.source_run_root / "gate1" / "NORMALIZATION.json", gate1_dir / "NORMALIZATION.json")
        write_once_json(gate1_dir / "DATA_LINEAGE.json", {
            "contract_sha256": contract.contract_sha256,
            "gate0_transition_sha256": sha256_file(config.gate0_transition_path),
            **data.lineage,
            "source_run_root": str(data.source_run_root),
            "train_years": list(contract.train_years),
            "selection_year": contract.selection_year,
            "selection_origin_count": contract.selection_origin_count,
            "evaluation_year_accessed": False,
        })
        if config.resume_from is not None:
            recovery = inspect_recovery_source(
                config.resume_from, contract, data, config.gate0_transition_path,
            )
            write_once_json(gate1_dir / "RECOVERY_PLAN.json", {
                "schema_version": "rsc-pf-complete-formal-gate1-recovery-plan-v1",
                "source_run_root": str(recovery.source_root),
                "candidate_states": [
                    {
                        "family": item.key.family,
                        "value": item.key.value,
                        "method_id": item.key.method_id,
                        "seed": item.key.seed,
                        "state": item.state,
                        "reason": item.reason,
                    }
                    for item in recovery.candidates
                ],
                "source_modified": False,
                "evaluation_year_accessed": False,
            })
        search = select_gate1_hyperparameters(data, contract, gate1_dir, smoke=smoke_mode, recovery=recovery)
        if recovery is not None and not smoke_mode:
            manifest = build_recovery_manifest(
                recovery, root, contract, data, config.gate0_transition_path, search,
            )
            recovery_manifest_path = gate1_dir / "RECOVERY_MANIFEST.json"
            write_once_json(recovery_manifest_path, manifest)
        artifacts = train_gate1_matrix(data, contract, gate1_dir, smoke=smoke_mode, selected_hyperparameters=search)
        rows = evaluate_gate1_matrix(data, contract, gate1_dir, artifacts, smoke=smoke_mode)
        if smoke_mode:
            audit_status = "smoke-not-authorized"
            failures: tuple[str, ...] = ()
            authorized = False
        else:
            audit = audit_complete_formal(contract, rows, origin_count=contract.selection_origin_count, stage="gate1")
            audit_status = audit.status
            failures = audit.failures
            audit_payload = {
                "stage": audit.stage,
                "status": audit.status,
                "execution_receipt_sha256": audit.execution_receipt_sha256,
                "recomputed_row_count": audit.recomputed_row_count,
                "failures": list(audit.failures),
                "evaluation_year_accessed": audit.evaluation_year_accessed,
                "excluded_year_accessed": audit.excluded_year_accessed,
                "rows": {label: asdict(item) for label, item in audit.rows.items()},
            }
            write_once_json(gate1_dir / "GATE1_AUDIT.json", audit_payload)
            authorized = audit.status == "pass" and len(rows) == 37
        write_once_json(gate1_dir / "ROWS.json", rows)
        evidence_selection_count = len(_smoke_data(data).selection) if smoke_mode else contract.selection_origin_count
        evidence = {
            "schema_version": "rsc-pf-complete-formal-gate1-evidence-v1",
            "run_id": config.run_id,
            "contract_sha256": contract.contract_sha256,
            "gate0_transition_sha256": sha256_file(config.gate0_transition_path),
            "row_count": len(rows),
            "expected_row_count": 37,
            "selection_origin_count": evidence_selection_count,
            "train_years": list(contract.train_years),
            "selection_year": contract.selection_year,
            "evaluation_year_accessed": False,
            "excluded_years_accessed": False,
            "synthetic": bool(smoke_mode),
            "paper_result": bool(authorized and not smoke_mode),
            "audit_status": audit_status,
            "failures": list(failures),
            "runtime_seconds": time.perf_counter() - started,
            "gate0_projected_total_hours": gate0.get("projected_total_hours"),
            "search_status": search.get("status"),
            "selected_rsc_decision_multiplier": search.get("rsc_selected_decision_multiplier"),
            "selected_difflp_learning_rate": search.get("difflp_selected_learning_rate"),
            "recovery_manifest_sha256": None if recovery_manifest_path is None else sha256_file(recovery_manifest_path),
        }
        write_once_json(gate1_dir / "GATE1_EVIDENCE.json", evidence)
        transition = {
            "schema_version": "rsc-pf-complete-formal-gate1-transition-v1",
            "run_id": config.run_id,
            "contract_sha256": contract.contract_sha256,
            "authorized_gate2": bool(authorized),
            "evaluation_year_accessed": False,
            "test_set_accessed": False,
            "synthetic": bool(smoke_mode),
            "paper_result": bool(authorized and not smoke_mode),
            "audit_status": audit_status,
            "row_count": len(rows),
            "gate1_evidence_sha256": sha256_file(gate1_dir / "GATE1_EVIDENCE.json"),
        }
        write_once_json(protocol_dir / "GATE1_TRANSITION.json", transition)
        return {**evidence, "authorized_gate2": bool(authorized), "transition_path": str(protocol_dir / "GATE1_TRANSITION.json")}
    except Exception as exc:
        write_once_json(gate1_dir / "GATE1_FAILURE.json", {
            "schema_version": "rsc-pf-complete-formal-gate1-failure-v1",
            "run_id": config.run_id,
            "contract_sha256": contract.contract_sha256,
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "evaluation_year_accessed": False,
            "resume_from": None if config.resume_from is None else str(config.resume_from.resolve()),
            "recovery_manifest_sha256": None if recovery_manifest_path is None else sha256_file(recovery_manifest_path),
        })
        raise


__all__ = [
    "Gate1DataBundle", "Gate1RunConfig", "evaluate_gate1_matrix", "evaluate_gate1_row",
    "expected_gate1_rows", "load_gate1_data", "run_complete_gate1", "train_gate1_matrix",
]
