"""Branch-authorized 2021 forecasting for the topology robustness protocol.

The module has one intentionally strict boundary: the Kitakyushu reader is
called only after a valid, immutable Phase-A branch-freeze file has been
loaded.  Nothing in this module can use 2021 validation evidence to choose a
model or a hyperparameter.  The test pass is a one-shot four-step forecast
using the same 24-to-4 interface as the main experiments.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, Tuple

import numpy as np

from .baselines import regression_metrics
from .data_pipeline import SplitSpec, build_protocol_windows, save_json, select_training_frame
from .formal_model_factory import build_formal_forecasting_model
from .kitakyushu_pipeline import (
    KITAKYUSHU_EXOG_COLUMNS,
    KITAKYUSHU_TASKS,
    clean_kitakyushu_dataframe,
    read_kitakyushu_canonical,
)
from .models import count_trainable_parameters
from .topology_protocol_contract import (
    FORMAL_SEEDS,
    PHASE_A_SEEDS as CONTRACT_PHASE_A_SEEDS,
    TASK_ORDER,
    VALID_BRANCHES,
    load_topology_contract,
    resolve_phase_b_matrix,
)
from .training import (
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    load_checkpoint,
    make_dataloader,
    set_reproducible,
)


LOOKBACK = 24
HORIZON = 4
PHASE_B_YEARS: Tuple[int, ...] = (2017, 2018, 2019, 2020, 2021)
TRAIN_YEARS: Tuple[int, ...] = (2017, 2018, 2019)
VALIDATION_YEAR = 2020
TEST_YEAR = 2021
PHASE_B_FORMAL_SEEDS: Tuple[int, ...] = tuple(FORMAL_SEEDS)
PHASE_A_SEEDS: Tuple[int, ...] = tuple(CONTRACT_PHASE_A_SEEDS)
PHASE_A_MODELS: Tuple[Tuple[str, str], ...] = (
    ("scheme2r", "H4"),
    ("dynamic_symmetric", "H1"),
)
JOINT_MODELS: Tuple[Tuple[str, str], ...] = (
    ("hard_share", "H2"),
    ("dynamic_symmetric", "H1"),
    ("mmoe_lite", "external_fixed"),
    ("ple_lite", "ple_lite_fixed_v1"),
    ("scheme2r", "H4"),
)
SPLIT = SplitSpec(
    train_start="2017-01-01 00:00:00",
    train_end="2019-12-31 23:00:00",
    validation_start="2020-01-01 00:00:00",
    validation_end="2020-12-31 23:00:00",
    test_start="2021-01-01 00:00:00",
    test_end="2021-12-31 23:00:00",
)

# These values are the already frozen Phase-A/Stage-7.7 configurations.  They
# are deliberately not searched or inferred from 2021 results.
PHASE_B_HYPERPARAMETERS: dict[tuple[str, str], dict[str, object]] = {
    ("hard_share", "H2"): {
        "candidate_id": "H2", "hidden_dim": 32, "kernel_size": 5,
        "dilations": [1, 2, 4], "dropout": 0.1,
        "learning_rate": 0.001, "prediction_head_hidden_dim": 16,
    },
    ("dynamic_symmetric", "H1"): {
        "candidate_id": "H1", "hidden_dim": 16, "kernel_size": 5,
        "dilations": [1, 2, 4], "dropout": 0.0,
        "learning_rate": 0.001, "prediction_head_hidden_dim": 16,
    },
    ("mmoe_lite", "external_fixed"): {
        "candidate_id": "external_fixed", "expert_count": 4,
        "expert_hidden_dim": 32, "representation_dim": 32,
        "dropout": 0.1, "learning_rate": 0.001,
        "prediction_head_hidden_dim": 16,
    },
    ("ple_lite", "ple_lite_fixed_v1"): {
        "candidate_id": "ple_lite_fixed_v1", "shared_expert_count": 2,
        "task_expert_count": 1, "expert_hidden_dim": 32,
        "representation_dim": 32, "dropout": 0.1,
        "learning_rate": 0.001, "prediction_head_hidden_dim": 16,
    },
    ("scheme2r", "H4"): {
        "candidate_id": "H4", "hidden_dim": 32, "kernel_size": 5,
        "dilations": [1, 2, 4], "dropout": 0.1,
        "learning_rate": 0.0005, "scheme2r_kernel_size": 5,
        "scheme2r_dilations": [1, 2, 4], "scheme2r_rank": 8,
        "scheme2r_gate_hidden_dim": 16, "scheme2r_step_embedding_dim": 4,
        "prediction_head_hidden_dim": 16,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _canonical_model(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def validate_phase_b_authorization(freeze: Mapping[str, object]) -> str:
    """Validate the immutable Phase-B authorization and return its branch."""

    branch = str(freeze.get("branch", ""))
    if branch not in VALID_BRANCHES:
        raise ValueError("Phase B requires a valid topology branch freeze")
    if freeze.get("test_year_accessed") is not False:
        raise ValueError("branch freeze must prove that 2021 was sealed during Phase A")
    if freeze.get("validation_year") != VALIDATION_YEAR:
        raise ValueError("branch freeze validation year must be 2020")
    matrix = freeze.get("phase_b_run_matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("branch freeze is missing phase_b_run_matrix")
    models = matrix.get("models")
    seeds = matrix.get("seeds")
    if not isinstance(models, list) or not isinstance(seeds, list):
        raise ValueError("branch freeze matrix must contain models and seeds")
    if tuple(int(year) for year in matrix.get("training_years", ())) != TRAIN_YEARS:
        raise ValueError("Phase B training years must be 2017-2019")
    if int(matrix.get("validation_year", -1)) != VALIDATION_YEAR:
        raise ValueError("Phase B validation year must be 2020")
    if int(matrix.get("test_year", -1)) != TEST_YEAR:
        raise ValueError("Phase B test year must be 2021")
    expected_models = JOINT_MODELS if branch != "core_conclusion_stable" else PHASE_A_MODELS
    expected_seeds = PHASE_B_FORMAL_SEEDS if branch != "core_conclusion_stable" else PHASE_A_SEEDS
    actual_models = tuple((_canonical_model(item.get("model")), str(item.get("candidate_id"))) for item in models if isinstance(item, Mapping))
    if actual_models != expected_models:
        raise ValueError(f"frozen matrix models do not match branch {branch}: {actual_models}")
    if tuple(int(seed) for seed in seeds) != expected_seeds:
        raise ValueError(f"frozen matrix seeds do not match branch {branch}")
    return branch


def load_branch_freeze(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, object]:
    """Load a branch freeze before any data reader is invoked."""

    freeze_path = Path(path)
    if not freeze_path.is_file():
        raise FileNotFoundError(freeze_path)
    if expected_sha256 is not None and _sha256(freeze_path) != expected_sha256:
        raise ValueError("branch-freeze SHA-256 mismatch")
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("branch freeze is not valid JSON") from exc
    if not isinstance(freeze, dict):
        raise ValueError("branch freeze must be a JSON object")
    validate_phase_b_authorization(freeze)
    return freeze


def validate_recorded_input_hashes(
    freeze: Mapping[str, object],
    *,
    contract_path: str | Path | None = None,
    audit_dir: str | Path | None = None,
    current_git_revision: str | None = None,
) -> None:
    """Verify immutable Phase-A inputs before any Phase-B data read."""

    recorded = freeze.get("input_hashes", {})
    if not isinstance(recorded, Mapping):
        raise ValueError("branch freeze input_hashes must be an object")
    if contract_path is not None and recorded.get("contract"):
        path = Path(contract_path)
        if not path.is_file() or _sha256(path) != str(recorded["contract"]):
            raise ValueError("topology contract SHA-256 mismatch")
    audit_hashes = recorded.get("audit_files")
    if audit_dir is not None and audit_hashes:
        root = Path(audit_dir)
        if not isinstance(audit_hashes, Mapping):
            raise ValueError("audit_files hash record must be an object")
        for relative, expected in audit_hashes.items():
            path = root / str(relative)
            if not path.is_file() or _sha256(path) != str(expected):
                raise ValueError(f"topology audit SHA-256 mismatch: {path}")
    if current_git_revision is not None and freeze.get("git_revision") not in {None, "unavailable", current_git_revision}:
        raise ValueError("repository revision differs from the branch-freeze revision")


def validate_phase_b_years(years: Sequence[int]) -> Tuple[int, ...]:
    selected = tuple(int(year) for year in years)
    if selected != PHASE_B_YEARS:
        raise ValueError(f"Phase B must read exactly {PHASE_B_YEARS}, got {selected}")
    return selected


def build_phase_b_run_plan(freeze: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    branch = validate_phase_b_authorization(freeze)
    matrix = freeze["phase_b_run_matrix"]
    assert isinstance(matrix, Mapping)
    rows = []
    for item in matrix["models"]:
        model = _canonical_model(item["model"])
        candidate = str(item["candidate_id"])
        for seed in matrix["seeds"]:
            rows.append({
                "branch": branch,
                "model": model,
                "candidate_id": candidate,
                "seed": int(seed),
                "run_id": f"{branch}/{model}/{candidate}/seed_{int(seed)}",
            })
    return tuple(rows)


def _load_data_after_authorization(
    data_dir: str | Path,
    freeze: Mapping[str, object],
    *,
    reader: Callable[..., tuple[Any, Mapping[str, object]]] = read_kitakyushu_canonical,
) -> tuple[Any, dict[str, object]]:
    """Read Phase-B data only after validating the branch freeze."""

    validate_phase_b_authorization(freeze)
    years = validate_phase_b_years(PHASE_B_YEARS)
    raw, metadata = reader(data_dir, years=years)
    cleaned, report = clean_kitakyushu_dataframe(raw)
    result = dict(metadata)
    result.update({"years_loaded": list(years), "cleaning_report": report})
    return cleaned, result


def build_phase_b_windows(frame) -> tuple[dict[str, dict[str, np.ndarray]], StandardizationStats]:
    raw = {
        name: build_protocol_windows(
            frame, SPLIT, split_name=name, lookback=LOOKBACK, horizon=HORIZON,
            exog_columns=KITAKYUSHU_EXOG_COLUMNS, task_columns=KITAKYUSHU_TASKS,
        ) for name in ("train", "validation", "test")
    }
    stats = StandardizationStats.fit(
        select_training_frame(frame, SPLIT),
        exog_columns=KITAKYUSHU_EXOG_COLUMNS,
        task_columns=KITAKYUSHU_TASKS,
    )
    return {name: stats.transform_windows(value) for name, value in raw.items()}, stats


def _run_one(
    run: Mapping[str, object], run_dir: Path,
    windows: Mapping[str, Mapping[str, np.ndarray]], stats: StandardizationStats,
    *, max_epochs: int = 100, patience: int = 12, threads: int = 8,
) -> dict[str, object]:
    model_name = str(run["model"])
    candidate = str(run["candidate_id"])
    seed = int(run["seed"])
    hp = PHASE_B_HYPERPARAMETERS[(model_name, candidate)]
    trainer = TrainerConfig(
        seed=seed, torch_threads=threads, learning_rate=float(hp["learning_rate"]),
        weight_decay=1e-4, grad_clip_norm=1.0, max_epochs=max_epochs,
        early_stopping_patience=min(patience, max_epochs),
    )
    # Initialize weights only after the frozen run seed is applied.
    set_reproducible(trainer)
    model = build_formal_forecasting_model(
        model_name, hp, exog_dim=len(KITAKYUSHU_EXOG_COLUMNS),
        lookback=LOOKBACK, horizon=HORIZON, task_count=len(KITAKYUSHU_TASKS),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_dataloader(windows["train"], 256, shuffle=True, seed=seed)
    validation_loader = make_dataloader(windows["validation"], 256, shuffle=False)
    test_loader = make_dataloader(windows["test"], 256, shuffle=False)
    started = time.perf_counter()
    history = fit_model(model, train_loader, validation_loader, trainer, run_dir / "best_model.pt", input_mode="loads_and_exog")
    fit_seconds = time.perf_counter() - started
    checkpoint = load_checkpoint(model, run_dir / "best_model.pt", "cpu")
    validation_loss, _, _ = evaluate_model(model, validation_loader, "cpu", input_mode="loads_and_exog")
    test_started = time.perf_counter()
    test_loss, prediction_std, target_std = evaluate_model(model, test_loader, "cpu", input_mode="loads_and_exog")
    test_seconds = time.perf_counter() - test_started
    prediction = stats.inverse_targets(prediction_std)
    target = stats.inverse_targets(target_std)
    stats.save(run_dir / "normalization_stats.npz")
    save_json({"history": history}, run_dir / "history.json")
    np.savez_compressed(
        run_dir / "predictions_test.npz", prediction=prediction, target=target,
        prediction_standardized=prediction_std, target_standardized=target_std,
        target_times=windows["test"]["target_times"],
    )
    metrics = regression_metrics(target, prediction, task_names=KITAKYUSHU_TASKS)
    metrics["test_loss_standardized"] = float(test_loss)
    metrics["validation_loss_standardized"] = float(validation_loss)
    save_json(metrics, run_dir / "metrics_test.json")
    files = ["best_model.pt", "normalization_stats.npz", "history.json", "predictions_test.npz", "metrics_test.json"]
    manifest = {
        "stage": "topology_protocol_pilot_phase_b", "status": "passed",
        "branch": run["branch"], "model": model_name, "candidate_id": candidate,
        "seed": seed, "dataset": "kitakyushu_energy_station",
        "years_loaded": list(PHASE_B_YEARS), "test_set_accessed": True,
        "branch_frozen_before_test": True,
        "test_used_for_selection": False, "future_exogenous_used": False,
        "tasks": list(KITAKYUSHU_TASKS),
        "window": {"lookback": LOOKBACK, "horizon": HORIZON, "output_shape": list(prediction.shape)},
        "sample_counts": {name: int(len(value["target"])) for name, value in windows.items()},
        "parameter_count": int(count_trainable_parameters(model)),
        "fit_seconds": float(fit_seconds), "test_seconds": float(test_seconds),
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_loss": float(checkpoint["best_validation_loss"]),
        "trainer_config": asdict(trainer), "files": files,
        "artifact_sha256": {name: _sha256(run_dir / name) for name in files},
        "git_revision": _git_revision(),
    }
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def _strict_resume_manifest(run_dir: Path, run: Mapping[str, object]) -> dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing Phase-B run manifest: {run_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Phase-B run manifest: {manifest_path}") from exc
    if manifest.get("status") != "passed" or manifest.get("branch") != run["branch"]:
        raise ValueError(f"cannot resume invalid Phase-B run: {run_dir}")
    for name, expected in (manifest.get("artifact_sha256") or {}).items():
        path = run_dir / str(name)
        if not path.is_file() or _sha256(path) != str(expected):
            raise ValueError(f"Phase-B artifact SHA-256 mismatch: {path}")
    if manifest.get("test_set_accessed") is not True or manifest.get("test_used_for_selection") is not False:
        raise ValueError(f"Phase-B run manifest violates test-use policy: {run_dir}")
    return manifest


def run_phase_b(
    data_dir: str | Path, branch_freeze_path: str | Path, output_dir: str | Path,
    *, contract_path: str | Path | None = None, smoke: bool = False,
    resume: bool = False, expected_freeze_sha256: str | None = None,
    audit_dir: str | Path | None = None,
) -> dict[str, object]:
    freeze = load_branch_freeze(branch_freeze_path, expected_sha256=expected_freeze_sha256)
    if contract_path is not None:
        contract = load_topology_contract(contract_path)
        branch = validate_phase_b_authorization(freeze)
        resolve_phase_b_matrix(contract, branch)
        validate_recorded_input_hashes(
            freeze,
            contract_path=contract_path,
            audit_dir=audit_dir,
            current_git_revision=_git_revision(),
        )
    frame, metadata = _load_data_after_authorization(data_dir, freeze)
    windows, stats = build_phase_b_windows(frame)
    if smoke:
        windows = {name: {key: value[:32] for key, value in split.items()} for name, split in windows.items()}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = build_phase_b_run_plan(freeze)
    completed = []
    for run in plan:
        run_dir = root / str(run["model"]) / str(run["candidate_id"]) / f"seed_{run['seed']}"
        manifest_path = run_dir / "run_manifest.json"
        if resume and manifest_path.is_file():
            existing = _strict_resume_manifest(run_dir, run)
            completed.append(existing)
            continue
        if manifest_path.exists() and not resume:
            raise FileExistsError(f"run exists; use --resume: {run_dir}")
        result = _run_one(run, run_dir, windows, stats, max_epochs=1 if smoke else 100, patience=1 if smoke else 12, threads=2 if smoke else 8)
        completed.append(result)
    manifest = {
        "stage": "topology_protocol_pilot_phase_b", "status": "passed",
        "branch": freeze["branch"], "branch_freeze_path": str(Path(branch_freeze_path)),
        "branch_frozen_before_test": True, "test_set_accessed": True,
        "test_used_for_selection": False, "years_loaded": list(PHASE_B_YEARS),
        "run_count_expected": len(plan), "run_count_completed": len(completed),
        "runs": completed, "data_metadata": metadata,
    }
    save_json(manifest, root / "phase_b_manifest.json")
    return manifest


__all__ = [
    "JOINT_MODELS", "PHASE_A_MODELS", "PHASE_A_SEEDS", "PHASE_B_FORMAL_SEEDS",
    "PHASE_B_YEARS", "build_phase_b_run_plan", "build_phase_b_windows",
    "load_branch_freeze", "run_phase_b", "validate_phase_b_authorization",
    "validate_phase_b_years", "validate_recorded_input_hashes",
]
