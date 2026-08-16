"""Validation-only runner for the post-gas-engine topology protocol.

The pilot deliberately stops at the 2020 validation year.  It is therefore
safe to use for deciding whether the equipment-regime change deserves a new
formal branch: no 2021 rows are loaded, no test DataLoader is constructed, and
no test prediction artifact is written.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

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
from .training import (
    StandardizationStats,
    TrainerConfig,
    evaluate_model,
    fit_model,
    load_checkpoint,
    make_dataloader,
)


LOOKBACK = 24
HORIZON = 4
BATCH_SIZE = 256
TORCH_THREADS = 8
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
INPUT_MODE = "loads_and_exog"
PROTOCOL_NAME = "post_ge_regular_operation"
PHASE_A_YEARS: Tuple[int, ...] = (2017, 2018, 2019, 2020)
FORBIDDEN_PHASE_A_YEARS: Tuple[int, ...] = (2021,)
PHASE_A_SPLIT = SplitSpec(
    train_start="2017-01-01 00:00:00",
    train_end="2019-12-31 23:00:00",
    validation_start="2020-01-01 00:00:00",
    validation_end="2020-12-31 23:00:00",
    # The sealed boundary is retained for contract validation only.  This
    # runner never uses it to construct a window or a DataLoader.
    test_start="2021-01-01 00:00:00",
    test_end="2021-12-31 23:00:00",
)
PHASE_A_MODELS: Tuple[Tuple[str, str], ...] = (
    ("scheme2r", "H4"),
    ("dynamic_symmetric", "H1"),
)
PHASE_A_SEEDS: Tuple[int, ...] = (2026, 2027, 2028)


def validate_phase_a_years(years: Sequence[int]) -> Tuple[int, ...]:
    """Validate and return the exact years allowed by Phase A."""

    selected = tuple(int(year) for year in years)
    if selected != PHASE_A_YEARS:
        raise ValueError(
            "Phase A must load exactly 2017, 2018, 2019 and 2020; "
            f"got {selected}"
        )
    if any(year in FORBIDDEN_PHASE_A_YEARS for year in selected):
        raise ValueError("Phase A cannot load the sealed 2021 test year")
    return selected


def load_phase_a_frame(
    data_dir: str | Path,
    years: Sequence[int] = PHASE_A_YEARS,
) -> tuple[pd.DataFrame, dict]:
    """Load and clean only the explicitly approved Phase-A years.

    Validation happens before entering :func:`read_kitakyushu_canonical`, so a
    request containing 2021 cannot cause even one sealed-year file to be read.
    """

    years = validate_phase_a_years(years)
    raw, metadata = read_kitakyushu_canonical(data_dir, years=years)
    cleaned, cleaning_report = clean_kitakyushu_dataframe(raw)
    metadata = dict(metadata)
    metadata["years_loaded"] = list(years)
    metadata["cleaning_report"] = cleaning_report
    return cleaned, metadata


def build_train_validation_windows(
    frame: pd.DataFrame,
    split_spec: SplitSpec = PHASE_A_SPLIT,
    *,
    sample_limit: int | None = None,
) -> tuple[Dict[str, Dict[str, np.ndarray]], StandardizationStats]:
    """Build standardized train/validation windows using train-only statistics.

    This function intentionally has no ``test`` branch.  A sample limit, when
    supplied, truncates each split deterministically after construction and is
    only intended for smoke tests.
    """

    if split_spec != PHASE_A_SPLIT:
        raise ValueError("Phase A requires the frozen post-ge regular-operation split")
    if sample_limit is not None and sample_limit <= 0:
        raise ValueError("sample_limit must be positive when supplied")
    raw_windows = {
        split_name: build_protocol_windows(
            frame,
            split_spec,
            split_name=split_name,
            lookback=LOOKBACK,
            horizon=HORIZON,
            exog_columns=KITAKYUSHU_EXOG_COLUMNS,
            task_columns=KITAKYUSHU_TASKS,
        )
        for split_name in ("train", "validation")
    }
    stats = StandardizationStats.fit(
        select_training_frame(frame, split_spec),
        exog_columns=KITAKYUSHU_EXOG_COLUMNS,
        task_columns=KITAKYUSHU_TASKS,
    )
    standardized = {
        split_name: stats.transform_windows(windows)
        for split_name, windows in raw_windows.items()
    }
    if sample_limit is not None:
        for split_name, windows in standardized.items():
            count = min(int(sample_limit), len(windows["target"]))
            standardized[split_name] = {
                key: value[:count] if key != "target_times" else value[:count]
                for key, value in windows.items()
            }
    return standardized, stats


def build_phase_a_run_plan(
    models: Sequence[Tuple[str, str]] = PHASE_A_MODELS,
    seeds: Sequence[int] = PHASE_A_SEEDS,
) -> Tuple[Dict[str, object], ...]:
    """Return the frozen six-run Phase-A matrix."""

    models = tuple((str(model), str(candidate)) for model, candidate in models)
    seeds = tuple(int(seed) for seed in seeds)
    if models != PHASE_A_MODELS:
        raise ValueError(f"Phase A model matrix is frozen as {PHASE_A_MODELS}")
    if seeds != PHASE_A_SEEDS:
        raise ValueError(f"Phase A seeds are frozen as {PHASE_A_SEEDS}")
    return tuple(
        {
            "protocol": PROTOCOL_NAME,
            "model": model,
            "candidate_id": candidate,
            "seed": seed,
            "run_id": f"{PROTOCOL_NAME}/{model}/{candidate}/seed_{seed}",
        }
        for model, candidate in models
        for seed in seeds
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _hyperparameters(model: str, candidate_id: str) -> dict[str, object]:
    if model == "scheme2r" and candidate_id == "H4":
        return {
            "candidate_id": "H4", "hidden_dim": 32, "kernel_size": 5,
            "dilations": [1, 2, 4], "dropout": 0.1, "learning_rate": 0.0005,
            "scheme2r_kernel_size": 5, "scheme2r_dilations": [1, 2, 4],
            "scheme2r_rank": 8, "scheme2r_gate_hidden_dim": 16,
            "scheme2r_step_embedding_dim": 4, "prediction_head_hidden_dim": 16,
        }
    if model == "dynamic_symmetric" and candidate_id == "H1":
        return {
            "candidate_id": "H1", "hidden_dim": 16, "kernel_size": 5,
            "dilations": [1, 2, 4], "dropout": 0.0,
            "learning_rate": 0.001, "prediction_head_hidden_dim": 16,
        }
    raise ValueError(f"unsupported Phase A candidate: {model}/{candidate_id}")


def _required_artifacts(run_dir: Path) -> Tuple[Path, ...]:
    return tuple(
        run_dir / name
        for name in (
            "best_model.pt",
            "normalization_stats.npz",
            "history.json",
            "metrics_validation.json",
            "predictions_validation.npz",
            "run_manifest.json",
        )
    )


def _complete_and_matching(run_dir: Path) -> bool:
    artifacts = _required_artifacts(run_dir)
    if not all(path.is_file() for path in artifacts):
        return False
    try:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        hashes = manifest["artifact_sha256"]
        return all(hashes.get(path.name) == _sha256(path) for path in artifacts[:-1])
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _run_one(
    run: Mapping[str, object],
    run_dir: Path,
    windows: Mapping[str, Mapping[str, np.ndarray]],
    stats: StandardizationStats,
    training_policy: Mapping[str, object],
) -> dict[str, object]:
    seed = int(run["seed"])
    model_name = str(run["model"])
    candidate_id = str(run["candidate_id"])
    hp = _hyperparameters(model_name, candidate_id)
    trainer_config = TrainerConfig(
        seed=seed,
        torch_threads=TORCH_THREADS,
        learning_rate=float(hp["learning_rate"]),
        weight_decay=float(training_policy["weight_decay"]),
        grad_clip_norm=float(training_policy["gradient_clip_norm"]),
        max_epochs=int(training_policy["max_epochs"]),
        early_stopping_patience=int(training_policy["early_stopping_patience"]),
    )
    model = build_formal_forecasting_model(
        model_name,
        hp,
        exog_dim=len(KITAKYUSHU_EXOG_COLUMNS),
        lookback=LOOKBACK,
        horizon=HORIZON,
        task_count=len(KITAKYUSHU_TASKS),
    )
    train_loader = make_dataloader(
        windows["train"], int(training_policy["batch_size"]), shuffle=True, seed=seed
    )
    validation_loader = make_dataloader(
        windows["validation"], int(training_policy["batch_size"]), shuffle=False
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    history = fit_model(
        model,
        train_loader,
        validation_loader,
        trainer_config,
        run_dir / "best_model.pt",
        input_mode=INPUT_MODE,
    )
    fit_seconds = time.perf_counter() - started
    load_checkpoint(model, run_dir / "best_model.pt", "cpu")
    validation_started = time.perf_counter()
    validation_loss, prediction_std, target_std = evaluate_model(
        model, validation_loader, "cpu", input_mode=INPUT_MODE
    )
    validation_seconds = time.perf_counter() - validation_started
    prediction = stats.inverse_targets(prediction_std)
    target = stats.inverse_targets(target_std)
    stats.save(run_dir / "normalization_stats.npz")
    save_json({"history": history}, run_dir / "history.json")
    np.savez_compressed(
        run_dir / "predictions_validation.npz",
        prediction=prediction,
        target=target,
        prediction_standardized=prediction_std,
        target_standardized=target_std,
        target_times=windows["validation"]["target_times"],
    )
    metrics = regression_metrics(target, prediction, task_names=KITAKYUSHU_TASKS)
    metrics["validation_loss_standardized"] = float(validation_loss)
    save_json(metrics, run_dir / "metrics_validation.json")
    artifacts = _required_artifacts(run_dir)
    manifest: dict[str, object] = {
        "stage": "topology_protocol_pilot_phase_a",
        "status": "passed",
        "protocol": PROTOCOL_NAME,
        "model": model_name,
        "candidate_id": candidate_id,
        "seed": seed,
        "dataset": "kitakyushu_energy_station",
        "years_loaded": list(PHASE_A_YEARS),
        "test_set_accessed": False,
        "future_exogenous_used": False,
        "tasks": list(KITAKYUSHU_TASKS),
        "window": {"lookback": LOOKBACK, "horizon": HORIZON, "output_shape": list(prediction.shape)},
        "sample_counts": {name: int(len(value["target"])) for name, value in windows.items()},
        "parameter_count": int(count_trainable_parameters(model)),
        "fit_seconds": float(fit_seconds),
        "validation_seconds": float(validation_seconds),
        "trainer_config": asdict(trainer_config),
        "git_revision": _git_revision(),
    }
    manifest["artifact_sha256"] = {path.name: _sha256(path) for path in artifacts[:-1]}
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def run_validation_pilot(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    sample_limit: int | None = None,
    max_epochs: int = MAX_EPOCHS,
    resume: bool = False,
) -> dict[str, object]:
    """Run or resume the six-run, validation-only pilot."""

    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    frame, metadata = load_phase_a_frame(data_dir)
    windows, stats = build_train_validation_windows(frame, sample_limit=sample_limit)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    policy = {
        "batch_size": BATCH_SIZE,
        "max_epochs": int(max_epochs),
        "early_stopping_patience": min(EARLY_STOPPING_PATIENCE, max_epochs),
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
        "loss": "SmoothL1Loss",
        "optimizer": "AdamW",
        "device": "cpu",
    }
    results = []
    for run in build_phase_a_run_plan():
        run_dir = output_root / str(run["model"]) / str(run["candidate_id"]) / f"seed_{run['seed']}"
        if run_dir.exists() and any(run_dir.iterdir()):
            if not resume:
                raise FileExistsError(f"run directory exists; use --resume: {run_dir}")
            if not _complete_and_matching(run_dir):
                raise ValueError(f"incomplete or hash-mismatched run cannot be resumed: {run_dir}")
            results.append(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")))
            continue
        results.append(_run_one(run, run_dir, windows, stats, policy))
    manifest = {
        "stage": "topology_protocol_pilot_phase_a",
        "status": "passed",
        "protocol": PROTOCOL_NAME,
        "years_loaded": metadata["years_loaded"],
        "test_set_accessed": False,
        "run_count": len(results),
        "runs": results,
        "sample_counts": {name: int(len(value["target"])) for name, value in windows.items()},
    }
    save_json(manifest, output_root / "phase_a_manifest.json")
    return manifest


__all__ = [
    "FORBIDDEN_PHASE_A_YEARS",
    "PHASE_A_MODELS",
    "PHASE_A_SEEDS",
    "PHASE_A_SPLIT",
    "PHASE_A_YEARS",
    "PROTOCOL_NAME",
    "build_phase_a_run_plan",
    "build_train_validation_windows",
    "load_phase_a_frame",
    "run_validation_pilot",
    "validate_phase_a_years",
]
