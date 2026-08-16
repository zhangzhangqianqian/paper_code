"""Strict registration and alignment of frozen 2020 validation artifacts.

This module is intentionally limited to validation artifacts from the existing
cross-topology protocol.  It never opens a test prediction file and never
copies the large NPZ files into the new protocol directory.
"""

from __future__ import annotations

import hashlib
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from .data_pipeline import save_json
from .protocol_statistics import (
    benjamini_hochberg,
    difference_in_differences_bootstrap,
    paired_protocol_bootstrap,
)


TASK_COUNT = 4
HORIZON = 4
EXPECTED_ORIGINS = 8781
EXPECTED_SEEDS: Tuple[int, ...] = (2026, 2027, 2028)
EXPECTED_MODELS: Tuple[Tuple[str, str], ...] = (
    ("scheme2r", "H4"),
    ("dynamic_symmetric", "H1"),
)
REQUIRED_KEYS = {"prediction", "target", "target_times"}
_PATH_RE = re.compile(
    r"(?:^|[\\/])full[\\/]([^\\/]+)[\\/]([^\\/]+)[\\/]seed_(20\d{2})[\\/]predictions_validation\.npz$",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidationArtifact:
    path: Path
    model: str
    candidate_id: str
    seed: int
    prediction: np.ndarray
    target: np.ndarray
    target_times: np.ndarray
    sha256: str

    @property
    def sample_count(self) -> int:
        return int(self.prediction.shape[0])

    @property
    def target_start(self) -> str:
        return str(self.target_times[0])

    @property
    def target_end(self) -> str:
        return str(self.target_times[-1])

    def subset(self, indices: np.ndarray) -> "ValidationArtifact":
        indices = np.asarray(indices, dtype=np.int64)
        return ValidationArtifact(
            path=self.path,
            model=self.model,
            candidate_id=self.candidate_id,
            seed=self.seed,
            prediction=self.prediction[indices],
            target=self.target[indices],
            target_times=self.target_times[indices],
            sha256=self.sha256,
        )


def _parse_path(path: Path, *, allow_post_path: bool = False) -> tuple[str, str, int]:
    if path.name.lower() != "predictions_validation.npz":
        raise ValueError("only predictions_validation.npz is accepted")
    if "predictions_test" in path.name.lower():
        raise ValueError("test prediction artifacts are forbidden")
    match = _PATH_RE.search(str(path))
    if match is None and allow_post_path:
        match = re.search(
            r"(?:^|[\\/])([^\\/]+)[\\/]([^\\/]+)[\\/]seed_(20\d{2})[\\/]predictions_validation\.npz$",
            str(path),
            re.IGNORECASE,
        )
    if match is None:
        raise ValueError(
            "validation artifact path must match full/<model>/<candidate>/seed_<seed>/"
            "predictions_validation.npz"
        )
    return match.group(1).lower().replace("-", "_"), match.group(2), int(match.group(3))


def load_validation_artifact(
    path: str | Path,
    *,
    expected_model: str | None = None,
    expected_candidate_id: str | None = None,
    expected_seed: int | None = None,
    require_expected_origin_count: bool = False,
    allow_post_path: bool = False,
) -> ValidationArtifact:
    """Load and validate one frozen validation NPZ without accepting test files."""

    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(artifact_path)
    model, candidate_id, seed = _parse_path(artifact_path, allow_post_path=allow_post_path)
    if expected_model is not None and model != expected_model.lower().replace("-", "_"):
        raise ValueError(f"model path mismatch: expected {expected_model}, got {model}")
    if expected_candidate_id is not None and candidate_id != str(expected_candidate_id):
        raise ValueError(f"candidate path mismatch: expected {expected_candidate_id}, got {candidate_id}")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError(f"seed path mismatch: expected {expected_seed}, got {seed}")
    with np.load(artifact_path, allow_pickle=False) as values:
        if not REQUIRED_KEYS.issubset(values.files):
            raise ValueError(f"artifact is missing required keys: {sorted(REQUIRED_KEYS - set(values.files))}")
        prediction = np.asarray(values["prediction"], dtype=np.float32)
        target = np.asarray(values["target"], dtype=np.float32)
        target_times = np.asarray(values["target_times"]).astype("datetime64[ns]")
    if prediction.ndim != 3 or tuple(prediction.shape[1:]) != (HORIZON, TASK_COUNT):
        raise ValueError(f"prediction must have shape [N,{HORIZON},{TASK_COUNT}]")
    if target.shape != prediction.shape:
        raise ValueError("prediction and target shapes must match")
    if target_times.ndim != 1 or len(target_times) != len(prediction):
        raise ValueError("target_times must be one-dimensional and match prediction length")
    if len(target_times) == 0:
        raise ValueError("validation artifact cannot be empty")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    if len(np.unique(target_times)) != len(target_times):
        raise ValueError("target_times must be unique")
    if np.any(np.diff(target_times) <= np.timedelta64(0, "ns")):
        raise ValueError("target_times must be strictly increasing")
    years = target_times.astype("datetime64[Y]").astype(int) + 1970
    if np.any(years != 2020):
        raise ValueError("cross-topology registration accepts only 2020 validation origins")
    if require_expected_origin_count and len(target_times) != EXPECTED_ORIGINS:
        raise ValueError(f"expected {EXPECTED_ORIGINS} validation origins, got {len(target_times)}")
    return ValidationArtifact(
        path=artifact_path,
        model=model,
        candidate_id=candidate_id,
        seed=seed,
        prediction=prediction,
        target=target,
        target_times=target_times,
        sha256=_sha256(artifact_path),
    )


def align_validation_origins(
    left: ValidationArtifact,
    right: ValidationArtifact,
) -> tuple[ValidationArtifact, ValidationArtifact]:
    """Intersect validation origins and verify target equality after alignment."""

    if left.seed != right.seed:
        raise ValueError("cannot align artifacts from different seeds")
    if left.target_times.shape == right.target_times.shape and np.array_equal(left.target_times, right.target_times):
        left_indices = right_indices = np.arange(len(left.target_times), dtype=np.int64)
    else:
        common, left_indices, right_indices = np.intersect1d(
            left.target_times, right.target_times, assume_unique=True, return_indices=True
        )
        if len(common) == 0:
            raise ValueError("validation artifacts have no common origins")
    left_aligned = left.subset(left_indices)
    right_aligned = right.subset(right_indices)
    if not np.array_equal(left_aligned.target_times, right_aligned.target_times):
        raise ValueError("aligned target origins disagree")
    if not np.allclose(left_aligned.target, right_aligned.target, rtol=0.0, atol=1e-3):
        raise ValueError("aligned target arrays disagree")
    return left_aligned, right_aligned


def _source_path(repo_root: Path, model: str, candidate: str, seed: int) -> Path:
    if (model, candidate) == ("scheme2r", "H4"):
        base = repo_root / "frame" / "reports" / "stage7r_3_kitakyushu_formal"
    elif (model, candidate) == ("dynamic_symmetric", "H1"):
        base = repo_root / "frame" / "reports" / "stage7r_7_joint_baselines_formal"
    else:
        raise ValueError(f"unsupported frozen cross-topology source: {model}/{candidate}")
    return base / "full" / model / candidate / f"seed_{seed}" / "predictions_validation.npz"


def register_cross_topology_artifacts(
    repo_root: str | Path,
    output_dir: str | Path,
    *,
    require_expected_origin_count: bool = True,
) -> dict[str, object]:
    """Validate and register the six existing 2020 artifacts."""

    root = Path(repo_root)
    output_root = Path(output_dir)
    artifacts: list[dict[str, object]] = []
    for model, candidate in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            path = _source_path(root, model, candidate, seed)
            artifact = load_validation_artifact(
                path,
                expected_model=model,
                expected_candidate_id=candidate,
                expected_seed=seed,
                require_expected_origin_count=require_expected_origin_count,
                allow_post_path=True,
            )
            artifacts.append({
                "path": str(path.relative_to(root)),
                "model": artifact.model,
                "candidate_id": artifact.candidate_id,
                "seed": artifact.seed,
                "target_start": artifact.target_start,
                "target_end": artifact.target_end,
                "sample_count": artifact.sample_count,
                "sha256": artifact.sha256,
            })
    registry = {
        "stage": "topology_protocol_pilot_task6",
        "protocol": "cross_topology",
        "validation_year": 2020,
        "test_set_accessed": False,
        "artifacts": artifacts,
    }
    save_json(registry, output_root / "cross_topology_artifact_registry.json")
    return registry


def load_registry_artifacts(
    registry_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[tuple[str, str, int], ValidationArtifact]:
    """Load every source named by a previously written registry."""

    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if registry.get("protocol") != "cross_topology" or registry.get("test_set_accessed") is not False:
        raise ValueError("registry is not a sealed, validation-only cross-topology registry")
    root = Path(repo_root)
    loaded: dict[tuple[str, str, int], ValidationArtifact] = {}
    for entry in registry.get("artifacts", []):
        path = root / str(entry["path"])
        artifact = load_validation_artifact(
            path,
            expected_model=str(entry["model"]),
            expected_candidate_id=str(entry["candidate_id"]),
            expected_seed=int(entry["seed"]),
            require_expected_origin_count=True,
        )
        if artifact.sha256 != entry.get("sha256"):
            raise ValueError(f"registry hash mismatch: {path}")
        loaded[(artifact.model, artifact.candidate_id, artifact.seed)] = artifact
    expected = {(m, c, s) for m, c in EXPECTED_MODELS for s in EXPECTED_SEEDS}
    if set(loaded) != expected:
        raise ValueError(f"registry matrix mismatch: expected {sorted(expected)}, got {sorted(loaded)}")
    return loaded


def load_post_protocol_artifacts(
    output_root: str | Path,
    *,
    require_expected_origin_count: bool = False,
) -> dict[tuple[str, str, int], ValidationArtifact]:
    """Load the Phase-A post-protocol validation artifacts only."""

    root = Path(output_root)
    loaded: dict[tuple[str, str, int], ValidationArtifact] = {}
    for model, candidate in EXPECTED_MODELS:
        for seed in EXPECTED_SEEDS:
            path = root / model / candidate / f"seed_{seed}" / "predictions_validation.npz"
            artifact = load_validation_artifact(
                path,
                expected_model=model,
                expected_candidate_id=candidate,
                expected_seed=seed,
                require_expected_origin_count=require_expected_origin_count,
                allow_post_path=True,
            )
            loaded[(model, candidate, seed)] = artifact
    return loaded


def align_protocol_matrix(
    cross: Mapping[tuple[str, str, int], ValidationArtifact],
    post: Mapping[tuple[str, str, int], ValidationArtifact],
) -> tuple[dict[tuple[str, str, int], ValidationArtifact], np.ndarray]:
    """Align all model/seed artifacts to one common 2020 origin axis."""

    keys = sorted(set(cross) & set(post))
    expected = {(m, c, s) for m, c in EXPECTED_MODELS for s in EXPECTED_SEEDS}
    if set(keys) != expected or set(cross) != expected or set(post) != expected:
        raise ValueError("cross and post artifact matrices must contain the frozen six runs")
    common = cross[keys[0]].target_times
    for key in keys:
        common = np.intersect1d(common, cross[key].target_times, assume_unique=True)
        common = np.intersect1d(common, post[key].target_times, assume_unique=True)
    if len(common) == 0:
        raise ValueError("cross and post protocols have no common validation origins")
    aligned: dict[tuple[str, str, int], ValidationArtifact] = {}
    reference_target: np.ndarray | None = None
    for key in keys:
        left, _ = _subset_to_times(cross[key], common)
        right, _ = _subset_to_times(post[key], common)
        if not np.allclose(left.target, right.target, rtol=0.0, atol=1e-3):
            raise ValueError(f"target disagreement between protocols for {key}")
        if reference_target is None:
            reference_target = left.target
        elif not np.allclose(reference_target, left.target, rtol=0.0, atol=1e-3):
            raise ValueError(f"target disagreement across runs for {key}")
        aligned[("cross_topology", *key)] = left
        aligned[("post_ge_regular_operation", *key)] = right
    assert reference_target is not None
    return aligned, common


def _subset_to_times(artifact: ValidationArtifact, times: np.ndarray) -> tuple[ValidationArtifact, np.ndarray]:
    positions = np.searchsorted(artifact.target_times, times)
    if np.any(positions >= len(artifact.target_times)) or not np.array_equal(artifact.target_times[positions], times):
        raise ValueError("requested common origins are not present in artifact")
    return artifact.subset(positions), positions


def _write_csv(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric_rows(prediction: np.ndarray, target: np.ndarray, protocol: str, model: str, seed: str) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    tasks = ("electricity", "cooling", "heating", "gas")
    metrics = ("MAE", "RMSE", "WAPE", "MAPE")
    overall = []
    per_task = []
    per_horizon = []
    task_horizon = []
    for metric in metrics:
        overall_value = float(np.mean([
            _single_metric(prediction[:, :, task_index], target[:, :, task_index], metric)
            for task_index in range(len(tasks))
        ]))
        overall.append({"protocol": protocol, "model": model, "seed": seed, "metric": metric, "value": overall_value})
    for task_index, task in enumerate(tasks):
        for metric in metrics:
            value = _single_metric(prediction[:, :, task_index], target[:, :, task_index], metric)
            per_task.append({"protocol": protocol, "model": model, "seed": seed, "task": task, "metric": metric, "value": value})
    for horizon_index in range(HORIZON):
        for metric in metrics:
            value = _single_metric(prediction[:, horizon_index, :], target[:, horizon_index, :], metric)
            per_horizon.append({"protocol": protocol, "model": model, "seed": seed, "horizon": horizon_index + 1, "metric": metric, "value": value})
        for task_index, task in enumerate(tasks):
            for metric in metrics:
                value = _single_metric(prediction[:, horizon_index, task_index], target[:, horizon_index, task_index], metric)
                task_horizon.append({"protocol": protocol, "model": model, "seed": seed, "task": task, "horizon": horizon_index + 1, "metric": metric, "value": value})
    return overall, per_task, per_horizon, task_horizon


def _single_metric(prediction: np.ndarray, target: np.ndarray, metric: str) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if metric == "MAE":
        return float(np.mean(np.abs(error)))
    if metric == "RMSE":
        return float(np.sqrt(np.mean(error ** 2)))
    if metric == "WAPE":
        return float(np.sum(np.abs(error)) / max(float(np.sum(np.abs(target))), 1e-12) * 100.0)
    return float(np.mean(np.abs(error) / np.maximum(np.abs(target), 1e-6)) * 100.0)


def _input_hashes(artifacts: Mapping[tuple[str, str, int], ValidationArtifact]) -> dict[str, str]:
    return {str(key): value.sha256 for key, value in artifacts.items()}


def compare_validation_protocols(
    cross_registry_path: str | Path,
    post_output_root: str | Path,
    output_dir: str | Path,
    *,
    repo_root: str | Path,
    bootstrap_replicates: int = 2000,
    block_length: int = 168,
    seed: int = 2026,
) -> dict[str, object]:
    """Generate validation-only comparison tables and bootstrap contrasts."""

    cross = load_registry_artifacts(cross_registry_path, repo_root=repo_root)
    post = load_post_protocol_artifacts(post_output_root)
    aligned, common_times = align_protocol_matrix(cross, post)
    root = Path(output_dir)
    all_overall: list[dict[str, object]] = []
    all_task: list[dict[str, object]] = []
    all_horizon: list[dict[str, object]] = []
    all_task_horizon: list[dict[str, object]] = []
    for protocol in ("cross_topology", "post_ge_regular_operation"):
        for model, candidate in EXPECTED_MODELS:
            for seed_value in EXPECTED_SEEDS:
                artifact = aligned[(protocol, model, candidate, seed_value)]
                rows = _metric_rows(artifact.prediction, artifact.target, protocol, model, str(seed_value))
                all_overall.extend(rows[0]); all_task.extend(rows[1]); all_horizon.extend(rows[2]); all_task_horizon.extend(rows[3])
    _write_csv(all_overall, root / "validation_overall.csv")
    _write_csv(all_task, root / "validation_per_task.csv")
    _write_csv(all_horizon, root / "validation_per_horizon.csv")
    _write_csv(all_task_horizon, root / "validation_task_horizon.csv")

    target = aligned[("cross_topology", "scheme2r", "H4", EXPECTED_SEEDS[0])].target
    seed_means: dict[tuple[str, str], np.ndarray] = {}
    for protocol in ("cross_topology", "post_ge_regular_operation"):
        for model, candidate in EXPECTED_MODELS:
            seed_means[(protocol, model)] = np.mean(
                np.stack([aligned[(protocol, model, candidate, seed_value)].prediction for seed_value in EXPECTED_SEEDS]), axis=0
            )
    effect_rows: list[dict[str, object]] = []
    task_names = ("electricity", "cooling", "heating", "gas")
    for model in ("scheme2r", "dynamic_symmetric"):
        left = seed_means[("cross_topology", model)]
        right = seed_means[("post_ge_regular_operation", model)]
        for metric in ("MAE", "RMSE", "WAPE"):
            estimate = paired_protocol_bootstrap(left, right, target, metric=metric, block_length=block_length, replicates=bootstrap_replicates, seed=seed)
            effect_rows.append({"scope": "overall", "task": "all", "horizon": "all", "model": model, "metric": metric, **_estimate_dict(estimate)})
        for task_index, task in enumerate(task_names):
            for metric in ("MAE", "RMSE", "WAPE"):
                estimate = paired_protocol_bootstrap(left[:, :, [task_index]], right[:, :, [task_index]], target[:, :, [task_index]], metric=metric, block_length=block_length, replicates=bootstrap_replicates, seed=seed + task_index)
                effect_rows.append({"scope": "task", "task": task, "horizon": "all", "model": model, "metric": metric, **_estimate_dict(estimate)})
    p_values = np.asarray([float(row["p_value"]) for row in effect_rows], dtype=np.float64)
    adjusted = np.full_like(p_values, np.nan)
    finite_mask = np.isfinite(p_values)
    if finite_mask.any():
        adjusted[finite_mask] = benjamini_hochberg(p_values[finite_mask])
    for row, value in zip(effect_rows, adjusted):
        row["p_value_bh"] = float(value) if np.isfinite(value) else None
    _write_csv(effect_rows, root / "protocol_effect_bootstrap.csv")

    did_rows: list[dict[str, object]] = []
    prediction_map = {
        ("cross_topology", "scheme2r"): seed_means[("cross_topology", "scheme2r")],
        ("cross_topology", "dynamic_symmetric"): seed_means[("cross_topology", "dynamic_symmetric")],
        ("post_ge_regular_operation", "scheme2r"): seed_means[("post_ge_regular_operation", "scheme2r")],
        ("post_ge_regular_operation", "dynamic_symmetric"): seed_means[("post_ge_regular_operation", "dynamic_symmetric")],
    }
    for metric in ("MAE", "RMSE", "WAPE"):
        estimate = difference_in_differences_bootstrap(prediction_map, target, metric=metric, block_length=block_length, replicates=bootstrap_replicates, seed=seed)
        did_rows.append({"scope": "overall", "task": "all", "horizon": "all", "metric": metric, **_estimate_dict(estimate)})
    _write_csv(did_rows, root / "model_gap_difference_in_differences.csv")

    reliability_rows: list[dict[str, object]] = []
    for protocol, source in (("cross_topology", cross), ("post_ge_regular_operation", post)):
        for key, artifact in source.items():
            model, candidate, seed_value = key
            manifest_path = artifact.path.parent / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            reliability_rows.append({"protocol": protocol, "model": model, "candidate_id": candidate, "seed": seed_value, "status": manifest.get("status", "artifact_only"), "sample_count": artifact.sample_count, "finite": bool(np.isfinite(artifact.prediction).all() and np.isfinite(artifact.target).all()), "test_set_accessed_for_analysis": False})
    _write_csv(reliability_rows, root / "training_reliability.csv")
    manifest = {
        "stage": "topology_protocol_pilot_task7",
        "protocols": ["cross_topology", "post_ge_regular_operation"],
        "validation_year": 2020,
        "test_set_accessed": False,
        "common_origin_count": int(len(common_times)),
        "bootstrap": {"replicates": int(bootstrap_replicates), "block_length": int(block_length), "seed": int(seed)},
        "input_artifact_hashes": {"cross": _input_hashes(cross), "post": _input_hashes(post)},
    }
    save_json(manifest, root / "validation_comparison_manifest.json")
    return manifest


def _estimate_dict(estimate) -> dict[str, object]:
    return {"estimate_post_minus_cross": float(estimate.estimate), "ci_low": float(estimate.ci_low), "ci_high": float(estimate.ci_high), "p_value": float(estimate.p_value), "relative_change_percent": float(estimate.relative_change), "n_origins": int(estimate.n_origins), "replicates": int(estimate.replicates), "block_length": int(estimate.block_length)}


def decide_topology_branch(
    protocol_effect_rows: Sequence[Mapping[str, object]],
    did_rows: Sequence[Mapping[str, object]],
    *,
    reliability_ok: bool = True,
) -> str:
    """Apply the frozen three-way branch rule using validation-only evidence."""

    if not reliability_ok:
        return "pilot_invalid"
    did_mae = next((row for row in did_rows if row.get("metric") == "MAE" and row.get("scope") == "overall"), None)
    did_significant = bool(
        did_mae
        and (
            float(did_mae["ci_low"]) > 0.0
            or float(did_mae["ci_high"]) < 0.0
        )
    )
    core_tasks = {"electricity", "cooling", "heating"}
    core = [row for row in protocol_effect_rows if row.get("scope") == "task" and row.get("metric") == "MAE" and row.get("task") in core_tasks]
    consistent_tasks = 0
    for task in core_tasks:
        task_rows = [row for row in core if row.get("task") == task and float(row.get("p_value_bh", 1.0)) < 0.05]
        signs = {np.sign(float(row["estimate_post_minus_cross"])) for row in task_rows}
        if len(task_rows) == 2 and len(signs) == 1:
            consistent_tasks += 1
    if did_significant or consistent_tasks >= 2:
        return "core_prediction_changed"
    gas = [row for row in protocol_effect_rows if row.get("scope") == "task" and row.get("metric") == "MAE" and row.get("task") == "gas" and float(row.get("p_value_bh", 1.0)) < 0.05]
    if gas:
        return "gas_only_changed"
    return "core_conclusion_stable"


def freeze_topology_branch(
    branch: str,
    comparison_manifest: Mapping[str, object],
    output_dir: str | Path,
    *,
    git_revision: str = "unavailable",
    phase_b_matrix: Mapping[str, object] | None = None,
    input_hashes: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write an immutable validation branch freeze (or pilot-invalid marker)."""

    if branch not in {"core_prediction_changed", "gas_only_changed", "core_conclusion_stable", "pilot_invalid"}:
        raise ValueError(f"unsupported topology branch: {branch}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    filename = "pilot_invalid.json" if branch == "pilot_invalid" else "topology_branch_freeze.json"
    path = root / filename
    payload = {
        "stage": "topology_protocol_pilot_task7",
        "branch": branch,
        "decision_rule_version": "topology_protocol_pilot_v1",
        "validation_year": 2020,
        "test_year_accessed": False,
        "git_revision": git_revision,
        "comparison_manifest": dict(comparison_manifest),
        "phase_b_run_matrix": dict(phase_b_matrix or {}),
        "input_hashes": dict(input_hashes or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"immutable branch freeze already differs: {path}")
    path.write_text(encoded, encoding="utf-8")
    return payload


def validate_phase_b_result_artifacts(
    branch_freeze_path: str | Path,
    prediction_root: str | Path,
    *,
    scheduling_manifest_path: str | Path | None = None,
) -> dict[str, object]:
    """Validate the completed, branch-locked 2021 prediction/scheduling set."""

    from .topology_scheduling_runner import build_scheduling_run_plan, validate_prediction_artifact
    from .topology_test_runner import build_phase_b_run_plan, load_branch_freeze

    freeze = load_branch_freeze(branch_freeze_path)
    forecast_plan = build_phase_b_run_plan(freeze)
    unique_forecasts = {(row["model"], row["candidate_id"], row["seed"]) for row in forecast_plan}
    root = Path(prediction_root)
    prediction_records = []
    for model, candidate, seed in sorted(unique_forecasts):
        run_dir = root / str(model) / str(candidate) / f"seed_{seed}"
        prediction_path = run_dir / "predictions_test.npz"
        artifact = validate_prediction_artifact(prediction_path)
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("branch_frozen_before_test") is not True:
            raise ValueError(f"branch freeze does not precede test: {manifest_path}")
        if manifest.get("test_used_for_selection") is not False:
            raise ValueError(f"test artifact was used for selection: {manifest_path}")
        prediction_records.append({
            "model": model, "candidate_id": candidate, "seed": int(seed),
            "path": str(prediction_path), "sha256": artifact["sha256"],
            "sample_count": int(len(artifact["prediction"])),
        })
    expected_schedule_count = len(build_scheduling_run_plan(freeze))
    schedule_count = None
    if scheduling_manifest_path is not None:
        schedule_path = Path(scheduling_manifest_path)
        if not schedule_path.is_file():
            raise FileNotFoundError(schedule_path)
        schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
        if schedule.get("branch_frozen_before_scheduling") is not True:
            raise ValueError("scheduling manifest does not reference a prior branch freeze")
        if schedule.get("lp_equations_modified") is not False:
            raise ValueError("topology routing changed the frozen LP equations")
        schedule_count = len(schedule.get("runs", []))
        if schedule_count != expected_schedule_count:
            raise ValueError(
                f"scheduling run count mismatch: expected {expected_schedule_count}, got {schedule_count}"
            )
    return {
        "stage": "topology_protocol_pilot_task11",
        "status": "passed",
        "branch": freeze["branch"],
        "branch_freeze_precedes_test": True,
        "test_tuning_detected": False,
        "test_year": 2021,
        "forecast_run_count": len(prediction_records),
        "expected_forecast_run_count": len(unique_forecasts),
        "scheduling_run_count": schedule_count,
        "expected_scheduling_run_count": expected_schedule_count,
        "prediction_artifacts": prediction_records,
        "test_set_accessed": True,
    }


__all__ = [
    "EXPECTED_MODELS",
    "EXPECTED_ORIGINS",
    "EXPECTED_SEEDS",
    "ValidationArtifact",
    "align_validation_origins",
    "align_protocol_matrix",
    "compare_validation_protocols",
    "decide_topology_branch",
    "freeze_topology_branch",
    "load_validation_artifact",
    "load_post_protocol_artifacts",
    "load_registry_artifacts",
    "register_cross_topology_artifacts",
    "validate_phase_b_result_artifacts",
]
